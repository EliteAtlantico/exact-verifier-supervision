"""exp_thinking_dataset.py -- Phase 0 of the THINKING-vs-DATA study ($0, no torch, no API).

Builds the training/eval corpora for "is training on AI thinking more efficient than training on
data points?" for three exact-gate tasks:

  prime -- 4-digit primality. Thinking = trial-division narrative over primes up to isqrt(n).
  valid -- propositional argument validity (Z3 gate via newaxiom.contradiction). Thinking = tableau
           narrative: assume premises true + conclusion false; VALID -> the Z3 unsat core is the
           contradiction; INVALID -> the Z3 satisfying assignment is the countermodel.
  div7  -- divisibility by 7. Thinking = digit-by-digit (10*r + d) mod 7 reduction; trace length is
           UNIFORM by construction (label-independent). The prior GATE_FINETUNE_STUDY proved answer
           labels teach div7 NOTHING -- the H4 "algorithm unlock" test.

Arms materialized per task (identical user prompt everywhere; only completions differ):
  A = "Answer: Yes/No"                      (data points)
  B = deterministic exact trace + answer    (thinking)
  C = scrambled control: traces PERMUTED ACROSS EXAMPLES first (so trace length cannot leak the
      label -- composites/invalids have shorter honest traces than primes/valids), then token-
      shuffled within each trace (seeded); the final "Answer: X" line keeps the example's TRUE
      label. Corpus-wide token multiset identical to B by construction.
  D = wrong-but-plausible thinking: a fluent trace from a DIFFERENT instance + the correct answer.

SELF-CHECKS (abort on failure -- an exact study never ships a corrupted corpus):
  * every B trace's final answer equals the exact-gate label (100%)
  * no prompt contains an answer leak ("Answer:")
  * test/train pools problem-disjoint; classes balanced
  * C preserves B's corpus-wide whitespace-token count exactly (same multiset, permuted)

Also freezes the PRE-REGISTRATION section of results/THINKING_VS_DATA.md (written before any GPU
run; never edited afterward).

writes: results/thinking_vs_data/datasets.json
"""
from __future__ import annotations

import json
import math
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from axuniv.newaxiom.contradiction import check_consistency, _Z3_OK  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "thinking_vs_data")
LETTERS = list(string.ascii_uppercase)
N_TEST_PER_CLASS = 120
SEED = 20260803

SUFFIX = " End your reply with 'Answer: Yes' or 'Answer: No'."


# --------------------------------------------------------------------------- helpers
def _small_primes(limit: int):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


_P100 = _small_primes(100)                       # primes up to 100 cover isqrt of any 4-digit n


def _is_prime4(n: int) -> bool:
    r = int(math.isqrt(n))
    for p in _P100:
        if p > r:
            return True
        if n % p == 0:
            return False
    return True


# --------------------------------------------------------------------------- task: prime
def prime_examples(rng):
    """4-digit primality with trial-division thinking. Exact by construction ($0 gate inline)."""
    nums = list(range(1000, 10000))
    rng.shuffle(nums)
    out = []
    for n in nums:
        label = "Yes" if _is_prime4(n) else "No"
        prompt = f"Is {n} a prime number?{SUFFIX}"
        r = int(math.isqrt(n))
        steps = []
        divisor = None
        for p in _P100:
            if p > r:
                break
            m = n % p
            steps.append(f"{n} mod {p} = {m}.")
            if m == 0:
                divisor = p
                break
        if divisor is not None:
            trace = (f"Check prime divisors up to isqrt({n}) = {r}. " + " ".join(steps) +
                     f" {divisor} divides {n}, so {n} is composite.")
        else:
            trace = (f"Check prime divisors up to isqrt({n}) = {r}. " + " ".join(steps) +
                     f" No prime divisor up to {r}, so {n} is prime.")
        out.append({"id": f"prime-{n}", "prompt": prompt, "label": label, "trace": trace})
    return out


# --------------------------------------------------------------------------- task: div7
def div7_examples(rng):
    """Divisible-by-7 with the digit-reduction algorithm. UNIFORM trace length by construction."""
    nums = list(range(1000, 10000))
    rng.shuffle(nums)
    out = []
    for n in nums:
        label = "Yes" if n % 7 == 0 else "No"
        prompt = f"Is {n} divisible by 7?{SUFFIX}"
        r = 0
        steps = []
        for d in str(n):
            nr = (10 * r + int(d)) % 7
            steps.append(f"r = (10*{r} + {d}) mod 7 = {nr}.")
            r = nr
        tail = (f"Final remainder 0, so {n} is divisible by 7." if r == 0 else
                f"Final remainder {r} is not 0, so {n} is not divisible by 7.")
        trace = f"Process the digits of {n} keeping a running remainder mod 7. " + " ".join(steps) + " " + tail
        out.append({"id": f"div7-{n}", "prompt": prompt, "label": label, "trace": trace})
    return out


# --------------------------------------------------------------------------- task: valid (Z3)
VALID_SCHEMAS = [
    ("modus_ponens", ["{A} -> {B}", "{A}"], "{B}", ["if {A} then {B}", "{A}"], "{B}", 2),
    ("modus_tollens", ["{A} -> {B}", "~{B}"], "~{A}", ["if {A} then {B}", "not {B}"], "not {A}", 2),
    ("hyp_syllogism", ["{A} -> {B}", "{B} -> {C}"], "{A} -> {C}",
     ["if {A} then {B}", "if {B} then {C}"], "if {A} then {C}", 3),
    ("disj_syllogism", ["{A} | {B}", "~{A}"], "{B}", ["{A} or {B}", "not {A}"], "{B}", 2),
    ("constr_dilemma", ["({A} -> {B}) & ({C} -> {D})", "{A} | {C}"], "{B} | {D}",
     ["if {A} then {B}, and if {C} then {D}", "{A} or {C}"], "{B} or {D}", 4),
    ("contraposition", ["{A} -> {B}"], "~{B} -> ~{A}", ["if {A} then {B}"], "if not {B} then not {A}", 2),
]
INVALID_SCHEMAS = [
    ("affirm_consequent", ["{A} -> {B}", "{B}"], "{A}", ["if {A} then {B}", "{B}"], "{A}", 2),
    ("deny_antecedent", ["{A} -> {B}", "~{A}"], "~{B}", ["if {A} then {B}", "not {A}"], "not {B}", 2),
    ("affirm_disjunct", ["{A} | {B}", "{A}"], "~{B}", ["{A} or {B}", "{A}"], "not {B}", 2),
    ("fake_chain", ["{A} -> {B}", "{C} -> {D}"], "{A} -> {D}",
     ["if {A} then {B}", "if {C} then {D}"], "if {A} then {D}", 4),
    ("converse_error", ["{A} -> {B}"], "{B} -> {A}", ["if {A} then {B}"], "if {B} then {A}", 2),
    ("illicit_conj", ["{A} | {B}"], "{A} & {B}", ["{A} or {B}"], "{A} and {B}", 2),
]


def _subst(tpl, mapping):
    for k, v in mapping.items():
        tpl = tpl.replace("{%s}" % k, v)
    return tpl


def valid_examples(rng, need: int):
    """Argument validity; the SCHEMA proposes, the Z3 gate DISPOSES the label AND supplies the
    thinking evidence (unsat core = the contradiction; satisfying model = the countermodel)."""
    out, seen = [], set()
    schemas = [("valid", s) for s in VALID_SCHEMAS] + [("invalid", s) for s in INVALID_SCHEMAS]
    tries = 0
    while len(out) < need and tries < 500_000:
        tries += 1
        _tag, (name, pf, cf, pt, ct, natoms) = rng.choice(schemas)
        atoms = rng.sample(LETTERS, natoms)
        mapping = {k: atoms[i] for i, k in enumerate(["A", "B", "C", "D"][:natoms])}
        prems = "; ".join("(%d) %s" % (i + 1, _subst(t, mapping)) for i, t in enumerate(pt))
        concl = _subst(ct, mapping)
        prompt = (f"Premises: {prems}. Conclusion: {concl}. Does the conclusion logically follow "
                  f"from the premises (is the argument valid)?{SUFFIX}")
        if prompt in seen:
            continue
        seen.add(prompt)
        clauses = [("p%d" % i, _subst(f, mapping)) for i, f in enumerate(pf)]
        neg_c = "~(%s)" % _subst(cf, mapping)
        r = check_consistency(clauses + [("negC", neg_c)])
        if r["status"] == "VERIFIED":                       # UNSAT -> valid
            label = "Yes"
            core = (r.get("receipt") or {}).get("unsat_core") or []
            core_txt = ", ".join(
                (c.get("formula") if isinstance(c, dict) else str(c)) for c in core) or "the premises"
            trace = (f"Assume every premise is true and the conclusion false, i.e. {neg_c} holds. "
                     f"Together these force a contradiction: {{{core_txt}}} cannot all hold at once. "
                     f"Since assuming the conclusion false is impossible, the conclusion follows.")
        elif r["status"] == "ABSTAIN":                      # SAT -> countermodel -> invalid
            label = "No"
            model = (r.get("receipt") or {}).get("model") or {}
            if isinstance(model, dict) and model:
                assign = ", ".join(f"{k}={'true' if v else 'false'}" for k, v in sorted(model.items()))
            else:
                assign = "an assignment exists"
            trace = (f"Assume every premise is true and the conclusion false. That is satisfiable: "
                     f"set {assign}. Under this assignment all premises hold while the conclusion is "
                     f"false, a countermodel, so the argument is invalid.")
        else:
            continue
        out.append({"id": f"valid-{len(out)}-{name}", "prompt": prompt, "label": label,
                    "trace": trace, "schema": name})
    return out


# --------------------------------------------------------------------------- arm materialization
def _tok(s: str) -> int:
    return len(s.split())


def build_arms(examples, rng):
    """From gate-labeled traced examples build A/B/C/D completion corpora."""
    A = [{"id": e["id"], "prompt": e["prompt"], "completion": f"Answer: {e['label']}"}
         for e in examples]
    B = [{"id": e["id"], "prompt": e["prompt"],
          "completion": f"{e['trace']}\nAnswer: {e['label']}"} for e in examples]
    # C: cross-example PERMUTATION of traces first (kills length<->label), then in-trace token
    # shuffle (kills content), answer line keeps the example's own true label.
    perm = list(range(len(examples)))
    rng.shuffle(perm)
    C = []
    for i, e in enumerate(examples):
        donor = examples[perm[i]]["trace"].split()
        rng.shuffle(donor)
        C.append({"id": e["id"], "prompt": e["prompt"],
                  "completion": " ".join(donor) + f"\nAnswer: {e['label']}"})
    # D: a FLUENT trace from a different instance (not shuffled) + the correct answer.
    shift = len(examples) // 2
    D = []
    for i, e in enumerate(examples):
        donor = examples[(i + shift) % len(examples)]["trace"]
        D.append({"id": e["id"], "prompt": e["prompt"],
                  "completion": donor + f"\nAnswer: {e['label']}"})
    return {"A": A, "B": B, "C": C, "D": D}


def balanced_split(examples, rng, n_test_per_class, max_train_per_class):
    yes = [e for e in examples if e["label"] == "Yes"]
    no = [e for e in examples if e["label"] == "No"]
    rng.shuffle(yes)
    rng.shuffle(no)
    test = yes[:n_test_per_class] + no[:n_test_per_class]
    train = (yes[n_test_per_class:n_test_per_class + max_train_per_class] +
             no[n_test_per_class:n_test_per_class + max_train_per_class])
    rng.shuffle(test)
    rng.shuffle(train)
    return train, test


PREREG = """# THINKING vs DATA -- pre-registration (frozen BEFORE any GPU run; never edited)

Hypothesis (user's): training on (problem -> thinking -> answer) is more efficient than training on
(problem -> answer). "The thinking itself is the data."

Arms: A answers-only; B deterministic exact-trace thinking; C length-matched scrambled-thinking
control (cross-example permutation then token shuffle; corpus token multiset identical to B);
A-tok answers-only at B@180's measured token budget; B-orn ornith:9b open-source thinking (STaR
rejection-sampled, coverage reported); B-mv thinking via multiverse.simulate_conversations with a
local reasoner; D wrong-but-plausible thinking (conditional: only where B beats both A and C).
Base model Qwen/Qwen2.5-1.5B-Instruct, LoRA r=8 a=16 targets q/k/v/o_proj, lr 2e-4, 3 epochs.
Tasks (all three reported; no averaging that hides a split): prime (4-digit), valid (Z3), div7.
Test = 240 problem-disjoint items/task (120/class), identical across arms; grading = exact gate on
the extracted final answer only (one parser for all arms).

Pre-registered endpoints:
- H1 matched examples: acc(B) > acc(A) at n=180 per task; paired McNemar on shared items; Holm
  correction across the 3 tasks.
- H2 content vs tokens: acc(B) > acc(C) at n=180. B ~= C means tokens, not thinking.
- H3 matched training tokens: accuracy at equal cumulative training tokens (B@60 vs A@540 points +
  one explicit A-tok run per task; actual token counts recorded) and matched wall-clock.
- H4 algorithm unlock: div7, where answer labels are proven null (GATE_FINETUNE_STUDY: 51.0% vs
  base 53.3%, shuffled 52.5%). B-trained div7 >= ~60% with A at chance = thinking teaches what
  labels cannot.
- Secondary: B-orn vs B-det; B-mv pilot.

Pre-declared readings: B>A and B>C = supported. B>A but B~=C = tokens not thinking; strong form
unsupported. B~=A = honest negative. Efficiency claims require the token-matched win (H3), not just
the example-matched one. Nulls and reversals get equal prominence. Adapters are tier `empirical`,
never `verified`.

(Results are appended BELOW this line after the grid completes; this section is never edited.)
---
"""


def main():
    assert _Z3_OK, "z3 unavailable; the valid task cannot be gate-labeled"
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = {}
    spec = {
        "prime": (prime_examples(rng), 660),      # >= A-tok headroom (941 primes exist; composites plenty)
        "div7": (div7_examples(rng), 700),
        "valid": (valid_examples(rng, 2600), 800),
    }
    for task, (examples, max_train) in spec.items():
        train, test = balanced_split(examples, rng, N_TEST_PER_CLASS, max_train)
        arms = build_arms(train, rng)

        # ---- self-checks (abort on failure) ----
        for e in train + test:
            assert e["prompt"].count("Answer:") == 0 or e["prompt"].endswith(SUFFIX), \
                f"answer leak in prompt: {e['id']}"
        for b in arms["B"]:
            src = next(e for e in train if e["id"] == b["id"])
            assert b["completion"].rstrip().endswith(f"Answer: {src['label']}"), \
                f"trace/label mismatch: {b['id']}"
        tok_b = sum(_tok(x["completion"]) for x in arms["B"])
        tok_c = sum(_tok(x["completion"]) for x in arms["C"])
        assert tok_b == tok_c, f"{task}: C token count {tok_c} != B {tok_b}"
        test_ids = {e["id"] for e in test}
        assert not test_ids & {e["id"] for e in train}, f"{task}: split not disjoint"
        ys = sum(1 for e in test if e["label"] == "Yes")
        assert ys == N_TEST_PER_CLASS, f"{task}: test imbalance {ys}"

        tasks[task] = {
            "test": [{"id": e["id"], "prompt": e["prompt"], "label": e["label"]} for e in test],
            "pools": arms,
            "meta": {
                "n_train_pool": len(train), "n_test": len(test),
                "ws_tokens": {a: sum(_tok(x["completion"]) for x in arms[a]) for a in arms},
                "mean_trace_ws_tokens": round(sum(_tok(e["trace"]) for e in train) / len(train), 1),
            },
        }
        print(f"{task}: pool={len(train)} test={len(test)} "
              f"mean_trace_ws_tok={tasks[task]['meta']['mean_trace_ws_tokens']} "
              f"B_ws_tok={tasks[task]['meta']['ws_tokens']['B']}")

    path = os.path.join(OUT_DIR, "datasets.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "tasks": tasks}, f)
    print("WROTE", path, f"({os.path.getsize(path) // 1024} KB)")

    # freeze the preregistration (write-once)
    md = os.path.join(os.path.dirname(__file__), "..", "results", "THINKING_VS_DATA.md")
    if not os.path.exists(md):
        with open(md, "w", encoding="utf-8") as f:
            f.write(PREREG)
        print("FROZE preregistration ->", md)
    else:
        print("preregistration already frozen; NOT rewriting", md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
