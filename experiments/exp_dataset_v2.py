"""exp_dataset_v2.py -- v2 corpora for the exact-verifier-supervision paper ($0, no torch, no API).

Writes results/v2/datasets_v2.json: ONE file serving every task.

  prime, valid, div7  COPIED UNCHANGED from results/thinking_vs_data/datasets.json (test, pools, meta).
                      div7 additionally gets pool "Bprime" (answer THEN trace; same ids/order/labels as B)
                      and a side table "step_truth" (existing entries are not touched).
  div2, div3, div11, div13
                      "Is N divisible by d?" for 4-digit N; trace = div7's long-division trace with 7 -> d
                      (running remainder after each digit). Arms A, B, C.
  div7_6d             6-digit N, div7 trace style (6 steps). Arms A, B, C.
  prime_hard          TEST ONLY, same prompt as prime: 120 primes + 120 odd composites with no prime factor
                      <= 7, 4-digit, disjoint from the prime train pool AND the prime test set. Every item
                      satisfies the shortcut rule "odd and no factor <= 7", so that rule scores exactly 50%.

Conventions mirrored from exp_thinking_dataset.py: SUFFIX, prompt wording, trace wording, B completion =
trace + "\\nAnswer: X", A = "Answer: X", C = cross-example trace permutation then in-trace whitespace-token
shuffle with the example's own true answer line (corpus token multiset identical to B).

Per-step ground truth (for p^k scoring): tasks[t]["step_truth"][id] = {"n", "d", "k", "rems"} for every test
and pool item of every div task (rems[i] = remainder after digit i+1). New-task test entries and B entries
also carry "k" and "rems" inline.

Fixed few-shot demos (for --prompt-mode fewshot4) live at top level: fewshot4[t] = 4 B-pool ids of task t,
order Yes, No, No, Yes.

Self-checks abort on failure. Deterministic (fixed integer seeds; Python's Mersenne Twister).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "results" / "thinking_vs_data" / "datasets.json"
OUT = REPO / "results" / "v2" / "datasets_v2.json"

SUFFIX = " End your reply with 'Answer: Yes' or 'Answer: No'."   # identical to exp_thinking_dataset.py
N_TEST_PER_CLASS = 120
N_TRAIN_PER_CLASS = 540                                            # pool = 1080 (n=540 draws possible)
SEED_V2 = 20260925
TASK_SEEDS = {"div2": 1, "div3": 2, "div11": 3, "div13": 4, "div7_6d": 5, "prime_hard": 6, "bprime": 7}
DIV_TASKS = {  # task -> (d, lo, hi)
    "div2": (2, 1000, 9999),
    "div3": (3, 1000, 9999),
    "div11": (11, 1000, 9999),
    "div13": (13, 1000, 9999),
    "div7_6d": (7, 100000, 999999),
}
COPIED = ("prime", "valid", "div7")


def _rng(name: str) -> random.Random:
    return random.Random(SEED_V2 * 100 + TASK_SEEDS[name])


def _tok(s: str) -> int:
    return len(s.split())


# --------------------------------------------------------------------------- div-d (mirrors div7_examples)
def div_trace(n: int, d: int):
    """Exactly div7_examples' trace with 7 replaced by d. Returns (trace, rems)."""
    r = 0
    steps, rems = [], []
    for ch in str(n):
        nr = (10 * r + int(ch)) % d
        steps.append(f"r = (10*{r} + {ch}) mod {d} = {nr}.")
        rems.append(nr)
        r = nr
    tail = (f"Final remainder 0, so {n} is divisible by {d}." if r == 0 else
            f"Final remainder {r} is not 0, so {n} is not divisible by {d}.")
    trace = f"Process the digits of {n} keeping a running remainder mod {d}. " + " ".join(steps) + " " + tail
    return trace, rems


def div_example(task: str, n: int, d: int):
    trace, rems = div_trace(n, d)
    return {"id": f"{task}-{n}", "n": n, "d": d, "prompt": f"Is {n} divisible by {d}?{SUFFIX}",
            "label": "Yes" if n % d == 0 else "No", "trace": trace, "k": len(rems), "rems": rems}


def div_split(task: str, d: int, lo: int, hi: int, rng: random.Random):
    """Balanced, problem-disjoint test (120/class) and train pool (540/class)."""
    need = N_TEST_PER_CLASS + N_TRAIN_PER_CLASS
    yes, no = [], []
    if hi - lo < 20000:                                   # 4-digit: enumerate everything, then shuffle
        nums = list(range(lo, hi + 1))
        rng.shuffle(nums)
        for n in nums:
            (yes if n % d == 0 else no).append(n)
    else:                                                 # 6-digit: rejection-sample distinct numbers
        seen = set()
        while len(yes) < need or len(no) < need:
            n = rng.randint(lo, hi)
            if n in seen:
                continue
            seen.add(n)
            bucket = yes if n % d == 0 else no
            if len(bucket) < need:
                bucket.append(n)
    assert len(yes) >= need and len(no) >= need, f"{task}: not enough items ({len(yes)} yes, {len(no)} no)"
    test = yes[:N_TEST_PER_CLASS] + no[:N_TEST_PER_CLASS]
    train = yes[N_TEST_PER_CLASS:need] + no[N_TEST_PER_CLASS:need]
    rng.shuffle(test)
    rng.shuffle(train)
    return ([div_example(task, n, d) for n in train], [div_example(task, n, d) for n in test])


def build_arms(examples, rng):
    """A/B/C exactly as exp_thinking_dataset.build_arms (D omitted). B carries k/rems inline."""
    A = [{"id": e["id"], "prompt": e["prompt"], "completion": f"Answer: {e['label']}"} for e in examples]
    B = [{"id": e["id"], "prompt": e["prompt"], "completion": f"{e['trace']}\nAnswer: {e['label']}",
          "k": e["k"], "rems": e["rems"]} for e in examples]
    perm = list(range(len(examples)))
    rng.shuffle(perm)
    C = []
    for i, e in enumerate(examples):
        donor = examples[perm[i]]["trace"].split()
        rng.shuffle(donor)
        C.append({"id": e["id"], "prompt": e["prompt"],
                  "completion": " ".join(donor) + f"\nAnswer: {e['label']}"})
    return {"A": A, "B": B, "C": C}


def step_truth_of(n: int, d: int):
    _, rems = div_trace(n, d)
    return {"n": n, "d": d, "k": len(rems), "rems": rems}


# --------------------------------------------------------------------------- prime_hard
def _is_prime(n: int) -> bool:                 # independent of exp_thinking_dataset's sieve
    if n < 2:
        return False
    for p in range(2, math.isqrt(n) + 1):
        if n % p == 0:
            return False
    return True


def _min_factor(n: int) -> int:
    for p in range(2, math.isqrt(n) + 1):
        if n % p == 0:
            return p
    return n


def prime_hard_test(old_prime, rng):
    used = {int(x["id"].split("-")[1]) for x in old_prime["pools"]["A"]}
    used |= {int(x["id"].split("-")[1]) for x in old_prime["test"]}
    primes = [n for n in range(1000, 10000) if n not in used and _is_prime(n)]
    hard = [n for n in range(1001, 10000, 2)
            if n not in used and not _is_prime(n) and _min_factor(n) > 7]
    rng.shuffle(primes)
    rng.shuffle(hard)
    assert len(primes) >= N_TEST_PER_CLASS and len(hard) >= N_TEST_PER_CLASS
    items = ([(n, "Yes", "prime") for n in primes[:N_TEST_PER_CLASS]] +
             [(n, "No", "hard_composite") for n in hard[:N_TEST_PER_CLASS]])
    rng.shuffle(items)
    test = [{"id": f"prime_hard-{n}", "prompt": f"Is {n} a prime number?{SUFFIX}", "label": lab,
             "kind": kind, "min_factor": _min_factor(n), "shortcut_label": "Yes"} for n, lab, kind in items]
    return test, {"available_primes": len(primes), "available_hard_composites": len(hard)}


# --------------------------------------------------------------------------- checks
STEP_RE = re.compile(r"r = \(10\*(\d+) \+ (\d)\) mod (\d+) = (\d+)\.")


def check_div_task(task, d, test, pools, step_truth):
    tr_n = [int(x["id"].rsplit("-", 1)[1]) for x in pools["A"]]
    te_n = [int(x["id"].rsplit("-", 1)[1]) for x in test]
    assert len(set(tr_n)) == len(tr_n) and len(set(te_n)) == len(te_n), f"{task}: duplicate numbers"
    assert not set(tr_n) & set(te_n), f"{task}: train/test overlap"
    ys = sum(x["label"] == "Yes" for x in test)
    assert ys == len(test) - ys == N_TEST_PER_CLASS, f"{task}: test imbalance {ys}/{len(test)}"
    for arm in ("A", "B", "C"):
        py = sum(x["completion"].endswith("Answer: Yes") for x in pools[arm])
        assert py * 2 == len(pools[arm]), f"{task}/{arm}: pool imbalance {py}/{len(pools[arm])}"
        assert [x["id"] for x in pools[arm]] == [x["id"] for x in pools["A"]], f"{task}/{arm}: order differs"
    for x in test + pools["A"]:
        assert x["prompt"].count("Answer:") == 0 or x["prompt"].endswith(SUFFIX), f"leak {x['id']}"
    for b in pools["B"]:
        n = int(b["id"].rsplit("-", 1)[1])
        trace, ans = b["completion"].rsplit("\nAnswer: ", 1)
        assert ans == ("Yes" if n % d == 0 else "No"), f"{task}: label wrong {b['id']}"
        steps = STEP_RE.findall(trace)
        digits = str(n)
        assert len(steps) == len(digits) == b["k"], f"{task}: step count {b['id']}"
        prev = 0
        for i, (r0, dig, dd, r1) in enumerate(steps):          # recompute every step independently
            assert int(dd) == d and int(r0) == prev and dig == digits[i], f"{task}: step input {b['id']}"
            assert int(r1) == int(digits[:i + 1]) % d == b["rems"][i], f"{task}: step value {b['id']}"
            prev = int(r1)
        assert prev == n % d
        assert step_truth[b["id"]]["rems"] == b["rems"]
    for x in test:
        n = int(x["id"].rsplit("-", 1)[1])
        assert x["rems"] == [int(str(n)[:i + 1]) % d for i in range(len(str(n)))], f"{task}: test rems"
        assert x["label"] == ("Yes" if n % d == 0 else "No")
    tb = sum(_tok(x["completion"]) for x in pools["B"])
    tc = sum(_tok(x["completion"]) for x in pools["C"])
    assert tb == tc, f"{task}: C tokens {tc} != B {tb}"
    mb = sorted(w for x in pools["B"] for w in x["completion"].split())
    mc = sorted(w for x in pools["C"] for w in x["completion"].split())
    assert mb == mc, f"{task}: C multiset differs from B"


def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    old_sha = hashlib.sha256(OLD.read_bytes()).hexdigest()
    tasks = {}
    for t in COPIED:
        tasks[t] = json.loads(json.dumps(old["tasks"][t]))       # deep copy, entries unchanged

    # ---- div7 additions: Bprime pool + step_truth side table (old entries untouched) ----
    div7 = tasks["div7"]
    bprime = []
    for b in div7["pools"]["B"]:
        trace, ans = b["completion"].rsplit("\nAnswer: ", 1)
        bprime.append({"id": b["id"], "prompt": b["prompt"], "completion": f"Answer: {ans}\n{trace}"})
    div7["pools"]["Bprime"] = bprime
    div7["step_truth"] = {x["id"]: step_truth_of(int(x["id"].split("-")[1]), 7)
                          for x in div7["test"] + div7["pools"]["A"]}

    # ---- new div tasks ----
    for task, (d, lo, hi) in DIV_TASKS.items():
        rng = _rng(task)
        train, test = div_split(task, d, lo, hi, rng)
        pools = build_arms(train, rng)
        test_rows = [{"id": e["id"], "prompt": e["prompt"], "label": e["label"], "k": e["k"], "rems": e["rems"]}
                     for e in test]
        st = {e["id"]: {"n": e["n"], "d": d, "k": e["k"], "rems": e["rems"]} for e in train + test}
        tasks[task] = {"test": test_rows, "pools": pools, "step_truth": st, "meta": {
            "d": d, "digits": len(str(lo)), "n_train_pool": len(train), "n_test": len(test),
            "ws_tokens": {a: sum(_tok(x["completion"]) for x in pools[a]) for a in pools},
            "mean_trace_ws_tokens": round(sum(_tok(e["trace"]) for e in train) / len(train), 1)}}

    # ---- prime_hard (test only) ----
    ph_test, ph_meta = prime_hard_test(old["tasks"]["prime"], _rng("prime_hard"))
    tasks["prime_hard"] = {"test": ph_test, "pools": {}, "meta": {
        "test_only": True, "train_task": "prime", "n_test": len(ph_test), **ph_meta,
        "note": "all items odd with no prime factor <= 7; the shortcut rule predicts Yes for every item"}}

    # ---- fixed few-shot demos (B pool of each trainable task): Yes, No, No, Yes ----
    fewshot4 = {}
    for t, v in tasks.items():
        if "B" not in v["pools"]:
            continue
        ys = [x["id"] for x in v["pools"]["B"] if x["completion"].endswith("Answer: Yes")][:2]
        ns = [x["id"] for x in v["pools"]["B"] if x["completion"].endswith("Answer: No")][:2]
        fewshot4[t] = [ys[0], ns[0], ns[1], ys[1]]

    # ======================= self-checks (abort on failure) =======================
    for t in COPIED:                                              # copies are byte-for-byte equal as JSON
        for k in ("test", "meta"):
            assert tasks[t][k] == old["tasks"][t][k], f"{t}.{k} changed"
        for a, p in old["tasks"][t]["pools"].items():
            assert tasks[t]["pools"][a] == p, f"{t}.pools.{a} changed"
    for b, bp in zip(div7["pools"]["B"], div7["pools"]["Bprime"]):
        trace, ans = b["completion"].rsplit("\nAnswer: ", 1)
        assert bp["id"] == b["id"] and bp["prompt"] == b["prompt"]
        assert bp["completion"] == f"Answer: {ans}\n{trace}" and bp["completion"].count("Answer:") == 1
    for x in div7["pools"]["B"]:                                  # old div7 traces agree with step_truth
        n = int(x["id"].split("-")[1])
        vals = [int(m[3]) for m in STEP_RE.findall(x["completion"])]
        assert vals == div7["step_truth"][x["id"]]["rems"], f"div7 step_truth mismatch {x['id']}"
    for task, (d, _lo, _hi) in DIV_TASKS.items():
        v = tasks[task]
        check_div_task(task, d, v["test"], v["pools"], v["step_truth"])
    ph = tasks["prime_hard"]["test"]
    ph_n = [int(x["id"].split("-")[1]) for x in ph]
    prime_pool_n = {int(x["id"].split("-")[1]) for x in old["tasks"]["prime"]["pools"]["A"]}
    prime_test_n = {int(x["id"].split("-")[1]) for x in old["tasks"]["prime"]["test"]}
    assert len(set(ph_n)) == 240 and not set(ph_n) & prime_pool_n and not set(ph_n) & prime_test_n
    for x, n in zip(ph, ph_n):
        assert 1000 <= n <= 9999 and n % 2 == 1 and all(n % p for p in (2, 3, 5, 7)), f"prime_hard {n}"
        assert x["label"] == ("Yes" if _is_prime(n) else "No")
        assert x["prompt"] == f"Is {n} a prime number?{SUFFIX}"
    assert sum(x["label"] == "Yes" for x in ph) == 120

    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"seed_v2": SEED_V2, "source": "results/thinking_vs_data/datasets.json",
                       "source_sha256": old_sha, "source_seed": old["seed"],
                       "v2_notes": {"div7": "pools.Bprime and step_truth added; all original entries unchanged"},
                       "fewshot4": fewshot4, "tasks": tasks})
    OUT.write_text(blob, encoding="utf-8")
    for t, v in tasks.items():
        pools = v["pools"]
        lens = {a: max(len(x["completion"]) for x in p) for a, p in pools.items()} if pools else {}
        print(f"{t:11s} pool={len(pools.get('A', []))} test={len(v['test'])} "
              f"max_completion_chars={lens}")
    print("WROTE", OUT, f"({OUT.stat().st_size // 1024} KB) sha256={hashlib.sha256(blob.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
