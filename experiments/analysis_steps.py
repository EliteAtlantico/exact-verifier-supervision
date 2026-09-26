"""analysis_steps.py -- step-level scoring of trace generations (div tasks and prime), and preregistered S5.

usage:  python experiments/analysis_steps.py
writes: results/analysis/steps.json (+ steps.md)

Torch-free, re-runnable; reads every file under results/v2/gens/<model_short>/ (analysis_common.load_gens)
whose generations contain parseable trace steps. Ground truth is recomputed from the item's number and
cross-checked against datasets_v2 step_truth / test-row "rems" where present.

div tasks (div2, div3, div7, div11, div13, div7_6d; any arm/prompt mode whose text has steps, e.g. B, Bprime,
base[fewshot4]): the model's steps "r = (10*a + d) mod m = r'." are parsed in order; step i is compared
with the true remainder after digit i.
  p_uncond   = fraction of the k true positions whose model remainder is correct (missing step = wrong)
  p_cond     = P(step i correct | step i-1 correct), pooled over positions (step 0 = the initial r = 0);
               the remainder is a Markov state, so P(whole trace correct) ~= p_cond^k
  p_cond_by_position, prod_p_i = product of the per-position conditional rates
  local_arith = non-circular local step accuracy over the k true positions: r_i == (10 * r_{i-1} + x_i) mod d with
               r_{i-1} the MODEL's own previous remainder (0 at step 1) and x_i the true digit (missing step = wrong).
               All k locally correct <=> trace fully correct, so local_arith^k is a non-circular predictor of the
               fully-correct fraction (exact under independence). arith_on_claimed_operands = the same check on the
               operands the model wrote in the step.
  error_anatomy = first wrong position among wrong traces, the model's final remainder in wrong traces, Yes-rate
  digit_copy  = fraction of parsed steps whose digit d equals the item's digit at that position
  carry       = fraction of parsed steps whose a equals the model's previous r' (0 for the first step)
  trace_correct = exactly k steps, all remainders correct
  answer_trace_consistency = final answer == ("Yes" iff the model's own last remainder is 0), over items with
               a parsed step and a Yes/No answer
  predicted_acc = p_cond^k (and prod_p_i) vs observed answer accuracy and observed trace_correct rate
S5 (PREDICTIONS.md): B's answer accuracy on 6-digit div7 within 10 pp of p^6, p = p_cond measured on the SAME
  model's 4-digit div7 B generations (same seed if present, else pooled over seeds). A sensitivity value with the
  near-trivial first step excluded (p' over steps 2..k, prediction p'^5) is reported but does not decide S5. Primary cell = the 4-digit
  B adapter scored on div7_6d ('div7->div7_6d'); a B trained on div7_6d itself ('div7_6d') is reported as
  secondary. The observed accuracy comes from the gens file or, if absent, the run row.
prime-family tasks (prime, prime_hard; arms whose text has "n mod p = x" steps): truncated traces (no
  "Answer:" line; cap hits) vs wrong answers with an answer line; accuracy / truncation vs the TRUE number of
  trial divisions (primes tried up to the first divisor, or all primes <= isqrt(n)); per-step arithmetic
  accuracy (x == n mod p) and divisor-sequence faithfulness; answer vs the trace's own conclusion.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import defaultdict

import analysis_common as C

C.fix_sys_path()

DIV_STEP = re.compile(r"r\s*=\s*\(\s*10\s*\*\s*(\d+)\s*\+\s*(\d+)\s*\)\s*mod\s*(\d+)\s*=\s*(\d+)")
PRIME_STEP = re.compile(r"(\d+)\s+mod\s+(\d+)\s*=\s*(\d+)")
ANS_LINE = re.compile(r"Answer:\s*(Yes|No)", re.I)
S5_TOL = 0.10
S5_P_FIXED = 0.955        # decision log 2026-09-25 21:27 (27ba807): p of 4-digit div7 B seed 0
S5_G_FIXED = 0.561        # same entry: g of 4-digit div7 B seed 0 (used only if those gens are absent)


def true_divisions(n):
    """Primes tried by the exact trial-division trace before stopping."""
    r = math.isqrt(n)
    tried = []
    for p in C.SMALL_PRIMES:
        if p > r:
            break
        tried.append(p)
        if n % p == 0:
            break
    return tried


# --------------------------------------------------------------------------- div tasks
def score_div(rows, test_by_id, d, step_truth):
    n_items = len(rows)
    per_pos_ok = defaultdict(int)          # position -> count correct
    cond_num = defaultdict(int)            # position -> correct & prev correct
    cond_den = defaultdict(int)            # position -> prev correct
    unc_ok = unc_tot = 0
    loc_ok = copy_ok = carry_ok = parsed_steps = 0
    trace_ok = parsed_items = exact_k = 0
    cons_ok = cons_tot = 0
    ans_ok = 0
    truth_mismatch = 0
    ans_ok_trace_ok = ans_ok_trace_bad = n_trace_bad = 0
    loc2_ok = loc2_tot = 0                 # non-circular local step accuracy over the k true positions
    loc2_pos = defaultdict(int)
    first_err = defaultdict(int)           # wrong traces: position of the first wrong remainder (or 'short')
    final_rem_wrong = defaultdict(int)     # wrong traces: the model's own last remainder
    yes = 0
    ks = []
    for row in rows:
        item = test_by_id.get(row.get("id"))
        if item is None:
            continue
        n = C.number_of(item["prompt"])
        truth = C.rems_of(n, d)
        ref = item.get("rems") or (step_truth.get(row["id"]) or {}).get("rems")
        if ref is not None and list(ref) != truth:
            truth_mismatch += 1
        k = len(truth)
        ks.append(k)
        digits = [int(c) for c in str(n)]
        pred = C.gen_pred(row)
        good_answer = pred == item["label"]
        ans_ok += good_answer
        yes += pred == "Yes"
        steps = [tuple(int(x) for x in m) for m in DIV_STEP.findall(C.gen_text(row))]
        if steps:
            parsed_items += 1
        exact_k += len(steps) == k
        prev_r, prev_ok = 0, True
        all_ok = len(steps) == k
        for i in range(k):
            if i < len(steps):
                a, dg, m, r = steps[i]
                ok = r == truth[i]
            else:
                ok = False
            unc_ok += ok
            unc_tot += 1
            per_pos_ok[i] += ok
            if prev_ok:
                cond_den[i] += 1
                cond_num[i] += ok
            prev_ok = ok
            all_ok &= ok
        prev_model_r = 0
        first_bad = None
        for i in range(k):                 # local: r_i == (10 * r_{i-1}^model + x_i) mod d, x_i = the TRUE digit
            if i < len(steps):
                ok_l = steps[i][3] == (10 * prev_model_r + digits[i]) % d
                prev_model_r = steps[i][3]
            else:
                ok_l = False
            loc2_ok += ok_l
            loc2_tot += 1
            loc2_pos[i] += ok_l
            if first_bad is None and (i >= len(steps) or steps[i][3] != truth[i]):
                first_bad = "short" if i >= len(steps) else str(i + 1)
        for i, (a, dg, m, r) in enumerate(steps):
            parsed_steps += 1
            loc_ok += r == (10 * a + dg) % m if m else 0
            copy_ok += i < k and dg == digits[i]
            carry_ok += a == prev_r
            prev_r = r
        trace_ok += all_ok
        if all_ok:
            ans_ok_trace_ok += good_answer
        else:
            n_trace_bad += 1
            ans_ok_trace_bad += good_answer
            first_err[first_bad or ("extra_steps" if len(steps) > k else "?")] += 1
            final_rem_wrong[str(steps[-1][3]) if steps else "none"] += 1
        if steps and pred in ("Yes", "No"):
            cons_tot += 1
            cons_ok += pred == ("Yes" if steps[-1][3] == 0 else "No")
    if not ks:
        return None
    kmax = max(ks)
    p_cond = sum(cond_num.values()) / max(1, sum(cond_den.values()))
    den1 = sum(v for i, v in cond_den.items() if i > 0)
    p_cond_after_first = sum(v for i, v in cond_num.items() if i > 0) / den1 if den1 else None
    p_by_pos = [cond_num[i] / cond_den[i] if cond_den[i] else None for i in range(kmax)]
    prod = 1.0
    for v in p_by_pos:
        prod *= v if v is not None else 0.0
    k_mode = max(set(ks), key=ks.count)
    return {"n_items": n_items, "k": k_mode, "answer_acc": ans_ok / n_items,
            "parse_rate": parsed_items / n_items, "exact_k_steps_rate": exact_k / n_items,
            "p_uncond": unc_ok / max(1, unc_tot), "p_cond": p_cond, "p_cond_by_position": p_by_pos,
            "p_cond_after_first_step": p_cond_after_first,
            "p_uncond_by_position": [per_pos_ok[i] / len(ks) for i in range(kmax)],
            "local_arith_acc": loc2_ok / max(1, loc2_tot),
            "local_arith_by_position": [loc2_pos[i] / len(ks) for i in range(kmax)],
            "local_arith_pow_k": (loc2_ok / max(1, loc2_tot)) ** k_mode,
            "arith_on_claimed_operands_acc": loc_ok / max(1, parsed_steps),
            "digit_copy_acc": copy_ok / max(1, parsed_steps),
            "carry_consistency": carry_ok / max(1, parsed_steps),
            "trace_correct_rate": trace_ok / n_items,
            "answer_trace_consistency": cons_ok / cons_tot if cons_tot else None, "n_consistency_items": cons_tot,
            "answer_acc_given_trace_correct": ans_ok_trace_ok / trace_ok if trace_ok else None,
            "answer_acc_given_trace_wrong": ans_ok_trace_bad / n_trace_bad if n_trace_bad else None,
            "predicted_acc_p_cond_pow_k": p_cond ** k_mode, "predicted_acc_prod_p_i": prod,
            "post_hoc_predicted_answer_acc_own_g": (None if n_trace_bad == 0 else
                                                     p_cond ** k_mode + (1 - p_cond ** k_mode) * (ans_ok_trace_bad / n_trace_bad)),
            "observed_minus_predicted_pp": 100 * (ans_ok / n_items - p_cond ** k_mode),
            "ground_truth_mismatches": truth_mismatch,
            "yes_rate": yes / n_items,
            "error_anatomy": {"n_wrong_traces": n_trace_bad,
                              "first_error_position": dict(sorted(first_err.items())),
                              "final_remainder_of_wrong_traces": dict(sorted(final_rem_wrong.items(),
                                                                             key=lambda kv: (len(kv[0]), kv[0]))),
                              "wrong_traces_ending_in_0": (final_rem_wrong.get("0", 0) / n_trace_bad) if n_trace_bad else None}}


# --------------------------------------------------------------------------- prime tasks
LEN_BINS = [(1, 1), (2, 3), (4, 8), (9, 16), (17, 99)]


def score_prime(rows, test_by_id):
    out_bins = {f"{a}-{b}": {"n": 0, "ok": 0, "trunc": 0, "model_steps": 0} for a, b in LEN_BINS}
    n = ok = trunc = trunc_wrong = cap = wrong_with_line = 0
    cls = {"Yes": [0, 0, 0], "No": [0, 0, 0]}          # gold class -> [n, correct, truncated]
    ar_ok = ar_tot = seq_ok = seq_tot = cons_ok = cons_tot = 0
    for row in rows:
        item = test_by_id.get(row.get("id"))
        if item is None:
            continue
        num = C.number_of(item["prompt"])
        txt = C.gen_text(row)
        pred = C.gen_pred(row)
        good = pred == item["label"]
        has_line = bool(ANS_LINE.search(txt))
        n += 1
        ok += good
        trunc += not has_line
        trunc_wrong += (not has_line) and not good
        cc = cls.setdefault(item["label"], [0, 0, 0])
        cc[0] += 1
        cc[1] += good
        cc[2] += not has_line
        cap += bool(row.get("hit_cap"))
        wrong_with_line += (not good) and has_line
        tried = true_divisions(num)
        L = len(tried)
        steps = [(int(a), int(p), int(x)) for a, p, x in PRIME_STEP.findall(txt)]
        for a, p, x in steps:
            ar_tot += 1
            ar_ok += a == num and p > 0 and x == num % p
        if steps:
            seq_tot += 1
            seq_ok += [p for _, p, _ in steps] == tried[:len(steps)]
        concl = re.search(r"is (prime|composite)", txt)
        if concl and pred in ("Yes", "No"):
            cons_tot += 1
            cons_ok += pred == ("Yes" if concl.group(1) == "prime" else "No")
        for a, b in LEN_BINS:
            if a <= L <= b:
                bb = out_bins[f"{a}-{b}"]
                bb["n"] += 1
                bb["ok"] += good
                bb["trunc"] += not has_line
                bb["model_steps"] += len(steps)
    if not n:
        return None
    bins = {k: {"n": v["n"], "acc": v["ok"] / v["n"] if v["n"] else None,
                "truncated_rate": v["trunc"] / v["n"] if v["n"] else None,
                "mean_model_steps": v["model_steps"] / v["n"] if v["n"] else None} for k, v in out_bins.items()}
    err = n - ok
    return {"n_items": n, "answer_acc": ok / n, "n_errors": err, "n_truncated_no_answer_line": trunc,
            "n_hit_cap": cap, "n_wrong_with_answer_line": wrong_with_line,
            "n_truncated_and_wrong": trunc_wrong,
            "per_class": {("primes" if k == "Yes" else "composites"): {"n": v[0], "acc": v[1] / v[0] if v[0] else None,
                                                                        "truncated_rate": v[2] / v[0] if v[0] else None}
                          for k, v in cls.items()},
            "truncated_share_of_errors": (trunc_wrong / err) if err else None,
            "step_arith_acc": ar_ok / ar_tot if ar_tot else None, "n_steps_parsed": ar_tot,
            "divisor_sequence_faithful_rate": seq_ok / seq_tot if seq_tot else None,
            "answer_vs_own_conclusion_consistency": cons_ok / cons_tot if cons_tot else None,
            "by_true_trial_divisions": bins}


# --------------------------------------------------------------------------- valid: claimed-contradiction audit
_TOK = re.compile(r"\s*(->|[A-Z]|[()~|&])")


def _parse_formula(txt):
    """Tiny parser for the trace formulas: ~, &, |, -> (right-assoc), parentheses, single-letter atoms.
    Returns (eval_fn, atoms) or None."""
    toks, pos = [], 0
    txt = txt.strip()
    while pos < len(txt):
        m = _TOK.match(txt, pos)
        if not m:
            return None
        toks.append(m.group(1))
        pos = m.end()
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def eat():
        nonlocal i
        i += 1
        return toks[i - 1]

    def imp():
        left = orr()
        if peek() == "->":
            eat()
            right = imp()
            return lambda v, a=left, b=right: (not a(v)) or b(v)
        return left

    def orr():
        left = andd()
        while peek() == "|":
            eat()
            right = andd()
            left = (lambda v, a=left, b=right: a(v) or b(v))
        return left

    def andd():
        left = un()
        while peek() == "&":
            eat()
            right = un()
            left = (lambda v, a=left, b=right: a(v) and b(v))
        return left

    def un():
        t = peek()
        if t == "~":
            eat()
            x = un()
            return lambda v, a=x: not a(v)
        if t == "(":
            eat()
            x = imp()
            if eat() != ")":
                raise ValueError
            return x
        if t and t.isalpha():
            eat()
            return lambda v, name=t: v[name]
        raise ValueError
    try:
        f = imp()
        if i != len(toks):
            return None
    except (ValueError, IndexError):
        return None
    return f, sorted({t for t in toks if t.isalpha()})


def _split_top(set_txt):
    parts, depth, cur = [], 0, ""
    for ch in set_txt:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def claimed_set_is_unsat(text):
    """True/False if the trace's claimed contradictory set {..} parses (unsat or not); None if unparseable."""
    m = re.search(r"contradiction:?\s*\{([^}]*)\}", text)
    if not m:
        return None
    fs = [_parse_formula(x) for x in _split_top(m.group(1))]
    if not fs or any(f is None for f in fs):
        return None
    atoms = sorted({a for _, at in fs for a in at})
    if len(atoms) > 10:
        return None
    from itertools import product
    for vals in product([False, True], repeat=len(atoms)):
        v = dict(zip(atoms, vals))
        if all(f(v) for f, _ in fs):
            return False
    return True


UNSAT_RE = re.compile(r"contradiction|cannot all hold|is impossible|conclusion follows", re.I)
SAT_RE = re.compile(r"satisfiable|countermodel", re.I)


def score_valid(rows, test_by_id):
    """Which template does each generation assert (contradiction = 'valid', countermodel = 'invalid'), by gold label."""
    by_gold = {"Yes": defaultdict(int), "No": defaultdict(int)}
    schema_err = defaultdict(lambda: [0, 0])
    claim_checked = claim_false = 0
    hybrid = 0
    n = ok = 0
    for row in rows:
        item = test_by_id.get(row.get("id"))
        if item is None:
            continue
        txt = C.gen_text(row)
        pred = C.gen_pred(row)
        good = pred == item["label"]
        n += 1
        ok += good
        u, sat = bool(UNSAT_RE.search(txt)), bool(SAT_RE.search(txt))
        kind = "asserts_contradiction" if u and not sat else ("asserts_countermodel" if sat and not u else
                                                              ("both" if u and sat else "neither"))
        g = by_gold.setdefault(item["label"], defaultdict(int))
        g[kind] += 1
        g["n"] += 1
        g["correct"] += good
        if re.search(r"conclusion false\.\s*That is(?! satisfiable)", txt) and u:
            hybrid += 1                     # countermodel-template opener that then asserts a contradiction
        sm = re.match(r"valid-\d+-(.+)$", item["id"])
        if sm:
            schema_err[sm.group(1)][0] += 1
            schema_err[sm.group(1)][1] += not good
        if u:
            r = claimed_set_is_unsat(txt)
            if r is not None:
                claim_checked += 1
                claim_false += not r
                g["claims_checked"] += 1
                g["claims_actually_satisfiable"] += not r
    if not n:
        return None
    return {"n_items": n, "answer_acc": ok / n,
            "by_gold": {k: {**v, "frac_asserts_contradiction": v["asserts_contradiction"] / v["n"] if v["n"] else None,
                            "frac_asserts_countermodel": v["asserts_countermodel"] / v["n"] if v["n"] else None,
                            "acc": v["correct"] / v["n"] if v["n"] else None} for k, v in by_gold.items() if v.get("n")},
            "hybrid_opener_then_contradiction": hybrid,
            "claimed_contradictions_checked": claim_checked, "claimed_contradictions_actually_satisfiable": claim_false,
            "errors_by_schema": {k: {"n": v[0], "errors": v[1]} for k, v in sorted(schema_err.items()) if v[1]}}


# --------------------------------------------------------------------------- STaR samples (arm S)
# POST HOC re-audit (analysis side only; the gate and exp_worker_star.audit_trace are unchanged). The worker's
# audit recognizes "a mod d = r", "(10*r + x) mod d = r'", "a divided by d ... remainder r" and "a / d = q" with q an
# integer or a terminating decimal; it misses LaTeX "\div", a mangled division sign, "approximately"/"\approx",
# and repeating or truncated decimals ("391.428571..."), and it counts a trace with no recognized claim as incorrect.
_XN = r"(\d[\d,]*)"
_XRES = [
    (re.compile(r"\(\s*10\s*\*\s*(\d+)\s*\+\s*(\d)\s*\)\s*(?:mod|%)\s*(\d+)\s*(?:=|is|equals)\s*(\d+)", re.I), "step"),
    (re.compile(_XN + r"\s*(?:mod|%)\s*(\d+)\s*(?:=|is|equals)\s*(\d+)", re.I), "mod"),
    (re.compile(_XN + r"\s*(?:divided by|÷|/)\s*(\d+)[^.\n]{0,40}?remainder(?:\s+(?:of|is))?\s*(\d+)", re.I), "rem"),
    (re.compile(_XN + r"\s*(?:divided by|÷|/)\s*(\d+)\s*(?:=|is|equals|≈)\s*(\d+(?:\.\d+)?)", re.I), "quot"),
]


def audit_trace_ext(text, n, d):
    t = (text or "").replace("\\div", "÷").replace("\ufffd", "÷").replace("\\approx", "≈")
    t = re.sub(r"\\[()\[\]]|\$", " ", t)
    t = re.sub(r"(?:is\s+)?approximately", "≈", t, flags=re.I)
    s, rems, r = str(n), [], 0
    for ch in s:
        r = (10 * r + int(ch)) % d
        rems.append(r)
    claims, seen, step_i = [], set(), 0
    for rx, kind in _XRES:
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b in seen):
                continue
            seen.add(m.span())
            if kind == "step":
                r0, dig, dd, rr = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
                if dd != d:
                    continue
                i, step_i = step_i, step_i + 1
                claims.append(i < len(rems) and dig == s[i] and r0 == (rems[i - 1] if i else 0) and rr == rems[i])
                continue
            a, dd, v = int(m.group(1).replace(",", "")), int(m.group(2)), m.group(3)
            if dd != d:
                continue
            if kind == "quot":
                if "." in v and not float(v).is_integer():
                    claims.append(a % d != 0 and abs(a / d - float(v)) < 0.01)   # truncated / repeating decimal
                else:
                    claims.append(a == d * int(float(v)))
            else:
                a_s = str(a)
                claims.append(int(v) == (rems[len(a_s) - 1] if s.startswith(a_s) and len(a_s) <= len(rems)
                                         else a % d))
    return {"n_claims": len(claims), "all_ok": bool(claims) and all(claims)}


def star_summary():
    root = os.path.join(C.V2, "star_samples")
    out = []
    if not os.path.isdir(root):
        return out
    for model in sorted(os.listdir(root)):
        for f in sorted(glob.glob(os.path.join(root, model, "*.jsonl"))):
            name = os.path.splitext(os.path.basename(f))[0]
            m = re.match(r"(.+)_S_(\d+)_(\d+)$", name)
            if not m:
                continue
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            if not rows:
                continue
            K = max(len(r.get("samples") or []) for r in rows)
            n_pass = sum(bool(x.get("pass")) for r in rows for x in r.get("samples") or [])
            n_samp = sum(len(r.get("samples") or []) for r in rows)
            kept = [r for r in rows if r.get("kept_index") is not None]
            aud = [r["audit"] for r in kept if r.get("audit")]
            parsed = [a for a in aud if a.get("n_claims", 0) > 0]
            trunc_rej = sum(bool(x.get("hit_cap")) and not x.get("pass") for r in rows for x in r.get("samples") or [])
            ext = {}
            dm = re.match(r"div(\d+)", m.group(1))
            if dm and kept:
                d = int(dm.group(1))
                ax = [audit_trace_ext(r["samples"][r["kept_index"]].get("text"), int(r["id"].split("-")[1]), d)
                      for r in kept]
                px = [a for a in ax if a["n_claims"]]
                ext = {"ext_kept_trace_parsed_frac": len(px) / len(kept),
                       "ext_kept_trace_correct_frac_all_kept": sum(a["all_ok"] for a in ax) / len(kept),
                       "ext_all_claims_correct_among_parsed": (sum(a["all_ok"] for a in px) / len(px)) if px else None,
                       "ext_n_parsed": len(px),
                       "ext_definition": "POST HOC re-audit (analysis_steps.audit_trace_ext): adds \\div, approximately, "
                                         "truncated/repeating decimals; parsed = >= 1 recognized claim"}
            out.append({**ext,"model": model, "task": m.group(1), "n": int(m.group(2)), "seed": int(m.group(3)),
                        "samples_path": os.path.relpath(f, C.ROOT).replace("\\", "/"), "n_items": len(rows), "K": K,
                        "keep_rate": n_pass / n_samp if n_samp else None, "item_keep_rate": len(kept) / len(rows),
                        "n_kept": len(kept), "n_kept_yes": sum(r["gold"] == "Yes" for r in kept),
                        "n_kept_no": sum(r["gold"] == "No" for r in kept), "n_trunc_rejected": trunc_rej,
                        "kept_trace_correct_frac": (sum(bool(a.get("all_ok")) and a.get("n_claims", 0) > 0 for a in aud)
                                                    / len(kept)) if aud else None,
                        "kept_trace_parsed_frac": (len(parsed) / len(kept)) if aud else None,
                        "all_claims_correct_among_parsed": (sum(bool(a.get("all_ok")) for a in parsed) / len(parsed))
                        if parsed else None,
                        "definition": "kept_trace_correct_frac = kept traces with >= 1 audited remainder/quotient claim, all "
                                      "exactly correct (exp_worker_star.audit_trace), over ALL kept traces"})
    return out


# --------------------------------------------------------------------------- main
def main():
    tasks = C.load_tasks()
    gens, notes = C.load_gens(tasks)
    runs, _ = C.load_runs()
    run_acc = {(r["task"], r["arm"], r["n"], r["seed"], C.short_model(r["model"])): r["k"] / r["n_test"] for r in runs}
    div_cells, prime_cells, valid_cells = [], [], []
    for key, g in sorted(gens.items()):
        label, arm, n, seed, model = key
        et = g["eval_task"]
        td = tasks.get(et)
        if not td:
            notes.append(f"{g['path']}: no test set for {et}")
            continue
        test_by_id = {t["id"]: t for t in td["test"]}
        meta = {"task": label, "eval_task": et, "train_task": g["train_task"], "arm": arm, "n": n, "seed": seed,
                "model": model, "prompt_mode": g["prompt_mode"], "gens_path": g["path"]}
        d = C.divisor_of_task(et)
        if d:
            if not any(DIV_STEP.search(C.gen_text(r)) for r in g["rows"]):
                continue                                   # answer-only / scrambled generations: no steps
            sc = score_div(g["rows"], test_by_id, d, (tasks[et].get("step_truth") or {}))
            if sc:
                div_cells.append({**meta, **sc})
        elif et.startswith("valid"):
            if arm in ("A", "A_tok") or not any(UNSAT_RE.search(C.gen_text(r)) or SAT_RE.search(C.gen_text(r))
                                                 for r in g["rows"]):
                continue
            sc = score_valid(g["rows"], test_by_id)
            if sc:
                valid_cells.append({**meta, **sc})
        elif et.startswith("prime"):
            if not any(PRIME_STEP.search(C.gen_text(r)) for r in g["rows"]) and not arm.startswith("B"):
                continue
            sc = score_prime(g["rows"], test_by_id)
            if sc:
                prime_cells.append({**meta, **sc})
    s5 = s5_check(div_cells, run_acc)
    trans = transitions(div_cells, tasks)
    C.ensure_out()
    json.dump(trans, open(os.path.join(C.OUT, "transitions.json"), "w", encoding="utf-8"), indent=1)
    out = {"generated_by": "experiments/analysis_steps.py", "notes": notes, "div": div_cells, "prime": prime_cells,
           "valid": valid_cells, "star": star_summary(),
           "S5": s5}
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "steps.json"), "w", encoding="utf-8"), indent=1)
    write_md(out)
    print(f"steps: {len(div_cells)} div cells, {len(prime_cells)} prime cells, S5 {s5['status']} -> "
          f"{os.path.relpath(C.OUT, C.ROOT)}/steps.json")
    for n in notes:
        print("  note:", n)


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - sse / sst) if sst else None


def transitions(div_cells, tasks):
    """S7 table: m = k * n / (10 * d) supervised transitions per (remainder, digit) table entry, with k = steps per
    TRAINING trace (the training task's digit count), n = training examples, d = divisor. One row per B-family
    generation file of a div task. Post-hoc fits over trained-and-tested plain B cells (distinct m needed):
      logistic:  logit(p) = a + b * ln m        power:  ln(1 - p) = a + b * ln m   (p clipped to [0.005, 0.995])."""
    rows = []
    for c in div_cells:
        if not c["arm"].startswith("B"):
            continue
        d = C.divisor_of_task(c["eval_task"])
        tt = tasks.get(c["train_task"], {}).get("test") or []
        k_train = len(str(C.number_of(tt[0]["prompt"]))) if tt else c["k"]
        m = k_train * c["n"] / (10 * d) if c["n"] else None
        trivial = sum((10 * r + x) % d == x % d for r in range(d) for x in range(10)) / (10 * d)
        rows.append({"task": c["task"], "train_task": c["train_task"], "eval_task": c["eval_task"], "arm": c["arm"],
                     "prompt_mode": c["prompt_mode"], "model": c["model"], "n": c["n"], "seed": c["seed"],
                     "k_train": k_train, "k_test": c["k"], "d": d, "m": m, "p": c["p_cond"],
                     "table_entries": 10 * d, "ten_mod_d": 10 % d,
                     "ten_mod_d_signed": min(((10 % d), (10 % d) - d), key=abs), "trivial_entry_fraction": trivial,
                     "max_intermediate": 10 * (d - 1) + 9, "local_arith": c["local_arith_acc"],
                     "local_arith_pow_k": c["local_arith_pow_k"], "yes_rate": c["yes_rate"],
                     "p_after_first_step": c["p_cond_after_first_step"], "trace_correct": c["trace_correct_rate"],
                     "accuracy": c["answer_acc"], "fit_eligible": bool(m) and c["arm"] == "B" and c["prompt_mode"] == "plain"
                     and c["train_task"] == c["eval_task"]})
    rows.sort(key=lambda r: (r["model"], r["m"] or 0))
    fits = {}
    for model in sorted({r["model"] for r in rows}):
        pts = [(r["m"], min(0.995, max(0.005, r["p"]))) for r in rows if r["model"] == model and r["fit_eligible"]]
        entry = {"n_points": len(pts), "label": "POST HOC fit (not preregistered)"}
        if len({x for x, _ in pts}) >= 3:
            lx = [math.log(x) for x, _ in pts]
            lg = _ols(lx, [math.log(p / (1 - p)) for _, p in pts])
            pw = _ols(lx, [math.log(1 - p) for _, p in pts])
            if lg:
                a, b, r2 = lg
                entry["logistic_logit_p_vs_ln_m"] = {
                    "a": a, "b": b, "r2_logit_scale": r2,
                    "m_for_p_0.95": math.exp((math.log(0.95 / 0.05) - a) / b) if b else None,
                    "fitted": {f"{x:.2f}": 1 / (1 + math.exp(-(a + b * math.log(x)))) for x, _ in pts}}
            if pw:
                a, b, r2 = pw
                entry["power_1_minus_p_vs_m"] = {"c": math.exp(a), "exponent": b, "r2_log_scale": r2,
                                                 "fitted": {f"{x:.2f}": 1 - math.exp(a) * x ** b for x, _ in pts}}
        else:
            entry["note"] = "fewer than 3 distinct m values; no fit"
        fits[model] = entry
    return {"generated_by": "experiments/analysis_steps.py", "definition": "m = k_train * n / (10 * d); p = P(step correct "
            "| previous step correct) (steps.json p_cond). Step difficulty: table_entries = 10 d; ten_mod_d = the multiplier "
            "10 mod d, ten_mod_d_signed = its smallest-magnitude representative (+1 = add the digit, -1 = subtract, 0 = only "
            "the last digit matters, |3| = multiply by 3); trivial_entry_fraction = share of (r, x) entries whose result equals "
            "x mod d (r = 0, or every entry when d divides 10); max_intermediate = 10 (d - 1) + 9.",
            "rows": rows, "post_hoc_fits": fits}


def s5_check(div_cells, run_acc):
    """S5 per the PREDICTIONS.md decision-log entry of 2026-09-25 21:27 (commit 27ba807, before any S5 result):
    PRIMARY = arm B trained AND tested on div7_6d, seeds 0 and 1: answer accuracy within 10 pp of p^6 with p FIXED
    at 0.955 (the value the decision log names: 4-digit div7 B seed 0); the same-seed 4-digit p is reported as a
    secondary comparison and does not decide S5. SECONDARY = the 4-digit div7 B adapter evaluated on
    div7_6d ('div7->div7_6d', evals file). POST HOC (labelled): fully-correct 6-digit trace fraction vs p^6, and
    answer accuracy vs p^6 + (1 - p^6) * g, g = answer-correct rate when the 4-digit trace is wrong."""
    four = defaultdict(dict)
    for c in div_cells:
        if c["task"] == "div7" and c["arm"] == "B" and c["prompt_mode"] == "plain" and c["n"] == 180:
            four[c["model"]][c["seed"]] = c       # the 4-digit reference is the n = 180 cell (not the S7 n = 90/270 cells)
    six_gens = {(c["task"], c["model"], c["seed"], c["n"]): c for c in div_cells
                if c["eval_task"] == "div7_6d" and c["arm"] == "B" and c["prompt_mode"] == "plain"}
    six_runs = {(k[0], k[4], k[3], k[2]): v for k, v in run_acc.items()
                if k[1] == "B" and k[0] in ("div7->div7_6d", "div7_6d")}
    cells = []
    for key in sorted(set(six_gens) | set(six_runs)):
        label, model, seed, n = key
        role = "primary" if label == "div7_6d" else "secondary"
        fd = four.get(model, {})
        # PRIMARY p is the value named in the decision log (4-digit div7 B seed 0: 0.955); same-seed p is secondary
        same = fd.get(seed)
        if model == C.short_model(C.DEFAULT_MODEL):
            p = S5_P_FIXED
        elif 0 in fd:                                   # other models: their OWN 4-digit div7 B seed-0 p (0.955 is a 1.5B value)
            p = fd[0]["p_cond"]
        else:
            cells.append({"task": label, "role": role, "model": model, "seed": seed, "n": n, "status": "PENDING",
                          "why": "no 4-digit div7 B seed-0 generations for this model"})
            continue
        g_src = fd.get(0) or same or (next(iter(fd.values())) if fd else None)
        g = g_src["answer_acc_given_trace_wrong"] if g_src else S5_G_FIXED
        p2 = g_src["p_cond_after_first_step"] if g_src else None
        p_src = ("fixed 0.955 = decision-log value (4-digit div7 B seed 0)" if model == C.short_model(C.DEFAULT_MODEL)
                 else f"this model's 4-digit div7 B seed 0 (secondary model)")
        obs = six_gens[key]["answer_acc"] if key in six_gens else six_runs[key]
        pred = p ** 6
        tc = six_gens[key]["trace_correct_rate"] if key in six_gens else None
        same_seed = None
        if same is not None:
            ps = same["p_cond"]
            same_seed = {"p": ps, "source": f"div7 B seed {seed} generations", "predicted_p6": ps ** 6,
                         "diff_pp": 100 * (obs - ps ** 6), "within_10pp": abs(obs - ps ** 6) <= S5_TOL}
        cells.append({"task": label, "role": role, "primary": role == "primary", "model": model, "seed": seed, "n": n,
                      "p": p, "p_source": p_src, "predicted_p6": pred, "observed_acc": obs,
                      "obs_source": "gens" if key in six_gens else "run row", "diff_pp": 100 * (obs - pred),
                      "within_10pp": abs(obs - pred) <= S5_TOL, "same_seed_p_secondary": same_seed,
                      "post_hoc": {"label": "post hoc (decision log 2026-09-25 21:27)",
                                   "trace_correct_6digit": tc,
                                   "trace_correct_minus_p6_pp": None if tc is None else 100 * (tc - pred),
                                   "g_4digit_answer_acc_when_trace_wrong": g,
                                   "predicted_answer_acc_with_guessing": None if g is None else pred + (1 - pred) * g,
                                   "answer_minus_guess_model_pp": None if g is None else 100 * (obs - (pred + (1 - pred) * g)),
                                   "p_after_first_step": p2, "p_after_first_step_pow5": p2 ** 5}})

    def verdict(cs, need_seeds):
        done = [c for c in cs if "within_10pp" in c]
        missing = sorted(set(need_seeds or []) - {c["seed"] for c in done})
        if any(not c["within_10pp"] for c in done):
            bad = sorted(c["seed"] for c in done if not c["within_10pp"])
            return f"FAIL (seed(s) {bad} outside 10 pp" + (f"; seed(s) {missing} not yet run)" if missing else ")")
        if need_seeds and not set(need_seeds) <= {c["seed"] for c in done}:
            return f"PENDING ({len(done)}/{len(need_seeds)} primary seeds done, all within 10 pp so far)" if done else "PENDING"
        return "PASS" if done else "PENDING"
    by_model = defaultdict(list)
    for c in cells:
        by_model[c["model"]].append(c)
    per_model = {m: {"primary": verdict([c for c in cs if c["role"] == "primary"], [0, 1]),
                     "secondary": verdict([c for c in cs if c["role"] == "secondary"], None)}
                 for m, cs in by_model.items()}
    m15 = C.short_model(C.DEFAULT_MODEL)
    status = per_model.get(m15, {}).get("primary", "PENDING")
    return {"status": status, "status_by_model": per_model, "tolerance_pp": 100 * S5_TOL,
            "rule": "PRIMARY = B trained+tested on div7_6d (seeds 0,1) vs p^6, p from 4-digit div7 B gens; "
                    "SECONDARY = 4-digit B adapter on div7_6d (PREDICTIONS.md decision log, 27ba807)",
            "cells": cells}


def write_md(o):
    L = ["# Step-level trace scoring (auto-generated by experiments/analysis_steps.py; do not edit)", ""]
    if o["div"]:
        L += ["## div tasks", "", "p_cond = P(step correct | previous step correct), pooled over positions; p after 1 = the same "
              "over steps 2..k (step 1 is r = digit mod d, near-trivial); guess model (post hoc) = p^k + (1 - p^k) g with g = this "
              "cell's answer accuracy when its trace is wrong.", "",
              "| task | model | arm | mode | n | seed | k | answer acc | trace correct | p_cond | p after 1 | p_uncond | "
              "p_cond^k | prod p_i | guess model | local arith | local^k | digit copy | answer-trace consistency | acc given trace wrong | "
              "Yes-rate | first-error position (wrong traces) | final remainder of wrong traces |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"   # noqa: E731
        for c in o["div"]:
            L.append(f"| {c['task']} | {c['model']} | {c['arm']} | {c['prompt_mode']} | {c['n']} | {c['seed']} | {c['k']} | "
                     f"{f(c['answer_acc'])} | {f(c['trace_correct_rate'])} | {f(c['p_cond'])} | {f(c['p_cond_after_first_step'])} | "
                     f"{f(c['p_uncond'])} | {f(c['predicted_acc_p_cond_pow_k'])} | {f(c['predicted_acc_prod_p_i'])} | "
                     f"{f(c['post_hoc_predicted_answer_acc_own_g'])} | {f(c['local_arith_acc'])} | {f(c['local_arith_pow_k'])} | "
                     f"{f(c['digit_copy_acc'])} | {f(c['answer_trace_consistency'])} | {f(c['answer_acc_given_trace_wrong'])} | "
                     f"{f(c['yes_rate'])} | {c['error_anatomy']['first_error_position']} | "
                     f"{c['error_anatomy']['final_remainder_of_wrong_traces']} |")
    if o["prime"]:
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"   # noqa: E731
        L += ["", "## prime tasks", "", "| task | model | arm | n | seed | acc | errors | truncated (no Answer line) | "
              "truncated & wrong | cap hits | wrong with Answer line | step arith | divisor seq faithful | acc by true #divisions (1 / 2-3 / 4-8 / 9-16 / 17+) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in o["prime"]:
            b = c["by_true_trial_divisions"]
            L.append(f"| {c['task']} | {c['model']} | {c['arm']} | {c['n']} | {c['seed']} | {f(c['answer_acc'])} | {c['n_errors']} | "
                     f"{c['n_truncated_no_answer_line']} | {c['n_truncated_and_wrong']} | {c['n_hit_cap']} | {c['n_wrong_with_answer_line']} | {f(c['step_arith_acc'])} | "
                     f"{f(c['divisor_sequence_faithful_rate'])} | " + " / ".join(
                         f"{f(v['acc'])} (n={v['n']})" for v in b.values()) + " |")
    if o.get("valid"):
        L += ["", "## valid: which template does the trace assert?", "",
              "| task | model | arm | seed | acc | gold Yes: asserts contradiction / countermodel | gold No: asserts contradiction / "
              "countermodel | claimed contradictions checked / actually satisfiable | errors by schema |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in o["valid"]:
            y, nn = c["by_gold"].get("Yes", {}), c["by_gold"].get("No", {})
            es = ", ".join(f"{k} {v['errors']}/{v['n']}" for k, v in c["errors_by_schema"].items())
            L.append(f"| {c['task']} | {c['model']} | {c['arm']} | {c['seed']} | {100 * c['answer_acc']:.1f} | "
                     f"{y.get('asserts_contradiction', 0)}/{y.get('n', 0)} / {y.get('asserts_countermodel', 0)}/{y.get('n', 0)} | "
                     f"{nn.get('asserts_contradiction', 0)}/{nn.get('n', 0)} / {nn.get('asserts_countermodel', 0)}/{nn.get('n', 0)} | "
                     f"{c['claimed_contradictions_checked']} / {c['claimed_contradictions_actually_satisfiable']} | "
                     f"{es or '-'} |")
    if o.get("star"):
        L += ["", "## Arm S (self-generated, answer-verified traces): STaR samples", "",
              "| model | task | n | seed | K | keep rate (samples) | items kept | kept Yes/No | truncated rejected | "
              "kept traces with all audited claims correct | kept traces with >= 1 audited claim |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"   # noqa: E731
        for c in o["star"]:
            L.append(f"| {c['model']} | {c['task']} | {c['n']} | {c['seed']} | {c['K']} | {f(c['keep_rate'])} | "
                     f"{c['n_kept']}/{c['n_items']} | {c['n_kept_yes']}/{c['n_kept_no']} | {c['n_trunc_rejected']} | "
                     f"{f(c['kept_trace_correct_frac'])} | {f(c['kept_trace_parsed_frac'])} |")
    L += ["", f"## S5 (primary, {C.short_model(C.DEFAULT_MODEL)}): {o['S5']['status']}", "", o["S5"]["rule"], ""]
    for m, v in o["S5"].get("status_by_model", {}).items():
        L.append(f"- {m}: primary {v['primary']}; secondary {v['secondary']}")
    L.append("")
    for c in o["S5"]["cells"]:
        if "p" in c:
            ph = c["post_hoc"]
            tc = "-" if ph["trace_correct_6digit"] is None else f"{100 * ph['trace_correct_6digit']:.1f}%"
            gm = ("-" if ph["predicted_answer_acc_with_guessing"] is None
                  else f"{100 * ph['predicted_answer_acc_with_guessing']:.1f}%")
            L.append(f"- [{c['role']}] {c['task']} {c['model']} n={c['n']} s{c['seed']}: p={c['p']:.4f} ({c['p_source']}) -> "
                     f"p^6={100 * c['predicted_p6']:.1f}%, observed answer acc {100 * c['observed_acc']:.1f}% ({c['obs_source']}), "
                     f"diff {c['diff_pp']:+.1f} pp -> {'within' if c['within_10pp'] else 'OUTSIDE'} 10 pp. "
                     f"Post hoc: fully correct 6-digit traces {tc} vs p^6; guess model p^6 + (1 - p^6) g = {gm}"
                     + (f". Same-seed p (secondary) {c['same_seed_p_secondary']['p']:.4f} -> p^6 "
                        f"{100 * c['same_seed_p_secondary']['predicted_p6']:.1f}% ({c['same_seed_p_secondary']['diff_pp']:+.1f} pp)"
                        if c.get("same_seed_p_secondary") else ""))
        else:
            L.append(f"- [{c['role']}] {c['task']} {c['model']} s{c['seed']}: PENDING ({c['why']})")
    if o["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in o["notes"]]
    open(os.path.join(C.OUT, "steps.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
