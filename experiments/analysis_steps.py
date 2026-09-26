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
  local_arith = fraction of parsed steps with r' == (10*a + d) mod m for the model's OWN a, d, m
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
            "local_arith_acc": loc_ok / max(1, parsed_steps), "digit_copy_acc": copy_ok / max(1, parsed_steps),
            "carry_consistency": carry_ok / max(1, parsed_steps),
            "trace_correct_rate": trace_ok / n_items,
            "answer_trace_consistency": cons_ok / cons_tot if cons_tot else None, "n_consistency_items": cons_tot,
            "answer_acc_given_trace_correct": ans_ok_trace_ok / trace_ok if trace_ok else None,
            "answer_acc_given_trace_wrong": ans_ok_trace_bad / n_trace_bad if n_trace_bad else None,
            "predicted_acc_p_cond_pow_k": p_cond ** k_mode, "predicted_acc_prod_p_i": prod,
            "observed_minus_predicted_pp": 100 * (ans_ok / n_items - p_cond ** k_mode),
            "ground_truth_mismatches": truth_mismatch}


# --------------------------------------------------------------------------- prime tasks
LEN_BINS = [(1, 1), (2, 3), (4, 8), (9, 16), (17, 99)]


def score_prime(rows, test_by_id):
    out_bins = {f"{a}-{b}": {"n": 0, "ok": 0, "trunc": 0, "model_steps": 0} for a, b in LEN_BINS}
    n = ok = trunc = trunc_wrong = cap = wrong_with_line = 0
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
            "truncated_share_of_errors": (trunc_wrong / err) if err else None,
            "step_arith_acc": ar_ok / ar_tot if ar_tot else None, "n_steps_parsed": ar_tot,
            "divisor_sequence_faithful_rate": seq_ok / seq_tot if seq_tot else None,
            "answer_vs_own_conclusion_consistency": cons_ok / cons_tot if cons_tot else None,
            "by_true_trial_divisions": bins}


# --------------------------------------------------------------------------- main
def main():
    tasks = C.load_tasks()
    gens, notes = C.load_gens(tasks)
    runs, _ = C.load_runs()
    run_acc = {(r["task"], r["arm"], r["n"], r["seed"], C.short_model(r["model"])): r["k"] / r["n_test"] for r in runs}
    div_cells, prime_cells = [], []
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
        elif et.startswith("prime"):
            if not any(PRIME_STEP.search(C.gen_text(r)) for r in g["rows"]) and not arm.startswith("B"):
                continue
            sc = score_prime(g["rows"], test_by_id)
            if sc:
                prime_cells.append({**meta, **sc})
    s5 = s5_check(div_cells, run_acc)
    out = {"generated_by": "experiments/analysis_steps.py", "notes": notes, "div": div_cells, "prime": prime_cells,
           "S5": s5}
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "steps.json"), "w", encoding="utf-8"), indent=1)
    write_md(out)
    print(f"steps: {len(div_cells)} div cells, {len(prime_cells)} prime cells, S5 {s5['status']} -> "
          f"{os.path.relpath(C.OUT, C.ROOT)}/steps.json")
    for n in notes:
        print("  note:", n)


def s5_check(div_cells, run_acc):
    """Per model: p from 4-digit div7 B gens (plain), observed = B accuracy on div7_6d."""
    four = defaultdict(dict)
    for c in div_cells:
        if c["task"] == "div7" and c["arm"] == "B":
            four[c["model"]][c["seed"]] = c
    six_gens = {(c["task"], c["model"], c["seed"], c["n"]): c for c in div_cells
                if c["eval_task"] == "div7_6d" and c["arm"] == "B" and c["prompt_mode"] == "plain"}
    six_runs = {(k[0], k[4], k[3], k[2]): v for k, v in run_acc.items()
                if k[1] == "B" and k[0] in ("div7->div7_6d", "div7_6d")}
    cells = []
    for key in sorted(set(six_gens) | set(six_runs)):
        label, model, seed, n = key
        fd = four.get(model, {})
        if not fd:
            cells.append({"task": label, "model": model, "seed": seed, "n": n, "status": "PENDING",
                          "why": "no 4-digit div7 B generations for this model"})
            continue
        src = [fd[seed]] if seed in fd else list(fd.values())          # same seed, else pooled (item-weighted)
        w = sum(c["n_items"] for c in src)
        p = sum(c["p_cond"] * c["n_items"] for c in src) / w
        p2 = sum((c["p_cond_after_first_step"] or 0) * c["n_items"] for c in src) / w
        p_src = f"div7 B seed {seed}" if seed in fd else f"div7 B pooled seeds {sorted(fd)}"
        obs = six_gens[key]["answer_acc"] if key in six_gens else six_runs[key]
        pred = p ** 6
        cells.append({"task": label, "primary": label == "div7->div7_6d", "model": model, "seed": seed, "n": n,
                      "p": p, "p_source": p_src, "predicted_p6": pred, "observed_acc": obs,
                      "obs_source": "gens" if key in six_gens else "run row", "diff_pp": 100 * (obs - pred),
                      "within_10pp": abs(obs - pred) <= S5_TOL,
                      "sensitivity_first_step_free": {"p_after_first": p2, "predicted": p2 ** 5,
                                                      "note": "step 1 (r = d mod 7) is near-trivial; p' over steps 2..k, "
                                                              "prediction p'^5 (not the preregistered rule)"},
                      "observed_trace_correct": six_gens[key]["trace_correct_rate"] if key in six_gens else None})
    prim = [c for c in cells if c.get("primary") and "within_10pp" in c]
    sec = [c for c in cells if not c.get("primary") and "within_10pp" in c]
    use = prim or sec
    status = ("PENDING" if not use else ("PASS" if all(c["within_10pp"] for c in use) else "FAIL"))
    if use and not prim:
        status += " (secondary cells only: B trained on div7_6d)"
    return {"status": status, "tolerance_pp": 100 * S5_TOL, "cells": cells}


def write_md(o):
    L = ["# Step-level trace scoring (auto-generated by experiments/analysis_steps.py; do not edit)", ""]
    if o["div"]:
        L += ["## div tasks", "", "| task | model | arm | mode | n | seed | k | answer acc | trace correct | p_cond | p_uncond | "
              "p_cond^k | prod p_i | local arith | digit copy | answer-trace consistency | acc given trace wrong |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"   # noqa: E731
        for c in o["div"]:
            L.append(f"| {c['task']} | {c['model']} | {c['arm']} | {c['prompt_mode']} | {c['n']} | {c['seed']} | {c['k']} | "
                     f"{f(c['answer_acc'])} | {f(c['trace_correct_rate'])} | {f(c['p_cond'])} | {f(c['p_uncond'])} | "
                     f"{f(c['predicted_acc_p_cond_pow_k'])} | {f(c['predicted_acc_prod_p_i'])} | {f(c['local_arith_acc'])} | "
                     f"{f(c['digit_copy_acc'])} | {f(c['answer_trace_consistency'])} | {f(c['answer_acc_given_trace_wrong'])} |")
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
    L += ["", f"## S5: {o['S5']['status']}", ""]
    for c in o["S5"]["cells"]:
        if "p" in c:
            L.append(f"- {c['task']} {c['model']} n={c['n']} s{c['seed']}: p={c['p']:.4f} ({c['p_source']}) -> p^6="
                     f"{100 * c['predicted_p6']:.1f}%, observed {100 * c['observed_acc']:.1f}% ({c['obs_source']}), "
                     f"diff {c['diff_pp']:+.1f} pp -> {'within' if c['within_10pp'] else 'OUTSIDE'} 10 pp")
        else:
            L.append(f"- {c['task']} {c['model']} s{c['seed']}: PENDING ({c['why']})")
    if o["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in o["notes"]]
    open(os.path.join(C.OUT, "steps.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
