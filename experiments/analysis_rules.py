"""analysis_rules.py -- which shortcut rule do the fine-tuned models' answers follow?

usage:  python experiments/analysis_rules.py
writes: results/analysis/rules.json (+ rules.md)

Torch-free, re-runnable. Two sources of per-item predictions:
 1. runs (runs.jsonl, runs_*.jsonl, results/v2/runs_*.jsonl) whose task has a test set on disk:
    predictions recovered from the correctness bitmap (wrong -> flipped label; an unparseable '?' answer
    is indistinguishable from a flip, so these are "recovered", not observed, predictions);
 2. generations under results/v2/gens/<model_short>/<cell>.jsonl written by exp_worker_v2.py (rows: id, gold,
    pred, correct, gen, n_gen_tokens, hit_cap). When a run has a matching gens file the prediction is OBSERVED
    (unparseable '?' counted, cap hits counted, and the bits are cross-checked); gens files without a run row
    are analysed too. Skipped gracefully when the directory does not exist.

For every run: agreement (fraction of items) of the model's predictions with each candidate rule, the
best-agreeing rule, and per-stratum behaviour:
  prime*: rules odd, last digit in {1,3,7,9}, no factor <= 3/5/7/11/13/..., true label; strata prime /
          even composite / odd composite with a factor <= 7 / hard composite (odd, no factor <= 7).
          S6 (PREDICTIONS.md): on hard composites, agreement with "no factor <= 7" (= P(pred prime)) vs with
          the true label, and whether accuracy there is < 60%.
  divK:   true label, last-digit rules, digit-sum rules, majority; plus the best agreement achievable by
          ANY function of the last digit (and of the digit sum), against a 1000-permutation null, i.e.
          "how much of the model's answer is explained by the last digit beyond chance".
  valid:  true label, majority, accuracy per argument schema.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

import analysis_common as C

C.fix_sys_path()

ARMS_FIRST = ("A", "B")


def agree(preds, ref):
    return sum(p == r for p, r in zip(preds, ref)) / len(preds)


def kappa(preds, ref):
    """Cohen's kappa: agreement corrected for the two marginal Yes-rates (0 = chance, 1 = identical)."""
    n = len(preds)
    po = agree(preds, ref)
    py, ry = sum(p == "Yes" for p in preds) / n, sum(r == "Yes" for r in ref) / n
    pe = py * ry + (1 - py) * (1 - ry)
    return 0.0 if pe >= 1 else (po - pe) / (1 - pe)


def best_fn_agreement(keys, preds):
    tab = defaultdict(Counter)
    for k, p in zip(keys, preds):
        tab[k][p] += 1
    return sum(max(c.values()) for c in tab.values()) / len(preds)


def perm_null(keys, preds, reps=1000, seed=0):
    rng = random.Random(seed)
    p = list(preds)
    vals = []
    for _ in range(reps):
        rng.shuffle(p)
        vals.append(best_fn_agreement(keys, p))
    vals.sort()
    return sum(vals) / reps, vals[int(0.95 * reps)]


def analyze(task, nums, preds, golds, prompts=None, ids=None):
    """nums: list of ints or None (text task); preds/golds: 'Yes'/'No'/'?'."""
    out = {"n_items": len(preds), "acc": agree(preds, golds),
           "p_pred_yes": sum(p == "Yes" for p in preds) / len(preds),
           "n_unparsed": sum(p == "?" for p in preds)}
    k = C.divisor_of_task(task)
    if task.startswith("prime") and nums:
        R = C.prime_rules()
        out["agreement"] = {name: agree(preds, [f(n) for n in nums]) for name, f in R.items()}
        out["kappa"] = {name: kappa(preds, [f(n) for n in nums]) for name, f in R.items()}
        strata = defaultdict(list)
        for i, n in enumerate(nums):
            strata[C.prime_stratum(n)].append(i)
        out["strata"] = {s: {"n": len(ix), "acc": agree([preds[i] for i in ix], [golds[i] for i in ix]),
                             "p_pred_yes": sum(preds[i] == "Yes" for i in ix) / len(ix)}
                         for s, ix in sorted(strata.items())}
        hard = strata.get("hard_composite_no_factor_le7", [])
        if hard:
            hp = [preds[i] for i in hard]
            a_rule = sum(p == "Yes" for p in hp) / len(hp)
            a_true = sum(p == "No" for p in hp) / len(hp)
            out["S6_hard_negatives"] = {"n_hard": len(hp), "agree_rule_no_factor_le7": a_rule,
                                        "agree_true_label": a_true,
                                        "rule_beats_truth": a_rule > a_true, "acc_below_60": a_true < 0.60}
    elif k and nums:
        R = C.divk_rules(k)
        out["agreement"] = {name: agree(preds, [f(n) for n in nums]) for name, f in R.items()}
        out["kappa"] = {name: kappa(preds, [f(n) for n in nums]) for name, f in R.items()}
        for fname, keyf in (("last_digit", lambda n: n % 10), ("digit_sum", C.digit_sum)):
            keys = [keyf(n) for n in nums]
            obs = best_fn_agreement(keys, preds)
            mu, q95 = perm_null(keys, preds)
            out[f"best_fn_of_{fname}"] = {"agreement": obs, "perm_null_mean": mu, "perm_null_q95": q95,
                                          "excess_over_null": obs - mu, "beyond_null_q95": obs > q95}
        by_ld = defaultdict(list)
        for n, p in zip(nums, preds):
            by_ld[n % 10].append(p == "Yes")
        out["p_pred_yes_by_last_digit"] = {str(d): sum(v) / len(v) for d, v in sorted(by_ld.items())}
    else:
        maj = Counter(golds).most_common(1)[0][0]
        out["agreement"] = {"true_label": agree(preds, golds), "always_" + maj: agree(preds, [maj] * len(preds))}
        if ids:
            by_schema = defaultdict(list)
            for i, iid in enumerate(ids):
                m = re.match(r"valid-\d+-(.+)$", iid or "")
                if m:
                    by_schema[m.group(1)].append(i)
            if by_schema:
                out["acc_by_schema"] = {s: {"n": len(ix), "acc": agree([preds[i] for i in ix], [golds[i] for i in ix])}
                                        for s, ix in sorted(by_schema.items())}
    ag, kp = out.get("agreement", {}), out.get("kappa", {})
    nontrivial = {r: v for r, v in ag.items() if not r.startswith("always_")}
    if nontrivial:                         # ranked by raw agreement; kappa guards against base-rate artefacts
        out["best_rule"] = max(nontrivial, key=lambda r: (nontrivial[r], r != "true_label"))
        out["best_rule_agreement"] = nontrivial[out["best_rule"]]
        shortcut = {r: v for r, v in nontrivial.items() if r != "true_label"}
        if shortcut and kp:
            out["best_shortcut_rule"] = max(shortcut, key=lambda r: kp.get(r, -1))
            out["best_shortcut_agreement"] = shortcut[out["best_shortcut_rule"]]
            out["best_shortcut_kappa"] = kp[out["best_shortcut_rule"]]
            out["true_label_kappa"] = kp.get("true_label")
    return out


def main():
    runs, problems = C.load_runs()
    tasks = C.load_tasks()
    gens, gnotes = C.load_gens(tasks)
    results, notes = [], list(problems) + gnotes
    used_gens = set()
    for r in sorted(runs, key=lambda r: (r["task"], r["arm"] not in ARMS_FIRST, r["arm"], r["n"], r["seed"])):
        test = tasks.get(r["eval_task"], {}).get("test")
        if not test or len(test) != r["n_test"]:
            notes.append(f"no test set on disk for {r['eval_task']} ({r['source']}); rule analysis skipped")
            continue
        golds = [t["label"] for t in test]
        gkey = (r["task"], r["arm"], r["n"], r["seed"], C.short_model(r["model"]))
        g = gens.get(gkey)
        extra = {}
        if g and [row.get("id") for row in g["rows"]] == [t["id"] for t in test]:
            preds = [row["pred"] for row in g["rows"]]
            src = "observed_generations"
            used_gens.add(gkey)
            gbits = [int(row.get("correct", p == gd)) for row, p, gd in zip(g["rows"], preds, golds)]
            extra = {"gens_path": g["path"], "gens_match_bits": gbits == r["bits_list"],
                     "n_hit_cap": sum(bool(row.get("hit_cap")) for row in g["rows"]),
                     "mean_gen_tokens": sum(row.get("n_gen_tokens") or 0 for row in g["rows"]) / len(g["rows"])}
        else:
            preds = C.recover_preds(r["bits_list"], test)
            src = "recovered_from_bits"
        nums = None if r["eval_task"].startswith("valid") else [C.number_of(t["prompt"]) for t in test]
        a = analyze(r["eval_task"], nums, preds, golds, ids=[t["id"] for t in test])
        results.append({"task": r["task"], "eval_task": r["eval_task"], "arm": r["arm"], "n": r["n"],
                        "seed": r["seed"], "model": r["model"], "pred_source": src, **extra, **a})
    for gkey, g in sorted(gens.items()):                   # generations with no run row
        if gkey in used_gens:
            continue
        label, arm, n, seed, model = gkey
        test = tasks.get(g["eval_task"], {}).get("test")
        if not test:
            notes.append(f"gens {g['path']}: no test set for {g['eval_task']}; skipped")
            continue
        by_id = {t["id"]: t for t in test}
        rows = [row for row in g["rows"] if row.get("id") in by_id]
        if not rows:
            continue
        items = [by_id[row["id"]] for row in rows]
        preds, golds = [row.get("pred", "?") for row in rows], [t["label"] for t in items]
        nums = None if g["eval_task"].startswith("valid") else [C.number_of(t["prompt"]) for t in items]
        a = analyze(g["eval_task"], nums, preds, golds, ids=[t["id"] for t in items])
        results.append({"task": label, "eval_task": g["eval_task"], "arm": arm, "n": n, "seed": seed, "model": model,
                        "pred_source": "observed_generations", "gens_path": g["path"], "no_run_row": True,
                        "n_hit_cap": sum(bool(row.get("hit_cap")) for row in rows), **a})
    s6 = [x for x in results if x["eval_task"].startswith("prime") and x["arm"] == "A" and "S6_hard_negatives" in x]
    out = {"generated_by": "experiments/analysis_rules.py", "notes": notes, "runs": results,
           "S6": [{"task": x["task"], "n": x["n"], "seed": x["seed"], "model": x["model"], "pred_source": x["pred_source"],
                   **x["S6_hard_negatives"]} for x in s6]}
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "rules.json"), "w", encoding="utf-8"), indent=1)
    write_md(out)
    print(f"rules: {len(results)} runs analysed -> {os.path.relpath(C.OUT, C.ROOT)}/rules.json, rules.md")
    for n in notes:
        print("  note:", n)


def write_md(o):
    L = ["# Rule agreement (auto-generated by experiments/analysis_rules.py; do not edit)", "",
         "Agreement = fraction of test items where the model's (recovered) answer equals the rule's answer. "
         "'best shortcut' = the non-trivial, non-truth rule with the highest Cohen's kappa (agreement corrected "
         "for the Yes-rates, so an always-No model does not 'agree' with rare-Yes rules).", ""]
    for task in sorted({x["task"] for x in o["runs"]}):
        rows = [x for x in o["runs"] if x["task"] == task]
        rules = list(rows[0].get("agreement", {}).keys())
        L += [f"## {task}", "", "| arm | n | seed | model | src | acc | P(Yes) | " + " | ".join(rules) + " | best shortcut |",
              "|---|---|---|---|---|---|---|" + "---|" * (len(rules) + 1)]
        for x in rows:
            L.append(f"| {x['arm']} | {x['n']} | {x['seed']} | {C.short_model(x['model'])} | "
                     f"{'bits' if x['pred_source'].startswith('rec') else 'gens'} | {100 * x['acc']:.1f} | {x['p_pred_yes']:.2f} | "
                     + " | ".join(f"{100 * x['agreement'].get(r, float('nan')):.1f}" for r in rules)
                     + (f" | {x['best_shortcut_rule']} ({100 * x['best_shortcut_agreement']:.1f}; kappa {x['best_shortcut_kappa']:.2f}"
                        f" vs truth {x['true_label_kappa']:.2f}) |" if "best_shortcut_rule" in x else " | - |"))
        if rows[0]["eval_task"].startswith("prime"):
            strata = sorted({s for x in rows for s in x.get("strata", {})})
            L += ["", "Per stratum: accuracy % / P(pred prime)", "", "| arm | n | seed | " + " | ".join(
                f"{s} (n={rows[0]['strata'].get(s, {}).get('n', '?')})" for s in strata) + " |",
                  "|---|---|---|" + "---|" * len(strata)]
            for x in rows:
                L.append(f"| {x['arm']} | {x['n']} | {x['seed']} | " + " | ".join(
                    f"{100 * x['strata'][s]['acc']:.0f} / {x['strata'][s]['p_pred_yes']:.2f}" if s in x.get("strata", {}) else "-"
                    for s in strata) + " |")
        if C.divisor_of_task(rows[0]["eval_task"]):
            L += ["", "Best agreement by ANY function of the last digit / digit sum (vs permutation-null mean, q95)", "",
                  "| arm | n | seed | last digit | null mean / q95 | digit sum | null mean / q95 |", "|---|---|---|---|---|---|---|"]
            for x in rows:
                a, b = x.get("best_fn_of_last_digit"), x.get("best_fn_of_digit_sum")
                if a and b:
                    L.append(f"| {x['arm']} | {x['n']} | {x['seed']} | {100 * a['agreement']:.1f} | {100 * a['perm_null_mean']:.1f} / "
                             f"{100 * a['perm_null_q95']:.1f} | {100 * b['agreement']:.1f} | {100 * b['perm_null_mean']:.1f} / {100 * b['perm_null_q95']:.1f} |")
        if any("acc_by_schema" in x for x in rows):
            schemas = sorted({s for x in rows for s in x.get("acc_by_schema", {})})
            L += ["", "Accuracy % by schema", "", "| arm | n | seed | " + " | ".join(schemas) + " |", "|---|---|---|" + "---|" * len(schemas)]
            for x in rows:
                L.append(f"| {x['arm']} | {x['n']} | {x['seed']} | " + " | ".join(
                    f"{100 * x['acc_by_schema'][s]['acc']:.0f}" if s in x.get("acc_by_schema", {}) else "-" for s in schemas) + " |")
        L.append("")
    if o["S6"]:
        L += ["## S6 check (hard negatives = odd composites with no factor <= 7)", "",
              "| task | n | seed | src | #hard | agree 'no factor<=7' | agree truth (=acc) | rule > truth | acc < 60% |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in o["S6"]:
            L.append(f"| {x['task']} | {x['n']} | {x['seed']} | {x["pred_source"]} | {x["n_hard"]} | {100 * x["agree_rule_no_factor_le7"]:.1f} | "
                     f"{100 * x['agree_true_label']:.1f} | {x['rule_beats_truth']} | {x['acc_below_60']} |")
        L.append("")
    if o["notes"]:
        L += ["## Notes", ""] + [f"- {n}" for n in o["notes"]]
    open(os.path.join(C.OUT, "rules.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
