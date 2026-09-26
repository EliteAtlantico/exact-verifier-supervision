"""analysis_stats.py -- accuracy CIs, exact McNemar, Holm, paired bootstrap for the thinking-vs-data runs.

usage:  python experiments/analysis_stats.py [--boot 10000]
writes: results/analysis/stats.json, results/analysis/stats.md

Torch-free, re-runnable: re-scans runs.jsonl, runs_*.jsonl and results/v2/runs_*.jsonl each time.

* Per run and pooled over seeds (per task x model x arm x n): accuracy with Wilson 95% CI. The pooled
  Wilson interval treats items x seeds as independent draws (it is narrower than the bootstrap below,
  which respects the item pairing).
* Per run: recovered per-item predictions (binary task: pred = gold if correct else flipped) -> P(pred=Yes),
  accuracy on gold-Yes and gold-No items.
* Exact McNemar (two-sided binomial on discordant pairs) per seed for B vs A, B vs C, A vs base, and where
  the runs exist B vs Bprime (answer-then-trace), B vs base[cot], B vs base[fewshot4] (base is seed-independent
  greedy decoding: each seed is paired with the base run of the same seed if it exists, else base seed 0).
  Runs are grouped by the test set they were scored on (analysis_common.cell_label: 'prime->prime_hard' =
  trained on prime, scored on prime_hard; 'div7[cot]' = cot prompt). Holm across tasks within each family (comparison, n, model, seed); a more
  conservative Holm across tasks x seeds (comparison, n, model) is also reported.
* Paired hierarchical bootstrap of the difference (resample seeds with replacement, and test items with
  replacement shared across seeds), 95% percentile CI.
(The preregistered decision rules S1-S6 are evaluated in analysis_report.py from this file, rules.json and steps.json.)
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import analysis_common as C

C.fix_sys_path()
import numpy as np  # noqa: E402

Z = 1.959963984540054
# (name, X arm, Y arm, Y prompt mode or None = same as X). Base runs are seed-independent greedy decoding.
COMPARISONS = [("B-A", "B", "A", None), ("B-C", "B", "C", None), ("A-base", "A", "base", None),
               ("B-Bprime", "B", "Bprime", None), ("B-base[cot]", "B", "base", "cot"),
               ("B-base[fewshot4]", "B", "base", "fewshot4")]


def wilson(k, n, z=Z):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar = binomial test on b+c discordant pairs at p=1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_bootstrap(pairs, n_boot, rng):
    """pairs: list of (bits_x, bits_y) per seed on the same items. Returns (point, lo, hi)."""
    D = np.array([np.array(x, float) - np.array(y, float) for x, y in pairs])   # S x N
    S, N = D.shape
    point = float(D.mean())
    items = rng.integers(0, N, size=(n_boot, N))
    counts = np.zeros((n_boot, N))
    np.add.at(counts, (np.repeat(np.arange(n_boot), N), items.ravel()), 1)
    M = counts @ D.T / N                                            # n_boot x S item-resampled per seed
    seeds = rng.integers(0, S, size=(n_boot, S))
    boot = np.take_along_axis(M, seeds, axis=1).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_acc(a, lo, hi):
    return f"{100 * a:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    n_boot = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 10000
    runs, problems = C.load_runs()
    tasks = C.load_tasks()
    by_key = {r["key"]: r for r in runs}

    # ---------------------------------------------------------------- per run
    per_run = []
    for r in sorted(runs, key=lambda r: (r["task"], r["model"], r["arm"], r["n"], r["seed"])):
        lo, hi = wilson(r["k"], r["n_test"])
        row = {"task": r["task"], "model": r["model"], "arm": r["arm"], "n": r["n"], "seed": r["seed"],
               "k": r["k"], "n_test": r["n_test"], "acc": r["k"] / r["n_test"], "wilson95": [lo, hi],
               "count_check": r["count_check"], "source": r["source"]}
        test = tasks.get(r["eval_task"], {}).get("test")
        if test and len(test) == r["n_test"]:
            preds = C.recover_preds(r["bits_list"], test)
            gy = [i for i, t in enumerate(test) if t["label"] == "Yes"]
            gn = [i for i, t in enumerate(test) if t["label"] == "No"]
            row["p_pred_yes"] = sum(p == "Yes" for p in preds) / len(preds)
            row["acc_on_yes"] = sum(r["bits_list"][i] for i in gy) / max(1, len(gy))
            row["acc_on_no"] = sum(r["bits_list"][i] for i in gn) / max(1, len(gn))
        per_run.append(row)

    # ---------------------------------------------------------------- pooled over seeds
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["model"], r["arm"], r["n"])].append(r)
    pooled = []
    for (t, m, a, n), rs in sorted(groups.items()):
        k = sum(r["k"] for r in rs)
        N = sum(r["n_test"] for r in rs)
        accs = [r["k"] / r["n_test"] for r in rs]
        lo, hi = wilson(k, N)
        pooled.append({"task": t, "model": m, "arm": a, "n": n, "seeds": sorted(r["seed"] for r in rs),
                       "k": k, "N": N, "acc": k / N, "wilson95": [lo, hi],
                       "seed_mean": float(np.mean(accs)),
                       "seed_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None})

    # ---------------------------------------------------------------- McNemar per seed
    tests = []
    for name, xa, ya, ymode in COMPARISONS:
        for r in runs:
            if r["arm"] != xa or (ymode and r["prompt_mode"] != "plain"):
                continue
            t, n, s, m = r["task"], r["n"], r["seed"], r["model"]
            if ya == "base":
                bl = C.cell_label(r["eval_task"], r["eval_task"], "base", ymode or r["prompt_mode"])
                ykey = next((k for k in ((bl, "base", 0, s, m), (bl, "base", 0, 0, m)) if k in by_key), None)
            else:
                ykey = (t, ya, n, s, m)
            y = by_key.get(ykey) if ykey else None
            if y is None or y["n_test"] != r["n_test"]:
                continue
            bx, by = r["bits_list"], y["bits_list"]
            b = sum(1 for u, v in zip(bx, by) if u and not v)
            c = sum(1 for u, v in zip(bx, by) if v and not u)
            tests.append({"comparison": name, "task": t, "model": m, "n": n, "seed": s,
                          "acc_x": r["k"] / r["n_test"], "acc_y": y["k"] / y["n_test"],
                          "diff": (b - c) / r["n_test"], "b_x_only": b, "c_y_only": c,
                          "p_exact": mcnemar_exact(b, c), "x_key": list(r["key"]), "y_key": list(ykey)})
    fam = defaultdict(list)
    fam_all = defaultdict(list)
    for i, tr in enumerate(tests):
        fam[(tr["comparison"], tr["n"], tr["model"], tr["seed"])].append(i)
        fam_all[(tr["comparison"], tr["n"], tr["model"])].append(i)
    for key, idx in fam.items():
        for i, p in zip(idx, holm([tests[i]["p_exact"] for i in idx])):
            tests[i]["p_holm_tasks"] = p
            tests[i]["holm_family_size"] = len(idx)
    for key, idx in fam_all.items():
        for i, p in zip(idx, holm([tests[i]["p_exact"] for i in idx])):
            tests[i]["p_holm_tasks_x_seeds"] = p
    tests.sort(key=lambda d: (d["comparison"], d["model"], d["n"], d["task"], d["seed"]))

    # ---------------------------------------------------------------- paired bootstrap
    rng = np.random.default_rng(20260925)
    boots = []
    pair_groups = defaultdict(list)
    for tr in tests:
        pair_groups[(tr["comparison"], tr["task"], tr["model"], tr["n"])].append(tr)
    for (name, t, m, n), trs in sorted(pair_groups.items()):
        pairs = [(by_key[tuple(tr["x_key"])]["bits_list"], by_key[tuple(tr["y_key"])]["bits_list"]) for tr in trs]
        point, lo, hi = paired_bootstrap(pairs, n_boot, rng)
        boots.append({"comparison": name, "task": t, "model": m, "n": n,
                      "seeds": sorted(tr["seed"] for tr in trs), "diff": point, "ci95": [lo, hi],
                      "n_boot": n_boot, "excludes_zero": bool(lo > 0 or hi < 0)})

    out = {"generated_by": "experiments/analysis_stats.py", "run_files": [os.path.relpath(f, C.ROOT)
           for f in C.run_files()], "n_runs": len(runs), "problems": problems,
           "per_run": per_run, "pooled": pooled, "mcnemar": tests, "bootstrap": boots,
           "notes": ["Predictions recovered from bits: wrong -> flipped label (an unparseable '?' answer "
                     "is indistinguishable from a flip in the bitmap).",
                     "Holm family = (comparison, n, model, seed) across tasks; p_holm_tasks_x_seeds also "
                     "pools seeds into the family (more conservative).",
                     "Bootstrap: seeds and items resampled with replacement; items shared across seeds."]}
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "stats.json"), "w", encoding="utf-8"), indent=1)
    write_md(out)
    print(f"stats: {len(runs)} runs, {len(tests)} McNemar tests, {len(boots)} bootstrap cells -> {os.path.relpath(C.OUT, C.ROOT)}")
    for p in problems:
        print("  note:", p)


def write_md(o):
    L = ["# Statistics (auto-generated by experiments/analysis_stats.py; do not edit)", "",
         f"Runs: {o['n_runs']} from {', '.join(o['run_files'])}. Accuracy in %, Wilson 95% CI.", ""]
    L.append("## Accuracy per run")
    L.append("")
    L.append("| task | model | arm | n | seed | acc % [95% CI] | P(pred=Yes) | acc on Yes | acc on No |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in o["per_run"]:
        py = f"{r['p_pred_yes']:.2f}" if "p_pred_yes" in r else "-"
        ay = f"{100 * r['acc_on_yes']:.1f}" if "acc_on_yes" in r else "-"
        an = f"{100 * r['acc_on_no']:.1f}" if "acc_on_no" in r else "-"
        L.append(f"| {r['task']} | {C.short_model(r['model'])} | {r['arm']} | {r['n']} | {r['seed']} | "
                 f"{fmt_acc(r['acc'], *r['wilson95'])} | {py} | {ay} | {an} |")
    L += ["", "### Pooled over seeds", "",
          "| task | model | arm | n | seeds | pooled acc % [95% CI] | seed mean (sd) |", "|---|---|---|---|---|---|---|"]
    for r in o["pooled"]:
        sd = f" ({100 * r['seed_sd']:.1f})" if r["seed_sd"] is not None else ""
        L.append(f"| {r['task']} | {C.short_model(r['model'])} | {r['arm']} | {r['n']} | "
                 f"{','.join(map(str, r['seeds']))} | {fmt_acc(r['acc'], *r['wilson95'])} | "
                 f"{100 * r['seed_mean']:.1f}{sd} |")
    L += ["", "## Exact McNemar per seed (diff = X - Y in pp; b = X-only correct, c = Y-only correct)", "",
          "Holm: across tasks within (comparison, n, model, seed); 'Holm t x s' also pools seeds.", "",
          "| comparison | task | model | n | seed | X % | Y % | diff pp | b | c | p exact | p Holm | p Holm t x s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in o["mcnemar"]:
        L.append(f"| {t['comparison']} | {t['task']} | {C.short_model(t['model'])} | {t['n']} | {t['seed']} | "
                 f"{100 * t['acc_x']:.1f} | {100 * t['acc_y']:.1f} | {100 * t['diff']:+.1f} | {t['b_x_only']} | "
                 f"{t['c_y_only']} | {t['p_exact']:.2g} | {t['p_holm_tasks']:.2g} | {t['p_holm_tasks_x_seeds']:.2g} |")
    L += ["", "## Paired bootstrap (items x seeds) of the difference, 95% percentile CI", "",
          "| comparison | task | model | n | seeds | diff pp [95% CI] |", "|---|---|---|---|---|---|"]
    for b in o["bootstrap"]:
        L.append(f"| {b['comparison']} | {b['task']} | {C.short_model(b['model'])} | {b['n']} | "
                 f"{','.join(map(str, b['seeds']))} | {100 * b['diff']:+.1f} [{100 * b['ci95'][0]:+.1f}, "
                 f"{100 * b['ci95'][1]:+.1f}] |")
    if o["problems"]:
        L += ["", "## Data notes", ""] + [f"- {p}" for p in o["problems"]]
    L += ["", "## Method notes", ""] + [f"- {n}" for n in o["notes"]]
    open(os.path.join(C.OUT, "stats.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
