"""analysis_coverage.py -- per-transition coverage law for divisibility traces (POST HOC, not preregistered).

usage:  python experiments/analysis_coverage.py
writes: results/analysis/coverage.json (+ coverage.md)

A div-by-d trace writes "r = (10*a + x) mod d = r'." after every digit, so each step is a lookup of one entry
(a, x) of a 10d table. For every trace-trained cell with saved generations (arm B, plain prompt, trained and
tested on the same task: div2/3/7/11/13, div7_6d; every n, seed and model size on disk) we

 1. rebuild the n training traces with analysis_common.balanced_sample (byte-for-byte replica of the worker's
    sampler, same datasets_v2 pools) and count c(a, x) = occurrences of the entry in those n traces
    (parsed with analysis_steps.DIV_STEP, so the first step's (0, x) entries are counted like any other);
 2. parse every test generation with the same regex and score each step.

Step correctness (two definitions, both reported; LOOKUP is primary, fixed before looking at results):
  LOOKUP = the entry the model actually wrote: (a, x) as claimed in the step; correct iff r' == (10a + x) mod d.
           Isolates the table lookup from copy/carry errors. Steps with m != d, a >= d or x > 9 are "off-table".
  LOCAL  = entry (model's own previous remainder, TRUE digit at that position), over the k true positions;
           correct iff r' == (10 r_prev + x) mod d (the non-circular local step of analysis_steps). Missing steps
           (trace too short) are counted separately and excluded from the curve.
"trivial" entries are those whose result equals x mod d without using a (a = 0, or d | 10 as in div2).

Analyses: binned accuracy vs c (per task, per model, trivial vs non-trivial); logistic regressions of
step-correct on log(1+c) with task and size terms (GLM binomial, standard errors clustered by cell x entry),
compared by AIC with (i) the pooled cell-level m = k n / (10 d) of transitions.json, (ii) cell fixed effects;
a within-cell test (cell fixed effects + log(1+c)); a seed-noise natural experiment (c split into its
structural expectation E[c] over 200 resampled seeds and the sample-specific deviation, which is random by
construction); c = 0 accuracy against chance 1/d and against the same entries' accuracy in cells where they
were seen; the threshold count above which lookups are nearly always right; d = 11 wrap entries (x < a).
Secondary cells (reported, not in the primary fit): the 4-digit div7 adapter scored on div7_6d (transfer),
arms Bprime (answer-first trace) and D (fluent trace of a different instance), and the base model with the
4 fixed few-shot demos (c counted from those 4 demos).
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import Counter, defaultdict

import numpy as np

import analysis_common as C
import analysis_steps as S

C.fix_sys_path()
warnings.filterwarnings("ignore")

BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10 ** 9)]
N_RESAMPLE = 200
NEAR_ALWAYS = 0.99


def bin_of(c):
    for lo, hi in BINS:
        if lo <= c <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}")
    return "?"


BIN_LABELS = [bin_of(lo) for lo, _ in BINS]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def acc_block(k, n):
    lo, hi = wilson(k, n)
    return {"n_steps": n, "n_correct": k, "acc": (k / n) if n else None, "ci95": [lo, hi]}


def size_b(model):
    for tag in ("0.5B", "1.5B", "3B", "7B"):
        if f"-{tag}" in model:
            return tag
    return model


def train_counts(pool, n, seed, d):
    """c(a, x) over the n picked training traces, plus parse diagnostics."""
    picked = C.balanced_sample(pool, n, seed)
    cnt = Counter()
    bad = 0
    for e in picked:
        for a, x, m, r in (tuple(int(v) for v in t) for t in S.DIV_STEP.findall(e["completion"])):
            if m != d or r != (10 * a + x) % d:
                bad += 1
            cnt[(a, x)] += 1
    return cnt, picked, bad


def expected_counts(pool, n, d, seeds):
    tot = Counter()
    for s in seeds:
        c, _, _ = train_counts(pool, n, s, d)
        tot.update(c)
    return {k: v / len(seeds) for k, v in tot.items()}


def score_steps(rows, test_by_id, d, cnt, exp_cnt, cell_id):
    """One record per step for LOOKUP and LOCAL definitions."""
    look, loc = [], []
    miss = offt = 0
    for row in rows:
        item = test_by_id.get(row.get("id"))
        if item is None:
            continue
        num = C.number_of(item["prompt"])
        digits = [int(ch) for ch in str(num)]
        truth = C.rems_of(num, d)
        k = len(truth)
        steps = [tuple(int(v) for v in t) for t in S.DIV_STEP.findall(C.gen_text(row))]
        for i, (a, x, m, r) in enumerate(steps):
            if m != d or a >= d or x > 9:
                offt += 1
                continue
            look.append({"cell": cell_id, "item": row["id"], "a": a, "x": x, "pos": i,
                         "c": cnt.get((a, x), 0), "ec": exp_cnt.get((a, x), 0.0),
                         "trivial": int(a == 0 or 10 % d == 0), "wrap11": int(d == 11 and x < a),
                         "out": r, "ok": int(r == (10 * a + x) % d)})
        prev = 0
        prev_on_path = True
        for i in range(k):
            if i >= len(steps):
                miss += 1
                break
            xi = digits[i]
            rr = steps[i][3]
            if prev < d:
                loc.append({"cell": cell_id, "item": row["id"], "a": prev, "x": xi, "pos": i,
                            "c": cnt.get((prev, xi), 0), "ec": exp_cnt.get((prev, xi), 0.0),
                            "trivial": int(prev == 0 or 10 % d == 0), "wrap11": int(d == 11 and xi < prev),
                            "on_path": int(prev_on_path), "ok": int(rr == (10 * prev + xi) % d)})
            prev_on_path = prev_on_path and rr == truth[i]
            prev = rr
    return look, loc, miss, offt


def curve(recs, key=None):
    """Binned accuracy vs c (optionally split by key(rec))."""
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        g = key(r) if key else "all"
        b = bin_of(r["c"])
        agg[(g, b)][0] += r["ok"]
        agg[(g, b)][1] += 1
    out = defaultdict(dict)
    for (g, b), (k, n) in agg.items():
        out[str(g)][b] = acc_block(k, n)
    return {g: {b: v[b] for b in BIN_LABELS if b in v} for g, v in sorted(out.items())}


def threshold(recs, target=NEAR_ALWAYS, min_steps=100):
    """Smallest c* such that the pooled accuracy over steps with c >= c* is >= target AND every bin at or above
    c*'s bin (with >= 30 steps) is >= target - 0.02; also the Wilson lower bound over c >= c*."""
    cs = sorted({r["c"] for r in recs})
    by_c = defaultdict(lambda: [0, 0])
    for r in recs:
        by_c[r["c"]][0] += r["ok"]
        by_c[r["c"]][1] += 1
    best = None
    for cstar in cs:
        k = sum(v[0] for c, v in by_c.items() if c >= cstar)
        n = sum(v[1] for c, v in by_c.items() if c >= cstar)
        if n < min_steps:
            break
        if k / n >= target:
            ok = True
            for lo, hi in BINS:
                kk = sum(v[0] for c, v in by_c.items() if c >= cstar and lo <= c <= hi)
                nn = sum(v[1] for c, v in by_c.items() if c >= cstar and lo <= c <= hi)
                if nn >= 30 and kk / nn < target - 0.02:
                    ok = False
            if ok:
                best = {"c_star": cstar, **acc_block(k, n),
                        "share_of_steps_at_or_above": n / len(recs),
                        "acc_below": acc_block(sum(v[0] for c, v in by_c.items() if c < cstar),
                                               sum(v[1] for c, v in by_c.items() if c < cstar))}
                break
    return best or {"c_star": None, "note": f"no c* with >= {min_steps} steps reaches {target}"}


def fit_glm(df, formula, cluster_col="cluster"):
    import patsy
    import patsy.builtins
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    env = patsy.EvalEnvironment([{"C": patsy.builtins.C, "Treatment": patsy.builtins.Treatment}])   # module name C shadows patsy's C
    try:
        m = smf.glm(formula, data=df, family=sm.families.Binomial(), eval_env=env)
        res = m.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col].astype("category").cat.codes})
    except Exception as e:                     # perfect separation etc.
        try:
            res = smf.glm(formula, data=df, family=sm.families.Binomial(), eval_env=env).fit()
        except Exception as e2:
            return {"formula": formula, "error": f"{e}; {e2}"}
    params = {k: float(v) for k, v in res.params.items()}
    ses = {k: float(v) for k, v in res.bse.items()}
    pv = {k: float(v) for k, v in res.pvalues.items()}
    return {"formula": formula, "n": int(res.nobs), "llf": float(res.llf), "aic": float(res.aic),
            "df_model": float(res.df_model), "params": params, "se_cluster": ses, "p": pv}


def strip(f):
    return {k: v for k, v in f.items() if k != "_res"}


def lr_test(small, big):
    from scipy.stats import chi2
    if "llf" not in small or "llf" not in big:
        return None
    stat = 2 * (big["llf"] - small["llf"])
    ddf = big["df_model"] - small["df_model"]
    return {"chi2": stat, "df": ddf, "p": float(chi2.sf(stat, ddf)) if ddf > 0 else None,
            "note": "likelihood-ratio (ignores clustering; the cluster-robust Wald p of the added terms is in params)"}


def c_for_p(fit, p, task, size, trivial=0):
    """Invert logit(p) = b0 + b1 log(1+c) + task + size (+ trivial) for c."""
    pr = fit["params"]
    b0 = pr.get("Intercept", 0.0)
    b1 = pr.get("lc")
    if not b1 or b1 <= 0:
        return None
    t = pr.get(f"C(task, Treatment('div7'))[T.{task}]", 0.0)
    s = pr.get(f"C(size, Treatment('1.5B'))[T.{size}]", 0.0)
    tv = pr.get("trivial", 0.0) * trivial
    lc = (math.log(p / (1 - p)) - b0 - t - s - tv) / b1
    return math.expm1(lc) if lc < 50 else float("inf")


def main():
    import pandas as pd
    tasks = C.load_tasks()
    gens, notes = C.load_gens(tasks)
    runs, _ = C.load_runs()
    run_tok = {(r["task"], r["arm"], int(r["n"]), int(r["seed"]), C.short_model(r["model"])): r.get("train_tokens")
               for r in runs}
    cells, look_all, loc_all = [], [], []
    exp_cache = {}
    for key, g in sorted(gens.items()):
        label, arm, n, seed, model = key
        et, tt = g["eval_task"], g["train_task"]
        d = C.divisor_of_task(et)
        if not d or arm not in ("B", "Bprime", "D", "base"):
            continue
        rows = g["rows"]
        if not any(S.DIV_STEP.search(C.gen_text(r)) for r in rows):
            continue
        if arm == "base":
            if g["prompt_mode"] != "fewshot4":
                continue
            ids = set(json.load(open(os.path.join(C.V2, "datasets_v2.json"), encoding="utf-8"))["fewshot4"][tt])
            pool = [e for e in tasks[tt]["pools"]["B"] if e.get("id") in ids] or \
                   [e for e in tasks[tt]["pools"]["B"] if f"{tt}-{C.number_of(e['prompt'])}" in ids]
            cnt = Counter()
            for e in pool:
                for a, x, m, r in (tuple(int(v) for v in t) for t in S.DIV_STEP.findall(e["completion"])):
                    cnt[(a, x)] += 1
            picked, bad, tok_ok, ec = pool, 0, None, {k: float(v) for k, v in cnt.items()}
            role = "base_fewshot4"
        else:
            pool = tasks[tt]["pools"][arm]
            cnt, picked, bad = train_counts(pool, n, seed, C.divisor_of_task(tt))
            tok = sum(len(e["completion"].split()) for e in picked)
            rt = run_tok.get((label, arm, n, seed, model))
            tok_ok = None if rt is None else (rt == tok)
            ck = (tt, arm, n)
            if ck not in exp_cache:
                exp_cache[ck] = expected_counts(pool, n, C.divisor_of_task(tt), range(1000, 1000 + N_RESAMPLE))
            ec = exp_cache[ck]
            if arm == "B" and g["prompt_mode"] == "plain" and tt == et:
                role = "primary"
            elif arm == "B" and g["prompt_mode"] == "plain":
                role = "transfer"
            else:
                role = f"arm_{arm}" if g["prompt_mode"] == "plain" else f"arm_{arm}[{g['prompt_mode']}]"
        cell_id = f"{model}|{label}|{arm}|{n}|{seed}"
        test_by_id = {t["id"]: t for t in tasks[et]["test"]}
        look, loc, miss, offt = score_steps(rows, test_by_id, d, cnt, ec, cell_id)
        k_train = len(str(C.number_of(picked[0]["prompt"]))) if picked else None
        meta = {"cell": cell_id, "role": role, "model": model, "size": size_b(model), "task": label,
                "train_task": tt, "eval_task": et, "arm": arm, "n": n, "seed": seed, "d": d,
                "k_train": k_train, "m_pooled": (k_train * n / (10 * d)) if (n and k_train) else None}
        for r in look + loc:
            r.update({"role": role, "model": model, "size": meta["size"], "task": et, "d": d, "n": n,
                      "m": meta["m_pooled"]})
        look_all += look
        loc_all += loc
        entries = 10 * d
        seen = sum(1 for a in range(d) for x in range(10) if cnt.get((a, x), 0) > 0)
        cells.append({**meta, "gens_path": g["path"], "train_trace_steps": sum(cnt.values()),
                      "train_steps_bad_arith": bad, "train_tokens_match_run_row": tok_ok,
                      "table_entries": entries, "entries_seen_in_training": seen,
                      "entries_unseen": entries - seen,
                      "lookup": acc_block(sum(r["ok"] for r in look), len(look)),
                      "local": acc_block(sum(r["ok"] for r in loc), len(loc)),
                      "missing_steps": miss, "off_table_steps": offt})
        print(f"{cell_id:55s} {role:14s} lookup {len(look):5d} acc "
              f"{(sum(r['ok'] for r in look) / max(1, len(look))):.3f}  unseen entries {entries - seen}")

    out = {"generated_by": "experiments/analysis_coverage.py", "label": "POST HOC analysis (not preregistered)",
           "definitions": {"c": "occurrences of entry (a, x) in the n training traces of the cell (balanced_sample replica)",
                           "LOOKUP": "claimed entry (a, x) of each parsed step; correct iff r' == (10a+x) mod d (PRIMARY)",
                           "LOCAL": "entry (model's previous remainder, true digit); correct iff r' == (10 r_prev + x) mod d",
                           "trivial": "a == 0 or d | 10 (result is x mod d)",
                           "E[c]": f"mean count over {N_RESAMPLE} resampled seeds (1000..{999 + N_RESAMPLE}) of the same pool and n",
                           "threshold": f"smallest c* with pooled acc >= {NEAR_ALWAYS} over steps with c >= c* (>= 100 steps) "
                                        "and no c-bin (>= 30 steps) at or above c* below target - 0.02"},
           "cells": cells, "notes": notes}

    res = {}
    for defn, recs in (("lookup", look_all), ("local", loc_all)):
        prim = [r for r in recs if r["role"] == "primary"]
        R = {"n_steps_primary": len(prim), "acc_primary": acc_block(sum(r["ok"] for r in prim), len(prim))}
        R["curve_all"] = curve(prim)
        R["curve_by_trivial"] = curve(prim, lambda r: "trivial" if r["trivial"] else "nontrivial")
        R["curve_by_task_nontrivial"] = curve([r for r in prim if not r["trivial"]], lambda r: r["task"])
        R["curve_by_size_nontrivial"] = curve([r for r in prim if not r["trivial"]], lambda r: r["size"])
        R["curve_by_task_size_nontrivial"] = curve([r for r in prim if not r["trivial"]],
                                                   lambda r: f"{r['task']}|{r['size']}")
        R["curve_secondary_roles"] = curve([r for r in recs if r["role"] != "primary"], lambda r: r["role"])
        R["threshold_pooled"] = threshold(prim)
        R["threshold_nontrivial"] = threshold([r for r in prim if not r["trivial"]])
        R["threshold_by_size_nontrivial"] = {s: threshold([r for r in prim if r["size"] == s and not r["trivial"]])
                                             for s in sorted({r["size"] for r in prim})}
        R["threshold_by_task_nontrivial"] = {t: threshold([r for r in prim if r["task"] == t and not r["trivial"]])
                                             for t in sorted({r["task"] for r in prim})}
        # ---- c = 0 : generalization vs lookup
        z = [r for r in prim if r["c"] == 0]
        zz = {}
        for t in sorted({r["task"] for r in z}):
            for s in sorted({r["size"] for r in z if r["task"] == t}):
                sub = [r for r in z if r["task"] == t and r["size"] == s]
                ents = {(r["a"], r["x"]) for r in sub}
                d = sub[0]["d"]
                # the same entries in same-task same-size primary cells where they WERE seen (c >= 1)
                seen_same = [r for r in prim if r["task"] == t and r["size"] == s and r["c"] >= 1
                             and (r["a"], r["x"]) in ents]
                ntriv = [r for r in sub if not r["trivial"]]
                zz[f"{t}|{s}"] = {"c0": acc_block(sum(r["ok"] for r in sub), len(sub)),
                                  "c0_nontrivial": acc_block(sum(r["ok"] for r in ntriv), len(ntriv)),
                                  "distinct_c0_entries": len(ents), "chance_1_over_d": 1 / d,
                                  "same_entries_when_seen_c_ge_1": acc_block(sum(r["ok"] for r in seen_same),
                                                                             len(seen_same)),
                                  "binomial_p_vs_chance": _binom_p(sum(r["ok"] for r in sub), len(sub), 1 / d)}
        R["c0"] = zz
        R["c0_pooled"] = acc_block(sum(r["ok"] for r in z), len(z))
        # ---- d = 11 wrap entries (x < a: needs the borrow in (x - a) mod 11)
        e11 = [r for r in prim if r["d"] == 11 and not r["trivial"]]
        R["div11_wrap"] = {"by_wrap": curve(e11, lambda r: "wrap_x_lt_a" if r["wrap11"] else "no_wrap_x_ge_a")}
        # ---- within-cell rank correlation of entry accuracy with c
        R["within_cell_spearman"] = within_cell_spearman(prim)
        # ---- lookup-regime tasks (no +-1 multiplier: d = 7, 13): same c, different n / task
        nt47 = [r for r in prim if not r["trivial"] and r["d"] in (7, 13)]
        R["curve_div13_by_n"] = curve([r for r in nt47 if r["task"] == "div13"], lambda r: f"n={r['n']}|{r['size']}")
        R["curve_div7_by_n"] = curve([r for r in nt47 if r["task"] == "div7"], lambda r: f"n={r['n']}|{r['size']}")
        R["curve_n180_4digit_by_task"] = curve([r for r in nt47 if r["n"] == 180 and r["task"] in ("div7", "div13")],
                                               lambda r: r["task"])
        # ---- base model (4 few-shot demos) on the entries that fine-tuned cells never saw
        base = [r for r in recs if r["role"] == "base_fewshot4"]
        btasks = {b["task"] for b in base}
        c0_ents = {(r["task"], r["a"], r["x"]) for r in z}
        bsame = [r for r in base if (r["task"], r["a"], r["x"]) in c0_ents]
        bnt = [r for r in base if not r["trivial"]]
        zb = [r for r in z if r["task"] in btasks]
        R["base_fewshot4_on_finetune_c0_entries"] = {
            "base_acc_same_entries": acc_block(sum(r["ok"] for r in bsame), len(bsame)),
            "finetuned_acc_same_entries_at_c0": acc_block(sum(r["ok"] for r in zb), len(zb)),
            "base_acc_all_nontrivial": acc_block(sum(r["ok"] for r in bnt), len(bnt)),
            "note": "base = Qwen2.5-1.5B-Instruct with the 4 fixed demos (div7 only on disk)"}
        # ---- error anatomy of wrong non-trivial lookups
        R["error_anatomy_nontrivial"] = error_anatomy([r for r in prim if not r["trivial"]]) if defn == "lookup" else None
        # ---- logistic fits
        R["fits"] = fits(pd, prim, recs)
        res[defn] = R
    out["results"] = res
    out["seed_noise_placebo"] = res["lookup"]["fits"].pop("_placebo_note", None)
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "coverage.json"), "w", encoding="utf-8"), indent=1, default=float)
    write_md(out)
    print("coverage ->", os.path.relpath(os.path.join(C.OUT, "coverage.json"), C.ROOT))


def _binom_p(k, n, p0):
    from scipy.stats import binomtest
    if n == 0:
        return None
    return float(binomtest(k, n, p0, alternative="greater").pvalue)


def error_anatomy(recs):
    """What does a wrong lookup output? Shares among wrong non-trivial lookups, by c band."""
    bands = [("c=0", 0, 0), ("c=1-4", 1, 4), ("c=5-16", 5, 16), ("c>=17", 17, 10 ** 9)]
    out = {}
    for name, lo, hi in bands:
        w = [r for r in recs if not r["ok"] and lo <= r["c"] <= hi]
        cnt, null = Counter(), Counter()

        def cats(a, x, d):
            return [("ignores_a_(x_mod_d)", {x % d}), ("10_as_1_((a+x)_mod_d)", {(a + x) % d}),
                    ("10_as_minus1_((x-a)_mod_d)", {(x - a) % d}),
                    ("neighbour_row_(a+-1)", {(10 * ((a + 1) % d) + x) % d, (10 * ((a - 1) % d) + x) % d}),
                    ("off_by_one", {(10 * a + x + 1) % d, (10 * a + x - 1) % d})]
        for r in w:
            a, x, d, o = r["a"], r["x"], r["d"], r["out"]
            right = (10 * a + x) % d
            if o >= d:
                cnt["out_of_range_ge_d"] += 1
            else:
                hit = next((nm for nm, vals in cats(a, x, d) if o in vals), "other")
                cnt[hit] += 1
            used = {right}                                    # null: a wrong output uniform over the d-1 wrong values
            for nm, vals in cats(a, x, d):
                new = vals - used
                null[nm] += len(new) / (d - 1)
                used |= new
            null["other"] += (d - len(used)) / (d - 1)
        n = len(w)
        n_in = sum(v for k, v in cnt.items() if k != "out_of_range_ge_d")
        out[name] = {"n_wrong": n, "shares": {k: v / n for k, v in cnt.most_common()} if n else {},
                     "uniform_null_shares_in_range": {k: v / n for k, v in null.most_common()} if n else {},
                     "n_in_range": n_in}
    out["note"] = ("categories checked in order; uniform_null = expected share if a wrong in-range output were uniform "
                   "over the d-1 wrong values (same order, same records)")
    return out


def within_cell_spearman(prim):
    from scipy.stats import spearmanr
    by = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in prim:
        if r["trivial"]:
            continue
        e = by[r["cell"]][(r["a"], r["x"])]
        e[0] += r["ok"]
        e[1] += 1
        e[2] = r["c"]
    rows = []
    for cell, ents in sorted(by.items()):
        pts = [(v[2], v[0] / v[1]) for v in ents.values() if v[1] >= 5]
        if len(pts) < 8 or len({p[1] for p in pts}) < 2 or len({p[0] for p in pts}) < 2:
            rows.append({"cell": cell, "n_entries": len(pts), "rho": None})
            continue
        rho, p = spearmanr([p[0] for p in pts], [p[1] for p in pts])
        rows.append({"cell": cell, "n_entries": len(pts), "rho": float(rho), "p": float(p)})
    rhos = [r["rho"] for r in rows if r["rho"] is not None]
    from scipy.stats import wilcoxon
    summ = {"n_cells": len(rhos), "median_rho": float(np.median(rhos)) if rhos else None,
            "n_positive": sum(1 for x in rhos if x > 0), "n_negative": sum(1 for x in rhos if x < 0),
            "wilcoxon_p_two_sided": float(wilcoxon(rhos).pvalue) if len(rhos) >= 6 else None,
            "note": "non-trivial entries with >= 5 test lookups; cells with >= 8 such entries and non-constant accuracy"}
    return {"summary": summ, "cells": rows}


def fits(pd, prim, allrecs):
    df = pd.DataFrame(prim)
    df["lc"] = np.log1p(df["c"])
    df["lm"] = np.log(df["m"])
    df["lec"] = np.log1p(df["ec"])
    df["dev"] = df["lc"] - df["lec"]                      # sample-specific (seed-noise) deviation of log(1+c)
    df["cluster"] = df["cell"] + "|" + df["a"].astype(str) + "," + df["x"].astype(str)
    df["pos1"] = (df["pos"] == 0).astype(int)
    T, Sz = "C(task, Treatment('div7'))", "C(size, Treatment('1.5B'))"
    F = {}
    F["null_task_size"] = fit_glm(df, f"ok ~ {T} + {Sz}")
    F["lc_only"] = fit_glm(df, "ok ~ lc")
    F["lc_task_size"] = fit_glm(df, f"ok ~ lc + {T} + {Sz}")
    F["lc_task_size_trivial"] = fit_glm(df, f"ok ~ lc + trivial + {T} + {Sz}")
    F["lc_x_task"] = fit_glm(df, f"ok ~ lc * {T} + trivial + {Sz}")
    F["lc_x_size"] = fit_glm(df, f"ok ~ lc * {Sz} + trivial + {T}")
    F["pooled_m_task_size_trivial"] = fit_glm(df, f"ok ~ lm + trivial + {T} + {Sz}")
    F["pooled_m_only"] = fit_glm(df, "ok ~ lm + trivial")
    F["lc_only_trivial"] = fit_glm(df, "ok ~ lc + trivial")
    F["lc_plus_m_task_size_trivial"] = fit_glm(df, f"ok ~ lc + lm + trivial + {T} + {Sz}")
    F["cell_FE"] = fit_glm(df, "ok ~ C(cell) + trivial")
    F["cell_FE_plus_lc"] = fit_glm(df, "ok ~ C(cell) + trivial + lc")
    # seed-noise natural experiment: structural expectation vs sample-specific deviation (random by construction)
    F["cell_FE_Ec_plus_dev"] = fit_glm(df, "ok ~ C(cell) + trivial + lec + dev")
    nt = df[df["trivial"] == 0].copy()
    F["nontrivial_lc_task_size"] = fit_glm(nt, f"ok ~ lc + {T} + {Sz}")
    F["nontrivial_cell_FE_Ec_plus_dev"] = fit_glm(nt, "ok ~ C(cell) + lec + dev")
    F["nontrivial_cell_FE_plus_lc"] = fit_glm(nt, "ok ~ C(cell) + lc")
    F["nontrivial_cell_FE"] = fit_glm(nt, "ok ~ C(cell)")
    lr = df[(df["trivial"] == 0) & (df["d"].isin([7, 13]))].copy()
    F["lookup_regime_lc_size"] = fit_glm(lr, f"ok ~ lc + {Sz}")
    F["lookup_regime_lc_task_size"] = fit_glm(lr, f"ok ~ lc + {T} + {Sz}")
    F["lookup_regime_lc_x_task"] = fit_glm(lr, f"ok ~ lc * {T} + {Sz}")
    F["lookup_regime_lm_task_size"] = fit_glm(lr, f"ok ~ lm + {T} + {Sz}")
    F["lookup_regime_lc_lm_task_size"] = fit_glm(lr, f"ok ~ lc + lm + {T} + {Sz}")
    F["lookup_regime_cell_FE_Ec_plus_dev"] = fit_glm(lr, "ok ~ C(cell) + lec + dev")
    tests = {"one_curve_task_interaction_LR": lr_test(F["lc_task_size_trivial"], F["lc_x_task"]),
             "size_interaction_LR": lr_test(F["lc_task_size_trivial"], F["lc_x_size"]),
             "task_size_terms_needed_LR": lr_test(F["lc_only_trivial"], F["lc_task_size_trivial"]),
             "within_cell_lc_LR": lr_test(F["cell_FE"], F["cell_FE_plus_lc"]),
             "within_cell_lc_nontrivial_LR": lr_test(F["nontrivial_cell_FE"], F["nontrivial_cell_FE_plus_lc"]),
             "lc_beyond_pooled_m_LR": lr_test(F["pooled_m_task_size_trivial"], F["lc_plus_m_task_size_trivial"]),
             "pooled_m_beyond_lc_LR": lr_test(F["lc_task_size_trivial"], F["lc_plus_m_task_size_trivial"]),
             "lookup_regime_task_needed_LR": lr_test(F["lookup_regime_lc_size"], F["lookup_regime_lc_task_size"]),
             "lookup_regime_task_interaction_LR": lr_test(F["lookup_regime_lc_task_size"], F["lookup_regime_lc_x_task"]),
             "lookup_regime_m_beyond_lc_LR": lr_test(F["lookup_regime_lc_task_size"], F["lookup_regime_lc_lm_task_size"]),
             "lookup_regime_lc_beyond_m_LR": lr_test(F["lookup_regime_lm_task_size"], F["lookup_regime_lc_lm_task_size"]),
             "aic": {k: v.get("aic") for k, v in F.items()}}
    # implied c for p = 0.95 / 0.99 from the lc + task + size + trivial fit, non-trivial entries
    f = F["lc_task_size_trivial"]
    implied = {}
    if "params" in f:
        for t in sorted(df["task"].unique()):
            for s in sorted(df[df["task"] == t]["size"].unique()):
                implied[f"{t}|{s}"] = {"c_for_p95": c_for_p(f, 0.95, t, s), "c_for_p99": c_for_p(f, 0.99, t, s)}
    keep = {}
    for k, v in F.items():
        v2 = strip(v)
        if "params" in v2 and "cell_FE" in k:
            v2["params"] = {p: x for p, x in v2.get("params", {}).items() if not p.startswith("C(cell)")}
            v2["se_cluster"] = {p: x for p, x in v2.get("se_cluster", {}).items() if not p.startswith("C(cell)")}
            v2["p"] = {p: x for p, x in v2.get("p", {}).items() if not p.startswith("C(cell)")}
        keep[k] = v2
    return {"models": keep, "tests": tests, "implied_c_nontrivial": implied,
            "note": "GLM binomial, SE clustered by cell x entry; reference levels task div7, size 1.5B"}


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def write_md(o):
    L = ["# Per-transition coverage law (auto-generated by experiments/analysis_coverage.py; do not edit)", "",
         "POST HOC analysis, not preregistered. c = times the table entry (a, x) appears in the cell's n training "
         "traces. LOOKUP (primary) = the entry the model wrote, correct iff r' = (10a+x) mod d. LOCAL = entry "
         "(model's previous remainder, true digit). Trivial = a = 0 or d | 10.", ""]
    L += ["## Cells", "", "| cell | role | n | lookup steps | lookup acc | local acc | unseen entries / 10d | "
          "missing | off-table | train tok match |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in o["cells"]:
        L.append(f"| {c['cell']} | {c['role']} | {c['n']} | {c['lookup']['n_steps']} | {fmt(c['lookup']['acc'])} | "
                 f"{fmt(c['local']['acc'])} | {c['entries_unseen']}/{c['table_entries']} | {c['missing_steps']} | "
                 f"{c['off_table_steps']} | {c['train_tokens_match_run_row']} |")
    for defn in ("lookup", "local"):
        R = o["results"][defn]
        L += ["", f"## {defn.upper()} (primary cells: arm B, plain, trained = tested)", "",
              f"Pooled step accuracy {fmt(R['acc_primary']['acc'])} over {R['n_steps_primary']} steps.", ""]

        def table(title, cv):
            groups = list(cv)
            L.extend([f"### {title}", "", "| c bin | " + " | ".join(groups) + " |",
                      "|---|" + "---|" * len(groups)])
            for b in BIN_LABELS:
                cells = []
                for g in groups:
                    v = cv[g].get(b)
                    cells.append("" if v is None else f"{fmt(v['acc'])} ({v['n_steps']})")
                if any(cells):
                    L.append(f"| {b} | " + " | ".join(cells) + " |")
            L.append("")
        table("Accuracy by c, trivial vs non-trivial", R["curve_by_trivial"])
        table("Non-trivial entries by task", R["curve_by_task_nontrivial"])
        table("Non-trivial entries by model size", R["curve_by_size_nontrivial"])
        if defn == "lookup":
            table("Non-trivial entries by task x size", R["curve_by_task_size_nontrivial"])
            table("Secondary cells (all entries)", R["curve_secondary_roles"])
            table("div11 non-trivial: wrap (x < a) vs no wrap", R["div11_wrap"]["by_wrap"])
            table("div13 non-trivial by training n (same c, more total training)", R["curve_div13_by_n"])
            table("div7 non-trivial by training n", R["curve_div7_by_n"])
            table("n = 180, 4-digit: div7 vs div13 non-trivial (equal n, equal c)", R["curve_n180_4digit_by_task"])
            ea = R["error_anatomy_nontrivial"]
            L.extend(["### What a wrong non-trivial lookup outputs", "", ea["note"], ""])
            for band, v in ea.items():
                if band != "note":
                    L.append(f"- {band} (n wrong {v['n_wrong']}): " +
                             ", ".join(f"{k} {fmt(x)} (null {fmt(v['uniform_null_shares_in_range'].get(k))})"
                                       for k, x in v["shares"].items()))
            L.append("")
        bb = R["base_fewshot4_on_finetune_c0_entries"]
        L.extend([f"Base model (4 demos) on the entries fine-tuned div7 cells never saw: "
                  f"{fmt(bb['base_acc_same_entries']['acc'])} ({bb['base_acc_same_entries']['n_steps']} steps) vs "
                  f"fine-tuned at c=0 {fmt(bb['finetuned_acc_same_entries_at_c0']['acc'])} "
                  f"({bb['finetuned_acc_same_entries_at_c0']['n_steps']}); base on all non-trivial entries "
                  f"{fmt(bb['base_acc_all_nontrivial']['acc'])} ({bb['base_acc_all_nontrivial']['n_steps']}).", ""])
        L += ["### Threshold", "", f"- pooled: {json.dumps(_thr(R['threshold_pooled']))}",
              f"- non-trivial: {json.dumps(_thr(R['threshold_nontrivial']))}"]
        for s, v in R["threshold_by_size_nontrivial"].items():
            L.append(f"- non-trivial, {s}: {json.dumps(_thr(v))}")
        for t, v in R["threshold_by_task_nontrivial"].items():
            L.append(f"- non-trivial, {t}: {json.dumps(_thr(v))}")
        L += ["", "### c = 0 (entry never supervised)", "",
              f"Pooled c = 0: {fmt(R['c0_pooled']['acc'])} over {R['c0_pooled']['n_steps']} steps.", "",
              "| task / size | c=0 acc (steps) | distinct entries | chance 1/d | binom p (> chance) | same entries when c>=1 |",
              "|---|---|---|---|---|---|"]
        for k, v in R["c0"].items():
            L.append(f"| {k} | {fmt(v['c0']['acc'])} ({v['c0']['n_steps']}) | {v['distinct_c0_entries']} | "
                     f"{fmt(v['chance_1_over_d'])} | {fmt(v['binomial_p_vs_chance'])} | "
                     f"{fmt(v['same_entries_when_seen_c_ge_1']['acc'])} ({v['same_entries_when_seen_c_ge_1']['n_steps']}) |")
        ws = R["within_cell_spearman"]["summary"]
        L += ["", f"### Within-cell rank correlation (entry accuracy vs c, non-trivial)", "",
              f"{ws['n_cells']} cells; median rho {fmt(ws['median_rho'])}; {ws['n_positive']} positive / "
              f"{ws['n_negative']} negative; Wilcoxon p {fmt(ws['wilcoxon_p_two_sided'])}.", ""]
        F = R["fits"]
        L += ["### Logistic fits (GLM binomial, SE clustered by cell x entry)", "",
              "| model | n | AIC | coef lc (SE, p) | other |", "|---|---|---|---|---|"]
        for k, v in F["models"].items():
            if "params" not in v:
                L.append(f"| {k} | - | - | error | {v.get('error', '')[:80]} |")
                continue
            pr, se, pv = v["params"], v["se_cluster"], v["p"]
            lc = f"{fmt(pr['lc'])} ({fmt(se['lc'])}, {fmt(pv['lc'])})" if "lc" in pr else "-"
            other = []
            for name in ("lm", "trivial", "lec", "dev"):
                if name in pr:
                    other.append(f"{name} {fmt(pr[name])} ({fmt(se[name])}, p {fmt(pv[name])})")
            L.append(f"| {k} | {v['n']} | {fmt(v['aic'], 1)} | {lc} | {'; '.join(other)} |")
        L += ["", "Tests: " + json.dumps({k: (None if v is None else {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                                          for kk, vv in v.items() if kk != 'note'})
                                          for k, v in F["tests"].items() if k != "aic"}), ""]
        if defn == "lookup":
            for fname in ("lc_task_size_trivial", "lc_plus_m_task_size_trivial", "lookup_regime_lc_task_size",
                          "lookup_regime_lc_lm_task_size"):
                ff = F["models"].get(fname, {})
                if "params" not in ff:
                    continue
                L.extend(["", f"Task / size terms of `{ff['formula']}` (log-odds vs div7, 1.5B):", ""])
                for q, x in ff["params"].items():
                    if q.startswith("C("):
                        L.append(f"- {q}: {fmt(x)} (SE {fmt(ff['se_cluster'][q])}, p {fmt(ff['p'][q])})")
            f = F["models"].get("lc_task_size_trivial", {})
            if "params" in f:
                L += ["", "Implied c for p = 0.95 / 0.99 on non-trivial entries (same fit):", ""]
                for k, v in F["implied_c_nontrivial"].items():
                    L.append(f"- {k}: {fmt(v['c_for_p95'], 1)} / {fmt(v['c_for_p99'], 1)}")
                L.append("")
    open(os.path.join(C.OUT, "coverage.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def _thr(t):
    if t.get("c_star") is None:
        return t
    return {"c_star": t["c_star"], "acc_at_or_above": round(t["acc"], 4), "n_steps": t["n_steps"],
            "ci95_lo": round(t["ci95"][0], 4), "share_steps": round(t["share_of_steps_at_or_above"], 3),
            "acc_below": None if t["acc_below"]["acc"] is None else round(t["acc_below"]["acc"], 4)}


if __name__ == "__main__":
    main()
