"""analysis_erasure.py -- item-level "erasure" of the base model's computation by fine-tuning.

usage:  python experiments/analysis_erasure.py
writes: results/analysis/erasure.json, results/analysis/erasure.md

CPU only, torch-free, re-scans every run file on each call (analysis_common.load_runs).

For every model that has a zero-shot base row on a test set, and every trained run of the same model scored
on that test set (arms A, B at every n, S, D, Bprime, C, ...), build the paired 2x2 over the 240 test items
from the per-item correctness bitmaps:

    kept  = base right, run right        lost   = base right, run wrong
    fixed = base wrong, run right        missed = base wrong, run wrong

keep rate = kept / (kept + lost)   (share of the items the base solved that the fine-tuned model still solves)
fix rate  = fixed / (fixed + missed)
McNemar   = exact two-sided binomial on the discordant pairs (lost vs fixed): did the run gain or lose net?
Dependence: if the fine-tuned model's correctness were independent of the base's, keep rate = fix rate
            (= the run's accuracy). keep - fix (the "inheritance gap") and a two-sided Fisher exact test on the
            2x2 measure whether the run still succeeds on the SAME items as the base, i.e. whether it inherited
            the base's item-level competence rather than replacing it.

Reference bases: the plain-prompt base (primary family; Holm across all run-level McNemar tests in it) and,
where they exist (1.5B), the chain-of-thought and few-shot base rows (secondary; arm S samples from the CoT
prompt, so base[cot] is its natural reference).

Erasure mechanisms (per trained run, from the recovered per-item answers; binary tasks with 120/120 labels):
  constant  : the run answers one label on >= 95% of items (Yes-rate <= 0.05 or >= 0.95)
  shortcut  : not constant, and the best surface rule (analysis_common.divk_rules / prime_rules, excluding the
              true label, the two constant rules and any rule that agrees with the truth on >= 95% of the test
              items, e.g. digit_sum_div_3 on div3, last_digit_even on div2, no_factor_le_11+ on prime) has Cohen's kappa >= 0.30 with the run's answers AND a
              higher kappa than the truth does
  computes  : not constant, not shortcut, kappa with the truth >= 0.50
  unstructured : everything else (non-constant answers that neither track the truth nor a listed rule)
Thresholds are fixed here, before looking at the table, and are stated in the outputs. valid has no surface
rules in analysis_common, so its runs are only constant / computes / unstructured.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import analysis_common as C

C.fix_sys_path()

CONST_THR = 0.95
SHORTCUT_KAPPA = 0.30
COMPUTES_KAPPA = 0.50
TRUTH_EQUIV = 0.95      # a surface rule that agrees with the true label on >= 95% of test items is not a shortcut
WEAK_BASE_ACC = 0.65     # bases below this are near chance: their "correct" items are partly lucky guesses
ALPHA = 0.05


# --------------------------------------------------------------------------- statistics
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables with prob <= observed)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    lo, hi = max(0, c1 - r2), min(r1, c1)
    den = math.comb(n, c1)

    def pr(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / den
    p_obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs * (1 + 1e-9)))


def phi(a, b, c, d):
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return 0.0 if den == 0 else (a * d - b * c) / den


def kappa(preds, ref):
    n = len(preds)
    po = sum(p == r for p, r in zip(preds, ref)) / n
    py, ry = sum(p == "Yes" for p in preds) / n, sum(r == "Yes" for r in ref) / n
    pe = py * ry + (1 - py) * (1 - ry)
    return 0.0 if pe >= 1 else (po - pe) / (1 - pe)


def rnd(x, k=4):
    return None if x is None else round(x, k)


# --------------------------------------------------------------------------- rules / mechanisms
def rules_for(task):
    base = task.split("->")[-1].split("[")[0]
    if base in ("prime", "prime_hard"):
        return C.prime_rules()
    k = C.divisor_of_task(base)
    return C.divk_rules(k) if k else None


def classify(run, test):
    preds = C.recover_preds(run["bits_list"], test)
    golds = [t["label"] for t in test]
    yes = sum(p == "Yes" for p in preds) / len(preds)
    k_truth = kappa(preds, golds)
    out = {"yes_rate": rnd(yes), "kappa_truth": rnd(k_truth), "best_rule": None, "best_rule_kappa": None,
           "best_rule_agree": None}
    rules = rules_for(run["task"])
    rule_preds = None
    if rules:
        nums = [C.number_of(t["prompt"]) for t in test]
        best = None
        for name, f in rules.items():
            if name in ("true_label", "always_No", "always_Yes"):
                continue
            rp = [f(x) for x in nums]
            if sum(r == g for r, g in zip(rp, golds)) / len(golds) >= TRUTH_EQUIV:
                continue    # truth-equivalent on this test set (digit_sum_div_3 on div3, no_factor_le_11+ on prime)
            kp = kappa(preds, rp)
            if best is None or kp > best[1]:
                best = (name, kp, rp)
        out["best_rule"], out["best_rule_kappa"] = best[0], rnd(best[1])
        out["best_rule_agree"] = rnd(sum(p == r for p, r in zip(preds, best[2])) / len(preds))
        rule_preds = best[2]
    if yes <= 1 - CONST_THR or yes >= CONST_THR:
        cls = "constant"
    elif rules and best[1] >= SHORTCUT_KAPPA and best[1] > k_truth:
        cls = "shortcut"
    elif k_truth >= COMPUTES_KAPPA:
        cls = "computes"
    else:
        cls = "unstructured"
    out["class"] = cls
    return out, preds, rule_preds


def gens_answer_stats(gens, run):
    """Yes / No / '?' counts from the saved generations of this run (if any), and agreement with the bitmap."""
    key = (run["task"], run["arm"], int(run["n"]), int(run["seed"]), C.short_model(run["model"]))
    g = gens.get(key)
    if not g or len(g["rows"]) != run["n_test"]:
        return None
    preds = [C.gen_pred(r) for r in g["rows"]]
    cnt = {a: preds.count(a) for a in ("Yes", "No", "?")}
    return {"path": g["path"], "yes": cnt["Yes"], "no": cnt["No"], "unparsed": cnt["?"]}


# --------------------------------------------------------------------------- main
def two_by_two(bb, rb):
    a = sum(1 for x, y in zip(bb, rb) if x and y)
    b = sum(1 for x, y in zip(bb, rb) if x and not y)
    c = sum(1 for x, y in zip(bb, rb) if not x and y)
    d = sum(1 for x, y in zip(bb, rb) if not x and not y)
    return a, b, c, d


def main():
    tasks = C.load_tasks()
    runs, problems = C.load_runs()
    gens, gnotes = C.load_gens(tasks)

    bases = {}      # (model, eval_task, prompt_mode) -> run
    for r in runs:
        if r["arm"] == "base":
            bases[(r["model"], r["eval_task"], r["prompt_mode"])] = r

    run_info = {}   # run key -> classification
    pairs = []
    for r in runs:
        if r["arm"] == "base" or r["prompt_mode"] != "plain":
            continue
        test = tasks[r["eval_task"]]["test"]
        info, preds, rule_preds = classify(r, test)
        info["gens"] = gens_answer_stats(gens, r)
        run_info[r["key"]] = info
        for mode in ("plain", "cot", "fewshot4"):
            br = bases.get((r["model"], r["eval_task"], mode))
            if br is None:
                continue
            a, b, c, d = two_by_two(br["bits_list"], r["bits_list"])
            gold = [t["label"] for t in test]
            ky = [(x, y) for x, y, g in zip(br["bits_list"], r["bits_list"], gold) if x and g == "Yes"]
            kn = [(x, y) for x, y, g in zip(br["bits_list"], r["bits_list"], gold) if x and g == "No"]
            # lost items split by whether the run's (wrong) answer equals the best surface rule's answer
            lost_idx = [i for i, (x, y) in enumerate(zip(br["bits_list"], r["bits_list"])) if x and not y]
            lost_rule = (sum(1 for i in lost_idx if preds[i] == rule_preds[i]) if rule_preds else None)
            lost_yes = sum(1 for i in lost_idx if preds[i] == "Yes")
            pairs.append({
                "model": C.short_model(r["model"]), "eval_task": r["eval_task"], "train_task": r["train_task"],
                "cell": r["task"], "arm": r["arm"], "n": int(r["n"]), "seed": int(r["seed"]), "ref": mode,
                "base_acc": rnd(br["acc"]), "run_acc": rnd(r["acc"]),
                "kept": a, "lost": b, "fixed": c, "missed": d,
                "keep_rate": rnd(a / (a + b)) if a + b else None,
                "fix_rate": rnd(c / (c + d)) if c + d else None,
                "keep_rate_goldYes": rnd(sum(y for _, y in ky) / len(ky)) if ky else None,
                "keep_rate_goldNo": rnd(sum(y for _, y in kn) / len(kn)) if kn else None,
                "n_base_correct_goldYes": len(ky), "n_base_correct_goldNo": len(kn),
                "mcnemar_p": mcnemar_exact(b, c), "net": c - b,
                "inherit_gap": rnd(a / (a + b) - c / (c + d)) if (a + b and c + d) else None,
                "phi": rnd(phi(a, b, c, d)), "fisher_p": fisher_two_sided(a, b, c, d),
                "lost_answered_yes": lost_yes, "lost_matching_best_rule": lost_rule,
                "run_class": info["class"], "run_yes_rate": info["yes_rate"],
                "weak_base": br["acc"] < WEAK_BASE_ACC,
                "base_capped": br.get("n_hit_cap") or 0, "run_source": r["source"], "base_source": br["source"],
            })

    # Holm across the primary (plain-base) family
    prim = [p for p in pairs if p["ref"] == "plain"]
    for p, h in zip(prim, holm([p["mcnemar_p"] for p in prim])):
        p["mcnemar_p_holm"] = h
    for p in pairs:
        p.setdefault("mcnemar_p_holm", None)
        p["mcnemar_p"] = float("%.3g" % p["mcnemar_p"])
        p["fisher_p"] = float("%.3g" % p["fisher_p"])
        if p["mcnemar_p_holm"] is not None:
            p["mcnemar_p_holm"] = float("%.3g" % p["mcnemar_p_holm"])

    # ---- cells: aggregate over seeds
    groups = defaultdict(list)
    for p in pairs:
        groups[(p["model"], p["cell"], p["eval_task"], p["arm"], p["n"], p["ref"])].append(p)
    cells = []
    for (m, cell, ev, arm, n, ref), ps in sorted(groups.items()):
        ps.sort(key=lambda x: x["seed"])
        kr = [p["keep_rate"] for p in ps if p["keep_rate"] is not None]
        fr = [p["fix_rate"] for p in ps if p["fix_rate"] is not None]
        A, B, Cc, D = (sum(p[k] for p in ps) for k in ("kept", "lost", "fixed", "missed"))
        cells.append({
            "model": m, "cell": cell, "eval_task": ev, "arm": arm, "n": n, "ref": ref, "seeds": [p["seed"] for p in ps],
            "base_acc": ps[0]["base_acc"], "weak_base": ps[0]["weak_base"], "base_capped": ps[0]["base_capped"],
            "run_acc_mean": rnd(sum(p["run_acc"] for p in ps) / len(ps)),
            "keep_mean": rnd(sum(kr) / len(kr)) if kr else None, "keep_min": min(kr) if kr else None,
            "keep_max": max(kr) if kr else None, "keep_pooled": rnd(A / (A + B)) if A + B else None,
            "fix_mean": rnd(sum(fr) / len(fr)) if fr else None,
            "inherit_gap_mean": rnd(sum(p["inherit_gap"] for p in ps if p["inherit_gap"] is not None) /
                                    max(1, sum(1 for p in ps if p["inherit_gap"] is not None))),
            "per_seed": [{"seed": p["seed"], "keep": p["keep_rate"], "fix": p["fix_rate"], "lost": p["lost"],
                          "fixed": p["fixed"], "net": p["net"], "p": p["mcnemar_p"], "p_holm": p["mcnemar_p_holm"],
                          "phi": p["phi"], "fisher_p": p["fisher_p"], "class": p["run_class"],
                          "yes_rate": p["run_yes_rate"], "keep_goldYes": p["keep_rate_goldYes"],
                          "keep_goldNo": p["keep_rate_goldNo"]} for p in ps],
            "n_seeds_sig_loss": sum(1 for p in ps if p["mcnemar_p"] < ALPHA and p["net"] < 0),
            "n_seeds_sig_gain": sum(1 for p in ps if p["mcnemar_p"] < ALPHA and p["net"] > 0),
            "n_seeds_sig_loss_holm": sum(1 for p in ps if (p["mcnemar_p_holm"] or 1) < ALPHA and p["net"] < 0),
            "n_seeds_sig_gain_holm": sum(1 for p in ps if (p["mcnemar_p_holm"] or 1) < ALPHA and p["net"] > 0),
            "n_seeds_dependent": sum(1 for p in ps if p["fisher_p"] < ALPHA and p["phi"] > 0),
        })

    # ---- dose: keep rate vs n (seed 0 series and seed means), with McNemar between consecutive n on the
    #      base-correct items only (does more data keep more of the base's items?)
    bits_of = {(C.short_model(r["model"]), r["task"], r["arm"], int(r["n"]), int(r["seed"])): r["bits_list"]
               for r in runs if r["prompt_mode"] == "plain"}
    base_bits = {(C.short_model(k[0]), k[1]): v["bits_list"] for k, v in bases.items() if k[2] == "plain"}
    dose = []
    for (m, cell, ev, arm, _n, ref) in sorted({(c["model"], c["cell"], c["eval_task"], c["arm"], 0, c["ref"])
                                               for c in cells if c["ref"] == "plain"}):
        cs = sorted([c for c in cells if (c["model"], c["cell"], c["arm"], c["ref"]) == (m, cell, arm, "plain")],
                    key=lambda c: c["n"])
        if len(cs) < 2:
            continue
        bb = base_bits[(m, ev)]
        series0 = []
        for c in cs:
            s0 = next((s for s in c["per_seed"] if s["seed"] == 0), None)
            series0.append({"n": c["n"], "keep_seed0": s0["keep"] if s0 else None, "keep_mean": c["keep_mean"],
                            "n_seeds": len(c["seeds"]), "fix_seed0": s0["fix"] if s0 else None,
                            "acc_mean": c["run_acc_mean"]})
        steps = []
        for c1, c2 in zip(cs, cs[1:]):
            x = bits_of.get((m, cell, arm, c1["n"], 0))
            y = bits_of.get((m, cell, arm, c2["n"], 0))
            if x is None or y is None:
                continue
            idx = [i for i, v in enumerate(bb) if v]
            b_ = sum(1 for i in idx if x[i] and not y[i])
            c_ = sum(1 for i in idx if not x[i] and y[i])
            steps.append({"from_n": c1["n"], "to_n": c2["n"], "base_correct_items": len(idx),
                          "kept_only_at_lower_n": b_, "kept_only_at_higher_n": c_,
                          "mcnemar_p": float("%.3g" % mcnemar_exact(b_, c_))})
        k0 = [s["keep_seed0"] for s in series0 if s["keep_seed0"] is not None]
        dose.append({"model": m, "cell": cell, "arm": arm, "base_acc": cs[0]["base_acc"],
                     "weak_base": cs[0]["weak_base"], "series": series0, "steps_seed0": steps,
                     "monotone_nondecreasing_seed0": all(a <= b for a, b in zip(k0, k0[1:])) if len(k0) > 1 else None,
                     "monotone_nondecreasing_mean": all(a["keep_mean"] <= b["keep_mean"]
                                                        for a, b in zip(series0, series0[1:]))})

    # ---- mechanism of erasure: lost base-correct items by the run's class (primary ref, non-weak bases)
    mech = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "lost": 0, "base_correct": 0}))
    for p in prim:
        if p["weak_base"]:
            continue
        g = mech[(p["model"], p["arm"])][p["run_class"]]
        g["runs"] += 1
        g["lost"] += p["lost"]
        g["base_correct"] += p["kept"] + p["lost"]
    mechanism = []
    for (m, arm), d in sorted(mech.items()):
        tot_lost = sum(v["lost"] for v in d.values())
        tot_bc = sum(v["base_correct"] for v in d.values())
        mechanism.append({"model": m, "arm": arm, "total_lost": tot_lost, "total_base_correct": tot_bc,
                          "by_class": {k: dict(v, share_of_lost=rnd(v["lost"] / tot_lost) if tot_lost else None)
                                       for k, v in sorted(d.items())}})

    # rule-consistency of lost items in non-constant runs of arm A (vs the null that a same-Yes-rate guesser
    # independent of the rule would match the rule on the lost items at the rule's base rate)
    rule_loss = []
    for p in prim:
        if p["arm"] != "A" or p["weak_base"] or p["lost_matching_best_rule"] is None or p["run_class"] == "constant":
            continue
        info = run_info[next(r["key"] for r in runs if C.short_model(r["model"]) == p["model"] and r["task"] == p["cell"]
                             and r["arm"] == p["arm"] and int(r["n"]) == p["n"] and int(r["seed"]) == p["seed"]
                             and r["prompt_mode"] == "plain")]
        rule_loss.append({"model": p["model"], "cell": p["cell"], "n": p["n"], "seed": p["seed"],
                          "class": p["run_class"], "best_rule": info["best_rule"],
                          "best_rule_kappa": info["best_rule_kappa"], "kappa_truth": info["kappa_truth"],
                          "lost": p["lost"], "lost_matching_best_rule": p["lost_matching_best_rule"],
                          "run_agree_with_rule_all_items": info["best_rule_agree"]})

    # ---- constant-collapse census for arm A (and all arms)
    collapse = defaultdict(lambda: {"runs": 0, "constant": 0, "shortcut": 0, "computes": 0, "unstructured": 0})
    for key, info in run_info.items():
        task, arm, n, seed, model = key
        g = collapse[(C.short_model(model), arm)]
        g["runs"] += 1
        g[info["class"]] += 1
    collapse_rows = [dict(model=m, arm=a, **v) for (m, a), v in sorted(collapse.items())]
    run_rows = [dict(model=C.short_model(k[4]), cell=k[0], arm=k[1], n=k[2], seed=k[3], **v)
                for k, v in sorted(run_info.items(), key=lambda kv: (kv[0][4], kv[0][0], kv[0][1], kv[0][2], kv[0][3]))]

    # ---- arm contrasts restricted to the base-correct items (same model, cell, n, seed): which supervision keeps
    #      more of what the base solved? exact McNemar on the base-correct items only.
    CONTRASTS = [("S", "A"), ("S", "B"), ("B", "A"), ("B", "D"), ("B", "Bprime"), ("S", "D")]
    contrasts = []
    for (m, ev), bb in sorted(base_bits.items()):
        idx = [i for i, v in enumerate(bb) if v]
        for x_arm, y_arm in CONTRASTS:
            for (mm, cell, arm, n, seed), xb in sorted(bits_of.items()):
                if (mm, arm) != (m, x_arm) or cell != ev:
                    continue
                yb = bits_of.get((m, cell, y_arm, n, seed))
                if yb is None:
                    continue
                b_ = sum(1 for i in idx if xb[i] and not yb[i])
                c_ = sum(1 for i in idx if not xb[i] and yb[i])
                contrasts.append({"model": m, "cell": cell, "x": x_arm, "y": y_arm, "n": n, "seed": seed,
                                  "base_correct_items": len(idx),
                                  "keep_x": rnd(sum(xb[i] for i in idx) / len(idx)),
                                  "keep_y": rnd(sum(yb[i] for i in idx) / len(idx)),
                                  "kept_only_by_x": b_, "kept_only_by_y": c_,
                                  "mcnemar_p": float("%.3g" % mcnemar_exact(b_, c_)),
                                  "weak_base": bases[next(k for k in bases if C.short_model(k[0]) == m and k[1] == ev
                                                          and k[2] == "plain")]["acc"] < WEAK_BASE_ACC})

    out = {
        "script": "experiments/analysis_erasure.py",
        "definitions": {"keep_rate": "kept/(kept+lost) over items the base answered correctly",
                        "fix_rate": "fixed/(fixed+missed) over items the base answered wrongly",
                        "inherit_gap": "keep_rate - fix_rate (0 if the run's correctness is independent of the base's)",
                        "mcnemar": "exact two-sided on lost vs fixed; Holm across all plain-reference run pairs",
                        "fisher": "two-sided Fisher exact on the 2x2 (dependence of run correctness on base correctness)",
                        "classes": {"constant": f"Yes-rate <= {1 - CONST_THR:.2f} or >= {CONST_THR:.2f}",
                                    "shortcut": f"not constant; best surface-rule kappa >= {SHORTCUT_KAPPA} and > kappa(truth)",
                                    "computes": f"not constant/shortcut; kappa(truth) >= {COMPUTES_KAPPA}",
                                    "unstructured": "otherwise"},
                        "weak_base": f"base accuracy < {WEAK_BASE_ACC} (base-correct items include lucky guesses)"},
        "n_pairs": len(pairs), "n_pairs_primary": len(prim),
        "bases": [{"model": C.short_model(k[0]), "eval_task": k[1], "prompt_mode": k[2], "acc": rnd(v["acc"]),
                   "n_hit_cap": v.get("n_hit_cap") or 0, "source": v["source"],
                   "yes_rate": rnd(sum(p == "Yes" for p in C.recover_preds(v["bits_list"], tasks[k[1]]["test"])) / v["n_test"])}
                  for k, v in sorted(bases.items())],
        "cells": cells, "dose": dose, "contrasts_on_base_correct": contrasts, "mechanism": mechanism,
        "rule_loss_A": rule_loss,
        "class_census": collapse_rows, "runs": run_rows, "pairs": pairs,
        "load_problems": problems, "gens_notes": gnotes[:20],
    }
    od = C.ensure_out()
    with open(os.path.join(od, "erasure.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_md(out, os.path.join(od, "erasure.md"))
    print(f"wrote {os.path.relpath(os.path.join(od, 'erasure.json'), C.ROOT)} and erasure.md "
          f"({len(pairs)} pairs, {len(cells)} cells)")


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def write_md(o, path):
    L = ["# Erasure of base-model computation, item by item (auto-generated by experiments/analysis_erasure.py; do not edit)", ""]
    d = o["definitions"]
    L += ["Paired 2x2 of each fine-tuned run against the zero-shot base of the same model on the same 240 test items.",
          f"keep = {d['keep_rate']}; fix = {d['fix_rate']}; gap = keep - fix ({d['inherit_gap']}).",
          f"McNemar: {d['mcnemar']}. Dependence: {d['fisher']}.",
          "Run classes: " + "; ".join(f"{k}: {v}" for k, v in d["classes"].items()) + ".",
          f"Weak base (marked *): {d['weak_base']}.", ""]
    # computed summary sentences (strong bases only; plain reference)
    L += ["## Summary (computed from the tables below)", ""]
    cmap = {(c["model"], c["cell"], c["arm"], c["n"]): c for c in o["cells"] if c["ref"] == "plain"}
    for m in sorted({c["model"] for c in o["cells"]}):
        for cell in ("div7", "div13", "prime", "valid"):
            here = sorted([c for (mm, ce, _a, _n), c in cmap.items() if mm == m and ce == cell and not c["weak_base"]],
                          key=lambda c: (c["arm"], c["n"]))
            if not here:
                continue
            parts = []
            for c in here:
                ks = "/".join(pct(s["keep"]) for s in c["per_seed"])
                cls = "/".join(s["class"] for s in c["per_seed"])
                parts.append(f"{c['arm']}@{c['n']} keeps {ks}% ({cls})")
            L.append(f"- {m} {cell} (base {pct(here[0]['base_acc'])}%): " + "; ".join(parts) + ".")
    L += [""]
    L += ["## Headline: strong bases (plain base accuracy >= 65%), per seed", "",
          "keep = share of base-solved items still solved; fix = share of base-failed items now solved; phi and Fisher p: "
          "dependence of the run's item-level correctness on the base's (phi near 0 = the run no longer tracks which "
          "items the base could do); keep|gold=Yes / keep|gold=No split the kept items by label (a constant-No run keeps "
          "every No and no Yes).", "",
          "| model | cell | base | arm | n | seed | acc | keep | fix | lost/fixed (McNemar p) | phi (Fisher p) | keep gold Yes / No | class | Yes-rate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted((p for p in o["pairs"] if p["ref"] == "plain" and not p["weak_base"]),
                    key=lambda p: (p["model"], p["cell"], p["arm"], p["n"], p["seed"])):
        L.append(f"| {p['model']} | {p['cell']} | {pct(p['base_acc'])}{' (cap ' + str(p['base_capped']) + ')' if p['base_capped'] else ''} | "
                 f"{p['arm']} | {p['n']} | {p['seed']} | {pct(p['run_acc'])} | {pct(p['keep_rate'])} | {pct(p['fix_rate'])} | "
                 f"{p['lost']}/{p['fixed']} ({p['mcnemar_p']:.3g}) | {p['phi']:.2f} ({p['fisher_p']:.3g}) | "
                 f"{pct(p['keep_rate_goldYes'])} / {pct(p['keep_rate_goldNo'])} | {p['run_class']} | {p['run_yes_rate']:.2f} |")
    L += ["", "## Arm contrasts on the base-correct items only (same model, cell, n, seed; exact McNemar)", "",
          "| model | cell | x vs y | n | seed | base-correct | keep x | keep y | kept only by x / only by y | p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in o["contrasts_on_base_correct"]:
        L.append(f"| {c['model']} | {c['cell']}{'*' if c['weak_base'] else ''} | {c['x']} vs {c['y']} | {c['n']} | {c['seed']} | "
                 f"{c['base_correct_items']} | {pct(c['keep_x'])} | {pct(c['keep_y'])} | {c['kept_only_by_x']} / "
                 f"{c['kept_only_by_y']} | {c['mcnemar_p']:.3g} |")
    L += ["", "## Base rows", "", "| model | test set | prompt | acc | Yes-rate | capped |", "|---|---|---|---|---|---|"]
    for b in o["bases"]:
        L.append(f"| {b['model']} | {b['eval_task']} | {b['prompt_mode']} | {pct(b['acc'])} | {b['yes_rate']:.2f} | {b['n_hit_cap']} |")
    for ref in ("plain", "cot", "fewshot4"):
        cs = [c for c in o["cells"] if c["ref"] == ref]
        if not cs:
            continue
        L += ["", f"## Keep rates vs base[{ref}]", "",
              "| model | cell | base | arm | n | seeds | acc | keep mean [min, max] | fix mean | gap | per-seed lost/fixed, McNemar p (Holm), class, Yes-rate |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cs:
            ps = "; ".join(f"s{s['seed']}: {s['lost']}/{s['fixed']}, p={s['p']:.3g}"
                           + (f" ({s['p_holm']:.3g})" if s["p_holm"] is not None else "")
                           + f", {s['class']}, Y={s['yes_rate']:.2f}" for s in c["per_seed"])
            L.append(f"| {c['model']} | {c['cell']} | {pct(c['base_acc'])}{'*' if c['weak_base'] else ''}"
                     f"{' (cap ' + str(c['base_capped']) + ')' if c['base_capped'] else ''} | {c['arm']} | {c['n']} | "
                     f"{len(c['seeds'])} | {pct(c['run_acc_mean'])} | {pct(c['keep_mean'])} [{pct(c['keep_min'])}, "
                     f"{pct(c['keep_max'])}] | {pct(c['fix_mean'])} | {pct(c['inherit_gap_mean'])} | {ps} |")
    L += ["", "## Dose: keep rate vs n (plain base)", "",
          "| model | cell | arm | base | n: keep seed0 (keep mean, seeds) | monotone s0 / mean | consecutive-n McNemar on base-correct items (seed 0) |",
          "|---|---|---|---|---|---|---|"]
    for r in o["dose"]:
        ser = "; ".join(f"{s['n']}: {pct(s['keep_seed0'])} ({pct(s['keep_mean'])}, {s['n_seeds']})" for s in r["series"])
        st = "; ".join(f"{s['from_n']}->{s['to_n']}: {s['kept_only_at_lower_n']} vs {s['kept_only_at_higher_n']}, "
                       f"p={s['mcnemar_p']:.3g}" for s in r["steps_seed0"])
        L.append(f"| {r['model']} | {r['cell']} | {r['arm']} | {pct(r['base_acc'])}{'*' if r['weak_base'] else ''} | "
                 f"{ser} | {r['monotone_nondecreasing_seed0']} / {r['monotone_nondecreasing_mean']} | {st} |")
    L += ["", "## Mechanism: lost base-correct items by run class (plain base, non-weak bases only)", "",
          "| model | arm | lost / base-correct | constant | shortcut | computes | unstructured |", "|---|---|---|---|---|---|---|"]
    for r in o["mechanism"]:
        bc = r["by_class"]

        def cellv(k):
            v = bc.get(k)
            return "--" if not v else f"{v['lost']} ({v['runs']} runs, {pct(v['share_of_lost'])}% of lost)"
        L.append(f"| {r['model']} | {r['arm']} | {r['total_lost']} / {r['total_base_correct']} | {cellv('constant')} | "
                 f"{cellv('shortcut')} | {cellv('computes')} | {cellv('unstructured')} |")
    L += ["", "## Arm A, non-constant runs: do the lost items follow a surface rule? (plain base, non-weak bases)", "",
          "| model | cell | n | seed | class | best rule (kappa) | kappa truth | lost | lost matching rule | run agrees with rule (all items) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in o["rule_loss_A"]:
        L.append(f"| {r['model']} | {r['cell']} | {r['n']} | {r['seed']} | {r['class']} | {r['best_rule']} ({r['best_rule_kappa']}) | "
                 f"{r['kappa_truth']} | {r['lost']} | {r['lost_matching_best_rule']} | {pct(r['run_agree_with_rule_all_items'])} |")
    L += ["", "## Run-class census (all plain-evaluated trained runs)", "",
          "| model | arm | runs | constant | shortcut | computes | unstructured |", "|---|---|---|---|---|---|---|"]
    for r in o["class_census"]:
        L.append(f"| {r['model']} | {r['arm']} | {r['runs']} | {r['constant']} | {r['shortcut']} | {r['computes']} | {r['unstructured']} |")
    if o["load_problems"]:
        L += ["", "## Loader notes", ""] + [f"- {p}" for p in o["load_problems"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
