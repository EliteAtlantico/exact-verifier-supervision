"""analysis_report.py -- refresh the analysis and compose results/analysis/README.md (CPU only, no GPU).

usage:  python experiments/analysis_report.py               # refresh everything that changed, then compose
        python experiments/analysis_report.py --no-refresh  # only re-compose README.md from the JSON files

Refresh = run, each in its own process: analysis_stats.py, analysis_rules.py, analysis_steps.py (seconds; they
re-scan every run/eval/gens file), then analysis_tokens.py and analysis_probe.py with --if-changed (skipped
unless the datasets or their code changed; a full probe run is ~20 min). Then README.md is composed from
stats.json, probe.json, rules.json, steps.json and tokens.json, including the preregistered decision rules
S1-S6 (PREDICTIONS.md) evaluated from whatever runs exist (PENDING when the needed cells are missing), with
one table per model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import analysis_common as C

C.fix_sys_path()

MAIN_N = 180
M15 = C.short_model(C.DEFAULT_MODEL)
# the "trivial rule" baselines named in the analysis plan (fixed rules, nothing fitted)
TRIVIAL = {"majority", "odd", "last_digit_1379", "no_factor_le_3", "no_factor_le_5", "no_factor_le_7",
           "last_digit_is_7", "last_digit_is_3", "last_digit_is_1", "last_digit_0_or_5", "last_digit_even",
           "digit_sum_div_7", "digit_sum_div_3", "digit_sum_div_11", "digit_sum_div_13", "digit_sum_div_2",
           "contains_digit_7"}
STEPS = [("analysis_stats.py", []), ("analysis_rules.py", []), ("analysis_steps.py", []),
         ("analysis_tokens.py", ["--if-changed"]), ("analysis_probe.py", ["--if-changed"])]


def refresh():
    for script, args in STEPS:
        p = subprocess.run([sys.executable, os.path.join(C.HERE, script)] + args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        tag = script[len("analysis_"):-3]
        first = next((ln for ln in p.stdout.splitlines() if ln.startswith(tag)),
                     (p.stdout.strip().splitlines() or [""])[-1])
        print(f"[{script}] rc={p.returncode} {first}")
        if p.returncode:
            print(p.stderr[-1500:])


def load(name):
    p = os.path.join(C.OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}"


# --------------------------------------------------------------------------- preregistered rules
def prereg(st, ru, sp):
    """S1-S6 from PREDICTIONS.md. Returns {rule: {"status", "detail": [...]}}; per model where meaningful."""
    tests = st["mcnemar"] if st else []
    pooled = st["pooled"] if st else []
    models = sorted({C.short_model(r["model"]) for r in pooled})
    acc = {(r["task"], C.short_model(r["model"]), r["arm"], r["n"], r["seed"]): r["acc"] for r in (st or {}).get("per_run", [])}

    def ba(task, model, n=MAIN_N):
        return [x for x in tests if x["comparison"] == "B-A" and x["task"] == task and x["n"] == n
                and C.short_model(x["model"]) == model]

    res = {}
    # S1: div7 B@180 seeds 1 and 2 >= 80%, B - A >= 20 pp, McNemar p < 0.01 (the 1.5B laptop model)
    det, ok, pending = [], True, False
    for s in (1, 2):
        tr = next((x for x in ba("div7", M15) if x["seed"] == s), None)
        if tr is None:
            pending = True
            det.append(f"seed {s}: pending")
            continue
        c = tr["acc_x"] >= 0.80 and tr["diff"] >= 0.20 and tr["p_exact"] < 0.01
        ok &= c
        det.append(f"seed {s}: B {pct(tr['acc_x'])} A {pct(tr['acc_y'])} diff {100 * tr['diff']:+.1f} pp p={tr['p_exact']:.1g} "
                   f"-> {'pass' if c else 'FAIL'}")
    res["S1"] = {"status": "FAIL" if not ok else ("PENDING" if pending else "PASS"), "detail": det}
    # S2 (per model): >= 2 of div3/div11/div13 with B - A >= 15 pp in every seed (>= 2 seeds)
    for m in models:
        det, wins, decided = [], 0, 0
        for t in ("div3", "div11", "div13"):
            trs = ba(t, m)
            if len(trs) < 2:
                det.append(f"{t}: {len(trs)} seed pair(s), pending")
                continue
            decided += 1
            w = all(x["diff"] >= 0.15 for x in trs)
            wins += w
            det.append(f"{t}: " + ", ".join(f"s{x['seed']} {100 * x['diff']:+.1f}" for x in trs) + (" -> win" if w else " -> no"))
        if decided == 0 and m != M15:
            continue
        st_ = "PASS" if wins >= 2 else ("FAIL" if wins + (3 - decided) < 2 else "PENDING")
        res[f"S2 [{m}]"] = {"status": st_, "detail": det}
    # S3 (per model): div2 A >= 95% (all seeds)
    for m in models:
        a2 = sorted((k[4], v) for k, v in acc.items() if k[0] == "div2" and k[1] == m and k[2] == "A" and k[3] == MAIN_N)
        if not a2 and m != M15:
            continue
        res[f"S3 [{m}]"] = {"status": "PENDING" if not a2 else ("PASS" if all(v >= 0.95 for _, v in a2) else "FAIL"),
                            "detail": [f"s{s}: A {pct(v)}" for s, v in a2] or ["no div2 A run yet"]}
    # S4: div7 B > A for Qwen2.5-3B (every seed)
    m3 = [m for m in models if "3B" in m]
    trs = [x for m in m3 for x in ba("div7", m)]
    res["S4"] = {"status": "PENDING" if not trs else ("PASS" if all(x["diff"] > 0 for x in trs) else "FAIL"),
                 "detail": [f"{C.short_model(x['model'])} s{x['seed']}: B {pct(x['acc_x'])} A {pct(x['acc_y'])} "
                            f"diff {100 * x['diff']:+.1f} pp p={x['p_exact']:.1g}" for x in trs] or ["no Qwen2.5-3B div7 A/B pair yet"]}
    # S5: from steps.json (primary/secondary per the PREDICTIONS.md decision log, 27ba807)
    if sp:
        cells = [c for c in sp["S5"]["cells"] if "p" in c]
        det = []
        for c in cells:
            ph = c.get("post_hoc", {})
            extra = []
            if ph.get("trace_correct_6digit") is not None:
                extra.append(f"fully correct traces {pct(ph['trace_correct_6digit'])}")
            if ph.get("predicted_answer_acc_with_guessing") is not None:
                extra.append(f"p^6+(1-p^6)g {pct(ph['predicted_answer_acc_with_guessing'])} "
                             f"(g={ph['g_4digit_answer_acc_when_trace_wrong']:.3f})")
            det.append(f"[{c.get('role', '?')}] {c['task']} {c['model']} s{c['seed']}: p={c['p']:.3f} ({c['p_source']}), "
                       f"p^6 {pct(c['predicted_p6'])} vs answer acc {pct(c['observed_acc'])} ({c['diff_pp']:+.1f} pp, "
                       f"{'within' if c['within_10pp'] else 'OUTSIDE'} 10)" + (f"; post hoc: {', '.join(extra)}" if extra else ""))
        sm = sp["S5"].get("status_by_model", {})
        res["S5"] = {"status": sp["S5"]["status"],
                     "detail": ([f"{m}: primary {v['primary']}, secondary {v['secondary']}" for m, v in sm.items()] + det)
                     or ["primary = B trained+tested on div7_6d (seeds 0,1); needs 4-digit div7 B gens and div7_6d B runs"]}
    else:
        res["S5"] = {"status": "PENDING", "detail": ["steps.json missing"]}
    # S6 (per model): prime A adapter on prime_hard: acc < 60% and agreement with 'odd & no factor <= 7' > truth
    s6 = [x for x in (ru or {}).get("runs", []) if x["task"] == "prime->prime_hard" and x["arm"] == "A"]
    for m in sorted({C.short_model(x["model"]) for x in s6}) or [M15]:
        xs = [x for x in s6 if C.short_model(x["model"]) == m]
        det, ok = [], True
        for x in sorted(xs, key=lambda x: (x["n"], x["seed"])):
            ag_rule, ag_true = x["agreement"]["no_factor_le_7"], x["agreement"]["true_label"]
            c = ag_true < 0.60 and ag_rule > ag_true
            ok &= c
            det.append(f"n={x['n']} s{x['seed']} ({x['pred_source'].split('_')[0]}): acc {pct(ag_true)}, agree rule {pct(ag_rule)}"
                       f" -> {'pass' if c else 'FAIL'}")
        res[f"S6 [{m}]"] = {"status": "PENDING" if not xs else ("PASS" if ok else "FAIL"),
                            "detail": det or ["needs prime A evaluated on prime_hard (prime->prime_hard)"]}
    return res


# --------------------------------------------------------------------------- README
def main():
    if "--no-refresh" not in sys.argv:
        refresh()
    st, pr, ru, tk, sp = (load(n) for n in ("stats.json", "probe.json", "rules.json", "tokens.json", "steps.json"))
    KEY = []
    L = ["# Analysis of the thinking-vs-data runs (auto-generated by experiments/analysis_report.py)", "",
         "Qwen2.5 LoRA runs graded by the exact gate; 240 test items per task (120 Yes / 120 No). "
         "All numbers come from stats.json, probe.json, rules.json, steps.json and tokens.json in this directory; "
         "full tables in stats.md, probe.md, rules.md, steps.md. Refresh: `python experiments/analysis_report.py`.", ""]
    pooled = {(r["task"], C.short_model(r["model"]), r["arm"], r["n"]): r for r in st["pooled"]} if st else {}
    boots = {(b["comparison"], b["task"], C.short_model(b["model"]), b["n"]): b for b in st["bootstrap"]} if st else {}
    tests = st["mcnemar"] if st else []
    rules_by = {}
    for x in (ru or {}).get("runs", []):
        rules_by.setdefault((x["task"], x["arm"], x["n"], C.short_model(x["model"])), []).append(x)
    models = sorted({k[1] for k in pooled if k[2] != "base"}, key=lambda m: (m != M15, m))

    # ---------------------------------------------------------------- 0. prereg
    pre = prereg(st, ru, sp)
    L += ["## Preregistered decision rules (PREDICTIONS.md), evaluated from the runs that exist", ""]
    L += [f"- **{k}: {v['status']}** -- " + "; ".join(v["detail"]) for k, v in pre.items()]
    L.append("")

    # ---------------------------------------------------------------- 1. main tables per model
    for m in models:
        L += [f"## 1. Main cell n = {MAIN_N}: {m}", "",
              "A = answer-only, B = trace then answer, C = scrambled trace. Accuracy pooled over seeds [Wilson 95%]; "
              "B-A = paired bootstrap over items and seeds [95% CI]; p = exact McNemar per seed (Holm across tasks within "
              "comparison, model and seed). Best probe = chosen by 5-fold CV on the same 180 training items (mean over "
              "seeds 0-2); best rule = best fixed trivial rule on the test set.", "",
              "| task | base | A | B | C | B-A pp [95% CI] | McNemar B vs A per seed | best probe | best trivial rule | A's closest shortcut rule |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        tasks = sorted({k[0] for k in pooled if k[1] == m and k[3] == MAIN_N},
                       key=lambda t: (t not in ("prime", "valid", "div7"), t))
        for t in tasks:
            cells = {}
            for arm in ("A", "B", "C"):
                r = pooled.get((t, m, arm, MAIN_N))
                cells[arm] = (f"{pct(r['acc'])} [{pct(r['wilson95'][0], 0)}, {pct(r['wilson95'][1], 0)}] "
                              f"(s{','.join(map(str, r['seeds']))})") if r else "-"
            if all(v == "-" for v in cells.values()):
                continue
            et = t.split("->")[-1].split("[")[0]
            base = pooled.get((C.cell_label(et, et, "base"), m, "base", 0))
            b = boots.get(("B-A", t, m, MAIN_N))
            ba = f"{100 * b['diff']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]" if b else "-"
            mc = "; ".join(f"s{x['seed']}: {100 * x['diff']:+.1f}, p={x['p_exact']:.1g} (Holm {x['p_holm_tasks']:.1g})"
                           for x in tests if x["comparison"] == "B-A" and x["task"] == t and C.short_model(x["model"]) == m
                           and x["n"] == MAIN_N) or "-"
            pt = (pr or {}).get("tasks", {}).get(et) if "->" not in t or t.startswith("prime->prime_hard") else None
            br = None
            if pt:
                bp = pt["summary"]["best_probe_cv_mean"]
                names = sorted({v["best_probe_cv"]["probe"] for v in pt["per_seed"].values()})
                probe = f"{pct(bp)} ({'/'.join(names)}; max on test {pct(pt['summary']['best_probe_test_mean'])})"
                rs = {k: v for k, v in pt["rules_summary"].items() if k in TRIVIAL}
                br = max(rs, key=rs.get) if rs else None
                rule = f"{pct(rs[br])} ({br})" if br else "-"
            else:
                probe = rule = "-"
            sh = [f"s{x['seed']}: {x['best_shortcut_rule']} {pct(x['best_shortcut_agreement'])}% "
                  f"(kappa {x['best_shortcut_kappa']:.2f} vs truth {x['true_label_kappa']:.2f})"
                  for x in rules_by.get((t, "A", MAIN_N, m), []) if "best_shortcut_rule" in x]
            ar = pooled.get((t, m, "A", MAIN_N))
            if ar and (t in ("prime", "valid", "div7", "div2", "div3", "div11", "div13") or b):
                kl = f"- **{t} [{m}]**: A {pct(ar['acc'])}%"
                if pt:
                    kl += (f" vs best CV-selected probe {pct(pt['summary']['best_probe_cv_mean'])}% "
                           f"(A - probe {100 * (ar['acc'] - pt['summary']['best_probe_cv_mean']):+.1f} pp)")
                if br:
                    kl += f"; best trivial rule {br} {pct(rs[br])}%"
                if b:
                    kl += f"; B - A {ba} pp (seeds {','.join(map(str, b['seeds']))})"
                KEY.append(kl + ".")
            L.append(f"| {t} | {pct(base['acc']) if base else '-'} | {cells['A']} | {cells['B']} | {cells['C']} | {ba} | "
                     f"{mc} | {probe} | {rule} | {'; '.join(sh) or '-'} |")
        other = sorted(b for b in boots if b[2] == m and b[0] not in ("B-A",) and b[3] == MAIN_N)
        if other:
            L += ["", "Other paired comparisons (bootstrap 95% CI, pp): " + "; ".join(
                f"{b[0]} {b[1]} {100 * boots[b]['diff']:+.1f} [{100 * boots[b]['ci95'][0]:+.1f}, {100 * boots[b]['ci95'][1]:+.1f}]"
                for b in other) + "."]
        L.append("")

    # ---------------------------------------------------------------- 2. A vs probe
    if pr:
        L += ["## 2. Is answer-only fine-tuning (A) about what a surface probe gets from the same 180 examples?", "",
              "| task | " + " | ".join(f"A [{m}]" for m in models) + " | logreg one-hot | logreg + engineered | MLP one-hot / eng. | "
              "kNN one-hot / eng. | best probe (CV) | full-pool logreg eng. |",
              "|---|" + "---|" * len(models) + "---|---|---|---|---|---|"]
        for t, r in pr["tasks"].items():
            lab = C.cell_label(r["train_task"], t, "A")
            s = r["summary"]
            fp = r.get("full_pool_reference", {}).get("logreg_b", {}).get("test_acc")
            L.append(f"| {t} | " + " | ".join(pct((pooled.get((lab, m, "A", MAIN_N)) or {}).get("acc")) for m in models)
                     + f" | {pct(s['logreg_a'])} | {pct(s['logreg_b'])} | {pct(s['mlp_a'])} / {pct(s['mlp_b'])} | "
                     f"{pct(s['knn_a'])} / {pct(s['knn_b'])} | {pct(s['best_probe_cv_mean'])} | {pct(fp)} |")
        L += ["", "Engineered number features = last digit even / in {0,5}, digit sum mod 3 and mod 9, alternating sum "
              "mod 11 (one-hot). For valid, 'one-hot' = bag of word tokens; 'engineered' adds character 1-3-grams and "
              "atom-abstracted word n-grams. prime_hard probes train on the prime pool."]
        tr = pr["tasks"].get("prime", {}).get("rules_summary")
        if tr:
            L += ["", "Trivial prime rules on the 240-item test set: " + ", ".join(
                f"{k} {pct(tr[k])}" for k in ("odd", "last_digit_1379", "no_factor_le_3", "no_factor_le_5",
                                              "no_factor_le_7") if k in tr) + "."]
        L.append("")

    # ---------------------------------------------------------------- 3. prime shortcut
    if ru:
        L += ["## 3. Which rule does A follow on prime?", "",
              "| model | task | arm | n | seed | src | acc | agree odd | agree last digit in {1,3,7,9} | no factor<=3 | <=5 | <=7 | "
              "truth | even comp. acc | odd comp. w/ factor<=7: P(prime) | hard comp.: P(prime) | primes acc |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in ru["runs"]:
            if x["eval_task"] not in ("prime", "prime_hard") or x["arm"] not in ("A", "A_tok", "B"):
                continue
            g, s = x["agreement"], x["strata"]
            even = pct(s["even_composite"]["acc"], 0) if "even_composite" in s else "-"
            oddf = f"{s['odd_composite_factor_le7']['p_pred_yes']:.2f}" if "odd_composite_factor_le7" in s else "-"
            hard = f"{s['hard_composite_no_factor_le7']['p_pred_yes']:.2f}" if "hard_composite_no_factor_le7" in s else "-"
            prim = pct(s["prime"]["acc"], 0) if "prime" in s else "-"
            L.append(f"| {C.short_model(x['model'])} | {x['task']} | {x['arm']} | {x['n']} | {x['seed']} | "
                     f"{'gens' if x['pred_source'].startswith('obs') else 'bits'} | {pct(x['acc'])} | {pct(g['odd'])} | "
                     f"{pct(g['last_digit_1379'])} | {pct(g['no_factor_le_3'])} | {pct(g['no_factor_le_5'])} | {pct(g['no_factor_le_7'])} | "
                     f"{pct(g['true_label'])} | {even} | {oddf} | {hard} | {prim} |")
            if x["task"] == "prime" and x["arm"] == "A" and x["n"] == MAIN_N:
                KEY.append(f"- **prime shortcut [{C.short_model(x['model'])} s{x['seed']}]**: A agrees with 'last digit in "
                           f"{{1,3,7,9}}' on {pct(g['last_digit_1379'])}% of items vs {pct(g['true_label'])}% with the truth "
                           f"(best-kappa shortcut {x.get('best_shortcut_rule')}, kappa {x.get('best_shortcut_kappa', 0):.2f}).")
        L += ["", "On prime_hard every item is odd with no factor <= 7, so 'odd', 'last digit in {1,3,7,9}' and 'no factor <= 7' "
              "all say Yes on every item (agreement with them = P(pred prime)).", ""]

    # ---------------------------------------------------------------- 4. steps
    if sp and (sp.get("div") or sp.get("prime")):
        L += ["## 4. Trace steps (generations)", ""]
        if sp.get("div"):
            L += ["p_cond = P(step correct | previous correct); p after 1 excludes the near-trivial first step; guess model "
                  "(post hoc) = p^k + (1 - p^k) g, g = the cell's answer accuracy when its trace is wrong.", "",
                  "| model | task | arm | mode | seed | k | answer acc | trace correct | p_cond | p after 1 | p_cond^k | guess model | "
                  "local arith | answer-trace consistency |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in sorted(sp["div"], key=lambda c: (c["model"], C.divisor_of_task(c["eval_task"]) or 0, c["task"], c["arm"], c["seed"])):
                L.append(f"| {c['model']} | {c['task']} | {c['arm']} | {c['prompt_mode']} | {c['seed']} | {c['k']} | "
                         f"{pct(c['answer_acc'])} | {pct(c['trace_correct_rate'])} | {pct(c['p_cond'])} | "
                         f"{pct(c.get('p_cond_after_first_step'))} | {pct(c['predicted_acc_p_cond_pow_k'])} | "
                         f"{pct(c.get('post_hoc_predicted_answer_acc_own_g'))} | {pct(c['local_arith_acc'])} | "
                         f"{pct(c['answer_trace_consistency'])} |")
        if sp.get("prime"):
            L += ["", "| model | task | arm | seed | acc | errors | truncated & wrong | wrong with Answer line | "
                  "acc by true #trial divisions (1 / 2-3 / 4-8 / 9-16 / 17+) |", "|---|---|---|---|---|---|---|---|---|"]
            for c in sp["prime"]:
                L.append(f"| {c['model']} | {c['task']} | {c['arm']} | {c['seed']} | {pct(c['answer_acc'])} | {c['n_errors']} | "
                         f"{c['n_truncated_and_wrong']} | {c['n_wrong_with_answer_line']} | " + " / ".join(
                             pct(v["acc"], 0) for v in c["by_true_trial_divisions"].values()) + " |")
        L.append("")

    # ---------------------------------------------------------------- 5. tokens
    if tk:
        dc = tk["digit_check"]
        L += ["## 5. Tokenizer and training tokens (real Qwen2.5 tokenizer)", "",
              f"- Digits: {dc['verdict']} ({dc['exhaustive_4digit_contexts_checked']} contexts checked: bare, "
              "'Is n a prime number?', 'Is n divisible by 7?'); 6-digit examples also split into single digits.",
              "- Loss-bearing tokens per 180-example draw (completion + EOS, one epoch; x3 for 3 epochs), mean over seeds 0-2:", "",
              "| task | A | B | C | Bprime | B / A | B tokens per whitespace token |", "|---|---|---|---|---|---|---|"]
        for t, d in tk["train_tokens"].items():
            arms = d["arms"]
            row = [f"{arms[a]['mean_completion_tokens']:.0f}" if a in arms else "-" for a in ("A", "B", "C", "Bprime")]
            b = arms.get("B")
            L.append(f"| {t} | " + " | ".join(row) + (f" | {b['ratio_vs_A']:.1f} | {b['seeds']['0']['tokens_per_ws_token']:.2f} |"
                                                     if b else " | - | - |"))
        ok = [v["ws_matches_recorded_run"] for d in tk["train_tokens"].values() for a in d["arms"].values()
              for v in a["seeds"].values() if v["ws_matches_recorded_run"] is not None]
        L += ["", f"- Sampling replica check: whitespace-token count of the replicated draw equals the worker's recorded "
              f"train_tokens in {sum(ok)}/{len(ok)} cells (as of the last tokens.json build); A, B and C of one seed train on "
              "the same 180 items. C is length-matched corpus-wide, not per draw.", ""]
    if KEY:
        at = next(i for i, ln in enumerate(L) if ln.startswith("## Preregistered"))
        L[at:at] = ["## Key numbers", ""] + KEY + [""]
    L += ["## Caveats", "",
          "- Without generation files, per-item predictions are recovered from correctness bitmaps (wrong = flipped label; an "
          "unparseable '?' counts as a flip). With results/v2/gens the observed predictions are used (src = gens).",
          "- 'max on test' probe numbers are selected on the test set (optimistic); the CV-selected probe is the honest one.",
          "- Engineered number features contain n mod 3 and n mod 11, so on div3/div11 the engineered probes are handed the "
          "answer; the one-hot probes are the fair surface baseline there. On prime they supply partial-sieve features.",
          "- Pooled Wilson intervals treat items x seeds as independent; the paired bootstrap respects item pairing and seed "
          "variation and is wide with 2 seeds.", ""]
    open(os.path.join(C.OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"report -> {os.path.relpath(os.path.join(C.OUT, 'README.md'), C.ROOT)}")


if __name__ == "__main__":
    main()
