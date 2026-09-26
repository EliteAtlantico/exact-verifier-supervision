"""prior_stats.py -- derived statistics from the frozen prior grid (CPU only, no model calls).

Reads results/thinking_vs_data/{datasets.json, runs.jsonl} (per-item correctness bitmaps) and writes
paper/numbers_prior.tex: providecommand lines for per-class accuracies, rule accuracies and rule
agreement, the hard-negative preview on the existing test set, trace lengths, and exact McNemar tests,
plus paper/tab_runs.tex (every logged run). Surface probes are NOT computed here; they come from
experiments/analysis_probe.py. main.tex reads numbers.tex first (if present), so anything numbers.tex
defines takes precedence.

usage: python paper/exp/prior_stats.py
"""
from __future__ import annotations

import json
import math
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "thinking_vs_data")
OUT = os.path.join(HERE, "..", "numbers_prior.tex")
ANS = re.compile(r"Answer:\s*(Yes|No)")

D = json.load(open(os.path.join(RES, "datasets.json"), encoding="utf-8"))["tasks"]
RUNS = [json.loads(l) for l in open(os.path.join(RES, "runs.jsonl"), encoding="utf-8")]


def bits(h, n=240):
    v = int(h, 16)
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def run_bits(task, arm, n, seed):
    for r in RUNS:
        if (r["task"], r["arm"], r["n"], r["seed"]) == (task, arm, n, seed) and r.get("bits"):
            return bits(r["bits"], r["n_test"])
    return None


def mcnemar(x, y):
    """Exact two-sided McNemar test on paired correctness vectors; returns (b, c, p)."""
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if j and not i)
    k, n = min(b, c), b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def pct(x):
    return "%.1f" % x


def sci(p):
    if p >= 0.1:
        return "%.2f" % p
    if p >= 0.001:
        return "%s" % float("%.2g" % p)
    e = math.floor(math.log10(p))
    return "%.0f\\times 10^{%d}" % (p / 10 ** e, e)


def main():
    out = {}
    # prime: trace lengths, rules on this test set, per-class accuracy, rule agreement, hard negatives
    te = D["prime"]["test"]
    nums = [int(e["id"].split("-")[1]) for e in te]
    lab = [e["label"] == "Yes" for e in te]
    small = [p for p in range(2, 100) if all(p % q for q in range(2, int(p ** 0.5) + 1))]

    def steps(n):
        r, k = math.isqrt(n), 0
        for p in small:
            if p > r:
                break
            k += 1
            if n % p == 0:
                return k
        return k
    sp = [steps(n) for n, l in zip(nums, lab) if l]
    sc = sorted(steps(n) for n, l in zip(nums, lab) if not l)
    out["primeStepsMin"], out["primeStepsMax"] = str(min(sp)), str(max(sp))
    out["primeStepsMeanPrimes"] = "%.1f" % (sum(sp) / len(sp))
    out["primeStepsMedComp"] = str(sc[len(sc) // 2])
    out["primeStepsMeanComp"] = "%.1f" % (sum(sc) / len(sc))
    hard = [i for i, n in enumerate(nums) if not lab[i] and all(n % p for p in (2, 3, 5, 7))]
    out["primeHardNegN"] = str(len(hard))
    out["ruleLastDigit"] = pct(100 * sum((n % 10 in (1, 3, 7, 9)) == l for n, l in zip(nums, lab)) / len(te))
    rules = (("ruleOdd", (2,)), ("ruleNoTwoThree", (2, 3)), ("ruleNoSeven", (2, 3, 5, 7)))
    for rule, ps in rules:
        out[rule] = pct(100 * sum((all(n % p for p in ps)) == l for n, l in zip(nums, lab)) / len(te))
    # the same rules on the earlier answer-only study's prime test set, regenerated exactly as
    # experiments/exp_gate_ft_worker.py builds it (random.Random(999), n in [100, 9999], 120 per class)
    rng = random.Random(999)
    pos, neg, seen = [], [], set()
    while len(pos) < 280 or len(neg) < 280:
        n = rng.randint(100, 9999)
        if n in seen:
            continue
        seen.add(n)
        (pos if n > 1 and all(n % q for q in range(2, math.isqrt(n) + 1)) else neg).append(n)
    old = [(x, True) for x in pos[:120]] + [(x, False) for x in neg[:120]]
    for rule, ps in rules:
        out[rule + "Prior"] = pct(100 * sum((all(n % p for p in ps)) == l for n, l in old) / len(old))
    acc = {}
    for arm in ("A", "B"):
        pc, cc, agree, agree_ld, hn = [], [], [], [], []
        for s in (0, 1):
            b = run_bits("prime", arm, 180, s)
            pred = [l if ok else (not l) for l, ok in zip(lab, b)]
            pc.append(100 * sum(ok for ok, l in zip(b, lab) if l) / 120)
            cc.append(100 * sum(ok for ok, l in zip(b, lab) if not l) / 120)
            agree.append(100 * sum(p == (n % 2 == 1) for p, n in zip(pred, nums)) / len(te))
            agree_ld.append(100 * sum(p == (n % 10 in (1, 3, 7, 9)) for p, n in zip(pred, nums)) / len(te))
            hn.append(sum(b[i] for i in hard))
        acc[arm] = (pc, cc, agree, hn, agree_ld)
    out["primeAprimesCorrect"] = pct(sum(acc["A"][0]) / 2)
    out["primeAcompCorrect"] = pct(sum(acc["A"][1]) / 2)
    out["primeAagreeOdd"] = pct(sum(acc["A"][2]) / 2)
    out["primeAagreeLastDigitLo"] = pct(min(acc["A"][4]))
    out["primeAagreeLastDigitHi"] = pct(max(acc["A"][4]))
    out["primeHardNegAcorrect"] = "%d and %d" % tuple(acc["A"][3])
    ev = {(r["task"], r["arm"], r["n"], r["seed"]): r.get("eval_s") for r in RUNS}
    out["baseEvalRatio"] = "%.0f" % (ev[("prime", "base", 0, 0)] / ev[("prime", "A", 180, 0)])
    out["primeBprimesCorrect"] = pct(sum(acc["B"][0]) / 2)
    out["primeBcompCorrect"] = pct(sum(acc["B"][1]) / 2)

    # div7 per class (B seed 0; A both seeds)
    lab7 = [e["label"] == "Yes" for e in D["div7"]["test"]]
    b = run_bits("div7", "B", 180, 0)
    out["divsevenByes"] = pct(100 * sum(ok for ok, l in zip(b, lab7) if l) / 120)
    out["divsevenBno"] = pct(100 * sum(ok for ok, l in zip(b, lab7) if not l) / 120)
    for s, tag in ((0, "sZero"), (1, "sOne")):
        a = run_bits("div7", "A", 180, s)
        out["divsevenAyes" + tag] = pct(100 * sum(ok for ok, l in zip(a, lab7) if l) / 120)
        out["divsevenAno" + tag] = pct(100 * sum(ok for ok, l in zip(a, lab7) if not l) / 120)

    # exact McNemar (paired on the 240 shared test items)
    def mc(task, x, y, sx, sy, name):
        bx, by = run_bits(task, x[0], x[1], sx), run_bits(task, y[0], y[1], sy)
        if bx and by:
            b_, c_, p = mcnemar(bx, by)
            out[name] = sci(p)
            out[name + "Disc"] = "%d/%d" % (b_, c_)
    for s, tag in ((0, "sZero"), (1, "sOne")):
        mc("prime", ("B", 180), ("A", 180), s, s, "primeBAp" + tag)
        mc("prime", ("B", 180), ("C", 180), s, s, "primeBCp" + tag)
        mc("valid", ("B", 180), ("A", 180), s, s, "validBAp" + tag)
        mc("valid", ("B", 180), ("C", 180), s, s, "validBCp" + tag)
        # the base run exists for seed 0 only; each A seed is paired with it
        mc("div7", ("A", 180), ("base", 0), s, 0, "divsevenABasep" + tag)
    mc("div7", ("B", 180), ("A", 180), 0, 0, "divsevenBApsZero")
    mc("div7", ("B", 180), ("C", 180), 0, 0, "divsevenBCpsZero")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% generated by paper/exp/prior_stats.py from results/thinking_vs_data -- do not edit\n")
        for k, v in out.items():
            f.write("\\providecommand{\\%s}{%s}\n" % (k, v))
    for k, v in out.items():
        print(k, v)
    write_run_table()


def write_run_table():
    """Appendix table of every run in runs.jsonl (one row per run, as logged)."""
    arm_tex = {"base": "base", "A": "A", "B": "B", "C": "C", "A_tok": "A-full", "B_orn": "B-ext"}
    order = {"prime": 0, "div7": 1, "valid": 2}
    rows = [r for r in RUNS if r.get("acc") is not None and r["arm"] in arm_tex]
    rows.sort(key=lambda r: (order.get(r["task"], 9), list(arm_tex).index(r["arm"]), r["n"], r["seed"]))
    task_tex = {"prime": "prime", "div7": "div7", "valid": "valid"}
    lines = ["% generated by paper/exp/prior_stats.py from results/thinking_vs_data/runs.jsonl -- do not edit",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "task & arm & $n$ & seed & train tokens & accuracy (\\%) \\\\", "\\midrule"]
    last = None
    for r in rows:
        if last is not None and r["task"] != last:
            lines.append("\\midrule")
        last = r["task"]
        lines.append("%s & %s & %d & %d & %s & %.1f \\\\" % (
            task_tex[r["task"]], arm_tex[r["arm"]], r["n"], r["seed"],
            "{:,}".format(r.get("train_tokens") or 0).replace(",", "{,}"), 100 * r["acc"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(HERE, "..", "tab_runs.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
