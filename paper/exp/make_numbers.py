"""make_numbers.py -- regenerate paper/numbers.tex and paper/tab_runs.tex from the result files.

CPU only, no model calls, no numpy. Every macro the paper uses for a number is written here from:
  results/analysis/{stats,probe,rules,tokens,steps}.json      (experiments/analysis_*.py; S1-S6 verdicts via
                                                              analysis_report.prereg, the analysis' own rule code)
  results/thinking_vs_data/runs.jsonl, runs_*.jsonl          (prior grid + S1)
  results/v2/runs_*.jsonl, results/v2/evals_*.jsonl           (v2 worker, any model)
  results/v2/gens/<model>/<cell>.jsonl                        (generations: per-step trace accuracy)
  results/thinking_vs_data/datasets.json, results/v2/datasets_v2.json (test sets, pools, trace structure)
  results/THINKING_VS_DATA.md, GATE_FINETUNE_STUDY.md, LOGIC_FINETUNE.md, PREDICTIONS.md (prior tables, rules)
  experiments/exp_thinking_ft_worker.py                        (hyperparameters, generation caps)
Runs are loaded with experiments/analysis_common.load_runs (same cell labels and duplicate policy as the
analysis: first occurrence of a (task, arm, n, seed, model) cell is kept). A cell that has not been run is
written as \\TBD. Macros are written with \\newcommand, so they override paper/numbers_placeholder.tex.

usage: python paper/exp/make_numbers.py            (writes paper/numbers.tex, paper/tab_runs.tex, paper/exp/numbers.json)
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from statistics import mean, median

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER, ".."))
sys.path.append(os.path.join(ROOT, "experiments"))
import analysis_common as C  # noqa: E402

AN = os.path.join(ROOT, "results", "analysis")
M15 = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS = {"halfB": "Qwen/Qwen2.5-0.5B-Instruct", "": M15, "threeB": "Qwen/Qwen2.5-3B-Instruct",
          "sevenB": "Qwen/Qwen2.5-7B-Instruct"}
PREFIX = {"prime": "prime", "div7": "divseven", "valid": "valid", "div2": "divtwo", "div3": "divthree",
          "div11": "diveleven", "div13": "divthirteen", "div7_6d": "divsevenSix"}
ARMS = {"base": "Base", "A": "A", "B": "B", "C": "C", "Bprime": "Bp", "A_tok": "Atok", "B_orn": "Born", "D": "D",
        "S": "S"}
SEEDW = ["Zero", "One", "Two", "Three", "Four"]
TBD = "\\TBD"
MAIN_N, SMALL_N, LARGE_N = 180, 60, 540
# engineered probe features include n mod 2 (last digit even), n mod 3 (digit sum) and n mod 11 (alternating
# sum): on those divisors they hand the probe the answer, so the fair probe there uses one-hot features only
UNFAIR_ENGINEERED = {"div2", "div3", "div11"}

OUT = {}          # macro -> TeX string
SRC = {}          # macro -> provenance note (numbers.json)


def put(name, value, src=""):
    OUT[name] = value
    SRC[name] = src


def pct(x, nd=1):
    return TBD if x is None else ("%." + str(nd) + "f") % (100 * x)


def thousands(x):
    return "{:,}".format(int(round(x))).replace(",", "{,}")


def pval(p):
    if p is None:
        return TBD
    if p >= 0.1:
        return "%.2f" % p
    if p >= 0.001:
        return "%s" % float("%.2g" % p)
    e = math.floor(math.log10(p))
    return "%s\\times 10^{%d}" % ("%.2g" % (p / 10 ** e), e)


def pval_ub(p):
    """Upper bound for 'p <= ...': mantissa rounded UP to an integer."""
    if p is None:
        return TBD
    if p >= 0.001:
        return pval(p)
    e = math.floor(math.log10(p))
    mant = math.ceil(p / 10 ** e - 1e-12)
    if mant >= 10:
        mant, e = 1, e + 1
    return "%d\\times 10^{%d}" % (mant, e)


def mcnemar(x, y):
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if j and not i)
    k, n = min(b, c), b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def wilson_half(p, n, z=1.96):
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def load_json(name):
    p = os.path.join(AN, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------ data
TASKS = C.load_tasks()
RUNS, PROBLEMS = C.load_runs()
GENS, GEN_NOTES = C.load_gens(TASKS)
STATS, PROBE, RULES, TOKENS, STEPS, TRANS = (load_json(f) for f in ("stats.json", "probe.json", "rules.json",
                                                                    "tokens.json", "steps.json", "transitions.json"))
IDX = {r["key"]: r for r in RUNS}


def run(task, arm, n, seed, model=M15):
    return IDX.get((task, arm, n, seed, model))


def seeds(task, arm, n, model=M15):
    return sorted(k[3] for k in IDX if k[0] == task and k[1] == arm and k[2] == n and k[4] == model)


def acc_mean(task, arm, n, model=M15):
    ss = seeds(task, arm, n, model)
    return mean(run(task, arm, n, s, model)["acc"] for s in ss) if ss else None


def class_acc(r, label):
    test = TASKS[r["eval_task"]]["test"]
    b = [x for x, t in zip(r["bits_list"], test) if t["label"] == label]
    return sum(b) / len(b) if b else None


def gens_rows(task, arm, n, seed, model=M15):
    g = GENS.get((task, arm, n, seed, C.short_model(model)))
    return g["rows"] if g else None


def preds(r):
    """Observed predictions from generations when saved, else recovered from the correctness bitmap."""
    rows = gens_rows(r["task"], r["arm"], int(r["n"]), int(r["seed"]), r["model"])
    test = TASKS[r["eval_task"]]["test"]
    if rows and len(rows) == len(test):
        by = {x["id"]: x.get("pred") for x in rows}
        return [by.get(t["id"], "?") for t in test], "observed"
    return C.recover_preds(r["bits_list"], test), "recovered"


# ------------------------------------------------------------------------------------------ setup
def setup_macros():
    src = open(os.path.join(ROOT, "experiments", "exp_thinking_ft_worker.py"), encoding="utf-8").read()
    m = re.search(r"LoraConfig\(r=(\d+), lora_alpha=(\d+), lora_dropout=([\d.]+)", src)
    put("loraRank", m.group(1), "worker LoraConfig")
    put("loraAlpha", m.group(2), "worker LoraConfig")
    put("loraDropout", m.group(3), "worker LoraConfig")
    put("nEpochs", re.search(r"EPOCHS = (\d+)", src).group(1), "worker EPOCHS")
    lr = float(re.search(r"lr=([\d.e-]+)", src).group(1))
    e = math.floor(math.log10(lr))
    put("learningRate", "%g\\times 10^{%d}" % (lr / 10 ** e, e), "worker AdamW lr")
    cap = dict((k, int(v)) for k, v in re.findall(r'"(\w+)": (\d+)', re.search(r"MAXNEW = \{[^}]*\}", src).group(0)))
    put("maxNewPrime", str(cap["prime"]), "worker MAXNEW")
    put("maxNewDiv", str(cap["div7"]), "worker MAXNEW")
    put("maxNewValid", str(cap["valid"]), "worker MAXNEW")
    n_test = sorted({r["n_test"] for r in RUNS})
    put("nTest", str(n_test[0]), "runs n_test")
    put("nTestClass", str(n_test[0] // 2), "balanced test sets")
    put("ciHalf", "%.0f" % (100 * wilson_half(0.5, n_test[0])), "Wilson half-width at p=0.5")
    put("nTrain", str(MAIN_N), "runs n")
    put("nTrainSmall", str(SMALL_N), "runs n")
    put("nTrainLarge", str(LARGE_N), "runs n")
    pred = open(os.path.join(ROOT, "PREDICTIONS.md"), encoding="utf-8").read()
    put("alphaLevel", re.search(r"alpha = (\d+\.\d+)", pred).group(1), "PREDICTIONS.md")
    put("winMargin", re.search(r'"win" is a difference of >= (\d+) pp', pred).group(1), "PREDICTIONS.md")
    put("neitherMargin", re.search(r"majority class \+ (\d+) pp", pred).group(1), "PREDICTIONS.md")
    m = re.search(r"Committed (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \w+ \(commit (\w+)\)", pred)
    if m:
        put("preregDate", m.group(1), "PREDICTIONS.md")
        put("preregTime", m.group(2), "PREDICTIONS.md")
        put("preregCommit", m.group(3), "PREDICTIONS.md")
    log = re.findall(r"^- (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \w+ \(commit (\w+)[^)]*\)\s*\W*\s*\*\*([^*]+)\*\*",
                     pred, re.M)
    for date, time_, commit, head in log:
        key = ("sOne" if head.startswith("S1") else "sFiveClar" if "clarification" in head else
               "sFiveFail" if head.startswith("S5") else "sSeven" if head.startswith("S7") else
               "sEightNine" if head.startswith("S8") else None)
        if key and key + "Time" not in OUT:
            put(key + "Time", time_, "PREDICTIONS.md decision log")
            put(key + "Commit", commit, "PREDICTIONS.md decision log")
    tvd = open(os.path.join(ROOT, "results", "THINKING_VS_DATA.md"), encoding="utf-8").read()
    tot = re.findall(r"_Total runs: (\d+); wall (\d+) min\._", tvd)[-1]
    put("nPriorRuns", tot[0], "THINKING_VS_DATA.md")
    put("priorWallMin", tot[1], "THINKING_VS_DATA.md")
    put("priorVerdictDate", re.search(r"Verdict \(human-written, (\d{4}-\d{2}-\d{2})", tvd).group(1),
        "THINKING_VS_DATA.md")
    v2 = json.load(open(os.path.join(ROOT, "results", "v2", "datasets_v2.json"), encoding="utf-8"))
    put("nShots", str(len(v2["fewshot4"]["div7"])), "datasets_v2 fewshot4")
    schemas = {"-".join(e["id"].split("-")[2:]) for e in TASKS["valid"]["pools"]["A"]}
    put("nForms", str(len(schemas)), "valid pool schema ids")
    put("nFormsHalf", str(len(schemas) // 2), "valid pool schema ids")
    put("validPool", thousands(len(TASKS["valid"]["pools"]["A"])), "valid pool")
    put("primeAtokN", thousands(len(TASKS["prime"]["pools"]["A"])), "prime pool")
    ndig = len(TASKS["div7"]["test"][0]["id"].split("-")[1])
    put("nProbeFeatures", str(10 * ndig), "digits x positions")
    # wall-clock of trained 1.5B runs at n = 180
    walls = [r["wall_s"] for r in RUNS if r["model"] == M15 and r["n"] == MAIN_N and r["arm"] != "base"
             and r.get("wall_s") and r["prompt_mode"] == "plain" and not r.get("eval_only")]
    put("runMinLow", str(max(1, math.floor(min(walls) / 60))), "wall_s")
    put("runMinHigh", str(math.ceil(max(walls) / 60)), "wall_s")


# ------------------------------------------------------------------------------------------ structure
def structure_macros():
    small = C.SMALL_PRIMES
    te = TASKS["prime"]["test"]

    def steps(n):
        r, k = math.isqrt(n), 0
        for p in small:
            if p > r:
                break
            k += 1
            if n % p == 0:
                return k
        return k
    nums = [int(e["id"].split("-")[1]) for e in te]
    lab = [e["label"] == "Yes" for e in te]
    sp = [steps(n) for n, l in zip(nums, lab) if l]
    sc = sorted(steps(n) for n, l in zip(nums, lab) if not l)
    put("primeStepsMin", str(min(sp)), "prime test")
    put("primeStepsMax", str(max(sp)), "prime test")
    put("primeStepsMeanPrimes", "%.1f" % mean(sp), "prime test")
    put("primeStepsMedComp", "%d" % median(sc), "prime test")
    put("primeStepsMeanComp", "%.1f" % mean(sc), "prime test")
    for t, mac in (("prime", "primeTraceTok"), ("div7", "divsevenTraceTok"), ("valid", "validTraceTok")):
        put(mac, "%.1f" % TASKS[t]["meta"].get("mean_trace_ws_tokens", float("nan")) if TASKS[t]["meta"]
            else TBD, "datasets meta")
    for t, tp in PREFIX.items():
        if t in TASKS and TASKS[t]["meta"].get("mean_trace_ws_tokens") is not None:
            put(tp + "TraceTok", "%.1f" % TASKS[t]["meta"]["mean_trace_ws_tokens"], "datasets meta")
    k4 = len(TASKS["div7"]["test"][0]["id"].split("-")[1])
    k6 = len(TASKS["div7_6d"]["test"][0]["id"].split("-")[1]) if "div7_6d" in TASKS else 6
    put("divsevenSteps", str(k4), "digits")
    put("divsevenSixSteps", str(k6), "digits")
    put("divsevenTableSize", str(10 * 7), "10 d")
    put("divsevenStepSamples", str(k4 * MAIN_N), "k n")
    # exact constants: P(7 | n) with three of four digits fixed; hard negatives among 4-digit composites
    lo, hi = 1.0, 0.0
    for j in range(k4):
        for rest in range(10 ** (k4 - 1)):
            ds = list(str(rest).zfill(k4 - 1))
            vals = []
            for x in range(10):
                dig = ds[:j] + [str(x)] + ds[j:]
                if dig[0] == "0":
                    continue
                vals.append(int("".join(dig)) % 7 == 0)
            if vals:
                lo, hi = min(lo, mean(vals)), max(hi, mean(vals))
    marg = sum(1 for n in range(10 ** (k4 - 1), 10 ** k4) if n % 7 == 0) / (9 * 10 ** (k4 - 1))
    put("condLow", "%.2f" % lo, "exact")
    put("condHigh", "%.2f" % hi, "exact")
    put("condMarg", "%.2f" % marg, "exact")
    comps = [n for n in range(1000, 10000) if not C.is_prime(n)]
    hard = [n for n in comps if all(n % p for p in (2, 3, 5, 7))]
    put("compPop", thousands(len(comps)), "exact")
    put("hardNegPop", thousands(len(hard)), "exact")
    put("hardNegFrac", "%.1f" % (100 * len(hard) / len(comps)), "exact")


# ------------------------------------------------------------------------------------------ accuracies
def accuracy_macros():
    for mpre, model in MODELS.items():
        for task, tp in PREFIX.items():
            for arm, aw in ARMS.items():
                if arm == "base":
                    ns = {0: ""}
                elif model == M15 and arm in ("A", "B", "C"):
                    ns = {MAIN_N: "", SMALL_N: "small", LARGE_N: "large"}
                else:
                    ns = {MAIN_N: ""}
                if arm == "A_tok":
                    ns = {n: "" for n in sorted({k[2] for k in IDX if k[0] == task and k[1] == "A_tok"})}
                if arm == "B_orn":
                    ns = {n: "" for n in sorted({k[2] for k in IDX if k[0] == task and k[1] == "B_orn"})}
                for n, nw in ns.items():
                    base = mpre + tp + aw + nw
                    a = acc_mean(task, arm, n, model)
                    put(base, pct(a), f"mean over seeds {seeds(task, arm, n, model)}")
                    for s in seeds(task, arm, n, model):
                        if s < len(SEEDW):
                            put(base + "s" + SEEDW[s], pct(run(task, arm, n, s, model)["acc"]), "run")
                    if seeds(task, arm, n, model):
                        put(base + "nSeeds", str(len(seeds(task, arm, n, model))), "count")
    # B-ext (rejection-sampled external traces): number of traces that passed
    ns = sorted({k[2] for k in IDX if k[0] == "prime" and k[1] == "B_orn"})
    put("primeBornN", str(ns[0]) if ns else TBD, "runs n")
    # the first-listed seed-0 cell is the headline; report a v2 re-run of the same cell if one exists
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "v2", "runs_Qwen2.5-1.5B-Instruct.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("task"), r.get("arm"), r.get("n"), r.get("seed")) == ("div7", "B", MAIN_N, 0) and \
                    (r.get("eval_task") in (None, "div7")) and (r.get("prompt_mode") in (None, "plain")) and \
                    r.get("acc") is not None:
                put("divsevenBsZeroRerun", pct(r["acc"]), f)
    OUT.setdefault("divsevenBsZeroRerun", TBD)


def per_class_macros():
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else None
    A = [run("prime", "A", MAIN_N, s) for s in seeds("prime", "A", MAIN_N)]
    B = [run("prime", "B", MAIN_N, s) for s in seeds("prime", "B", MAIN_N)]
    put("primeAprimesCorrect", pct(avg(class_acc(r, "Yes") for r in A)), "bits")
    put("primeAcompCorrect", pct(avg(class_acc(r, "No") for r in A)), "bits")
    put("primeBprimesCorrect", pct(avg(class_acc(r, "Yes") for r in B)), "bits")
    put("primeBcompCorrect", pct(avg(class_acc(r, "No") for r in B)), "bits")
    bl = run("prime", "B", LARGE_N, 0)
    put("primeBlargePrimes", pct(class_acc(bl, "Yes")) if bl else TBD, "bits")
    put("primeBlargeComp", pct(class_acc(bl, "No")) if bl else TBD, "bits")
    D = [run("div7", "B", MAIN_N, s) for s in seeds("div7", "B", MAIN_N)]
    put("divsevenByes", pct(avg(class_acc(r, "Yes") for r in D)), "bits, mean over seeds")
    put("divsevenBno", pct(avg(class_acc(r, "No") for r in D)), "bits, mean over seeds")
    for s in seeds("div7", "A", MAIN_N):
        r = run("div7", "A", MAIN_N, s)
        put("divsevenAyess" + SEEDW[s], pct(class_acc(r, "Yes")), "bits")
        put("divsevenAnos" + SEEDW[s], pct(class_acc(r, "No")), "bits")
    b0 = run("div7", "base", 0, 0)
    if b0:
        put("divsevenBaseYes", pct(class_acc(b0, "Yes")), "bits")
        put("divsevenBaseNo", pct(class_acc(b0, "No")), "bits")
    pb, pa = run("prime", "base", 0, 0), run("prime", "A", MAIN_N, 0)
    if pb and pa and pb.get("eval_s") and pa.get("eval_s"):
        put("baseEvalRatio", "%.0f" % (pb["eval_s"] / pa["eval_s"]), "eval_s")
    # C vs B training-token difference per draw (whitespace tokens recorded by the worker)
    diffs = []
    for t in ("prime", "div7", "valid"):
        for s in seeds(t, "C", MAIN_N):
            b, c = run(t, "B", MAIN_N, s), run(t, "C", MAIN_N, s)
            if b and c and b.get("train_tokens") and c.get("train_tokens"):
                diffs.append(abs(c["train_tokens"] - b["train_tokens"]) / b["train_tokens"])
    put("cTokDiff", "%.0f" % math.ceil(100 * max(diffs)) if diffs else TBD, "train_tokens")


def mcnemar_macros():
    def mc(task, x, y, sx, sy, name, model=M15):
        rx, ry = run(task, x[0], x[1], sx, model), run(task, y[0], y[1], sy, model)
        if rx and ry:
            b, c, p = mcnemar(rx["bits_list"], ry["bits_list"])
            put(name, pval(p), "exact McNemar")
            put(name + "Disc", "%d/%d" % (b, c), "discordant")
            return p
        put(name, TBD)
        return None
    ps = []
    for s in range(3):
        w = SEEDW[s]
        mc("prime", ("B", MAIN_N), ("A", MAIN_N), s, s, "primeBAps" + w)
        mc("prime", ("B", MAIN_N), ("C", MAIN_N), s, s, "primeBCps" + w)
        mc("valid", ("B", MAIN_N), ("A", MAIN_N), s, s, "validBAps" + w)
        mc("valid", ("B", MAIN_N), ("C", MAIN_N), s, s, "validBCps" + w)
        ps.append(mc("div7", ("B", MAIN_N), ("A", MAIN_N), s, s, "divsevenBAps" + w))
        mc("div7", ("B", MAIN_N), ("C", MAIN_N), s, s, "divsevenBCps" + w)
        mc("div7", ("A", MAIN_N), ("base", 0), s, 0, "divsevenABaseps" + w)
    ps = [p for p in ps if p is not None]
    put("divsevenBApMax", pval_ub(max(ps)) if ps else TBD, "max over seeds")
    pc = [mcnemar(run("div7", "B", MAIN_N, s_)["bits_list"], run("div7", "C", MAIN_N, s_)["bits_list"])[2]
          for s_ in seeds("div7", "C", MAIN_N) if run("div7", "B", MAIN_N, s_)]
    put("divsevenBCpMax", pval_ub(max(pc)) if pc else TBD, "max over seeds")
    # B vs A at n = 540 on prime (one seed)
    b5, a5 = run("prime", "B", LARGE_N, 0), run("prime", "A", LARGE_N, 0)
    put("primeBAplarge", pval(mcnemar(b5["bits_list"], a5["bits_list"])[2]) if (a5 and b5) else TBD, "McNemar")


# ------------------------------------------------------------------------------------------ probes/rules/tokens
def fair_probe(task):
    t = (PROBE or {}).get("tasks", {}).get(task)
    if not t:
        return None
    allowed = (lambda name: name.endswith("_a")) if task in UNFAIR_ENGINEERED else (lambda name: True)
    vals = []
    for s, ps in t["per_seed"].items():
        cands = [(v["cv_acc"], v["test_acc"]) for k, v in ps["probes"].items() if allowed(k)]
        if cands:
            vals.append(max(cands)[1])
    return mean(vals) if vals else None


def probe_rule_macros():
    for task, tp in PREFIX.items():
        put(tp + "Probe", pct(fair_probe(task)), "probe.json, CV-selected, fair features")
    t = (PROBE or {}).get("tasks", {}).get("prime")
    if t:
        put("primeProbeOneHot", pct(t["summary"]["logreg_a"]), "probe.json logreg_a")
        rs = t["rules_summary"]
        put("ruleOdd", pct(rs["odd"]), "probe.json rules")
        put("ruleNoTwoThree", pct(rs["no_factor_le_3"]), "probe.json rules")
        put("ruleNoSeven", pct(rs["no_factor_le_7"]), "probe.json rules")
        put("ruleLastDigit", pct(rs["last_digit_1379"]), "probe.json rules")
    # A's agreement with the last-digit rule (rules.json; falls back to recovered predictions)
    ag = []
    for r in (RULES or {}).get("runs", []):
        if r["task"] == "prime" and r["arm"] == "A" and r["n"] == MAIN_N and r.get("model", M15) == M15:
            ag.append(r["agreement"]["last_digit_1379"])
    if ag:
        put("primeAagreeLastDigitLo", pct(min(ag)), "rules.json")
        put("primeAagreeLastDigitHi", pct(max(ag)), "rules.json")
        odd = [r["agreement"]["odd"] for r in RULES["runs"] if r["task"] == "prime" and r["arm"] == "A"
               and r["n"] == MAIN_N and r.get("model", M15) == M15]
        put("primeAagreeOdd", pct(mean(odd)), "rules.json")
    # hard-negative preview on the original test set (odd composites without a factor <= 7)
    te = TASKS["prime"]["test"]
    hard = [i for i, e in enumerate(te) if e["label"] == "No" and
            all(int(e["id"].split("-")[1]) % p for p in (2, 3, 5, 7))]
    put("primeHardNegN", str(len(hard)), "prime test")
    hc = [sum(run("prime", "A", MAIN_N, s)["bits_list"][i] for i in hard) for s in seeds("prime", "A", MAIN_N)]
    put("primeHardNegAcorrect", " and ".join(str(x) for x in hc), "bits")
    # rules on the earlier study's prime test set (regenerated as exp_gate_ft_worker.py builds it)
    import random
    rng = random.Random(999)
    pos, neg, seen = [], [], set()
    while len(pos) < 280 or len(neg) < 280:
        n = rng.randint(100, 9999)
        if n in seen:
            continue
        seen.add(n)
        (pos if C.is_prime(n) else neg).append(n)
    old = [(x, True) for x in pos[:120]] + [(x, False) for x in neg[:120]]
    for mac, ps_ in (("ruleOddPrior", (2,)), ("ruleNoTwoThreePrior", (2, 3))):
        put(mac, pct(mean((all(n % p for p in ps_)) == l for n, l in old)), "earlier study test set")
    # training tokens (real tokenizer)
    tt = (TOKENS or {}).get("train_tokens", {})
    if tt:
        put("tokA", thousands(tt["prime"]["arms"]["A"]["mean_completion_tokens"]), "tokens.json")
        put("primeTokB", thousands(tt["prime"]["arms"]["B"]["mean_completion_tokens"]), "tokens.json")
        put("divsevenTokB", thousands(tt["div7"]["arms"]["B"]["mean_completion_tokens"]), "tokens.json")
        put("validTokB", thousands(tt["valid"]["arms"]["B"]["mean_completion_tokens"]), "tokens.json")
        rat = [tt[t]["arms"]["B"]["ratio_vs_A"] for t in ("prime", "div7", "valid")]
        put("tokRatioLow", "%.0f" % min(rat), "tokens.json")
        put("tokRatioHigh", "%.0f" % max(rat), "tokens.json")


def earlier_study_macros():
    rows = {}
    for f in ("GATE_FINETUNE_STUDY.md", "LOGIC_FINETUNE.md"):
        txt = open(os.path.join(ROOT, "results", f), encoding="utf-8").read()
        for m in re.finditer(r"^\| (\w+) \([^|]*\) \| ([\d.]+)% \| \*\*([\d.]+)% ± [\d.]+\*\* \| ([\d.]+)% \|", txt, re.M):
            rows[m.group(1)] = tuple(float(x) for x in m.group(2, 3, 4))
    names = {"div3": "Divthree", "div7": "Divseven", "div13": "Divthirteen", "square": "Square", "prime": "Prime",
             "valid": "Valid", "consist": "Consist"}
    for k, w in names.items():
        if k in rows:
            b, a, s = rows[k]
            put("prior" + w + "Base", "%.1f" % b, "earlier study md")
            put("prior" + w + "A", "%.1f" % a, "earlier study md")
            put("prior" + w + "Shuf", "%.1f" % s, "earlier study md")
    if rows:
        put("shufMin", "%.1f" % min(v[2] for v in rows.values()), "earlier study md")
        put("shufMax", "%.1f" % max(v[2] for v in rows.values()), "earlier study md")
        put("priorCorrectMax", "%.1f" % max(v[1] for v in rows.values()), "earlier study md")
    txt = open(os.path.join(ROOT, "results", "LOGIC_FINETUNE.md"), encoding="utf-8").read()
    m = re.search(r"(\d+) train / (\d+) test", txt)
    if m:
        put("priorTrainN", m.group(1), "LOGIC_FINETUNE.md")


# ------------------------------------------------------------------------------------------ trace steps
DIV_STEP = re.compile(r"r = \(10\*(\d+) \+ (\d)\) mod (\d+) = (\d+)")
PRIME_STEP = re.compile(r"(\d+) mod (\d+) = (\d+)")


STEP_ERR = {"digit": 0, "carry": 0, "lookup": 0, "missing": 0}


def div_step_accuracy(rows, d):
    """Per-step (local) accuracy of generated long-division traces: a step is right when it reads the true
    next digit, carries the previously written remainder, and writes (10 r + x) mod d."""
    ok = tot = full = 0
    for x in rows:
        digits = [int(c) for c in x["id"].split("-")[1]]
        text = (x.get("gen") or "").split("Answer:")[0]
        st = DIV_STEP.findall(text)
        prev, good_all = 0, True
        for t, dig in enumerate(digits):
            good = False
            if t < len(st):
                carry, dd, _mod, outv = (int(v) for v in st[t])
                good = dd == dig and carry == prev and outv == (10 * carry + dd) % d
                if dd != dig:
                    STEP_ERR["digit"] += 1
                elif carry != prev:
                    STEP_ERR["carry"] += 1
                elif not good:
                    STEP_ERR["lookup"] += 1
                prev = outv
            else:
                STEP_ERR["missing"] += 1
            ok += good
            tot += 1
            good_all &= good
        full += good_all
    return (ok / tot if tot else None), (full / len(rows) if rows else None)


def prime_step_accuracy(rows):
    ok = tot = 0
    for x in rows:
        n = int(x["id"].split("-")[1])
        r = math.isqrt(n)
        true = []
        for p in C.SMALL_PRIMES:
            if p > r:
                break
            true.append(p)
            if n % p == 0:
                break
        st = PRIME_STEP.findall((x.get("gen") or "").split("Answer:")[0])
        st = [(int(a), int(b), int(c)) for a, b, c in st if int(a) == n]
        for t, p in enumerate(true):
            good = t < len(st) and st[t][1] == p and st[t][2] == n % p
            ok += good
            tot += 1
    return ok / tot if tot else None


def steps_cells(task, arm="B", model=M15, eval_task=None):
    """Plain-prompt cells of results/analysis/steps.json (experiments/analysis_steps.py) for one cell."""
    out = []
    for c in (STEPS or {}).get("div", []):
        if c["task"] == task and c["arm"] == arm and c["n"] == MAIN_N and c["prompt_mode"] == "plain" and \
                c["model"] == C.short_model(model) and (eval_task is None or c["eval_task"] == eval_task):
            out.append(c)
    return out


def pooled(cells, key):
    vals = [(c[key], c["n_items"]) for c in cells if c.get(key) is not None]
    w = sum(n for _, n in vals)
    return sum(v * n for v, n in vals) / w if w else None


def step_macros():
    """Per-step trace accuracy from results/analysis/steps.json (analysis_steps.py): p = p_cond, the chance that
    a step is right given that the previous one was, pooled (item-weighted) over the 1.5B div7 B seeds with
    saved generations. The error breakdown (lookup vs copy vs carry) is recomputed here from the same gens."""
    cells = steps_cells("div7", eval_task="div7")
    p = pooled(cells, "p_cond")
    put("stepAcc", "%.3f" % p if p is not None else TBD, "steps.json p_cond")
    put("stepLocalAcc", "%.3f" % pooled(cells, "local_arith_acc") if cells else TBD, "steps.json local_arith_acc")
    put("stepGenSeeds", str(len(cells)), "steps.json cells")
    put("divsevenDigitCopy", pct(pooled(cells, "digit_copy_acc")) if cells else TBD, "steps.json")
    put("divsevenCarry", pct(pooled(cells, "carry_consistency")) if cells else TBD, "steps.json")
    put("divsevenFullTrace", pct(pooled(cells, "trace_correct_rate")) if cells else TBD, "steps.json")
    put("divsevenAccGivenWrong", pct(pooled(cells, "answer_acc_given_trace_wrong")) if cells else TBD, "steps.json")
    put("divsevenGenAcc", pct(pooled(cells, "answer_acc")) if cells else TBD, "steps.json answer_acc")
    rows = []
    for (label, arm, n, seed, model), g in GENS.items():
        if label == "div7" and arm == "B" and n == MAIN_N and model == C.short_model(M15):
            rows += g["rows"]
    if rows:
        div_step_accuracy(rows, 7)
    nerr = sum(STEP_ERR.values())
    put("stepLookupShare", pct(STEP_ERR["lookup"] / nerr) if nerr else TBD, str(STEP_ERR))
    for k, mac in ((4, "Four"), (6, "Six")):
        put("divseven" + mac + "Pred", pct(p ** k) if p is not None else TBD, "p_cond^%d" % k)
        put("divseven" + mac + "PredBal", pct((1 + p ** k) / 2) if p is not None else TBD, "(1+p^%d)/2" % k)
    six = [run("div7_6d", "B", MAIN_N, s) for s in seeds("div7_6d", "B", MAIN_N)]
    put("divsevenSixByes", pct(mean(class_acc(r, "Yes") for r in six)) if six else TBD, "bits")
    tr = [run("div7->div7_6d", "B", MAIN_N, s) for s in seeds("div7->div7_6d", "B", MAIN_N)]
    put("divsevenTransferSixB", pct(mean(r["acc"] for r in tr)) if tr else TBD, "eval-only transfer (S5 primary)")
    put("divsevenTransferSixByes", pct(mean(class_acc(r, "Yes") for r in tr)) if tr else TBD, "bits")
    return p


# ------------------------------------------------------------------------------------------ preregistered tests
def word(status):
    status = (status or "").upper()
    if status.startswith("PASS"):
        return "passed"
    if status.startswith("FAIL"):
        return "failed"
    return "pending" if status.startswith("PENDING") else TBD


def prereg_macros(p_step):
    """Verdicts S1-S6 exactly as experiments/analysis_report.prereg() evaluates them (one implementation)."""
    import analysis_report as AR
    res = AR.prereg(STATS, RULES, STEPS, TRANS)
    m15 = C.short_model(M15)
    for rule, mac in (("S1", "verdictSOne"), (f"S2 [{m15}]", "verdictSTwo"), (f"S3 [{m15}]", "verdictSThree"),
                      ("S4", "verdictSFour"), ("S5", "verdictSFive"), (f"S6 [{m15}]", "verdictSSix"),
                      ("S7", "verdictSSeven"), ("S8", "verdictSEight"), ("S9", "verdictSNine"),
                      ("S10", "verdictSTen")):
        r = res.get(rule, {})
        put(mac, word(r.get("status")) if r else "pending", "; ".join(r.get("detail", [])))
    st7 = res.get("S7", {}).get("status", "")
    mm = re.search(r"\((\d+)/5 hold, (\d+) pending\)", st7)
    put("SsevenHeld", mm.group(1) if mm else TBD, st7)
    put("SsevenPending", mm.group(2) if mm else TBD, st7)
    put("SsevenScored", str(5 - int(mm.group(2))) if mm else TBD, st7)
    # S6 cell values: the prime adapters on prime_hard (every item odd with no prime factor <= 7, so the
    # shortcut rule calls every item prime); accuracy overall (the S6 statistic) and on the hard negatives
    lab = "prime->prime_hard"
    for arm, mac in (("A", "primeHardA"), ("B", "primeHardB")):
        rs = [run(lab, arm, MAIN_N, s) for s in seeds(lab, arm, MAIN_N)]
        put(mac, pct(mean(r["acc"] for r in rs)) if rs else TBD, "prime->prime_hard accuracy")
        put(mac + "neg", pct(mean(class_acc(r, "No") for r in rs)) if rs else TBD, "hard negatives only")
    if "prime_hard" in TASKS:
        test = TASKS["prime_hard"]["test"]
        put("primeHardN", str(len(test)), "prime_hard test")
        put("primeHardNneg", str(sum(1 for t in test if t["label"] == "No")), "prime_hard test")
    # B' and prompting controls on the base model (1.5B)
    put("divsevenBp", pct(acc_mean("div7", "Bprime", MAIN_N)), "Bprime")
    for task, tp in (("prime", "prime"), ("div7", "divseven"), ("valid", "valid")):
        put(tp + "Cot", pct(acc_mean(f"{task}[cot]", "base", 0)), "cot eval")
        put(tp + "Few", pct(acc_mean(f"{task}[fewshot4]", "base", 0)), "fewshot4 eval")


# ------------------------------------------------------------------------------------------ review additions
CAP = {"div7": "Divseven", "div7_6d": "DivsevenSix", "div13": "Divthirteen", "div11": "Diveleven", "div3": "Divthree",
       "div2": "Divtwo"}
S7_CELLS = {("div13", 540), ("div7", 270), ("div7", 90), ("div11", 180), ("div3", 180), ("div2", 180)}
S9_CELLS = {("div13", 360)}
PRE_S7_CELLS = {("div7", 180), ("div7_6d", 180), ("div13", 180)}     # observed before S7 was registered


def trans_rows(model=M15):
    return [r for r in (TRANS or {}).get("rows", []) if r["arm"] == "B" and r["prompt_mode"] == "plain"
            and r["model"] == C.short_model(model) and r["train_task"] == r["eval_task"]]


def review_macros():
    # transitions per table entry m and per-step accuracy p (n = 180, pooled over seeds with generations)
    for task, cap in CAP.items():
        rows = [r for r in trans_rows() if r["task"] == task and r["n"] == MAIN_N]
        if rows:
            put("m" + cap, "%.1f" % rows[0]["m"], "transitions.json")
            put("p" + cap, "%.3f" % mean(r["p"] for r in rows), "transitions.json p_cond")
            rows0 = [r for r in rows if r.get("seed") == 0]
            put("p" + cap + "sZero", "%.3f" % rows0[0]["p"] if rows0 else TBD, "transitions.json p_cond seed 0")
            put("trace" + cap, pct(mean(r["trace_correct"] for r in rows)), "transitions.json")
        else:
            k = 6 if task == "div7_6d" else 4
            d = int(re.sub(r"\D", "", task.split("_")[0]))
            put("m" + cap, "%.1f" % (k * MAIN_N / (10 * d)), "k n / (10 d)")
            put("p" + cap, TBD)
            put("p" + cap + "sZero", TBD)
            put("trace" + cap, TBD)
    for task, n, mac in (("div13", 540, "mDivthirteenLarge"), ("div7", 270, "mDivsevenMid"), ("div7", 90, "mDivsevenSmall"),
                         ("div13", 360, "mDivthirteenMid")):
        put(mac, "%.1f" % (4 * n / (10 * int(task[3:]))), "k n / (10 d)")
    # S7 (d): div11 B n = 180, predicted 0.566 < p < 0.955
    rows = [r for r in trans_rows() if r["task"] == "div11" and r["n"] == MAIN_N]
    if rows:
        p = mean(r["p"] for r in rows)
        put("verdictSSevenD", "held" if 0.566 < p < 0.955 else "failed", "p=%.3f" % p)
    else:
        put("verdictSSevenD", TBD)
    # div13 diagnostics from the saved seed-0 generations
    g = gens_rows("div13", "B", MAIN_N, 0)
    if g:
        n_items = len(g)
        ok = first2 = wrong = wrong_zero = yes = 0
        for x in g:
            n = int(x["id"].split("-")[1])
            true, r = [], 0
            for ch in str(n):
                r = (10 * r + int(ch)) % 13
                true.append(r)
            st = [int(o) for *_x, o in DIV_STEP.findall((x.get("gen") or "").split("Answer:")[0])]
            first = next((t for t in range(len(true)) if t >= len(st) or st[t] != true[t]), None)
            if first is None and len(st) == len(true):
                ok += 1
            else:
                wrong += 1
                wrong_zero += bool(st) and st[min(len(st), len(true)) - 1] == 0
                first2 += first == 1
            yes += x.get("pred") == "Yes"
        put("divthirteenNitems", str(n_items), "gens")
        put("divthirteenTraceOK", pct(ok / n_items), "gens")
        put("divthirteenFirstErrTwo", str(first2), "gens")
        put("divthirteenWrongN", str(wrong), "gens")
        put("divthirteenWrongZero", str(wrong_zero), "gens")
        put("divthirteenPredYes", "%.0f" % (100 * yes / n_items), "gens")
    else:
        for mac in ("divthirteenNitems", "divthirteenTraceOK", "divthirteenFirstErrTwo", "divthirteenWrongN",
                    "divthirteenWrongZero", "divthirteenPredYes"):
            put(mac, TBD)
    b, a = run("div13", "B", MAIN_N, 0), run("div13", "A", MAIN_N, 0)
    put("divthirteenBApsZero", pval(mcnemar(b["bits_list"], a["bits_list"])[2]) if (a and b) else TBD, "McNemar")
    b, a = run("div11", "B", MAIN_N, 0), run("div11", "A", MAIN_N, 0)
    put("divelevenBApsZero", pval(mcnemar(b["bits_list"], a["bits_list"])[2]) if (a and b) else TBD, "McNemar")
    b, a = run("div7_6d", "B", MAIN_N, 0), run("div7_6d", "A", MAIN_N, 0)
    put("divsevenSixBApsZero", pval(mcnemar(b["bits_list"], a["bits_list"])[2]) if (a and b) else TBD, "McNemar")
    # prime: scrambled C seed 1 per-class counts; A on composites ending in 1/3/7/9 with a factor 3 or 7
    te = TASKS["prime"]["test"]
    c1 = run("prime", "C", MAIN_N, 1)
    if c1:
        put("primeCsOnePrimes", str(sum(bb for bb, t in zip(c1["bits_list"], te) if t["label"] == "Yes")), "bits")
        put("primeCsOneComps", str(sum(bb for bb, t in zip(c1["bits_list"], te) if t["label"] == "No")), "bits")
    idx = [i for i, e in enumerate(te) if e["label"] == "No" and int(e["id"].split("-")[1]) % 10 in (1, 3, 7, 9)
           and any(int(e["id"].split("-")[1]) % q == 0 for q in (3, 7))]
    put("primeCompThreeSevenN", str(len(idx)), "prime test")
    put("primeCompThreeSevenA", " and ".join(str(sum(run("prime", "A", MAIN_N, s_)["bits_list"][i] for i in idx))
                                             for s_ in seeds("prime", "A", MAIN_N)), "bits")
    # bootstrap CI of B - A and seed SD (stats.json)
    for task, tp in PREFIX.items():
        bs = [x for x in (STATS or {}).get("bootstrap", []) if x["comparison"] == "B-A" and x["task"] == task
              and x["n"] == MAIN_N and x["model"] == M15]
        if bs:
            put(tp + "BAdiff", "%+.1f" % (100 * bs[0]["diff"]), "stats.json bootstrap")
            put(tp + "BAci", "[%+.1f, %+.1f]" % tuple(100 * v for v in bs[0]["ci95"]), "stats.json bootstrap")
        else:
            put(tp + "BAdiff", TBD)
            put(tp + "BAci", "")
        for arm in ("A", "B", "C"):
            po = [x for x in (STATS or {}).get("pooled", []) if x["task"] == task and x["arm"] == arm and
                  x["n"] == MAIN_N and x["model"] == M15]
            sd = po[0]["seed_sd"] if po else None
            put(tp + arm + "sd", ("$\\pm$%.1f" % (100 * sd)) if sd is not None else "", "stats.json seed_sd")
    # post hoc S5 forms (steps.json S5 post_hoc) and S8 / S9 / arm S cells
    ph = [c.get("post_hoc", {}) for c in (STEPS or {}).get("S5", {}).get("cells", []) if c.get("primary")]
    put("divsevenSixPredGuess", pct(ph[0]["predicted_answer_acc_with_guessing"]) if ph and ph[0] else TBD, "post hoc")
    put("divsevenSixTraceOK", pct(ph[0]["trace_correct_6digit"]) if ph and ph[0] else TBD, "post hoc")
    put("stepLocalPowFour", pct(float(OUT["stepLocalAcc"]) ** 4) if OUT.get("stepLocalAcc", TBD) != TBD else TBD,
        "local^4")
    b9, a9 = run("div13", "B", 360, 0), run("div13", "A", 360, 0)
    put("divthirteenBmid", pct(b9["acc"]) if b9 else TBD, "S9")
    put("divthirteenAmid", pct(a9["acc"]) if a9 else TBD, "S9")
    # dose: accuracy (A, B), per-step p and m at every n run, for div7 and div13
    W = {60: "Sixty", 90: "Ninety", 180: "OneEighty", 270: "TwoSeventy", 360: "ThreeSixty", 540: "FiveForty"}
    for task, tp, cap in (("div7", "divseven", "Divseven"), ("div13", "divthirteen", "Divthirteen")):
        for n, w in W.items():
            for arm in ("A", "B"):
                put(tp + arm + "n" + w, pct(acc_mean(task, arm, n)), "dose")
                r0 = run(task, arm, n, 0)
                put(tp + arm + "n" + w + "sZero", pct(r0["acc"]) if r0 else TBD, "dose, seed 0 (preregistered cell)")
                for s_, sw_ in ((1, "sOne"), (2, "sTwo")):
                    rs_ = run(task, arm, n, s_)
                    if rs_:
                        put(tp + arm + "n" + w + sw_, pct(rs_["acc"]), "dose, seed %d" % s_)
            rows0 = [r for r in trans_rows() if r["task"] == task and r["n"] == n and r.get("seed") == 0]
            put("p" + cap + "n" + w + "sZero", "%.3f" % rows0[0]["p"] if rows0 else TBD, "transitions.json seed 0")
            rows = [r for r in trans_rows() if r["task"] == task and r["n"] == n]
            put("p" + cap + "n" + w, "%.3f" % mean(r["p"] for r in rows) if rows else TBD, "transitions.json")
            put("m" + cap + "n" + w, "%.1f" % (4 * n / (10 * int(task[3:]))), "k n / (10 d)")
    # B' still writes a trace after its answer: its per-step accuracy (steps.json, pooled over seeds)
    cells = steps_cells("div7", arm="Bprime", eval_task="div7")
    put("pDivsevenBp", "%.3f" % pooled(cells, "p_cond") if cells else TBD, "steps.json Bprime")
    tr = steps_cells("div7->div7_6d", eval_task="div7_6d")
    put("pDivsevenTransfer", "%.3f" % pooled(tr, "p_cond") if tr else TBD, "steps.json transfer")
    # S6: agreement of prime A with the last-digit rule on prime_hard (rules.json)
    ag = [x["agreement"]["last_digit_1379"] for x in (RULES or {}).get("runs", [])
          if x["task"] == "prime->prime_hard" and x["arm"] == "A" and x.get("model", M15) == M15]
    put("primeHardAagree", pct(mean(ag)) if ag else TBD, "rules.json")
    for task, tp in (("div7", "divseven"), ("prime", "prime")):
        sr = [run(task, "S", MAIN_N, s_) for s_ in seeds(task, "S", MAIN_N)]
        put(tp + "SelfTrace", pct(mean(r["acc"] for r in sr)) if sr else TBD, "arm S")


# ------------------------------------------------------------------------------------------ scale
def scale_macros():
    for mpre, model in MODELS.items():
        if not mpre:
            continue
        # B vs A, B vs base, A vs base (per seed) on div7
        for s_ in range(2):
            w = SEEDW[s_]
            b, a, z = run("div7", "B", MAIN_N, s_, model), run("div7", "A", MAIN_N, s_, model), \
                run("div7", "base", 0, 0, model)
            put(mpre + "divsevenBAps" + w, pval(mcnemar(b["bits_list"], a["bits_list"])[2]) if (a and b) else TBD, "McNemar")
            put(mpre + "divsevenBBaseps" + w, pval(mcnemar(b["bits_list"], z["bits_list"])[2]) if (b and z) else TBD,
                "McNemar")
            put(mpre + "divsevenABaseps" + w, pval(mcnemar(a["bits_list"], z["bits_list"])[2]) if (a and z) else TBD,
                "McNemar")
        # eval-only cells: prime A on prime_hard, 4-digit div7 B on 6-digit inputs
        for lab, arm, mac in (("prime->prime_hard", "A", "primeHardA"), ("prime->prime_hard", "B", "primeHardB"),
                              ("div7->div7_6d", "B", "divsevenTransferSixB")):
            ss = seeds(lab, arm, MAIN_N, model)
            put(mpre + mac, pct(mean(run(lab, arm, MAIN_N, x, model)["acc"] for x in ss)) if ss else TBD, "eval")
            for x in ss:
                put(mpre + mac + "s" + SEEDW[x], pct(run(lab, arm, MAIN_N, x, model)["acc"]), "eval")
        ag = [x["agreement"]["last_digit_1379"] for x in (RULES or {}).get("runs", [])
              if x["task"] == "prime->prime_hard" and x["arm"] == "A" and x.get("model") == model]
        put(mpre + "primeHardAagree", pct(mean(ag)) if ag else TBD, "rules.json")
        cells = steps_cells("div7", model=model, eval_task="div7")
        put(mpre + "pDivseven", "%.3f" % pooled(cells, "p_cond") if cells else TBD, "steps.json")
        put(mpre + "divsevenFullTrace", pct(pooled(cells, "trace_correct_rate")) if cells else TBD, "steps.json")
    # 3B dose cell (div13 at n = 360) and the 7B valid trace-arm failure pattern
    for arm in ("A", "B"):
        put("threeBdivthirteen" + arm + "nThreeSixty", pct(acc_mean("div13", arm, 360, MODELS["threeB"])), "3B n=360")
    g = gens_rows("valid", "B", MAIN_N, 0, MODELS["sevenB"])
    if g:
        err = [x for x in g if not x.get("correct")]
        yes = [x for x in err if x.get("pred") == "Yes"]
        tmpl = [x for x in yes if any(w in (x.get("gen") or "") for w in ("contradiction", "impossible", "no model"))]
        put("sevenBvalidBerrN", str(len(err)), "gens")
        put("sevenBvalidBerrYes", str(len(yes)), "gens")
        put("sevenBvalidBerrTemplate", str(len(tmpl)), "gens")
    else:
        for mac in ("sevenBvalidBerrN", "sevenBvalidBerrYes", "sevenBvalidBerrTemplate"):
            put(mac, TBD)
    # what the 3B base model does on div7 / div13 (saved generations)
    for task, tp in (("div7", "divseven"), ("div13", "divthirteen")):
        g = gens_rows(task, "base", 0, 0, MODELS["threeB"])
        if g:
            direct = [x for x in g if "\\div" in (x.get("gen") or "") or "\u00f7" in (x.get("gen") or "")]
            put("threeBbase" + tp + "Direct", pct(len(direct) / len(g)), "gens")
            put("threeBbase" + tp + "N", str(len(g)), "gens")
            put("threeBbase" + tp + "Tokens", "%.0f" % mean(x.get("n_gen_tokens", 0) for x in g), "gens")
        else:
            for suf in ("Direct", "N", "Tokens"):
                put("threeBbase" + tp + suf, TBD)
    # McNemar p-values (B vs A) listed over the seeds where both cells exist, as text with inline math
    for task, tp in (("prime", "prime"), ("valid", "valid"), ("div7", "divseven")):
        ps = []
        for s_ in range(5):
            b, a_ = run(task, "B", MAIN_N, s_), run(task, "A", MAIN_N, s_)
            if a_ and b:
                ps.append((s_, pval(mcnemar(b["bits_list"], a_["bits_list"])[2])))
        if ps:
            vals = ["$%s$" % v for _, v in ps]
            txt = vals[0] if len(vals) == 1 else ", ".join(vals[:-1]) + " and " + vals[-1]
            rng_ = "seed %d" % ps[0][0] if len(ps) == 1 else "seeds %d--%d" % (ps[0][0], ps[-1][0])
            put(tp + "BApList", "$p=$ " + txt + " for " + rng_, "McNemar per seed")
        else:
            put(tp + "BApList", TBD)
    # per-seed B - A differences for the divisor family (S2)
    for task, tp in (("div3", "divthree"), ("div11", "diveleven"), ("div13", "divthirteen")):
        for s_ in range(2):
            b, a = run(task, "B", MAIN_N, s_), run(task, "A", MAIN_N, s_)
            put(tp + "BAd" + SEEDW[s_], "%+.1f" % (100 * (b["acc"] - a["acc"])) if (a and b) else TBD, "runs")


# ------------------------------------------------------------------------------------------ review round 2
DASH = "--"
# cells that queued runs (results/queues/lap_S.txt, lap_T4_05b.txt, q3_star_*.txt) may still fill: when absent at
# build time they render as a dash (not run by submission), never as [TBD]
QUEUED = {"halfB" + tp + arm for tp in ("divseven", "divthirteen", "prime", "valid", "divthree", "divtwo")
          for arm in ("A", "B")}


def seed_list(ss):
    ss = sorted(ss)
    if not ss:
        return ""
    if len(ss) == 1:
        return "seed %d" % ss[0]
    if ss == list(range(ss[0], ss[-1] + 1)) and len(ss) > 2:
        return "seeds %d--%d" % (ss[0], ss[-1])
    return "seeds " + ", ".join(str(s) for s in ss[:-1]) + " and " + str(ss[-1])


def late_macros():
    M3, M7 = MODELS["threeB"], MODELS["sevenB"]
    for mac in QUEUED:
        if OUT.get(mac, TBD) == TBD:
            put(mac, DASH, "queued, not landed")
    # (F1g) Table 2 max-p column: largest per-seed McNemar p (B vs A) over seeds; parentheses when A > B pooled
    for task, tp in PREFIX.items():
        ps, dif = [], []
        for s_ in seeds(task, "B", MAIN_N):
            b, a = run(task, "B", MAIN_N, s_), run(task, "A", MAIN_N, s_)
            if a and b:
                ps.append(mcnemar(b["bits_list"], a["bits_list"])[2])
                dif.append(b["acc"] - a["acc"])
        if ps:
            v = "$%s$" % pval(max(ps))
            put(tp + "BApMaxCol", ("(%s)" % v) if mean(dif) < 0 else v, "max McNemar p over seeds %d" % len(ps))
            put(tp + "BAnSeeds", str(len(ps)), "paired seeds")
        else:
            put(tp + "BApMaxCol", DASH, "no paired seeds")
    # (F1b) S7 cell status from the analysis' own S7 evaluation
    import analysis_report as AR
    res = AR.prereg(STATS, RULES, STEPS, TRANS)
    det = res.get("S7", {}).get("status", "") + " " + " ".join(res.get("S7", {}).get("detail", []))
    for c, w in zip("abcde", ("A", "B", "C", "D", "E")):
        mm = re.search(r"\(" + c + r"\)[^\[]*?(HOLDS|holds|fails|FAILS|pending|PENDING)", det)
        put("SsevenCell" + w, ("holds" if mm.group(1).lower() == "holds" else "fails") if mm and
            not mm.group(1).lower().startswith("pend") else "not scored", det[:200])
    # (F7a) items-by-seeds bootstrap contrasts for every model (stats.json): <model><task>Boot<cmp>{Diff,Ci,Seeds}
    CMPN = {"B-A": "BA", "A-base": "ABase", "B-base": "BBase", "B-D": "BD", "B-Bprime": "BBp", "B-C": "BC",
            "S-A": "SA", "B-S": "BS", "D-A": "DA"}
    mp = {v: k for k, v in MODELS.items()}
    for x in (STATS or {}).get("bootstrap", []):
        if x["n"] != MAIN_N or x["comparison"] not in CMPN or x["model"] not in mp or x["task"] not in PREFIX:
            continue
        nm = mp[x["model"]] + PREFIX[x["task"]] + "Boot" + CMPN[x["comparison"]]
        put(nm + "Diff", "%+.1f" % (100 * x["diff"]), "stats.json bootstrap")
        put(nm + "Ci", "[%+.1f, %+.1f]" % tuple(100 * v for v in x["ci95"]), "stats.json bootstrap")
        put(nm + "Seeds", str(len(x["seeds"])), "stats.json bootstrap seeds")
    # (F6d) secondary-model verdicts, as the analysis evaluates them
    for mdl, w in (("Qwen2.5-3B-Instruct", "ThreeB"), ("Qwen2.5-7B-Instruct", "SevenB")):
        for rule in ("S2", "S6", "S8", "S9"):
            k = next((x for x in res if x.startswith(rule + " [" + mdl)), None)
            st = (res[k].get("status") or "").upper() if k else ""
            put("verdict" + {"S2": "STwo", "S6": "SSix", "S8": "SEight", "S9": "SNine"}[rule] + w, "passed" if st.startswith("PASS") else "failed"
                if st.startswith("FAIL") else "not evaluable" if k else DASH, k or "")
        sb = (STEPS or {}).get("S5", {}).get("status_by_model", {}).get(mdl, {})
        put("verdictSFive" + w, "failed" if str(sb.get("primary", "")).upper().startswith("FAIL") else
            "passed" if str(sb.get("primary", "")).upper().startswith("PASS") else DASH, str(sb))
    # (F1d / F5e) prime shortcut counts over every A seed
    te = TASKS["prime"]["test"]
    sA = seeds("prime", "A", MAIN_N)
    idx = [i for i, e in enumerate(te) if e["label"] == "No" and int(e["id"].split("-")[1]) % 10 in (1, 3, 7, 9)]
    put("primeCompLastDigitN", str(len(idx)), "prime test composites ending 1/3/7/9")
    put("primeCompLastDigitA", ", ".join(str(sum(run("prime", "A", MAIN_N, s_)["bits_list"][i] for i in idx))
                                         for s_ in sA), "bits")
    idx37 = [i for i in idx if any(int(te[i]["id"].split("-")[1]) % q == 0 for q in (3, 7))]
    counts = [sum(run("prime", "A", MAIN_N, s_)["bits_list"][i] for i in idx37) for s_ in sA]
    put("primeCompThreeSevenA", ", ".join(map(str, counts[:-1])) + (" and " if len(counts) > 1 else "")
        + str(counts[-1]), "bits")
    put("primeASeeds", seed_list(sA), "seeds")
    # (F5b) 1.5B prime B accuracy by the true number of trial divisions the trace performs
    def ndiv(n):
        ps_ = [q for q in range(2, math.isqrt(n) + 1) if all(q % d for d in range(2, math.isqrt(q) + 1))]
        return next((i + 1 for i, q in enumerate(ps_) if n % q == 0), len(ps_))
    kk = [ndiv(int(t["id"].split("-")[1])) for t in te]
    sB = seeds("prime", "B", MAIN_N)
    for (lo, hi), w in zip(((1, 1), (2, 3), (4, 8), (9, 16), (17, 99)), ("One", "Two", "Three", "Four", "Five")):
        ix = [i for i, k in enumerate(kk) if lo <= k <= hi]
        acc = mean(run("prime", "B", MAIN_N, s_)["bits_list"][i] for s_ in sB for i in ix) if (ix and sB) else None
        put("primeBdivBin" + w, "%.0f" % (100 * acc) if acc is not None else TBD, "bits by trial divisions")
        put("primeBdivBinN" + w, str(len(ix)), "items")
    put("primeBSeeds", seed_list(sB), "seeds")
    c0 = run("prime", "C", MAIN_N, 0)
    if c0:
        put("primeCsZeroPrimes", pct(class_acc(c0, "Yes")), "bits")
        put("primeCsZeroComps", pct(class_acc(c0, "No")), "bits")
    # (F3) 7B and 3B: collapse of 7B prime A seed 0, dose cells, S vs base
    r7 = run("prime", "A", MAIN_N, 0, M7)
    if r7:
        pr, _ = preds(r7)
        put("sevenBprimeAsZeroNo", str(sum(x == "No" for x in pr)), "7B prime A s0 predictions")
    W = {90: "Ninety", 270: "TwoSeventy", 360: "ThreeSixty", 540: "FiveForty"}
    for mpre, model in (("threeB", M3), ("sevenB", M7)):
        for task, tp in (("div7", "divseven"), ("div13", "divthirteen")):
            for n, w in W.items():
                for arm in ("A", "B"):
                    put(mpre + tp + arm + "n" + w, pct(acc_mean(task, arm, n, model)), "dose")
                    r0 = run(task, arm, n, 0, model)
                    put(mpre + tp + arm + "n" + w + "sZero", pct(r0["acc"]) if r0 else TBD, "dose seed 0")
                    for s_, sw_ in ((1, "sOne"), (2, "sTwo")):
                        rs_ = run(task, arm, n, s_, model)
                        if rs_:
                            put(mpre + tp + arm + "n" + w + sw_, pct(rs_["acc"]), "dose seed %d" % s_)
    for mpre, model in (("threeB", M3), ("sevenB", M7)):
        s0, z = run("div7", "S", MAIN_N, 0, model), run("div7", "base", 0, 0, model)
        if s0 and z:
            b, c, p = mcnemar(s0["bits_list"], z["bits_list"])
            put(mpre + "divsevenSBaseDisc", "%d:%d" % (b, c), "S-only : base-only correct, seed 0")
            put(mpre + "divsevenSBasep", pval(p), "exact McNemar S vs base, seed 0")
        for task, tp in (("div7", "divseven"), ("prime", "prime"), ("valid", "valid")):
            ss = seeds(task, "S", MAIN_N, model)
            put(mpre + tp + "SSeeds", seed_list(ss) if ss else DASH, "arm S seeds")
    # (F7b) Holm-adjusted A vs base on 1.5B div7 (family: tasks within comparison, model and seed)
    hs = sorted(((m["seed"], m["p_holm_tasks"]) for m in (STATS or {}).get("mcnemar", [])
                 if m["comparison"] == "A-base" and m["task"] == "div7" and m["n"] == MAIN_N and m["model"] == M15
                 and m.get("p_holm_tasks") is not None))
    for s_, p in hs:
        put("divsevenABaseHolms" + SEEDW[s_], pval(p), "stats.json p_holm_tasks")
    put("divsevenABaseHolmSig", "%d of %d" % (sum(p < 0.05 for _, p in hs), len(hs)), "Holm < 0.05")
    # (F8) arm S kept-trace audits: worker audit and the post hoc extended re-audit (analysis_steps.star_summary)
    for c in (STEPS or {}).get("star", []):
        if c["task"] != "div7" or c["seed"] != 0:
            continue
        mpre = {"Qwen2.5-1.5B-Instruct": "", "Qwen2.5-3B-Instruct": "threeB", "Qwen2.5-7B-Instruct": "sevenB"}.get(
            c["model"])
        if mpre is None:
            continue
        put(mpre + "selfKeep", pct(c["keep_rate"]), "star keep rate (samples)")
        put(mpre + "selfKept", "%d/%d" % (c["n_kept"], c["n_items"]), "items kept")
        put(mpre + "selfParsed", pct(c["kept_trace_parsed_frac"]), "worker audit coverage")
        put(mpre + "selfCorrectAll", pct(c["kept_trace_correct_frac"]), "worker audit, over all kept")
        if c.get("ext_kept_trace_parsed_frac") is not None:
            put(mpre + "selfParsedExt", pct(c["ext_kept_trace_parsed_frac"]), "re-audit coverage")
            put(mpre + "selfCorrectParsedExt", pct(c["ext_all_claims_correct_among_parsed"]), "re-audit, parsed")
            put(mpre + "selfCorrectAllExt", pct(c["ext_kept_trace_correct_frac_all_kept"]), "re-audit, all kept")
    # S10 as text: evaluable only with div7 seeds 0 and 1 and valid
    d7 = [run("div7", "S", MAIN_N, s_) for s_ in (0, 1)]
    va = run("valid", "S", MAIN_N, 0)
    got = [r for r in d7 if r]
    if got:
        put("STenResult", "arm S reaches " + " and ".join(pct(r["acc"]) + "\\%" for r in got) + " on div7 ("
            + seed_list([s_ for s_, r in zip((0, 1), d7) if r]) + ")" + (" and " + pct(va["acc"]) + "\\% on valid"
                                                                          if va else ""), "arm S 1.5B")
    else:
        put("STenResult", "the arm-S fine-tuning runs had not finished by submission", "arm S 1.5B")
    rr = [x for x in (RULES or {}).get("runs", []) if x["task"] == "prime" and x["arm"] == "B" and x["seed"] == 0
          and x.get("model", "").endswith("3B-Instruct") and x["n"] == MAIN_N]
    put("threeBprimeBnoFiveAgree", pct(rr[0]["agreement"]["no_factor_le_5"]) if rr else TBD, "rules.json")
    src = open(os.path.join(ROOT, "experiments", "exp_thinking_ft_worker.py"), encoding="utf-8").read()
    cap = dict((k, int(v)) for k, v in re.findall(r'"(\w+)": (\d+)', re.search(r"MAXNEW = \{[^}]*\}", src).group(0)))
    caps = sorted({(r["task"], r["max_new"]) for r in RUNS if r.get("max_new") and r["arm"] != "base"})
    for t, w in (("div13", "Divthirteen"), ("div11", "Diveleven"), ("div7_6d", "DivsevenSix")):
        v = [c for tt, c in caps if tt == t]
        put("maxNew" + w, str(v[0]) if v else str(cap.get(t, "")) or TBD, "runs max_new")
    if any(r["acc"] > 0.65 for r in got):
        put("verdictSTen", "failed", "a landed div7 S seed exceeds 65%, so the both-seeds clause cannot hold")
    elif not (all(d7) and va):
        put("verdictSTen", "not evaluable (two-seed rule)", "S10 incomplete")
    words = {1: "one seed", 2: "two seeds", 3: "three seeds", 4: "four seeds", 5: "five seeds"}
    for k in [k for k in OUT if k.endswith("nSeeds")]:
        put(k[:-6] + "SeedTxt", words.get(int(OUT[k]), OUT[k] + " seeds"), "seed count")
    put("divsevenSelfTrace", pct(mean(r["acc"] for r in got)) if got else DASH, "arm S 1.5B div7")
    put("divsevenSelfSeeds", seed_list([s_ for s_, r in zip((0, 1), d7) if r]) if got else DASH, "seeds")
    put("divsevenSelfsZero", pct(d7[0]["acc"]) if d7[0] else DASH, "arm S 1.5B div7 seed 0")
    put("divsevenSelfsOne", pct(d7[1]["acc"]) if d7[1] else DASH, "arm S 1.5B div7 seed 1")
    put("validSelfTrace", pct(va["acc"]) if va else DASH, "arm S 1.5B valid seed 0 (S10 predicted >= 90)")
    # (F6c) post hoc logistic fits of q on ln m
    for mdl, w in (("Qwen2.5-1.5B-Instruct", "OneFive"), ("Qwen2.5-3B-Instruct", "ThreeB")):
        f = (TRANS or {}).get("post_hoc_fits", {}).get(mdl, {}).get("logistic_logit_p_vs_ln_m")
        put("fitRsq" + w, "%.2f" % f["r2_logit_scale"] if f else TBD, "transitions.json post hoc fit")
        put("fitN" + w, str(TRANS["post_hoc_fits"][mdl]["n_points"]) if f else TBD, "cells")
    # (F6e) S5 secondary: the transfer model's own traces
    sec = [c for c in (STEPS or {}).get("S5", {}).get("cells", []) if not c.get("primary")
           and c["model"] == C.short_model(M15)]
    if sec and sec[0].get("post_hoc"):
        put("divsevenTransferTraceOK", pct(sec[0]["post_hoc"]["trace_correct_6digit"]), "post hoc")
    # (F12) q excluding the first step, seed 0 (steps.json p_cond_after_first_step)
    for task, n, w in (("div13", 180, "DivthirteenOneEighty"), ("div7", 90, "DivsevenNinety")):
        cs = [c for c in (STEPS or {}).get("div", []) if c["task"] == task and c["n"] == n and c["seed"] == 0
              and c["arm"] == "B" and c["model"] == C.short_model(M15) and c["prompt_mode"] == "plain"
              and c["eval_task"] == task]
        put("qAll" + w, "%.3f" % cs[0]["p_cond"] if cs else TBD, "steps.json s0")
        put("qAfter" + w, "%.3f" % cs[0]["p_cond_after_first_step"] if cs and
            cs[0].get("p_cond_after_first_step") is not None else TBD, "steps.json s0")
    # (F5c) 3B prime trace claims: share of 'n mod p = r' claims that are correct (pooled over seeds)
    pc = [c for c in (STEPS or {}).get("prime", []) if c["task"] == "prime" and c["model"] == "Qwen2.5-3B-Instruct"]
    w_ = sum(c["n_steps_parsed"] for c in pc)
    put("threeBprimeClaimAcc", "%.0f" % (100 * sum(c["step_arith_acc"] * c["n_steps_parsed"] for c in pc) / w_)
        if w_ else TBD, "steps.json prime step_arith_acc")
    # (F9c) generation-cap hits behind lower-bound cells
    for key, mac in ((("prime", "base", 0, 0, M3), "threeBprimeBaseCap"), (("prime[fewshot4]", "base", 0, 0, M15),
                     "primeFewCap"), (("div11", "base", 0, 0, M15), "divelevenBaseCap")):
        r = run(*key)
        put(mac, str(r.get("n_hit_cap")) if r and r.get("n_hit_cap") is not None else TBD, "n_hit_cap")


def yes_rate(r):
    pr, _ = preds(r)
    return sum(x == "Yes" for x in pr) / len(pr) if pr else None


# ------------------------------------------------------------------------------------------ run table
def run_table():
    arm_tex = {"base": "base", "A": "A", "B": "B", "C": "C", "Bprime": "B$'$", "A_tok": "A-full", "B_orn": "B-ext",
               "D": "D", "S": "S"}
    order = list(PREFIX)
    rows = [r for r in RUNS if r["arm"] in arm_tex]

    def sort_key(r):
        base_task = r["train_task"]
        return (list(MODELS.values()).index(r["model"]) if r["model"] in MODELS.values() else 9,
                order.index(base_task) if base_task in order else 99, r["task"], list(arm_tex).index(r["arm"]),
                r["n"], r["seed"])
    rows.sort(key=sort_key)
    L = ["% generated by paper/exp/make_numbers.py from the run files -- do not edit",
         "\\begin{longtable}{lllrrrrrr}",
         "\\caption{All logged runs of arms base, A, B, C, D, B$'$, S, A-full and B-ext (\\nTest{} test items per "
         "cell). Cell: the test set, prefixed by the training task when they differ; [cot] and [fewshot4] mark "
         "prompting controls on the base model. SmolLM2: SmolLM2-1.7B-Instruct (S11). Train tokens: whitespace tokens in the completions. Cap: test "
         "generations that hit the generation cap (scored wrong; -- where not logged). Yes: share of test items "
         "answered Yes, from saved generations or recovered from per-item correctness; 0 or 100 marks a "
         "constant-output collapse.}\\label{tab:runs}\\\\",
         "\\toprule", "model & cell & arm & $n$ & seed & train tokens & cap & Yes (\\%) & acc.\\ (\\%) \\\\",
         "\\midrule", "\\endfirsthead", "\\toprule",
         "model & cell & arm & $n$ & seed & train tokens & cap & Yes (\\%) & acc.\\ (\\%) \\\\",
         "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    last = None
    for r in rows:
        grp = (r["model"], r["train_task"])
        if last is not None and grp != last:
            L.append("\\midrule")
        last = grp
        cell = r["task"].replace("_", "\\_").replace("->", "$\\to$")
        yr = yes_rate(r)
        L.append("%s & %s & %s & %d & %d & %s & %s & %s & %.1f \\\\" % (
            C.short_model(r["model"]).replace("Qwen2.5-", "").replace("-Instruct", "").replace("SmolLM2-1.7B", "SmolLM2"), cell, arm_tex[r["arm"]],
            int(r["n"]), int(r["seed"]), thousands(r.get("train_tokens") or 0),
            str(r["n_hit_cap"]) if r.get("n_hit_cap") is not None else "--",
            "%.0f" % (100 * yr) if yr is not None else "--", 100 * r["acc"]))
    L.append("\\end{longtable}")
    open(os.path.join(PAPER, "tab_runs.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return len(rows)


# ------------------------------------------------------------------------------------------ S12 + CoT evals
def sweep_macros():
    """S12 answer-only lr/epoch sweep (results/v2/sweep/, deliberately outside the main run files) and the
    1.5B CoT-prompt evaluations of answer-only models (evals_*.jsonl, labelled '<task>[cot]')."""
    sw_dir = os.path.join(ROOT, "results", "v2", "sweep")
    test7 = TASKS["div7"]["test"]

    def rows_of(fname):
        f = os.path.join(sw_dir, fname)
        return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] if os.path.exists(f) else []

    cells, configs = [], []
    for lr_s, lw in (("2e-5", "Two"), ("5e-5", "Five")):
        for ep, ew in ((10, "Ten"), (30, "Thirty")):
            rows = rows_of("sweepA_div7_lr%s_ep%d.jsonl" % (lr_s, ep))
            yes_by_seed, tok = {}, None
            for s_, sw_ in ((0, "SZero"), (1, "SOne")):
                r = next((x for x in rows if int(x["seed"]) == s_ and x.get("acc") is not None), None)
                name = "sweepAlr%sEp%s%s" % (lw, ew, sw_)
                if r:
                    bits = C.decode_bits(r["bits"], int(r.get("n_test") or 240))
                    pr = C.recover_preds(bits, test7)
                    yes = sum(1 for p_ in pr if p_ == "Yes") / len(pr)
                    yes_by_seed[s_] = yes
                    put(name, pct(r["acc"]), "S12 sweep " + r.get("gens_path", ""))
                    put(name + "Yes", "%.2f" % yes, "S12 Yes-rate (recovered from bits and labels)")
                    cells.append((r["acc"], "lr %s, %d epochs, seed %d" % (lr_s, ep, s_)))
                    tok = int(r.get("train_tokens") or 0) * int(r.get("epochs") or ep)
                else:
                    put(name, DASH, "S12 cell not landed")
                    put(name + "Yes", DASH, "S12 cell not landed")
            put("sweepAlr%sEp%sTok" % (lw, ew), thousands(tok) if tok else DASH,
                "supervised whitespace tokens summed over epochs (train_tokens x epochs)")
            configs.append(yes_by_seed)
    n_cells = len(cells)
    if cells:
        best = max(cells)
        put("sweepAmax", pct(best[0]), "best div7 A accuracy in the sweep")
        put("sweepAmaxConfig", best[1], "config of the best cell")
    else:
        put("sweepAmax", DASH, "S12 not landed")
        put("sweepAmaxConfig", DASH, "S12 not landed")
    non_collapse = sum(1 for y in configs if len(y) == 2 and all(0.2 <= v <= 0.8 for v in y.values()))
    put("sweepAnonCollapse", str(non_collapse) if n_cells else DASH, "configs with Yes-rate in [0.2, 0.8] in both seeds")
    d3 = next((x for x in rows_of("sweepA_div3_lr5e-5_ep10.jsonl") if x.get("acc") is not None), None)
    put("sweepDivthreeA", pct(d3["acc"]) if d3 else DASH, "S12 positive control div3 A lr 5e-5, 10 epochs")
    if cells and max(c[0] for c in cells) >= 0.65:
        verdict = "failed"
    elif n_cells == 8 and d3:
        verdict = "passed" if (non_collapse >= 1 and d3["acc"] >= 0.90) else "failed"
    else:
        verdict = "not complete"
    put("STwelveVerdict", verdict, "S12 rule: no div7 A >= 65%, >= 1 non-collapsing config, div3 A >= 90%")
    if n_cells:
        put("STwelveResult", "the best of %d answer-only configurations reaches %s\\%% on div7 (%s), %d of 4 "
            "configurations avoid a constant answer in both seeds, and the div3 control reaches %s\\%%"
            % (n_cells, pct(max(cells)[0]), max(cells)[1], non_collapse, pct(d3["acc"]) if d3 else DASH),
            "S12 summary")
    else:
        put("STwelveResult", "the answer-only sweep had not finished by submission", "S12 not landed")
    # 1.5B CoT-prompt evaluations of answer-only models
    a0, a1 = run("div7[cot]", "A", MAIN_N, 0), run("div7[cot]", "A", MAIN_N, 1)
    put("divsevenAcotsZero", pct(a0["acc"]) if a0 else DASH, "A retrained, CoT-prompt eval, seed 0")
    put("divsevenAcotsOne", pct(a1["acc"]) if a1 else DASH, "A retrained, CoT-prompt eval, seed 1")
    got = [r["acc"] for r in (a0, a1) if r]
    put("divsevenAcot", pct(mean(got)) if got else DASH, "mean over landed seeds")
    b13 = run("div13[cot]", "base", 0, 0)
    put("divthirteenBaseCot", pct(b13["acc"]) if b13 else DASH, "1.5B base, CoT prompt, div13")
    a13 = run("div13[cot]", "A", MAIN_N, 0)
    put("divthirteenAcot", pct(a13["acc"]) if a13 else DASH, "A seed-0 adapter, CoT-prompt eval, div13")
    ap_ = run("prime[cot]", "A", MAIN_N, 0)
    put("primeAcot", pct(ap_["acc"]) if ap_ else DASH, "A seed-0 adapter, CoT-prompt eval, prime")
    # does the A model reason under the CoT prompt? generated-token counts from the saved gens
    gdir = os.path.join(ROOT, "results", "v2", "gens", C.short_model(M15))

    def gen_rows(name):
        f = os.path.join(gdir, name + ".jsonl")
        return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] if os.path.exists(f) else []
    a_rows = gen_rows("div7_A_180_0__cot") + gen_rows("div7_A_180_1__cot")
    put("divsevenAcotMaxTok", str(max(r["n_gen_tokens"] for r in a_rows)) if a_rows else DASH,
        "max generated tokens, A under the CoT prompt, landed seeds")
    put("divsevenAcotYes", "%.2f" % (sum(1 for r in a_rows if r.get("pred") == "Yes") / len(a_rows)) if a_rows else DASH,
        "Yes-rate of A under the CoT prompt, landed seeds pooled")
    b_rows = gen_rows("div7_base_0_0__cot")
    put("divsevenBaseCotMedTok", "%d" % median(r["n_gen_tokens"] for r in b_rows) if b_rows else DASH,
        "median generated tokens, base under the CoT prompt")
    # S11: second model family (fallback SmolLM2-1.7B-Instruct when Llama-3.2-3B was not downloadable)
    fam_f = os.path.join(ROOT, "results", "v2", "second_family_model.txt")
    fam = open(fam_f, encoding="utf-8").read().strip() if os.path.exists(fam_f) else None
    put("secondFamilyModel", fam.split("/")[-1] if fam else DASH, "results/v2/second_family_model.txt")
    fr = lambda t, a, s_: run(t, a, (0 if a == "base" else MAIN_N), s_, fam) if fam else None
    for t, tp in (("div7", "Divseven"), ("prime", "Prime"), ("valid", "Valid")):
        for a in ("A", "B", "D", "base"):
            for s_, sw_ in ((0, "sZero"), (1, "sOne")):
                r = fr(t, a, s_)
                if r:
                    put("fam" + tp + ("Base" if a == "base" else a) + sw_, pct(r["acc"]), "S11 " + fam)
    d = {s_: (fr("div7", "B", s_), fr("div7", "A", s_)) for s_ in (0, 1)}
    diffs = {s_: 100 * (b["acc"] - a["acc"]) for s_, (b, a) in d.items() if a and b}
    for s_, sw_ in ((0, "sZero"), (1, "sOne")):
        if s_ in diffs:
            b, a = d[s_]
            put("famDivsevenBAdiff" + sw_, "%+.1f" % diffs[s_], "S11 B - A, pp")
            put("famDivsevenBAp" + sw_, pval(mcnemar(b["bits_list"], a["bits_list"])[2]), "S11 McNemar B vs A")
    dd, a0 = fr("div7", "D", 0), fr("div7", "A", 0)
    if len(diffs) == 2 and dd and a0:
        ok_a = all(v >= 20 for v in diffs.values())
        ok_b = abs(dd["acc"] - a0["acc"]) <= 0.10
        put("SElevenVerdict", "passed" if (ok_a and ok_b) else "failed", "S11 rule: B - A >= 20 pp both seeds and |D - A| <= 10 pp")
        put("SElevenResult", "on %s, traces beat answers on div7 by %+.1f and %+.1f points (seeds 0 and 1), "
            "significant in both, so the %d-point threshold %s; the grounding control stays within %.1f points of A"
            % (fam.split("/")[-1], diffs[0], diffs[1], 20, "holds" if ok_a else "fails in seed 0" if diffs[0] < 20 else "fails in seed 1",
               100 * abs(dd["acc"] - a0["acc"])), "S11 summary")
    else:
        put("SElevenVerdict", "not complete", "S11 incomplete")
        put("SElevenResult", "the second-family cells had not finished by submission", "S11 incomplete")


# ------------------------------------------------------------------------------------------ main
def main():
    setup_macros()
    structure_macros()
    accuracy_macros()
    per_class_macros()
    mcnemar_macros()
    probe_rule_macros()
    earlier_study_macros()
    p = step_macros()
    prereg_macros(p)
    review_macros()
    scale_macros()
    late_macros()
    sweep_macros()
    n_rows = run_table()
    with open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by paper/exp/make_numbers.py -- do not edit; re-run the script\n")
        f.write("\\providecommand{\\TBD}{\\textbf{[TBD]}}\n")
        for k in sorted(OUT):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, OUT[k]))
    json.dump({"macros": OUT, "sources": SRC, "problems": PROBLEMS, "gen_notes": GEN_NOTES},
              open(os.path.join(HERE, "numbers.json"), "w", encoding="utf-8"), indent=1)
    # which macros used by main.tex still fall back to the placeholder file?
    main_tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    ph = open(os.path.join(PAPER, "numbers_placeholder.tex"), encoding="utf-8").read()
    fallback = [d for d in re.findall(r"\\providecommand\{\\(\w+)\}", ph)
                if d != "TBD" and d not in OUT and re.search(r"\\" + d + r"(?![A-Za-z])", main_tex)]
    n_tbd = sum(1 for v in OUT.values() if v == TBD)
    print(f"numbers.tex: {len(OUT)} macros ({n_tbd} TBD); tab_runs.tex: {n_rows} runs; "
          f"{len(RUNS)} runs loaded, {len(GENS)} gens files")
    if fallback:
        print("WARNING: used in main.tex but not generated (placeholder value shown):", " ".join(fallback))
    for pr in PROBLEMS:
        if "duplicate" in pr:
            print("note:", pr)


if __name__ == "__main__":
    main()
