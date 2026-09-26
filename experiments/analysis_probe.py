"""analysis_probe.py -- surface-learnability probes from the SAME 180 balanced training examples.

usage:  python experiments/analysis_probe.py [--seeds 0,1,2] [--n 180] [--no-full] [--tasks prime,div7] [--if-changed]
        (--if-changed: exit immediately if datasets + probe code are unchanged since the last full run)
        (--no-full skips the full-pool reference; --tasks restricts the tasks)
writes: results/analysis/probe.json (+ probe.md)

Torch-free. For each task with a train pool (datasets.json, plus results/v2/*.json with the same schema)
and each seed, draws the training set with a byte-for-byte replica of exp_thinking_ft_worker.balanced_sample
on pool "A" (pools A/B/C share item order and labels, so A, B and C of one seed see the same 180 items),
fits each probe on those 180 items only (hyperparameters by 5-fold CV on the same 180), and scores it on
the task's 240 test items.

Number tasks (prime, divK, ...):  features from the zero-padded digit string
  (a) logreg on one-hot (position x digit)
  (b) logreg on (a) + engineered: last digit even, last digit in {0,5}, digit sum mod 3 and mod 9,
      alternating digit sum mod 11 (one-hot residues)
  (c) MLP (1 hidden layer, lbfgs; width and L2 by CV) on (a) and on (b)
  (d) k-NN on (a) and on (b)
Validity task: (a) bag of word tokens, (b) (a) + character 1-3-grams + atom-abstracted word 1-3-grams
  (each propositional letter renamed V1, V2, ... by first appearance), (c) MLP on (b), (d) k-NN on (a), (b).
A test-only task (meta.test_only, e.g. prime_hard) is probed with training draws from its meta.train_task pool.
Also: the same logreg/MLP on (b) fitted on the FULL train pool (reference: more data, same surface), and
trivial rule baselines scored on the test set (no training) plus train-fitted lookup rules.
"best_probe_cv" = the probe with the best 5-fold CV accuracy on the 180 training items (an honest choice);
"best_probe_test" = max test accuracy over probes (optimistic upper bound: selected on test).
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict

import analysis_common as C

C.fix_sys_path()
import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import MaxAbsScaler  # noqa: E402
from scipy import sparse  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
SKIP_FULL = False


# --------------------------------------------------------------------------- featurizers
def digits_of(n, L):
    return [int(c) for c in str(n).zfill(L)]


def feat_onehot(nums, L):
    X = np.zeros((len(nums), L * 10))
    for i, n in enumerate(nums):
        for p, d in enumerate(digits_of(n, L)):
            X[i, p * 10 + d] = 1
    return X


def feat_engineered(nums, L):
    rows = []
    for n in nums:
        ds = digits_of(n, L)
        s = sum(ds)
        alt = sum(d if i % 2 == 0 else -d for i, d in enumerate(reversed(ds))) % 11
        v = [float(ds[-1] % 2 == 0), float(ds[-1] in (0, 5))]
        v += [float(s % 3 == r) for r in range(3)]
        v += [float(s % 9 == r) for r in range(9)]
        v += [float(alt == r) for r in range(11)]
        rows.append(v)
    return np.hstack([feat_onehot(nums, L), np.array(rows)])


def abstract_atoms(prompt):
    """Rename single capital-letter atoms V1, V2, ... by first appearance (keeps the schema, drops letters)."""
    body = prompt.split(" Does the conclusion")[0]
    mapping = {}

    def sub(m):
        a = m.group(0)
        if a not in mapping:
            mapping[a] = f"v{len(mapping) + 1}"
        return mapping[a]
    return re.sub(r"\b[A-Z]\b", sub, body.replace("Premises:", "premises").replace("Conclusion:", "conclusion"))


class TextFeats:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, prompts):
        self.w = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[;,.()]", lowercase=False)
        self.w.fit(prompts)
        if self.kind == "b":
            self.c = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False).fit(prompts)
            self.a = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[;,.()]", ngram_range=(1, 3),
                                     lowercase=True).fit([abstract_atoms(p) for p in prompts])
        return self

    def transform(self, prompts):
        X = self.w.transform(prompts)
        if self.kind == "b":
            X = sparse.hstack([X, self.c.transform(prompts),
                               self.a.transform([abstract_atoms(p) for p in prompts])]).tocsr()
        return X


# --------------------------------------------------------------------------- models
def cv_for(y):
    k = min(5, min(Counter(y).values()))
    return StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=0)


def make_models(seed, y):
    cv = cv_for(y)
    return {
        "logreg": LogisticRegressionCV(Cs=10, cv=cv, max_iter=5000, scoring="accuracy"),
        "mlp": GridSearchCV(MLPClassifier(solver="lbfgs", max_iter=1000, random_state=seed),
                            {"hidden_layer_sizes": [(16,), (64,)], "alpha": [1e-4, 1e-2, 1.0]},
                            cv=cv, scoring="accuracy"),
        "knn": GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9, 15, 25]},
                            cv=cv, scoring="accuracy"),
    }


def fit_score(model, Xtr, ytr, Xte, yte, cv):
    model.fit(Xtr, ytr)
    acc = float(np.mean(model.predict(Xte) == yte))
    if hasattr(model, "best_score_"):
        cvacc = float(model.best_score_)
        params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in model.best_params_.items()}
    elif hasattr(model, "C_"):
        cvacc = float(np.max(np.mean(list(model.scores_.values())[0], axis=0)))
        params = {"C": float(model.C_[0])}
    else:
        cvacc = float(np.mean(cross_val_score(model, Xtr, ytr, cv=cv)))
        params = {}
    return {"test_acc": acc, "cv_acc": cvacc, "params": params}


# --------------------------------------------------------------------------- trivial rules
def lookup_rule(train_keys, train_y, test_keys, test_y):
    """Majority label per key learned on train (ties/unseen -> train majority)."""
    maj = Counter(train_y).most_common(1)[0][0]
    tab = defaultdict(Counter)
    for k, y in zip(train_keys, train_y):
        tab[k][y] += 1
    pred = []
    for k in test_keys:
        c = tab.get(k)
        if not c or (len(c) == 2 and c["Yes"] == c["No"]):
            pred.append(maj)
        else:
            pred.append(c.most_common(1)[0][0])
    return float(np.mean(np.array(pred) == np.array(test_y)))


def oracle_lookup(test_keys, test_y):
    """Best achievable test accuracy by ANY function of the key (upper bound; fitted on test)."""
    tab = defaultdict(Counter)
    for k, y in zip(test_keys, test_y):
        tab[k][y] += 1
    return float(sum(max(c.values()) for c in tab.values()) / len(test_y))


def number_rules(task, tr_nums, tr_y, te_nums, te_y):
    out = {}
    if task.startswith("prime"):
        R = C.prime_rules()
        for name in ("always_No", "odd", "last_digit_1379", "no_factor_le_3", "no_factor_le_5",
                     "no_factor_le_7", "no_factor_le_11", "no_factor_le_13", "true_label"):
            out[name] = float(np.mean([R[name](n) == y for n, y in zip(te_nums, te_y)]))
    k = C.divisor_of_task(task)
    if k:
        R = C.divk_rules(k)
        for name, f in R.items():
            out[name] = float(np.mean([f(n) == y for n, y in zip(te_nums, te_y)]))
    out["majority"] = float(max(Counter(te_y).values()) / len(te_y))
    out["train_lookup_last_digit"] = lookup_rule([n % 10 for n in tr_nums], tr_y, [n % 10 for n in te_nums], te_y)
    out["train_lookup_digit_sum"] = lookup_rule([C.digit_sum(n) for n in tr_nums], tr_y,
                                                [C.digit_sum(n) for n in te_nums], te_y)
    out["oracle_any_fn_of_last_digit"] = oracle_lookup([n % 10 for n in te_nums], te_y)
    out["oracle_any_fn_of_digit_sum"] = oracle_lookup([C.digit_sum(n) for n in te_nums], te_y)
    return out


# --------------------------------------------------------------------------- main
def probe_task(task, td, seeds, n, tasks):
    pool, test = td["pools"].get("A"), td["test"]
    train_task = task
    if not pool and td.get("train_task") in tasks:           # test-only task (prime_hard): train on its source
        train_task = td["train_task"]
        pool = tasks[train_task]["pools"].get("A")
    if not pool:
        return None
    te_y = np.array([t["label"] for t in test])
    is_text = not all(C.number_of(t["prompt"]) is not None for t in test[:20]) or task.startswith("valid")
    res = {"task": task, "train_task": train_task, "n_train": n, "n_test": len(test), "pool_size": len(pool), "per_seed": {},
           "featurization": "text" if is_text else "digits", "source": td["source"]}
    L = None
    if not is_text:
        all_nums = [C.number_of(e["prompt"]) for e in pool] + [C.number_of(t["prompt"]) for t in test]
        L = max(len(str(x)) for x in all_nums)

    def build(train_rows, kind):
        if is_text:                                      # counts, max-abs scaled on the train rows only
            f = TextFeats(kind).fit([e["prompt"] for e in train_rows])
            sc = MaxAbsScaler().fit(f.transform([e["prompt"] for e in train_rows]))
            return (sc.transform(f.transform([e["prompt"] for e in train_rows])),
                    sc.transform(f.transform([t["prompt"] for t in test])))
        tr_n = [C.number_of(e["prompt"]) for e in train_rows]
        te_n = [C.number_of(t["prompt"]) for t in test]
        fn = feat_onehot if kind == "a" else feat_engineered
        return fn(tr_n, L), fn(te_n, L)

    for s in seeds:
        picked = C.balanced_sample(pool, n, s)
        ytr = np.array([C.extract(e["completion"]) for e in picked])
        cv = cv_for(ytr)
        per = {"train_ids_head": [e["id"] for e in picked[:5]], "train_balance": dict(Counter(ytr.tolist()))}
        feats = {k: build(picked, k) for k in ("a", "b")}
        probes = {}
        for fk in ("a", "b"):
            Xtr, Xte = feats[fk]
            for mname, model in make_models(s, ytr).items():
                probes[f"{mname}_{fk}"] = fit_score(model, Xtr, ytr, Xte, te_y, cv)
        per["probes"] = probes
        best_cv = max(probes, key=lambda k: (probes[k]["cv_acc"], -len(k)))
        best_te = max(probes, key=lambda k: probes[k]["test_acc"])
        per["best_probe_cv"] = {"probe": best_cv, **probes[best_cv]}
        per["best_probe_test"] = {"probe": best_te, **probes[best_te]}
        if not is_text:
            per["rules"] = number_rules(task, [C.number_of(e["prompt"]) for e in picked], ytr.tolist(),
                                        [C.number_of(t["prompt"]) for t in test], te_y.tolist())
        else:
            per["rules"] = {"majority": float(max(Counter(te_y.tolist()).values()) / len(te_y))}
        res["per_seed"][str(s)] = per
        print(f"  {task} seed {s}: " + ", ".join(f"{k}={v['test_acc']:.3f}" for k, v in probes.items()), flush=True)

    # full-pool reference (seed-independent)
    yfull = np.array([C.extract(e["completion"]) for e in pool])
    Xtr, Xte = build(pool, "b")
    full = {}
    for mname in (() if SKIP_FULL else ("logreg", "mlp")):
        model = make_models(0, yfull)[mname]
        full[f"{mname}_b"] = fit_score(model, Xtr, yfull, Xte, te_y, cv_for(yfull))
    res["full_pool_reference"] = full

    # seed summaries
    names = list(next(iter(res["per_seed"].values()))["probes"].keys())
    res["summary"] = {nm: float(np.mean([res["per_seed"][s]["probes"][nm]["test_acc"] for s in res["per_seed"]]))
                      for nm in names}
    res["summary"]["best_probe_cv_mean"] = float(np.mean([v["best_probe_cv"]["test_acc"] for v in res["per_seed"].values()]))
    res["summary"]["best_probe_test_mean"] = float(np.mean([v["best_probe_test"]["test_acc"] for v in res["per_seed"].values()]))
    rule_names = list(next(iter(res["per_seed"].values()))["rules"].keys())
    res["rules_summary"] = {r: float(np.mean([v["rules"][r] for v in res["per_seed"].values()])) for r in rule_names}
    return res


def main():
    global SKIP_FULL
    SKIP_FULL = "--no-full" in sys.argv
    digest = C.inputs_hash(os.path.abspath(__file__), (C.balanced_sample, C.extract, C.number_of, C.load_tasks,
                                                       C.prime_rules, C.divk_rules, C.has_factor_le, C.digit_sum))
    if C.cached_and_unchanged("probe.json", digest) and "--tasks" not in sys.argv:
        print("probe: inputs unchanged -> keeping results/analysis/probe.json")
        return
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [0, 1, 2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 180
    tasks = C.load_tasks()
    out = {"generated_by": "experiments/analysis_probe.py", "seeds": seeds, "n_train": n, "tasks": {},
           "inputs_hash": None if ("--tasks" in sys.argv or SKIP_FULL or seeds != [0, 1, 2] or n != 180) else digest}
    only = sys.argv[sys.argv.index("--tasks") + 1].split(",") if "--tasks" in sys.argv else None
    prev = os.path.join(C.OUT, "probe.json")
    if only and os.path.exists(prev):                  # partial run: update those tasks, keep the others
        try:
            old = json.load(open(prev, encoding="utf-8"))
            if old.get("n_train") == n:
                out["tasks"] = {t: v for t, v in old.get("tasks", {}).items() if t not in only}
        except Exception:
            pass
    for t, td in tasks.items():
        if only and t not in only:
            continue
        r = probe_task(t, td, seeds, n, tasks)
        if r is None:
            continue
        out["tasks"][t] = r
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "probe.json"), "w", encoding="utf-8"), indent=1)
    write_md(out)
    print("probe -> results/analysis/probe.json, probe.md")


def write_md(o):
    L = ["# Surface probes (auto-generated by experiments/analysis_probe.py; do not edit)", "",
         f"Train = the worker's balanced n={o['n_train']} draw per seed; test = 240 items. "
         "Mean test accuracy over seeds, %.", ""]
    for t, r in o["tasks"].items():
        L += [f"## {t} ({r['featurization']} features" + (f"; trained on {r['train_task']} pool" if r["train_task"] != t else "") + ")", ""]
        L.append("(Fine-tuned accuracies for comparison: results/analysis/README.md, section 2.)")
        L += ["", "| probe | " + " | ".join(f"seed {s}" for s in r["per_seed"]) + " | mean |",
              "|---|" + "---|" * (len(r["per_seed"]) + 1)]
        for nm, m in r["summary"].items():
            if nm.startswith("best"):
                continue
            L.append(f"| {nm} | " + " | ".join(f"{100 * v['probes'][nm]['test_acc']:.1f}" for v in r["per_seed"].values())
                     + f" | {100 * m:.1f} |")
        L.append("| best by train-CV | " + " | ".join(f"{100 * v['best_probe_cv']['test_acc']:.1f} ({v['best_probe_cv']['probe']})"
                                                    for v in r["per_seed"].values()) + f" | {100 * r['summary']['best_probe_cv_mean']:.1f} |")
        L.append("| best on test (optimistic) | " + " | ".join(f"{100 * v['best_probe_test']['test_acc']:.1f}"
                                                             for v in r["per_seed"].values()) + f" | {100 * r['summary']['best_probe_test_mean']:.1f} |")
        L += ["", "Full-pool reference (features b): " + ", ".join(f"{k}={100 * v['test_acc']:.1f}" for k, v in r["full_pool_reference"].items()),
              "", "Rule baselines (test accuracy, %): " + ", ".join(f"{k}={100 * v:.1f}" for k, v in r["rules_summary"].items()), ""]
    open(os.path.join(C.OUT, "probe.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
