"""check_v2.py -- independent static checks of results/v2/datasets_v2.json and the v2 tooling (CPU, no model).

python experiments/check_v2.py            # data checks only (no torch import)
python experiments/check_v2.py --worker   # also import exp_worker_v2 and exercise argument parsing
                                          # (sets CUDA_VISIBLE_DEVICES=-1 first; never loads a model)
Re-derives every label, remainder and split property from the numbers themselves; shares no code with
exp_dataset_v2.py. Exits non-zero on the first failure.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "results" / "v2" / "datasets_v2.json"
OLD = REPO / "results" / "thinking_vs_data" / "datasets.json"
DIV = {"div2": 2, "div3": 3, "div7": 7, "div11": 11, "div13": 13, "div7_6d": 7}
STEP = re.compile(r"r = \(10\*(\d+) \+ (\d)\) mod (\d+) = (\d+)\.")


def num(item_id):
    return int(item_id.rsplit("-", 1)[1])


def ans(c):
    m = re.findall(r"Answer:\s*(Yes|No)", c)
    return m[-1] if m else None


def picks(pool, n, seed):                     # the worker's balanced_sample, re-implemented
    yes = [e for e in pool if ans(e["completion"]) == "Yes"]
    no = [e for e in pool if ans(e["completion"]) == "No"]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    p = yes[:n // 2] + no[:n // 2]
    rng.shuffle(p)
    return [e["id"] for e in p]


def check_data():
    d = json.loads(V2.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    T = d["tasks"]
    for t in ("prime", "valid", "div7"):
        assert T[t]["test"] == old["tasks"][t]["test"] and T[t]["meta"] == old["tasks"][t]["meta"], t
        for a in old["tasks"][t]["pools"]:
            assert T[t]["pools"][a] == old["tasks"][t]["pools"][a], (t, a)
    report = {}
    for t, dv in DIV.items():
        v = T[t]
        test, pools = v["test"], v["pools"]
        tr = [num(x["id"]) for x in pools["A"]]
        te = [num(x["id"]) for x in test]
        assert len(te) == 240 and sum(x["label"] == "Yes" for x in test) == 120, f"{t} test balance"
        assert len(tr) >= 540 and sum(ans(x["completion"]) == "Yes" for x in pools["A"]) * 2 == len(tr), f"{t} pool"
        assert len(set(tr)) == len(tr) and len(set(te)) == 240 and not set(tr) & set(te), f"{t} disjoint"
        digits = 6 if t == "div7_6d" else 4
        assert all(len(str(x)) == digits for x in tr + te), f"{t} digit count"
        for x in test:
            assert x["label"] == ("Yes" if num(x["id"]) % dv == 0 else "No"), f"{t} test label"
            assert f"Is {num(x['id'])} divisible by {dv}?" in x["prompt"]
        arms = [a for a in ("A", "B", "C", "Bprime") if a in pools]
        for a in arms:
            assert [x["id"] for x in pools[a]] == [x["id"] for x in pools["A"]], f"{t}/{a} order"
            assert [ans(x["completion"]) for x in pools[a]] == [ans(x["completion"]) for x in pools["A"]]
        for seed in (0, 1, 2):                    # every arm trains on the SAME problems at a given seed
            ref = picks(pools["A"], 180, seed)
            assert len(ref) == 180 and all(picks(pools[a], 180, seed) == ref for a in arms), f"{t} picks"
        for x in pools["B"] + pools.get("Bprime", []):
            n = num(x["id"])
            steps = STEP.findall(x["completion"])
            assert len(steps) == digits, f"{t} steps {x['id']}"
            r = 0
            for (r0, dig, dd, r1), ch in zip(steps, str(n)):
                assert int(r0) == r and dig == ch and int(dd) == dv
                r = (10 * r + int(ch)) % dv
                assert int(r1) == r, f"{t} wrong remainder {x['id']}"
            assert (r == 0) == (ans(x["completion"]) == "Yes"), f"{t} trace/answer {x['id']}"
            st = v["step_truth"][x["id"]]
            assert st["k"] == digits and st["rems"] == [int(s[3]) for s in steps]
        for x in test:
            st = v["step_truth"][x["id"]]
            assert st["rems"] == [int(str(num(x["id"]))[:i + 1]) % dv for i in range(digits)], f"{t} test truth"
        if "Bprime" in pools:
            for b, bp in zip(pools["B"], pools["Bprime"]):
                tr_txt, a_ = b["completion"].rsplit("\nAnswer: ", 1)
                assert bp["completion"] == f"Answer: {a_}\n{tr_txt}"
        wb = sorted(w for x in pools["B"] for w in x["completion"].split())
        wc = sorted(w for x in pools["C"] for w in x["completion"].split())
        assert wb == wc, f"{t} C multiset"
        report[t] = {"pool": len(tr), "test": 240, "k": digits,
                     "max_B_chars": max(len(x["completion"]) for x in pools["B"])}
    # prime_hard
    ph = T["prime_hard"]["test"]
    used = {num(x["id"]) for x in old["tasks"]["prime"]["pools"]["A"]} | {num(x["id"]) for x in old["tasks"]["prime"]["test"]}
    ns = [num(x["id"]) for x in ph]
    assert len(set(ns)) == 240 and not set(ns) & used, "prime_hard overlaps the prime pool/test"

    def isp(n):
        return n > 1 and all(n % p for p in range(2, int(n ** 0.5) + 1))
    for x, n in zip(ph, ns):
        assert 1000 <= n <= 9999 and all(n % p for p in (2, 3, 5, 7)), f"prime_hard factor <= 7: {n}"
        assert x["label"] == ("Yes" if isp(n) else "No")
        assert x["prompt"] == f"Is {n} a prime number? End your reply with 'Answer: Yes' or 'Answer: No'."
    assert sum(x["label"] == "Yes" for x in ph) == 120
    assert all(len(v) == 4 for v in d["fewshot4"].values())
    for t, ids in d["fewshot4"].items():
        pool_ids = {x["id"] for x in T[t]["pools"]["B"]}
        test_ids = {x["id"] for x in T[t]["test"]}
        assert set(ids) <= pool_ids and not set(ids) & test_ids, f"fewshot4 {t}"
    report["prime_hard"] = {"test": 240, "primes": 120, "hard_composites": 120,
                            "min_factor_of_composites": min(x["min_factor"] for x in ph if x["label"] == "No")}
    return report


def check_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                  # "" is deleted on Windows
    sys.path.insert(0, str(REPO / "experiments"))
    import exp_worker_v2 as w                                 # imports torch on CPU; loads no model
    assert w.DEV == "cpu"
    a = w.parse_args(["div7", "B", "180", "0"])
    assert (a.model, a.eval_task, a.prompt_mode, a.eval_bs, a.eval_only) == (w.DEFAULT_MODEL, "div7", "plain", 8, None)
    assert w.cell_name(a) == "div7_B_180_0"
    assert w.adapter_dir_for(a) == REPO / "results/adapters/Qwen2.5-1.5B-Instruct/div7_B_180_0"
    a = w.parse_args(["div7", "B", "180", "1", "--eval-only", "auto", "--eval-task", "div7_6d", "--eval-bs", "32"])
    assert w.cell_name(a) == "div7_B_180_1__on-div7_6d__reeval" and a.eval_bs == 32
    a = w.parse_args(["prime", "A", "180", "0", "--eval-only", "none", "--prompt-mode", "fewshot4"])
    assert (a.arm, a.n) == ("base", 0) and w.cell_name(a) == "prime_base_0_0__fewshot4"
    a = w.parse_args(["div7", "Bprime", "180", "0", "--model", "Qwen/Qwen2.5-3B-Instruct"])
    assert w.model_short(a.model) == "Qwen2.5-3B-Instruct"
    d = json.loads(V2.read_text(encoding="utf-8"))
    for t in d["tasks"]:
        assert t in w.MAXNEW, f"MAXNEW missing {t}"
    assert w.load_pool(d, "div7", "Bprime")[0]["completion"].startswith("Answer: ")
    msgs = w.eval_messages("Q?", "fewshot4", [{"prompt": "p", "completion": "c"}] * 4)
    assert len(msgs) == 9 and msgs[-1] == {"role": "user", "content": "Q?"}
    assert w.eval_messages("Q?", "cot", [])[0]["content"].endswith("'Answer: No'.")
    import run_queue2 as q
    ck = q.cell_key("prime", "A", 180, 0, ["--model", "M", "--eval-only", "auto", "--eval-task", "prime_hard"])
    rk = q.result_key({"task": "prime", "arm": "A", "n": 180, "req_n": 180, "seed": 0, "model": "M",
                       "prompt_mode": "plain", "eval_task": "prime_hard", "eval_only": True})
    assert ck == rk
    assert q.cell_key("div7", "B", 180, 0, ["--model", "M"]) != ck
    assert q.cell_key("div7", "X", 5, 0, ["--eval-only", "none"])[:3] == ("div7", "base", 0)
    assert not q.stdlib_shadows(), q.stdlib_shadows()
    return "worker/queue2 arg parsing OK"


if __name__ == "__main__":
    for k, v in check_data().items():
        print(k, v)
    if "--worker" in sys.argv:
        print(check_worker())
    print("ALL CHECKS PASSED")
