"""analysis_common.py -- shared, torch-free loaders for the analysis_*.py scripts.

IMPORTANT: experiments/queue.py shadows the stdlib ``queue`` module for any script launched from this
directory (sys.path[0] == experiments/), which breaks torch, concurrent.futures, joblib/sklearn, ...
Every analysis entry point therefore calls ``fix_sys_path()`` BEFORE importing numpy/scipy/sklearn:
it drops experiments/ from the front of sys.path and re-appends it at the END (so this module stays
importable while the stdlib ``queue`` wins).

Nothing here imports numpy or torch (numpy + torch in one process segfaults on this machine).

Run sources (re-scanned on every call, so new runs are picked up automatically):
  results/thinking_vs_data/runs.jsonl        (original 36 runs, model implicit = Qwen2.5-1.5B-Instruct)
  results/thinking_vs_data/runs_*.jsonl      (e.g. runs_L1.jsonl)
  results/v2/runs_*.jsonl                    (same schema plus "model")
Test sets: results/thinking_vs_data/datasets.json plus any results/v2/*.json with the same
{"tasks": {task: {"test": [...], "pools": {...}}}} schema (later files add tasks; they never
override a task already defined by datasets.json).
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TVD = os.path.join(ROOT, "results", "thinking_vs_data")
# env overrides exist only for dry-running the pipeline on synthetic runs (never needed in normal use)
V2 = os.environ.get("ANALYSIS_V2_DIR") or os.path.join(ROOT, "results", "v2")
OUT = os.environ.get("ANALYSIS_OUT_DIR") or os.path.join(ROOT, "results", "analysis")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ANS_RE = re.compile(r"Answer:\s*(Yes|No)", re.I)
STANDALONE = re.compile(r"\b(Yes|No)\b", re.I)
NUM_RE = re.compile(r"\b(\d{2,})\b")


def fix_sys_path():
    """Remove experiments/ from the head of sys.path (stdlib queue shadowing); keep it at the end."""
    here = os.path.normcase(HERE)
    sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or os.getcwd())) != here]
    sys.path.append(HERE)
    if "queue" in sys.modules and getattr(sys.modules["queue"], "__file__", "") and \
            os.path.normcase(os.path.dirname(os.path.abspath(sys.modules["queue"].__file__))) == here:
        del sys.modules["queue"]


def extract(text: str):
    """Identical to exp_thinking_ft_worker.extract (the ONE answer extractor)."""
    m = ANS_RE.findall(text)
    if m:
        return m[-1].capitalize()
    m2 = STANDALONE.search(text)
    if m2:
        return m2.group(1).capitalize()
    return "?"


def balanced_sample(pool, n, seed):
    """Byte-for-byte replica of exp_thinking_ft_worker.balanced_sample."""
    yes = [e for e in pool if extract(e["completion"]) == "Yes"]
    no = [e for e in pool if extract(e["completion"]) == "No"]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    half = n // 2
    picked = yes[:half] + no[:half]
    rng.shuffle(picked)
    return picked


def number_of(prompt: str):
    m = NUM_RE.search(prompt)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- datasets
def load_tasks():
    """task -> {"test", "pools", "source", "meta", "notes"}. datasets.json first; results/v2/*.json add new
    tasks and ADD pools that an existing task lacks (e.g. div7 Bprime); a task's test set is never replaced
    (a v2 test set whose ids differ from the original is noted). A test-only task (meta.train_task, e.g.
    prime_hard) gets "train_task" so probes can train on that task's pool."""
    tasks = {}
    files = [os.path.join(TVD, "datasets.json")] + sorted(glob.glob(os.path.join(V2, "*.json")))
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not isinstance(d.get("tasks"), dict):
            continue
        src = os.path.relpath(f, ROOT).replace("\\", "/")
        for t, td in d["tasks"].items():
            if not isinstance(td, dict) or "test" not in td:
                continue
            if t in tasks:
                cur = tasks[t]
                if [x["id"] for x in td["test"]] != [x["id"] for x in cur["test"]]:
                    cur["notes"].append(f"{src} has a DIFFERENT test set for {t}; kept {cur['source']}")
                if td.get("step_truth") and not cur.get("step_truth"):
                    cur["step_truth"] = td["step_truth"]
                for arm, pool in (td.get("pools") or {}).items():
                    if arm not in cur["pools"]:
                        cur["pools"][arm] = pool
                        cur["notes"].append(f"pool {arm} added from {src}")
                continue
            meta = td.get("meta") or {}
            tasks[t] = {"test": td["test"], "pools": dict(td.get("pools") or {}), "source": src, "meta": meta,
                        "notes": [], "train_task": meta.get("train_task") if meta.get("test_only") else None,
                        "step_truth": td.get("step_truth") or {}}
    return tasks


# --------------------------------------------------------------------------- runs
def run_files():
    fs = [os.path.join(TVD, "runs.jsonl")]
    fs += sorted(f for f in glob.glob(os.path.join(TVD, "runs_*.jsonl")))
    fs += sorted(glob.glob(os.path.join(V2, "runs_*.jsonl")))
    fs += sorted(glob.glob(os.path.join(V2, "evals_*.jsonl")))
    return [f for f in fs if os.path.exists(f)]


def cell_label(train_task, eval_task, arm, prompt_mode="plain"):
    """The analysis 'task' of a run = the test set it was scored on, prefixed by the training task when that
    differs (e.g. 'prime->prime_hard'), suffixed by a non-plain prompt mode (e.g. 'div7[cot]'). Base runs are
    labelled by their test set alone (no training task)."""
    lab = eval_task if (arm == "base" or train_task == eval_task) else f"{train_task}->{eval_task}"
    return lab if prompt_mode in (None, "", "plain") else f"{lab}[{prompt_mode}]"


def decode_bits(hexstr: str, n: int):
    """Hex bitmap (MSB = first test item, leading zeros dropped) -> list of n ints."""
    v = int(hexstr, 16)
    if v.bit_length() > n:
        raise ValueError(f"bitmap has {v.bit_length()} bits > n_test={n}")
    return [int(c) for c in format(v, "0%db" % n)]


def load_runs():
    """All usable runs with decoded bits. Returns (runs, problems)."""
    runs, problems, seen = [], [], {}
    for f in run_files():
        for ln_no, line in enumerate(open(f, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                problems.append(f"{os.path.basename(f)}:{ln_no} bad json ({e})")
                continue
            if r.get("acc") is None or not r.get("bits"):
                problems.append(f"{os.path.basename(f)}:{ln_no} skipped {r.get('task')}/{r.get('arm')} "
                                f"({r.get('skip', 'no acc/bits')})")
                continue
            r = dict(r)
            if r.get("eval_limit"):
                problems.append(f"{os.path.basename(f)}:{ln_no} skipped (smoke test, eval_limit={r['eval_limit']})")
                continue
            r["model"] = r.get("model") or DEFAULT_MODEL
            r["source"] = os.path.relpath(f, ROOT).replace("\\", "/")
            r["train_task"] = r["task"]
            r["eval_task"] = r.get("eval_task") or r["task"]
            r["prompt_mode"] = r.get("prompt_mode") or "plain"
            r["task"] = cell_label(r["train_task"], r["eval_task"], r["arm"], r["prompt_mode"])
            n_test = int(r.get("n_test") or 240)
            bits = decode_bits(r["bits"], n_test)
            k = sum(bits)
            r["k"], r["n_test"], r["bits_list"] = k, n_test, bits
            r["count_check"] = (k == round(r["acc"] * n_test))
            if not r["count_check"]:
                problems.append(f"{r['source']}:{ln_no} decoded {k} != round(acc*n)={round(r['acc'] * n_test)}")
            key = (r["task"], r["arm"], int(r["n"]), int(r["seed"]), r["model"])
            if key in seen:
                first = next(x for x in runs if x["key"] == key)
                problems.append(f"duplicate {key}: keeping first ({seen[key]}, acc {first['acc']}), ignoring "
                                f"{r['source']}:{ln_no} (acc {r['acc']})")
                continue
            seen[key] = f"{r['source']}:{ln_no}"
            r["key"] = key
            runs.append(r)
    return runs, problems


def recover_preds(bits, test):
    """Binary tasks: pred = gold if correct else the other label. (A wrong answer might in principle
    have been unparseable '?'; the bitmap cannot distinguish that, so '?' is folded into 'flipped'.)"""
    flip = {"Yes": "No", "No": "Yes"}
    return [t["label"] if b else flip[t["label"]] for b, t in zip(bits, test)]


def short_model(m):
    return re.split(r"[/\\]", m.rstrip("/\\\\"))[-1]


def load_gens(tasks):
    """results/v2/gens/<model_short>/<task>_<arm>_<n>_<seed>[__on-<eval_task>][__<mode>][__reeval].jsonl
    (rows: id, gold, pred, correct, gen, n_gen_tokens, hit_cap) -> {(label, arm, n, seed, model_short): rows}."""
    root = os.path.join(V2, "gens")
    out, notes = {}, []
    if not os.path.isdir(root):
        return out, [f"no {os.path.relpath(root, ROOT)} directory (generation-level analyses skipped)"]
    known = sorted(tasks, key=len, reverse=True)
    for f in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        name = os.path.splitext(os.path.basename(f))[0]
        parts = name.split("__")
        head, flags = parts[0], parts[1:]
        try:
            ta, n, seed = head.rsplit("_", 2)
            n, seed = int(n), int(seed)
        except ValueError:
            notes.append(f"unparsed gens file name {name}")
            continue
        train_task = next((t for t in known if ta.startswith(t + "_")), None)
        if train_task is None:
            notes.append(f"gens {name}: unknown task")
            continue
        arm = ta[len(train_task) + 1:]
        eval_task, mode, reeval = train_task, "plain", False
        for fl in flags:
            if fl.startswith("on-"):
                eval_task = fl[3:]
            elif fl == "reeval":
                reeval = True
            else:
                mode = fl
        rows = []
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        model = os.path.basename(os.path.dirname(f))
        key = (cell_label(train_task, eval_task, arm, mode), arm, n, seed, model)
        if key in out and not reeval:
            continue
        out[key] = {"rows": rows, "path": os.path.relpath(f, ROOT).replace("\\", "/"), "eval_task": eval_task,
                    "train_task": train_task, "prompt_mode": mode, "reeval": reeval}
    return out, notes


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


# --------------------------------------------------------------------------- candidate rules
SMALL_PRIMES = [p for p in range(2, 100) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def has_factor_le(n: int, bound: int) -> bool:
    return any(n % p == 0 and n != p for p in SMALL_PRIMES if p <= bound)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % p for p in range(2, int(n ** 0.5) + 1))


def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(n))


def prime_rules():
    """name -> f(n) -> 'Yes'/'No' (Yes = 'prime'). Nested sieve rules; 'true_label' is exact primality."""
    R = {"always_No": lambda n: "No", "always_Yes": lambda n: "Yes",
         "odd": lambda n: "Yes" if n % 2 else "No",
         "last_digit_1379": lambda n: "Yes" if n % 10 in (1, 3, 7, 9) else "No"}
    for b in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        R[f"no_factor_le_{b}"] = (lambda bb: (lambda n: "No" if has_factor_le(n, bb) else "Yes"))(b)
    R["true_label"] = lambda n: "Yes" if is_prime(n) else "No"
    return R


def divk_rules(k: int):
    """Rules for 'Is n divisible by k?' (Yes = divisible)."""
    return {"always_No": lambda n: "No", "always_Yes": lambda n: "Yes",
            f"last_digit_is_{k % 10}": lambda n: "Yes" if n % 10 == k % 10 else "No",
            "last_digit_0_or_5": lambda n: "Yes" if n % 5 == 0 else "No",
            "last_digit_even": lambda n: "Yes" if n % 2 == 0 else "No",
            f"digit_sum_div_{k}": lambda n: "Yes" if digit_sum(n) % k == 0 else "No",
            "digit_sum_div_3": lambda n: "Yes" if digit_sum(n) % 3 == 0 else "No",
            "contains_digit_7": lambda n: "Yes" if "7" in str(n) else "No",
            "true_label": lambda n: "Yes" if n % k == 0 else "No"}


def divisor_of_task(task: str):
    m = re.match(r"div(\d+)", task)
    return int(m.group(1)) if m else None


def prime_stratum(n: int) -> str:
    if is_prime(n):
        return "prime"
    if n % 2 == 0:
        return "even_composite"
    if has_factor_le(n, 7):
        return "odd_composite_factor_le7"
    return "hard_composite_no_factor_le7"


# --------------------------------------------------------------------------- generation rows
def gen_text(row):
    """Generated text of a gens row (exp_worker_v2 writes 'gen'; 'text' accepted too)."""
    for k in ("gen", "text", "output", "completion", "generation"):
        if isinstance(row.get(k), str):
            return row[k]
    return ""


def gen_pred(row):
    """Predicted answer of a gens row ('pred' or 'predicted'; else re-extracted from the text)."""
    for k in ("pred", "predicted"):
        if row.get(k) in ("Yes", "No", "?"):
            return row[k]
    return extract(gen_text(row))


def rems_of(n: int, d: int):
    """Ground-truth running remainders of the digit-by-digit (10*r + digit) mod d procedure."""
    r, out = 0, []
    for c in str(n):
        r = (10 * r + int(c)) % d
        out.append(r)
    return out


# --------------------------------------------------------------------------- input fingerprint (caching)
def inputs_hash(script_path, funcs=()):
    """sha256 over the dataset files, the calling script and the source of the helper functions it relies on.
    Used by analysis_probe.py / analysis_tokens.py --if-changed to skip recomputation when nothing changed."""
    import hashlib
    import inspect
    h = hashlib.sha256()
    files = [os.path.join(TVD, "datasets.json")] + sorted(glob.glob(os.path.join(V2, "*.json")))
    for f in files + [script_path]:
        if os.path.exists(f):
            h.update(os.path.basename(f).encode())
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    for fn in funcs:
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def cached_and_unchanged(out_name, digest):
    p = os.path.join(OUT, out_name)
    if "--if-changed" not in sys.argv or not os.path.exists(p):
        return False
    try:
        return json.load(open(p, encoding="utf-8")).get("inputs_hash") == digest
    except Exception:
        return False
