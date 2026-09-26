"""run_queue2.py -- run_queue.py plus per-line worker flags and a resume key that knows the v2 fields.

usage: python experiments/run_queue2.py <queue-file> <out-jsonl> [--worker exp_worker_v2.py]
                                        [--timeout 3600] [--min-free-mb 800] [--extra ARGS...]
queue line: "<task> <arm> <n> <seed> [worker flags...]"   ('#' comments allowed), e.g.
    prime A 180 0 --eval-only auto --eval-task prime_hard
    div7 base 0 0 --eval-only none --prompt-mode cot
--extra ARGS go to every cell (e.g. --model Qwen/Qwen2.5-3B-Instruct --eval-bs 32); line flags come after
them, so a line flag wins. Resume key = (task, arm, req_n, seed, model, prompt_mode, eval_task, eval_only).
Keep eval-only / cot / --eval-task rows OUT of results/v2/runs_*.jsonl: analysis_common.load_runs keys runs
on (task, arm, n, seed, model) and would score them against the wrong test set. Use evals_<model>.jsonl.
Failures are appended (stderr tail) to <out-jsonl>.failures.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"          # must equal exp_worker_v2.DEFAULT_MODEL


def _cell_flags(tokens):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-mode", default="plain")
    ap.add_argument("--eval-task", default=None)
    ap.add_argument("--eval-only", default=None)
    f, _ = ap.parse_known_args(tokens)
    return f


def cell_key(t, a, n, s, tokens):
    f = _cell_flags(tokens)
    if f.eval_only is not None and f.eval_only.lower() == "none":
        a, n = "base", 0
    return (t, a, int(n), int(s), f.model, f.prompt_mode, f.eval_task or t, f.eval_only is not None)


def result_key(r):
    return (r["task"], r["arm"], int(r.get("req_n", r["n"])), int(r["seed"]), r.get("model") or DEFAULT_MODEL,
            r.get("prompt_mode", "plain"), r.get("eval_task") or r["task"], bool(r.get("eval_only", False)))


def stdlib_shadows():
    names = getattr(sys, "stdlib_module_names", ())
    return sorted(p.name for p in HERE.glob("*.py") if p.stem in names)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--extra" in argv:
        i = argv.index("--extra")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile")
    ap.add_argument("out")
    ap.add_argument("--worker", default="exp_worker_v2.py")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--min-free-mb", type=int, default=800)
    a = ap.parse_args(argv)
    bad = stdlib_shadows()
    if bad:
        sys.exit(f"ABORT: {bad} in {HERE} shadow stdlib modules and break `import torch` in workers")
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    qfile = Path(a.qfile) if Path(a.qfile).is_absolute() else REPO / a.qfile
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(result_key(json.loads(line)))
            except Exception:
                pass
    cells = []
    for line in open(qfile, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            parts = line.split()
            cells.append((parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4:]))
    for t, arm, n, s, flags in cells:
        tokens = extra + flags
        k = cell_key(t, arm, n, s, tokens)
        label = " ".join([t, arm, str(n), str(s)] + flags)
        if k in done:
            print(f"skip {label} (done)", flush=True)
            continue
        free = shutil.disk_usage(out.parent).free
        if free < a.min_free_mb * 2 ** 20:
            print(f"STOP: free disk {free / 2**20:.0f} MB < {a.min_free_mb} MB", flush=True)
            return 2
        cmd = [sys.executable, str(HERE / a.worker), t, arm, str(n), str(s)] + tokens
        t0 = time.time()
        print(f"run {label} [{k[4]}] ...", flush=True)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=a.timeout, cwd=str(REPO))
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            with open(str(out) + ".failures.log", "a", encoding="utf-8") as fh:
                fh.write(f"=== {label} TIMEOUT after {a.timeout}s\n")
            continue
        res = None
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                res = json.loads(ln[7:])
        if res is None or res.get("acc") is None:
            tail = (p.stderr or "")[-3000:]
            print("  FAILED rc=%s\n%s" % (p.returncode, tail[-1500:]), flush=True)
            with open(str(out) + ".failures.log", "a", encoding="utf-8") as fh:
                fh.write(f"=== {label} rc={p.returncode}\n{tail}\n")
            if res is None:
                continue
        res["wall_s"] = round(time.time() - t0, 1)
        res.setdefault("model", k[4])
        res.setdefault("prompt_mode", k[5])
        res.setdefault("eval_task", k[6])
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
        done.add(result_key(res))
        if res.get("acc") is not None:
            print(f"  acc {res['acc']:.4f}  ({res['wall_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
