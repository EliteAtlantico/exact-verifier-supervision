"""Run a list of cells through an isolated worker subprocess each, appending RESULT json lines.

usage: python experiments/queue.py <queue-file> <out-jsonl> [--worker exp_thinking_ft_worker.py] [--extra ARGS...]
queue file: one cell per line: "<task> <arm> <n> <seed>"; '#' comments allowed.
Skips cells already present in the output file (resumable). Stops if free disk < 800 MB.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def key(r):
    return (r["task"], r["arm"], int(r["n"]), int(r["seed"]), r.get("model", ""))


def main():
    qfile, out = sys.argv[1], sys.argv[2]
    worker = "exp_thinking_ft_worker.py"
    extra = []
    if "--worker" in sys.argv:
        worker = sys.argv[sys.argv.index("--worker") + 1]
    if "--extra" in sys.argv:
        extra = sys.argv[sys.argv.index("--extra") + 1:]
    model = ""
    if "--model" in extra:
        model = extra[extra.index("--model") + 1]
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                done.add(key(json.loads(line)))
            except Exception:
                pass
    cells = []
    for line in open(qfile, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            t, a, n, s = line.split()
            cells.append((t, a, int(n), int(s)))
    for t, a, n, s in cells:
        if (t, a, n, s, model) in done:
            print(f"skip {t} {a} {n} {s} (done)", flush=True)
            continue
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(out)) or ".").free
        if free < 800 * 2 ** 20:
            print(f"STOP: free disk {free / 2**20:.0f} MB < 800 MB", flush=True)
            return
        cmd = [sys.executable, os.path.join(HERE, worker), t, a, str(n), str(s)] + extra
        t0 = time.time()
        print(f"run {t} {a} {n} {s} {model} ...", flush=True)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=3600, cwd=HERE)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            continue
        res = None
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                res = json.loads(ln[7:])
        if res is None:
            print("  FAILED rc=%s\n%s" % (p.returncode, (p.stderr or "")[-1500:]), flush=True)
            continue
        res["wall_s"] = round(time.time() - t0, 1)
        if model:
            res.setdefault("model", model)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
        print(f"  acc {res['acc']:.4f}  ({res['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
