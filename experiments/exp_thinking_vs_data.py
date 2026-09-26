"""exp_thinking_vs_data.py -- Phases 1-3 driver for the THINKING-vs-DATA study.

Spawns exp_thinking_ft_worker.py as one ISOLATED subprocess per run (the only pattern that avoids
the multi-run OOM on the 8GB laptop GPU). Resumable: appends results/thinking_vs_data/runs.jsonl and
skips completed (task,arm,n,seed). Refuses to start a worker while Ollama still holds VRAM (torch-free
/api/ps check). After the grid: Wilson CIs, paired McNemar from the per-item bitmaps, accuracy-vs-
examples / vs-tokens / vs-wall tables -> appended below the frozen preregistration in THINKING_VS_DATA.md.

usage:
  python exp_thinking_vs_data.py pilot     # 4 runs (prime base + A/B/C @180 s0)
  python exp_thinking_vs_data.py full      # the 35-run core grid
  python exp_thinking_vs_data.py report    # (re)build the markdown from runs.jsonl
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "results", "thinking_vs_data")
RUNS = os.path.join(OUT_DIR, "runs.jsonl")
WORKER = os.path.join(HERE, "exp_thinking_ft_worker.py")
MD = os.path.join(HERE, "..", "results", "THINKING_VS_DATA.md")
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TIMEOUT = 1800
TASKS = ("prime", "valid", "div7")


def _grid_full():
    runs = []
    for t in TASKS:
        runs.append((t, "base", 0, 0))
    for t in TASKS:
        for n in (60, 540):
            runs.append((t, "A", n, 0)); runs.append((t, "B", n, 0))
        for s in (0, 1):
            runs.append((t, "A", 180, s)); runs.append((t, "B", 180, s)); runs.append((t, "C", 180, s))
        runs.append((t, "A_tok", 0, 0))                  # driver fills n below
    runs.append(("prime", "B_orn", 180, 0)); runs.append(("prime", "B_orn", 180, 1))
    runs.append(("valid", "B_mv", 60, 0))
    return runs


def _grid_pilot():
    return [("prime", "base", 0, 0), ("prime", "A", 180, 0),
            ("prime", "B", 180, 0), ("prime", "C", 180, 0)]


def _a_tok_n(task):
    """n for the A-tok arm = examples needed so answers-only tokens match B@180's tokens."""
    d = json.load(open(os.path.join(OUT_DIR, "datasets.json"), encoding="utf-8"))["tasks"][task]
    b_tok = d["meta"]["ws_tokens"]["B"]
    b_pool = len(d["pools"]["B"])
    b_tok_180 = b_tok * (180 / b_pool)                   # B@180 token estimate
    a_tok_per = d["meta"]["ws_tokens"]["A"] / len(d["pools"]["A"])   # ~2 ws-tokens ("Answer: Yes")
    n = int(round(b_tok_180 / max(1e-6, a_tok_per)))
    return min(n - n % 2, (b_pool // 2) * 2)             # even, within pool


def _done():
    got = {}
    if os.path.exists(RUNS):
        for line in open(RUNS, encoding="utf-8"):
            try:
                r = json.loads(line)
                got[(r["task"], r["arm"], r["req_n"], r["seed"])] = r
            except Exception:  # noqa: BLE001
                pass
    return got


def _ollama_busy():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=4) as r:
            return bool((json.loads(r.read()) or {}).get("models"))
    except Exception:  # noqa: BLE001
        return False                                     # ollama not running -> not busy


def run_grid(grid):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = _done()
    for _ in range(30):
        if not _ollama_busy():
            break
        print("Ollama still holds VRAM; waiting 30s before spawning GPU workers...")
        time.sleep(30)
    for (task, arm, n, seed) in grid:
        req_n = _a_tok_n(task) if arm == "A_tok" else n
        key = (task, arm, req_n, seed)
        if key in done:
            print(f"skip (done): {key}")
            continue
        cmd = [sys.executable, WORKER, task, arm, str(req_n), str(seed)]
        print(f"RUN {task} {arm} n={req_n} seed={seed}")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=TIMEOUT, cwd=HERE)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {TIMEOUT}s"); continue
        except Exception as exc:  # noqa: BLE001 -- a decode/spawn error must not abort the whole grid
            print(f"  worker error: {str(exc)[:120]}"); continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print("  no RESULT. stderr tail:\n", (proc.stderr or "")[-800:]); continue
        rec = json.loads(line[len("RESULT "):])
        rec["req_n"] = req_n
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[key] = rec
        print(f"  acc={rec.get('acc')} n={rec.get('n')} train_tok={rec.get('train_tokens')} "
              f"wall={rec['wall_s']}s")
    return done


# --------------------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar(bits_a, bits_b):
    """Exact binomial two-sided p for discordant pairs. bits are hex correctness strings."""
    a = bin(int(bits_a, 16))[2:].zfill(len(bits_a) * 4)
    b = bin(int(bits_b, 16))[2:].zfill(len(bits_b) * 4)
    L = min(len(a), len(b))
    a, b = a[-L:], b[-L:]
    b01 = sum(1 for i in range(L) if a[i] == "1" and b[i] == "0")  # A right, B wrong
    b10 = sum(1 for i in range(L) if a[i] == "0" and b[i] == "1")  # A wrong, B right
    nd = b01 + b10
    if nd == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    p = 2 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)
    return min(1.0, p), b01, b10


def _by(runs):
    m = {}
    for r in runs:
        m.setdefault((r["task"], r["arm"], r["req_n"]), []).append(r)
    return m


def report():
    runs = [json.loads(l) for l in open(RUNS, encoding="utf-8")] if os.path.exists(RUNS) else []
    if not runs:
        print("no runs yet"); return
    by = _by(runs)

    def acc(task, arm, n):
        rs = [r for r in by.get((task, arm, n), []) if r.get("acc") is not None]
        if not rs:
            return None
        a = sum(r["acc"] for r in rs) / len(rs)
        nt = rs[0]["n_test"]
        lo, hi = wilson(round(a * nt), nt)
        return a, lo, hi, rs

    lines = ["", "## Results (appended after the grid; preregistration above is unchanged)", ""]
    lines.append(f"_Total runs: {len(runs)}; wall {sum(r.get('wall_s', 0) for r in runs) / 60:.0f} min._")
    lines.append("")
    lines.append("### H1/H2 at n=180 (matched examples; McNemar on shared items, seed 0)")
    lines.append("| task | base | A (data) | B (thinking) | C (scrambled) | B>A p | B>C p | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        b = acc(t, "base", 0); A = acc(t, "A", 180); B = acc(t, "B", 180); C = acc(t, "C", 180)
        def cell(x):
            return f"{x[0]*100:.1f}% [{x[1]*100:.0f},{x[2]*100:.0f}]" if x else "-"
        pBA = pBC = "-"
        vr = "-"
        if A and B:
            ra = next((r for r in A[3] if r["seed"] == 0), None)
            rb = next((r for r in B[3] if r["seed"] == 0), None)
            if ra and rb and ra.get("bits") and rb.get("bits"):
                p, _, _ = mcnemar(rb["bits"], ra["bits"]); pBA = f"{p:.3f}"
        if B and C:
            rb = next((r for r in B[3] if r["seed"] == 0), None)
            rc = next((r for r in C[3] if r["seed"] == 0), None)
            if rb and rc and rb.get("bits") and rc.get("bits"):
                p, _, _ = mcnemar(rb["bits"], rc["bits"]); pBC = f"{p:.3f}"
        if A and B and C:
            if B[0] > A[0] and B[0] > C[0]:
                vr = "thinking wins"
            elif B[0] > A[0]:
                vr = "tokens, not thinking"
            else:
                vr = "no thinking gain"
        lines.append(f"| {t} | {cell(b)} | {cell(A)} | {cell(B)} | {cell(C)} | {pBA} | {pBC} | {vr} |")

    lines.append("")
    lines.append("### H3 efficiency: accuracy vs training examples and tokens")
    lines.append("| task | arm | n | ex | train_ws_tok | acc | wall_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in TASKS:
        for arm in ("A", "B", "A_tok"):
            for key in sorted(k for k in by if k[0] == t and k[1] == arm):
                for r in by[key]:
                    if r.get("acc") is not None:
                        lines.append(f"| {t} | {arm} | {r['req_n']} | {r.get('n')} | "
                                     f"{r.get('train_tokens')} | {r['acc']*100:.1f}% | {r.get('wall_s')} |")

    lines.append("")
    lines.append("### H4 div7 (answer-labels proven null; does the procedure trace unlock it?)")
    b = acc("div7", "base", 0); A = acc("div7", "A", 180); B = acc("div7", "B", 180)
    def c2(x):
        return f"{x[0]*100:.1f}%" if x else "-"
    lines.append(f"- base {c2(b)} | A (labels) {c2(A)} | B (thinking) {c2(B)} -> "
                 + ("**thinking teaches what labels cannot**" if (A and B and B[0] - A[0] > 0.08)
                    else "no unlock"))

    lines.append("")
    lines.append("### Open-source AI thinking")
    bo = acc("prime", "B_orn", 180); bd = acc("prime", "B", 180)
    bmv = acc("valid", "B_mv", 60)
    lines.append(f"- prime: B-det {c2(bd)} vs B-orn (ornith:9b) {c2(bo)}")
    lines.append(f"- valid: B-mv (multiverse+ornith) {c2(bmv)}")

    with open(MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("APPENDED results ->", MD)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "report":
        report(); return 0
    grid = _grid_pilot() if mode == "pilot" else _grid_full()
    run_grid(grid)
    if mode == "full":
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
