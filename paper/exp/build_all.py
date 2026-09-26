"""build_all.py -- one command from result files to paper/main.pdf (CPU only; never touches the GPU).

  1. analysis:  experiments/analysis_stats.py, analysis_probe.py (only for tasks missing from probe.json,
                or all tasks with --probe), analysis_rules.py, analysis_tokens.py, analysis_report.py
  2. paper/exp/make_numbers.py  -> paper/numbers.tex, paper/tab_runs.tex
  3. paper/exp/make_figures.py  -> paper/fig_map.pdf, paper/fig_arms.pdf
  4. pdflatex, bibtex, pdflatex, pdflatex in paper/
  5. report: total pages, the page on which the main text ends, overfull boxes, undefined references

usage: python paper/exp/build_all.py [--probe] [--skip-analysis]
Exit status is non-zero if a step failed or the build has undefined references / main text > 9 pages.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER, ".."))
EXP = os.path.join(ROOT, "experiments")
PY = sys.executable
FAILED = []


def step(name, cmd, cwd=ROOT, timeout=3600, quiet=False):
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")   # CPU only
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        ok = p.returncode == 0
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT"
    print("[%s] %-28s %5.0fs" % ("ok" if ok else "FAIL", name, time.time() - t0), flush=True)
    if not ok:
        FAILED.append(name)
        print("   " + "\n   ".join(out.strip().splitlines()[-12:]))
    elif not quiet and out.strip():
        for line in out.strip().splitlines()[-6:]:
            print("   " + line)
    return ok


def probe_tasks_needed():
    """Tasks that have runs or a test set but no entry in results/analysis/probe.json."""
    sys.path.append(EXP)
    import analysis_common as C
    have = set()
    p = os.path.join(ROOT, "results", "analysis", "probe.json")
    if os.path.exists(p):
        have = set(json.load(open(p, encoding="utf-8")).get("tasks", {}))
    tasks = C.load_tasks()
    return sorted(t for t in tasks if t not in have)


def main():
    skip_analysis = "--skip-analysis" in sys.argv
    if not skip_analysis:
        step("analysis_stats", [PY, os.path.join(EXP, "analysis_stats.py")])
        if "--probe" in sys.argv:
            step("analysis_probe (all)", [PY, os.path.join(EXP, "analysis_probe.py")], timeout=7200)
        else:
            need = probe_tasks_needed()
            if need:
                step("analysis_probe " + ",".join(need),
                     [PY, os.path.join(EXP, "analysis_probe.py"), "--tasks", ",".join(need)], timeout=7200)
            else:
                print("[skip] analysis_probe               (probe.json covers every task; --probe to refresh)")
        step("analysis_rules", [PY, os.path.join(EXP, "analysis_rules.py")])
        step("analysis_tokens", [PY, os.path.join(EXP, "analysis_tokens.py")])
        step("analysis_report", [PY, os.path.join(EXP, "analysis_report.py")])
    step("make_numbers", [PY, os.path.join(HERE, "make_numbers.py")])
    step("make_figures", [PY, os.path.join(HERE, "make_figures.py")])
    for ext in ("aux", "bbl", "blg", "out"):
        f = os.path.join(PAPER, "main." + ext)
        if os.path.exists(f):
            os.remove(f)
    tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    step("pdflatex 1", tex, cwd=PAPER, quiet=True)
    step("bibtex", ["bibtex", "main"], cwd=PAPER, quiet=True)
    step("pdflatex 2", tex, cwd=PAPER, quiet=True)
    step("pdflatex 3", tex, cwd=PAPER, quiet=True)

    log = open(os.path.join(PAPER, "main.log"), encoding="latin-1").read() \
        if os.path.exists(os.path.join(PAPER, "main.log")) else ""
    m = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    pages = int(m.group(1)) if m else None
    overfull = len(re.findall(r"^Overfull \\hbox", log, re.M))
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
    undef_cs = len(re.findall(r"Undefined control sequence", log))
    main_end = refs = None
    try:
        import fitz  # PyMuPDF, optional
        doc = fitz.open(os.path.join(PAPER, "main.pdf"))
        for i, pg in enumerate(doc):
            t = pg.get_text()
            if main_end is None and "Limitations." in t:
                main_end = i + 1
            if refs is None and "REFERENCES" in t:
                refs = i + 1
        tbd = sum(pg.get_text().count("[TBD]") for pg in doc)
    except Exception:
        tbd = None
    print("\n== build report")
    print("pages: %s total; main text (through Limitations) ends on page %s; references start on page %s"
          % (pages, main_end, refs))
    print("overfull hboxes: %d; undefined refs/cites: %s; undefined control sequences: %d; [TBD] cells: %s"
          % (overfull, ", ".join(undef) or "none", undef_cs, tbd))
    bad = FAILED or undef or undef_cs or (main_end is not None and main_end > 9)
    if FAILED:
        print("failed steps:", ", ".join(FAILED))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
