"""make_supplementary.py -- build the anonymized ICLR supplementary zip (CPU only, re-runnable).

usage: python scripts/make_supplementary.py [--out ZIP] [--keep-staging] [--no-verify] [--max-mb 60]

What it does
  1. copies code, data, run records, analysis outputs, generations and the preregistration files into a
     fresh staging folder (never touches the repository files themselves);
  2. scrubs the copies: absolute paths (C:\\Users\\..., /home/...) -> repo-relative, commit hashes in
     PREDICTIONS.md and elsewhere -> the paper's neutral labels P1..P6 (other hashes removed), timestamp
     locations -> time zone, names of people / internal packages -> neutral words;
  3. writes README_SUPPLEMENT.md (folder guide, reproduction commands, environment read from the installed
     packages, hardware, preregistration protocol with a hash-free timestamp manifest from git log);
  4. zips the staging folder, then re-opens the zip and scans every member (names and contents, including
     JSONL fields such as adapter paths) case-insensitively for forbidden terms, absolute paths, git metadata
     and commit hashes; exits non-zero if anything remains or the zip exceeds --max-mb;
  5. (unless --no-verify) runs paper/exp/make_numbers.py inside a scratch copy of the staging folder to check
     that the numbers rebuild from the anonymized files alone.

The zip is written to --out (default: ~/Downloads/PerfectLabels_ICLR2027_supplementary.zip).
The scan list below necessarily contains the terms it scans for; this script is NOT copied into the zip.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = "supplementary"                       # top-level folder inside the zip
DEFAULT_OUT = Path.home() / "Downloads" / "PerfectLabels_ICLR2027_supplementary.zip"
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".tex", ".bib", ".sty", ".bst"}

# ----------------------------------------------------------------------------------------------- inputs
# (glob relative to repo root). Everything is filtered again by EXCLUDE and TEXT_EXT.
INCLUDE = [
    "experiments/*.py",
    "paper/exp/*.py",
    "paper/main.tex", "paper/refs.bib", "paper/math_commands.tex", "paper/numbers_placeholder.tex",
    "paper/numbers.tex", "paper/tab_runs.tex", "paper/iclr2027_conference.sty",
    "paper/iclr2027_conference.bst", "paper/fancyhdr.sty", "paper/natbib.sty",
    "PREDICTIONS.md",
    "results/THINKING_VS_DATA.md", "results/GATE_FINETUNE.md", "results/GATE_FINETUNE_STUDY.md",
    "results/LOGIC_FINETUNE.md",
    "results/logic_dataset.json",
    "results/queue_L1.txt", "results/queues/*.txt",
    "results/thinking_vs_data/*.json", "results/thinking_vs_data/*.jsonl",
    "results/v2/*.json", "results/v2/*.txt",
    "results/v2/runs_*.jsonl", "results/v2/evals_*.jsonl",
    "results/v2/gens/**/*.jsonl", "results/v2/star_samples/**/*.jsonl",
    "results/analysis/*",
]
EXCLUDE_RE = re.compile(r"(__pycache__|\.pyc$|\.log$|\.cmd$|\.sh$|(^|/)\.git|KHALIL|failures)", re.I)

# --------------------------------------------------------------------------------------------- scrubbing
# commit hash -> neutral label used in the paper (Appendix: "anonymized identifiers P1--P6").
HASH_LABELS = {"b0370f5": "P1", "fcaa05d": "P2", "27ba807": "P3", "59e3989": "P4", "1163562": "P5",
               "8c0cfff": "P6"}
SEP = r"(?:\\\\|\\|/)"                      # path separator: JSON-escaped backslash, backslash, slash
PATH_SUBS = [
    # <drive>:\Users\<user>\<repo dir>\  ->  ''   (JSON-escaped or raw, either slash)
    (re.compile(r"[A-Za-z]:" + SEP + r"Users" + SEP + r"[^\\/\s\"'`<>|]+" + SEP + r"[^\\/\s\"'`<>|]+" + SEP), ""),
    # /home/<user>/[Projects/]<repo dir>/  ->  ''
    (re.compile(r"/home/[^/\s\"'`]+/(?:Projects/)?[^/\s\"'`]+/"), ""),
]


def _case_sub(pattern, repl_lower):
    """Replace keeping the case style (lower / Title / UPPER) of the match."""
    rx = re.compile(pattern, re.I)

    def f(m):
        s = m.group(0)
        if s.isupper():
            return repl_lower.upper()
        if s[:1].isupper():
            return repl_lower[:1].upper() + repl_lower[1:]
        return repl_lower
    return rx, f


WORD_SUBS = [
    # internal package / product names (identify the authors' other projects)
    (re.compile(r"axuniv\.newaxiom\."), "exactgate."),
    (re.compile(r"axuniv\.newaxiom"), "exactgate"),
    (re.compile(r"newaxiom\."), "exactgate."),
    (re.compile(r"\bNew Axiom\b"), "the exact-gate engine"),
    (re.compile(r"newaxiom", re.I), "exactgate"),
    (re.compile(r"axuniv", re.I), "exactgate"),
    (re.compile(r"axiomatic-universe[\w.-]*", re.I), "prior-repo"),
    _case_sub(r"reality(?=[- ]gat)", "exact"),       # reality-gated / -gate / -gating / reality gate
    # people
    (re.compile(r"Khalil's"), "the RTX 5090 run's"),
    (re.compile(r"\bKhalil\b"), "the RTX 5090 run"),
    # location in time stamps (the time zone is kept; make_numbers.py parses one word there)
    (re.compile(r"\bToronto\b"), "EDT"),
]


def repo_hashes():
    """All commit hashes of the repository (short 7 and full), to strip and to scan for."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "--all", "--format=%H"], capture_output=True,
                             text=True, check=True).stdout.split()
    except Exception:
        out = []
    return out


def scrub_text(s, hashes_short):
    for rx, rep in PATH_SUBS:
        s = rx.sub(rep, s)
    for rx, rep in WORD_SUBS:
        s = rx.sub(rep, s)
    # known hashes -> labels, any other repo hash -> removed
    def hsub(m):
        h = m.group(0)
        for k, lab in HASH_LABELS.items():
            if h.startswith(k):
                return lab
        return "[commit id removed]"
    if hashes_short:
        s = HASH_RX.sub(hsub, s)
    return s


HASH_RX = None   # built in main() from the repo's hashes


# ------------------------------------------------------------------------------------------------- scan
SCAN_ALLOW = ["Alexander Ratner"]        # a cited author in refs.bib whose first name matches a scan term


def scan_terms():
    host = socket.gethostname()
    subs = ["vicol", "alexander", "chaghouri", "khalil", "steve", "alexa", "libritor", "eliteatlantico",
            "yahoo", "gmail", "utoronto", "toronto", "axuniv", "newaxiom", "new axiom", "axiomatic",
            "reality-gat", "reality gat", "may27clean", "logical-truth", "mersiv", "claude-session",
            "claude.ai/code", "github.com/", "gitlab.com/", "/home/", "projects/exact", "\\users\\",
            "/users/", "\\\\users\\\\", host.lower()]
    words = ["mann", "kc"]
    return [re.compile(re.escape(t), re.I) for t in subs if t] + \
           [re.compile(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", re.I) for w in words] + \
           [re.compile(r"[A-Za-z]:(?:\\\\|\\|/)(?:Users|home)", re.I)]


def scan_zip(zpath, hashes):
    terms = scan_terms()
    hash_rx = re.compile(r"(?<![0-9a-f])(?:" + "|".join(sorted({h[:7] for h in hashes})) + r")[0-9a-f]{0,33}(?![0-9a-f])") \
        if hashes else None
    hits = []
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for name in names:
            if re.search(r"(^|/)\.git", name):
                hits.append((name, "<name>", "git metadata"))
            for rx in terms:
                if rx.search(name):
                    hits.append((name, "<name>", rx.pattern))
            txt = z.read(name).decode("utf-8", errors="replace")
            for allowed in SCAN_ALLOW:              # cited third-party authors, not the paper's authors
                txt = txt.replace(allowed, "")
            for rx in terms:
                m = rx.search(txt)
                if m:
                    ctx = txt[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                    hits.append((name, rx.pattern, ctx))
            if hash_rx:
                m = hash_rx.search(txt)
                if m:
                    hits.append((name, "commit hash", txt[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
    return names, hits


# ----------------------------------------------------------------------------------------------- README
def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def prereg_manifest():
    """Hash-free manifest of the commits that touched PREDICTIONS.md: label, commit time (with UTC offset),
    subject (scrubbed). Labels P1..P6 are the paper's; other commits are shown with '-'."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "--reverse", "--format=%h\t%ai\t%s", "--",
                              "PREDICTIONS.md"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return "(git history unavailable when this README was generated)\n"
    rows = ["| label | commit time | subject |", "|---|---|---|"]
    for line in out.strip().splitlines():
        h, t, subj = line.split("\t", 2)
        lab = next((v for k, v in HASH_LABELS.items() if h.startswith(k)), "-")
        subj = subj.replace("|", "/")
        rows.append("| %s | %s | %s |" % (lab, t, subj))
    return "\n".join(rows) + "\n"


def count(pattern, stage):
    return len(glob.glob(str(stage / pattern), recursive=True))


def readme(stage):
    v = {k: pkg_version(k) for k in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy",
                                     "scipy", "scikit-learn", "matplotlib")}
    runs = sorted(os.path.basename(p) for p in glob.glob(str(stage / "results/v2/runs_*.jsonl")))
    evals = sorted(os.path.basename(p) for p in glob.glob(str(stage / "results/v2/evals_*.jsonl")))
    n_rows = 0
    for p in glob.glob(str(stage / "results/v2/*_*.jsonl")) + glob.glob(str(stage / "results/thinking_vs_data/runs*.jsonl")):
        n_rows += sum(1 for line in open(p, encoding="utf-8") if line.strip())
    return f"""# Supplementary material: code, data and run records

Anonymized for double-blind review. Absolute paths were made repository-relative, commit hashes were
replaced by the neutral labels P1..P6 used in the paper (other hashes removed), and internal package and
people names were replaced by neutral words. Nothing else was changed; every table and figure in the paper
is regenerated from the files in this folder.

## Folder guide

| path | contents |
|---|---|
| `experiments/` | all experiment and analysis code. Workers: `exp_thinking_ft_worker.py` (prior grid and S1 cells), `exp_worker_v2.py` (all later cells: arms base/A/B/C/B'/D, eval-only controls), `exp_worker_star.py` (arm S: sample K solutions, keep those whose final answer the exact verifier accepts, fine-tune). Queue runners: `run_queue.py`, `run_queue2.py`. Dataset generators: `exp_thinking_dataset.py`, `exp_dataset_v2.py` (fixed seeds), `check_v2.py` (dataset invariants). Analysis: `analysis_common.py` (loaders), `analysis_stats.py` (bootstrap CIs, McNemar, Holm), `analysis_steps.py` (per-step trace parsing, per-step accuracy q), `analysis_rules.py` (shortcut rules), `analysis_probe.py` (surface probe), `analysis_tokens.py`, `analysis_report.py`. `exp_gate_*` and `exp_logic_*`: the earlier answer-only study reported in the appendix. |
| `paper/exp/` | `build_all.py` (one command: analysis -> numbers -> figures -> PDF), `make_numbers.py` (every number in the text is a macro written to `paper/numbers.tex` and `paper/tab_runs.tex`), `make_figures.py`, `make_fig_story.py`, `numbers.json` (each macro with its source). |
| `paper/` | LaTeX source of the submission and the style files needed to build it. |
| `PREDICTIONS.md` | second preregistration (S1..S6) and its append-only decision log (S7..S11, outcomes, corrections). |
| `results/THINKING_VS_DATA.md` | first preregistration (H1..H4, prior grid) with its results appended below it. |
| `results/GATE_FINETUNE*.md`, `results/LOGIC_FINETUNE.md` | write-ups of the earlier answer-only study (appendix table). |
| `results/thinking_vs_data/` | prior-grid dataset (`datasets.json`), B-ext traces (`ornith_traces.jsonl`), prior-grid runs (`runs.jsonl`) and the S1 replication runs (`runs_L1.jsonl`). |
| `results/v2/datasets_v2.json` | all tasks of the main study: train pools per arm and trace format, problem-disjoint 240-item test sets, 4-shot demos. |
| `results/v2/runs_<model>.jsonl` | one row per training run ({len(runs)} files: {", ".join(runs)}). |
| `results/v2/evals_<model>.jsonl` | eval-only rows (zero-shot CoT, 4-shot, transfer to other test sets) ({", ".join(evals)}). |
| `results/v2/gens/<model>/` | saved generations, one row per test item: id, gold, pred, correct, gen, n_gen_tokens, hit_cap. File name `<task>_<arm>_<n>_<seed>[__on-<eval_task>][__<prompt_mode>][__reeval].jsonl`. |
| `results/v2/star_samples/<model>/` | arm S: the K sampled solutions per training item and which were kept by the verifier. |
| `results/analysis/` | outputs of the analysis scripts (stats, per-step accuracy, shortcut rules, probe, token counts). |
| `results/queues/`, `results/queue_L1.txt` | the exact cell lists that were run (one cell per line: `<task> <arm> <n> <seed> [worker flags]`). |
| `results/logic_dataset.json` | dataset of the earlier logic study. |

Run records: {n_rows} rows in total. Each row has `task, arm, n, seed, acc, n_test, model` and `bits`, a hex
bitmap of per-item correctness over the fixed test set (item order of `datasets_v2.json`), from which every
paired test (McNemar, bootstrap over items and seeds) is recomputed. `adapter` fields are repo-relative;
the LoRA adapters themselves (several hundred MB) are not included and are re-created by training.

## Reproduce one cell (GPU)

```
# arm B, div7, n = 180, seed 0, Qwen2.5-1.5B-Instruct; prints RESULT {{json}}, writes generations + adapter
python experiments/exp_worker_v2.py div7 B 180 0 --model Qwen/Qwen2.5-1.5B-Instruct
# eval-only controls
python experiments/exp_worker_v2.py div7 base 0 0 --eval-only none --prompt-mode cot
python experiments/exp_worker_v2.py div7 B 180 0 --eval-only auto --eval-task div7_6d
# arm S (self-generated, verifier-filtered traces; K = 4, temperature 0.7)
python experiments/exp_worker_star.py div7 S 180 0 --model Qwen/Qwen2.5-3B-Instruct
# prior grid / S1 worker
python experiments/exp_thinking_ft_worker.py div7 B 180 1
# a whole queue, appended to a run file, resumable (skips cells already present)
python experiments/run_queue2.py results/queues/q4_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl \\
    --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-3B-Instruct --eval-bs 32
```

The worker loads the model from the Hugging Face cache with `HF_HUB_OFFLINE=1` (download the model once
beforehand, or unset the variable). Training: LoRA r = 8, alpha = 16 on q/k/v/o, lr 2e-4, 3 epochs,
batch 1, bf16, prompt tokens masked; decoding greedy. Eval-only rows belong in `evals_<model>.jsonl`, not
in `runs_<model>.jsonl` (see `run_queue2.py`).

## Rebuild every table and figure (CPU only)

```
python paper/exp/build_all.py                 # analysis -> paper/numbers.tex, tab_runs.tex -> figures -> PDF
python paper/exp/build_all.py --skip-analysis # reuse results/analysis/*
# or step by step
python experiments/analysis_stats.py ; python experiments/analysis_steps.py ; python experiments/analysis_rules.py
python paper/exp/make_numbers.py ; python paper/exp/make_figures.py
```

`build_all.py` needs pdflatex and bibtex for the PDF step; the numbers and figures need only the Python
packages below. It exits non-zero on an undefined reference, an unfilled number macro, or a failed step.
The dataset generators for prime/valid used a Z3-based consistency checker from an in-house package that is
not included (its module name was replaced by `exactgate`); the generated datasets are included, so no
regeneration is needed to reproduce any cell.

## Environment

- Python {platform.python_version()} (analysis and paper build), {platform.system()} and Linux
- torch {v["torch"]}, transformers {v["transformers"]}, peft {v["peft"]}, accelerate {v["accelerate"]},
  safetensors {v["safetensors"]}
- analysis: numpy {v["numpy"]}, scipy {v["scipy"]}, scikit-learn {v["scikit-learn"]}, matplotlib {v["matplotlib"]}
- versions read from the installed packages when this folder was built; the RTX 5090 runs pinned the same
  torch / transformers / peft versions (CUDA 12.8 wheels).

## Hardware

- NVIDIA RTX 4060 Laptop GPU (8 GB): Qwen2.5-1.5B-Instruct cells (the preregistered model) and part of the
  0.5B cells.
- NVIDIA RTX 5090 desktop GPU (32 GB): Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct cells, 0.5B cells, and
  extra seeds.
- Analysis and the paper build run on CPU.

## Preregistration protocol

1. **First preregistration** (`results/THINKING_VS_DATA.md`): written before any run of the prior grid and
   never edited afterwards; results were appended below it. Endpoints H1..H4 (arms A, B, C on prime, valid,
   div7; 240 problem-disjoint test items; paired McNemar, Holm across tasks).
2. **Second preregistration** (`PREDICTIONS.md`): decision rules S1..S6, committed (P1) before any of the
   runs it governs. S1 was a go/no-go replication of the prior div7 result, run first.
3. **Append-only decision log**: each new hypothesis (S7, S8/S9, S10, S11) was committed to the repository
   before any cell it governs had started, and each outcome was logged when its cells finished, including
   every failed prediction (S5, S10) and a post hoc correction of an audit (S10). Earlier entries were never
   rewritten, except that estimated clock times were replaced by the commit times on the day they were written.
4. The commit is the time stamp. Hashes are replaced by labels here; the times below come from the version
   history (author time with UTC offset). P1..P6 are the labels used in the paper's appendix; commits marked
   `-` changed only wording or time stamps, or appended outcome entries written "time = this commit".

{prereg_manifest()}
5. Analyses not named in either file (the surface probe, extended trace re-audit, derived forms of the
   compounding model) are labelled post hoc in the paper.
"""


# ------------------------------------------------------------------------------------------------- main
def main():
    global HASH_RX
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--max-mb", type=float, default=60.0)
    a = ap.parse_args()

    hashes = repo_hashes()
    if hashes:
        HASH_RX = re.compile(r"(?<![0-9a-f])(?:" + "|".join(sorted({h[:7] for h in hashes})) +
                             r")[0-9a-f]{0,33}(?![0-9a-f])")

    tmp = Path(tempfile.mkdtemp(prefix="supp_"))
    stage = tmp / TOP
    stage.mkdir()
    files = set()
    for pat in INCLUDE:
        for p in glob.glob(str(REPO / pat), recursive=True):
            rel = Path(p).relative_to(REPO).as_posix()
            if os.path.isfile(p) and not EXCLUDE_RE.search(rel) and Path(p).suffix.lower() in TEXT_EXT:
                files.add(rel)
    dropped_lines = 0
    for rel in sorted(files):
        src = REPO / rel
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        txt = src.read_bytes().decode("utf-8")
        if rel.endswith(".jsonl"):
            # a queue may be appending: keep only complete, parseable rows
            keep = []
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    keep.append(line)
                except json.JSONDecodeError:
                    if i == len(lines) - 1:
                        dropped_lines += 1
                        print("  dropped incomplete last line of", rel)
                    else:
                        raise SystemExit("unparseable line %d in %s" % (i + 1, rel))
            txt = "\n".join(keep) + ("\n" if keep else "")
        dst.write_text(scrub_text(txt, hashes), encoding="utf-8", newline="\n")
    (stage / "README_SUPPLEMENT.md").write_text(scrub_text(readme(stage), hashes), encoding="utf-8", newline="\n")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, (Path(TOP) / p.relative_to(stage)).as_posix())

    names, hits = scan_zip(out, hashes)
    size = out.stat().st_size
    raw = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print("zip: %s" % out)
    print("size: %.2f MB compressed (%.2f MB uncompressed), %d files" % (size / 2**20, raw / 2**20, len(names)))
    for d in ("experiments", "paper", "results/v2/gens", "results/v2/star_samples", "results/analysis"):
        print("  %-26s %4d files" % (d, sum(1 for n in names if n.startswith(TOP + "/" + d + "/"))))
    ok = True
    if hits:
        ok = False
        print("ANONYMIZATION SCAN: FAIL (%d hits)" % len(hits))
        for h in hits[:60]:
            print("  ", h)
    else:
        print("ANONYMIZATION SCAN: PASS (%d files, names and contents; %d terms + path/hash patterns, "
              "%d repo commit hashes checked)" % (len(names), len(scan_terms()), len(hashes)))
    if size > a.max_mb * 2**20:
        ok = False
        print("SIZE: FAIL (> %.0f MB)" % a.max_mb)

    if not a.no_verify:
        chk = tmp / "verify"
        shutil.copytree(stage, chk)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, str(chk / "paper/exp/make_numbers.py")], cwd=chk, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env, timeout=1800)
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()[-4:]
        print("VERIFY make_numbers.py on the anonymized copy: %s" % ("ok" if p.returncode == 0 else "FAIL"))
        for line in tail:
            print("   " + line)
        if p.returncode != 0:
            ok = False
    if a.keep_staging:
        print("staging kept at", stage)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
