# RTX 5090 run for the exact-verifier-supervision paper (Khalil)

One script covers everything: it installs dependencies, downloads the models, runs the queues, commits and
pushes. It can be resumed: if it dies, run the same command again and finished cells are skipped.

## Requirements
- Linux, NVIDIA driver recent enough for CUDA 12.8 (Blackwell / sm_120), Python >= 3.10 with `venv`.
- Disk: about 7 GB for the 3B model, 1 GB for 0.5B, 15 GB more for 7B, 3 GB for the venv, 0.5 GB for adapters.
- RAM: 16 GB (32 GB if `RUN_7B=1`). Nothing else should be using the GPU.

## What to run
```bash
git clone https://github.com/Libritor/exact-verifier-supervision
cd exact-verifier-supervision
git checkout main
bash scripts/run_5090.sh                  # default: 3B grid, 0.5B, 3B extras
# or, if time allows, also the two 7B cells:
RUN_7B=1 bash scripts/run_5090.sh
```
Options: `USE_SYSTEM_PY=1` uses your existing python3 (with torch cu128 already installed) instead of
creating `.venv_5090`; `SKIP_INSTALL=1` skips pip; `RUN_EXTRA=0` skips the extras; `NO_PUSH=1` commits
without pushing; `HF_HOME=/big/disk/hf` puts the model cache somewhere else.

## What it runs (in this order)
| stage | queue file | model | cells |
|---|---|---|---|
| 1 | results/queues/q_5090_3b.txt | Qwen2.5-3B-Instruct | div7, div13, prime, valid x A/B x seeds 0,1 at n=180 (16) |
| 2 (only if RUN_7B=1) | results/queues/q_5090_7b.txt | Qwen2.5-7B-Instruct | div7 A/B seed 0 (2) |
| 3 | results/queues/q_5090_05b.txt | Qwen2.5-0.5B-Instruct | div7, prime x A/B seed 0 (4) |
| 4 (extras, RUN_EXTRA=1) | q_5090_3b_extra_base.txt, q_5090_3b_extra_evals.txt | 3B | 4 zero-shot baselines; 7 eval-only cells on the saved 3B adapters (prime to prime_hard, div7 to div7_6d) |

Every cell is one LoRA fine-tune (r=8, 3 epochs, bf16) plus greedy evaluation on 240 held-out items, with
`--eval-bs 32`. The data is the committed `results/v2/datasets_v2.json`; the script never regenerates it.

## Expected time
Setup, including the 3B and 0.5B downloads: 10-20 min. 3B grid: about 1-1.5 h (2-6 min per cell; prime B
takes longest). 0.5B: about 10 min. Extras: about 20-30 min. 7B, if enabled: another 20-40 min. Total is
roughly 2-2.5 h without 7B and 2.5-3 h with it. These figures are extrapolated from the laptop runs and are
not measured on a 5090.

## What comes back (pushed automatically)
The script commits `results/v2/` after each stage, then runs `git pull --rebase` and `git push origin
HEAD:main`. If that push is rejected, it pushes to the branch `khalil-5090-results`. If both fail, it writes
`results_v2_<host>.tgz`; please send us that file.
- `results/v2/runs_Qwen2.5-3B-Instruct.jsonl`, `runs_Qwen2.5-0.5B-Instruct.jsonl` (and `runs_Qwen2.5-7B-Instruct.jsonl`):
  one RESULT row per cell (accuracy, per-item correctness bitmap, token counts, timings).
- `results/v2/evals_Qwen2.5-3B-Instruct.jsonl`: rows from the eval-only extras.
- `results/v2/gens/<model>/*.jsonl`: the full generated text for every test item (needed for per-step scoring).
- `results/v2/run_5090_<host>.log` and any `*.failures.log`, which hold the stderr of failed cells.

LoRA adapters are written to `results/adapters/<model>/` and are gitignored (about 15 MB each for 3B). Please
keep them until the paper is submitted. If we ask for them:
`tar czf adapters_5090.tgz results/adapters`.

## If something fails
- `CUDA check failed`: the torch build lacks sm_120. Run
  `.venv_5090/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`
  and then rerun the script.
- A single cell `FAILED`: the queue continues. The traceback is in the log and in `results/v2/*.failures.log`.
  Rerunning retries only the missing cells.
- Out of memory on 7B: rerun with `EVAL_BS=8 RUN_7B=1 bash scripts/run_5090.sh`.
