#!/usr/bin/env bash
# Fourth RTX 5090 package: extra seeds for single-seed headline cells. Same setup as run_5090_third.sh.
# Resumable (re-run skips finished cells); commits and pushes after each stage. About 30 minutes on an RTX 5090.
# 1.5B results go to their own file (runs_Qwen2.5-1.5B-Instruct_5090.jsonl) so they never conflict with the laptop's.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; EVAL_BS="${EVAL_BS:-32}"; HOST="${HOSTNAME:-5090}"
PY="$REPO/.venv_5090/bin/python"; [ -x "$PY" ] || PY="${PY_FALLBACK:-python3}"
LOG="$REPO/.run_5090_fourth.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090_fourth.sh start $(date -Is) commit=$(git rev-parse --short HEAD)"
git pull --rebase --autostash https://github.com/Libritor/exact-verifier-supervision main || true
commit_stage() { cp "$LOG" "results/v2/run_5090_fourth_${HOST}.log"; git add results/v2 2>/dev/null
  git diff --cached --quiet || git -c user.name=khalil-5090 -c user.email=khalil-5090@users.noreply.github.com commit -q -m "5090 fourth: $1"
  git push origin HEAD:main 2>/dev/null || git push origin HEAD:khalil-5090-results 2>/dev/null || true; }
M15=Qwen/Qwen2.5-1.5B-Instruct; M3=Qwen/Qwen2.5-3B-Instruct; M7=Qwen/Qwen2.5-7B-Instruct
# 1.5B seeds (q4_15b.txt) moved to the laptop (scripts/run_laptop4.sh); not run here, to avoid duplicate cells.
"$PY" experiments/run_queue2.py results/queues/q4_7b.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
commit_stage "7B seed 1 (valid, D, B', div11)"
"$PY" experiments/run_queue2.py results/queues/q4_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M3 --eval-bs "$EVAL_BS"
commit_stage "3B seed 1 (dose curve, D, B')"
"$PY" experiments/run_queue2.py results/queues/q4_star.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
commit_stage "7B arm S seed 1"
echo "=== run_5090_fourth.sh done $(date -Is)"
