#!/usr/bin/env bash
# Third RTX 5090 package (same setup as run_5090.sh / run_5090_more.sh). Resumable; commits per stage; pushes.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; EVAL_BS="${EVAL_BS:-32}"
PY="$REPO/.venv_5090/bin/python"; [ -x "$PY" ] || PY="${PY_FALLBACK:-python3}"
LOG="$REPO/.run_5090_third.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090_third.sh start $(date -Is) commit=$(git rev-parse --short HEAD)"
git pull --rebase --autostash origin main || true
commit_stage() { cp "$LOG" "results/v2/run_5090_third_$(hostname).log"; git add results/v2 2>/dev/null
  git diff --cached --quiet || git -c user.name=khalil-5090 -c user.email=khalil-5090@users.noreply.github.com commit -q -m "5090 third: $1 ($(hostname))"
  git push origin HEAD:main 2>/dev/null || git push origin HEAD:khalil-5090-results 2>/dev/null || true; }
M3=Qwen/Qwen2.5-3B-Instruct; M7=Qwen/Qwen2.5-7B-Instruct
"$PY" experiments/run_queue2.py results/queues/q3_7b.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
commit_stage "7B baselines, seeds, controls"
"$PY" experiments/run_queue2.py results/queues/q3_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --extra --model $M3 --eval-bs "$EVAL_BS"
commit_stage "3B dose curve"
"$PY" experiments/run_queue2.py results/queues/q3_star_7b.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
commit_stage "7B arm S"
"$PY" experiments/run_queue2.py results/queues/q3_star_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200 --extra --model $M3 --eval-bs "$EVAL_BS"
commit_stage "3B arm S"
echo "=== run_5090_third.sh done $(date -Is)"
