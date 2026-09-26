#!/usr/bin/env bash
# Second RTX 5090 package. Reuses the venv and model cache from scripts/run_5090.sh (run that first, or at least
# its setup). Resumable; commits after each stage and pushes like run_5090.sh.
#   RUN_7B=1 bash scripts/run_5090_more.sh
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; RUN_7B="${RUN_7B:-0}"; EVAL_BS="${EVAL_BS:-32}"
PY="$REPO/.venv_5090/bin/python"; [ -x "$PY" ] || PY="${PY_FALLBACK:-python3}"
LOG="$REPO/.run_5090_more.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090_more.sh start $(date -Is) host=$(hostname) commit=$(git rev-parse --short HEAD)"
git pull --rebase --autostash origin main || true
dl() { HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$1"; }
dl Qwen/Qwen2.5-3B-Instruct || { echo "3B download failed"; exit 1; }
[ "$RUN_7B" = "1" ] && { dl Qwen/Qwen2.5-7B-Instruct || echo "WARN 7B download failed"; }
commit_stage() { cp "$LOG" "results/v2/run_5090_more_$(hostname).log"; git add results/v2 2>/dev/null
  git diff --cached --quiet || git -c user.name=khalil-5090 -c user.email=khalil-5090@users.noreply.github.com commit -q -m "5090 more: $1 ($(hostname))"; }
M3=Qwen/Qwen2.5-3B-Instruct; M7=Qwen/Qwen2.5-7B-Instruct
"$PY" experiments/run_queue2.py results/queues/q_5090_more_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --extra --model $M3 --eval-bs "$EVAL_BS"
commit_stage "3B family + controls"
if [ "$RUN_7B" = "1" ]; then
  "$PY" experiments/run_queue2.py results/queues/q_5090_more_7b.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
  commit_stage "7B core"
fi
"$PY" experiments/run_queue2.py results/queues/q_5090_more_star.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200 --extra --model $M3 --eval-bs "$EVAL_BS"
commit_stage "3B arm S"
if [ "${NO_PUSH:-0}" != "1" ]; then
  git pull --rebase --autostash origin main && git push origin HEAD:main && { echo "PUSHED to main"; exit 0; }
  git push origin HEAD:khalil-5090-results && { echo "PUSHED to branch khalil-5090-results"; exit 0; }
fi
tar czf "results_v2_more_$(hostname).tgz" results/v2 && echo "NOT PUSHED: send results_v2_more_$(hostname).tgz"
