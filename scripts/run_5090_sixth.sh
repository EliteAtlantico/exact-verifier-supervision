#!/usr/bin/env bash
# Sixth RTX 5090 package (~25 min): (1) the reviewers' erasure-vs-format check at 3B and 7B -- answer-only (A) models
# retrained and evaluated with the CoT prompt; (2) S11, a second model family (preregistered in PREDICTIONS.md).
# Commits and pushes after each stage. Paper files are generated on Alexander's side: if git reports conflicts in
# paper/, keep the Libritor version (git checkout --theirs paper/ ; git add paper/ ; git commit --no-edit).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; EVAL_BS="${EVAL_BS:-32}"; HOST="${HOSTNAME:-5090}"
PY="$REPO/.venv_5090/bin/python"; [ -x "$PY" ] || PY="${PY_FALLBACK:-python3}"
LOG="$REPO/.run_5090_sixth.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090_sixth.sh start $(date -Is) commit=$(git rev-parse --short HEAD)"
commit_stage() { cp "$LOG" "results/v2/run_5090_sixth_${HOST}.log"; git add results/v2 2>/dev/null
  git diff --cached --quiet || git -c user.name=khalil-5090 -c user.email=khalil-5090@users.noreply.github.com commit -q -m "5090 sixth: $1"
  git push origin HEAD:main 2>/dev/null || git push origin HEAD:khalil-5090-results 2>/dev/null || true; }
M3=Qwen/Qwen2.5-3B-Instruct; M7=Qwen/Qwen2.5-7B-Instruct
"$PY" experiments/run_queue2.py results/queues/q6_cot_big.txt results/v2/evals_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --timeout 3600 --extra --model $M3 --eval-bs "$EVAL_BS"
"$PY" experiments/run_queue2.py results/queues/q6_cot_big.txt results/v2/evals_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 3600 --extra --model $M7 --eval-bs "$EVAL_BS"
commit_stage "3B/7B answer-only models under the CoT prompt"
bash scripts/run_5090_fifth.sh
echo "=== run_5090_sixth.sh done $(date -Is)"
