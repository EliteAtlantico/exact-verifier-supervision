#!/usr/bin/env bash
# Fifth RTX 5090 package: second model family (preregistered as S11 in PREDICTIONS.md before any cell ran).
# Model: meta-llama/Llama-3.2-3B-Instruct if this machine's Hugging Face token can download it (gated),
# otherwise HuggingFaceTB/SmolLM2-1.7B-Instruct (ungated). Both are Llama-architecture, so the worker runs unchanged.
# About 15-20 minutes on an RTX 5090. Resumable; commits and pushes when done.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"; EVAL_BS="${EVAL_BS:-32}"; HOST="${HOSTNAME:-5090}"
PY="$REPO/.venv_5090/bin/python"; [ -x "$PY" ] || PY="${PY_FALLBACK:-python3}"
LOG="$REPO/.run_5090_fifth.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090_fifth.sh start $(date -Is) commit=$(git rev-parse --short HEAD)"
git pull --rebase --autostash https://github.com/Libritor/exact-verifier-supervision main || true
M="${SECOND_FAMILY_MODEL:-}"
if [ -z "$M" ]; then
  if "$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-3.2-3B-Instruct','config.json')" 2>/dev/null; then
    M=meta-llama/Llama-3.2-3B-Instruct; else M=HuggingFaceTB/SmolLM2-1.7B-Instruct; fi
fi
SHORT="${M##*/}"; echo "second family model: $M"; echo "$M" > results/v2/second_family_model.txt
"$PY" experiments/run_queue2.py results/queues/q5_family.txt "results/v2/runs_${SHORT}.jsonl" --worker exp_worker_v2.py --timeout 3600 --extra --model "$M" --eval-bs "$EVAL_BS"
cp "$LOG" "results/v2/run_5090_fifth_${HOST}.log"; git add results/v2 2>/dev/null
git diff --cached --quiet || git -c user.name=khalil-5090 -c user.email=khalil-5090@users.noreply.github.com commit -q -m "5090 fifth: second family ($SHORT)"
git push origin HEAD:main 2>/dev/null || git push origin HEAD:khalil-5090-results 2>/dev/null || true
echo "=== run_5090_fifth.sh done $(date -Is)"
