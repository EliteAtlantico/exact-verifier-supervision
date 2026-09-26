#!/usr/bin/env bash
# One rented GPU (A100/H100, >= 40 GB): everything still waiting on a large GPU, in priority order.
# Usage on the instance:  git clone https://github.com/Libritor/exact-verifier-supervision && cd exact-verifier-supervision
#                         pip install -q "transformers>=4.44" peft accelerate   (torch with CUDA must already be installed)
#                         export HF_TOKEN=...   (optional; needed only for Llama-3.2-3B, else SmolLM2 is used)
#                         bash scripts/run_cloud.sh
# Results stay in results/v2/ (no git push needed); copy them back with scp or tar.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"; EVAL_BS="${EVAL_BS:-32}"
M3=Qwen/Qwen2.5-3B-Instruct; M7=Qwen/Qwen2.5-7B-Instruct
LOG=results/v2/run_cloud.log; exec > >(tee -a "$LOG") 2>&1
echo "=== run_cloud.sh start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# 1) erasure vs format at 3B and 7B (retrains A, evaluates with the CoT prompt; rows go to evals files)
"$PY" experiments/run_queue2.py results/queues/q6_cot_big.txt results/v2/evals_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --timeout 3600 --extra --model $M3 --eval-bs "$EVAL_BS"
"$PY" experiments/run_queue2.py results/queues/q6_cot_big.txt results/v2/evals_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 3600 --extra --model $M7 --eval-bs "$EVAL_BS"
# 2) S11 second model family (preregistered)
M="${SECOND_FAMILY_MODEL:-}"
if [ -z "$M" ]; then
  if "$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-3.2-3B-Instruct','config.json')" 2>/dev/null; then
    M=meta-llama/Llama-3.2-3B-Instruct; else M=HuggingFaceTB/SmolLM2-1.7B-Instruct; fi
fi
echo "second family model: $M"; echo "$M" > results/v2/second_family_model.txt
"$PY" experiments/run_queue2.py results/queues/q5_family.txt "results/v2/runs_${M##*/}.jsonl" --worker exp_worker_v2.py --timeout 3600 --extra --model "$M" --eval-bs "$EVAL_BS"
# 3) extra seeds: 7B (valid, D, B', div11), 3B (dose curve, D, B'), 7B arm S seed 1
"$PY" experiments/run_queue2.py results/queues/q4_7b.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
"$PY" experiments/run_queue2.py results/queues/q4_3b.txt results/v2/runs_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py --timeout 7200 --extra --model $M3 --eval-bs "$EVAL_BS"
"$PY" experiments/run_queue2.py results/queues/q4_star.txt results/v2/runs_Qwen2.5-7B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200 --extra --model $M7 --eval-bs "$EVAL_BS"
tar czf cloud_results.tgz results/v2/runs_*.jsonl results/v2/evals_*.jsonl results/v2/gens results/v2/star_samples results/v2/second_family_model.txt results/v2/run_cloud.log 2>/dev/null
echo "=== run_cloud.sh done $(date -Is)  (cloud_results.tgz ready)"
