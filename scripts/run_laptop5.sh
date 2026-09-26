#!/usr/bin/env bash
# Stage 5 on the laptop: waits for stage 4, then CoT-prompt evaluations of answer-only models (erasure vs format),
# then the 0.5B grid. Output goes to results/laptop_queue5.log.
cd "$(dirname "$0")/.."
until grep -q "LAPTOP STAGE 4 DONE" results/laptop_queue4.log 2>/dev/null; do sleep 30; done
python experiments/run_queue2.py results/queues/lap_T6_cot.txt results/v2/evals_Qwen2.5-1.5B-Instruct.jsonl --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_T7_05b.txt results/v2/runs_Qwen2.5-0.5B-Instruct.jsonl --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-0.5B-Instruct
echo "LAPTOP STAGE 5 DONE $(date)"
