#!/usr/bin/env bash
# Stage 3 on the laptop (replaces stage 2): waits for run_laptop.sh to finish, then arm S (STaR), then the 0.5B grid.
cd "$(dirname "$0")/.."
until grep -q "LAPTOP QUEUE DONE" results/laptop_queue.log 2>/dev/null; do sleep 30; done
# --timeout 7200: prime S (sampling 720 completions at cap 512, then a CoT-length eval) can pass the 1 h default.
python experiments/run_queue2.py results/queues/lap_S.txt results/v2/runs_Qwen2.5-1.5B-Instruct.jsonl --worker exp_worker_star.py --timeout 7200
python experiments/run_queue2.py results/queues/lap_T4_05b.txt results/v2/runs_Qwen2.5-0.5B-Instruct.jsonl --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-0.5B-Instruct
echo "LAPTOP STAGE 3 DONE $(date)"
