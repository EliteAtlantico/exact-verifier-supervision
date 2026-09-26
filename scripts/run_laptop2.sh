#!/usr/bin/env bash
# Stage 2 on the laptop: waits for run_laptop.sh to finish, then the 0.5B grid.
cd "$(dirname "$0")/.."
until grep -q "LAPTOP QUEUE DONE" results/laptop_queue.log 2>/dev/null; do sleep 30; done
python experiments/run_queue2.py results/queues/lap_T4_05b.txt results/v2/runs_Qwen2.5-0.5B-Instruct.jsonl --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-0.5B-Instruct
echo "LAPTOP STAGE 2 DONE $(date)"
