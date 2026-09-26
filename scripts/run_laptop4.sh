#!/usr/bin/env bash
# Stage 4 on the laptop: waits for stage 3 (arm S; its 0.5B queue was emptied), then extra 1.5B seeds for the
# single-seed dose/S7/S9 cells, then the 0.5B grid. Output goes to results/laptop_queue4.log.
cd "$(dirname "$0")/.."
until grep -q "LAPTOP STAGE 3 DONE" results/laptop_queue3.log 2>/dev/null; do sleep 30; done
python experiments/run_queue2.py results/queues/lap_T5_seeds15.txt results/v2/runs_Qwen2.5-1.5B-Instruct.jsonl --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_T4b_05b.txt results/v2/runs_Qwen2.5-0.5B-Instruct.jsonl --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-0.5B-Instruct
echo "LAPTOP STAGE 4 DONE $(date)"
