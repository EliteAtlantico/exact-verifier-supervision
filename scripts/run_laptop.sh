#!/usr/bin/env bash
# Laptop queue after the S1 checkpoint (Qwen2.5-1.5B, worker v2). Resumable.
cd "$(dirname "$0")/.."
until grep -q "^exit=" results/L1.log 2>/dev/null; do sleep 20; done
M=Qwen2.5-1.5B-Instruct
R=results/v2/runs_$M.jsonl; E=results/v2/evals_$M.jsonl
python experiments/run_queue2.py results/queues/lap_T1.txt $R --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_E1.txt $E --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_T2.txt $R --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_T3.txt $R --worker exp_worker_v2.py
echo "LAPTOP QUEUE DONE $(date)"
