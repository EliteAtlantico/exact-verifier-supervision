#!/usr/bin/env bash
# Stage 6 on the laptop (replaces stage 5): waits for stage 4's div13 B 540 seed-1 cell, stops stage 4 at the next
# cell boundary, then runs: CoT-prompt evals of answer-only models, the S12 answer-only sweep, the remaining stage-4
# seed cells, and the 0.5B grid. Output: results/laptop_queue5.log.
cd "$(dirname "$0")/.."
STAGE4_PID="${STAGE4_PID:-54900}"
until grep -q "run div7 B 90 2" results/laptop_queue4.log 2>/dev/null; do sleep 10; done
echo "stage 6: div13 B 540 seed 1 finished; stopping stage 4 (cmd pid $STAGE4_PID) at $(date)"
taskkill //F //T //PID "$STAGE4_PID" || true
sleep 8
python experiments/run_queue2.py results/queues/lap_T6_cot.txt results/v2/evals_Qwen2.5-1.5B-Instruct.jsonl --worker exp_worker_v2.py
mkdir -p results/v2/sweep
for q in div7_lr5e-5_ep30 div7_lr2e-5_ep30 div7_lr5e-5_ep10 div7_lr2e-5_ep10 div3_lr5e-5_ep10; do
  python experiments/run_queue2.py "results/queues/sweepA/$q.txt" "results/v2/sweep/sweepA_$q.jsonl" --worker exp_worker_v2.py --timeout 3600
done
echo "S12 SWEEP DONE $(date)"
python experiments/run_queue2.py results/queues/lap_T8_rest.txt results/v2/runs_Qwen2.5-1.5B-Instruct.jsonl --worker exp_worker_v2.py
python experiments/run_queue2.py results/queues/lap_T7_05b.txt results/v2/runs_Qwen2.5-0.5B-Instruct.jsonl --worker exp_worker_v2.py --extra --model Qwen/Qwen2.5-0.5B-Instruct
echo "LAPTOP STAGE 6 DONE $(date)"
