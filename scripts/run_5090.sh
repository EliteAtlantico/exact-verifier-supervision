#!/usr/bin/env bash
# run_5090.sh -- Khalil's RTX 5090 package for the exact-verifier-supervision paper (Linux, bash).
#
#   git clone https://github.com/Libritor/exact-verifier-supervision && cd exact-verifier-supervision
#   bash scripts/run_5090.sh                 # 3B grid, then 0.5B, then the optional 3B extras
#   RUN_7B=1 bash scripts/run_5090.sh        # also the 7B div7 A/B cells (needs ~16 GB VRAM, ~20 GB RAM)
#
# env knobs: RUN_7B=0|1 (default 0)  RUN_EXTRA=0|1 (default 1)  USE_SYSTEM_PY=0|1 (default 0: make .venv_5090)
#            PY=python3  EVAL_BS=32  SKIP_INSTALL=0|1  NO_PUSH=0|1  HF_HOME (model cache location)
# Resumable: re-running skips cells already in the output jsonl files.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RUN_7B="${RUN_7B:-0}"
RUN_EXTRA="${RUN_EXTRA:-1}"
EVAL_BS="${EVAL_BS:-32}"
PY="${PY:-python3}"
LOG="$REPO/.run_5090.log"            # live log stays untracked (a tracked, growing file blocks git pull)
mkdir -p "$REPO/results/v2"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_5090.sh start $(date -Is) host=$(hostname) commit=$(git rev-parse --short HEAD)"

# ---------------------------------------------------------------- python env
if [ "${USE_SYSTEM_PY:-0}" != "1" ]; then
  if [ ! -x .venv_5090/bin/python ]; then
    "$PY" -m venv .venv_5090 || { echo "venv failed (apt install python3-venv?)"; exit 1; }
  fi
  PY="$REPO/.venv_5090/bin/python"
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || { echo "need Python >= 3.10"; exit 1; }
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  "$PY" -m pip install -U pip
  # torch 2.11.0 is what the laptop runs; cu128 wheels support Blackwell (sm_120). Fall back to latest cu128.
  "$PY" -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 \
    || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    || { echo "torch install failed"; exit 1; }
  "$PY" -m pip install "transformers==5.12.1" "peft==0.19.1" accelerate huggingface_hub safetensors \
    || { echo "transformers/peft install failed"; exit 1; }
fi
"$PY" - <<'EOF' || { echo "CUDA check failed"; exit 1; }
import torch, transformers, peft
assert torch.cuda.is_available(), "no CUDA device visible"
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| bf16", torch.cuda.is_bf16_supported(), "| transformers", transformers.__version__, "| peft", peft.__version__)
torch.ones(8, device="cuda", dtype=torch.bfloat16).sum().item()      # fails fast on an sm_120-less build
EOF

# ---------------------------------------------------------------- data (committed; never regenerated here)
test -f results/v2/datasets_v2.json || { echo "results/v2/datasets_v2.json missing -- pull main"; exit 1; }
sha256sum results/v2/datasets_v2.json

# ---------------------------------------------------------------- models
download() {  # $1 = HF id
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PY" -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$1"
}
download Qwen/Qwen2.5-3B-Instruct || { echo "3B download failed"; exit 1; }
download Qwen/Qwen2.5-0.5B-Instruct || echo "WARN: 0.5B download failed; its queue will fail"
if [ "$RUN_7B" = "1" ]; then download Qwen/Qwen2.5-7B-Instruct || echo "WARN: 7B download failed"; fi

# ---------------------------------------------------------------- queues
commit_stage() {  # local commit after each stage so a crash never loses finished cells
  cp "$LOG" "results/v2/run_5090_$(hostname).log"
  git add results/v2 2>/dev/null
  if ! git diff --cached --quiet; then
    git -c user.name="${GIT_AUTHOR_NAME:-khalil-5090}" -c user.email="${GIT_AUTHOR_EMAIL:-khalil-5090@users.noreply.github.com}" \
      commit -q -m "5090 results: $1 ($(hostname), $(date -Is))" && echo "committed: $1"
  fi
}
runq() {  # $1 queue file  $2 HF model id
  local short; short="$(basename "$2")"
  echo "=== queue $1 model $2 -> results/v2/runs_${short}.jsonl  $(date -Is)"
  "$PY" experiments/run_queue.py "$1" "results/v2/runs_${short}.jsonl" --worker exp_worker_v2.py \
    --extra --model "$2" --eval-bs "$EVAL_BS"
  commit_stage "$(basename "$1") ${short}"
}
runq results/queues/q_5090_3b.txt Qwen/Qwen2.5-3B-Instruct
if [ "$RUN_7B" = "1" ]; then runq results/queues/q_5090_7b.txt Qwen/Qwen2.5-7B-Instruct; fi
runq results/queues/q_5090_05b.txt Qwen/Qwen2.5-0.5B-Instruct
if [ "$RUN_EXTRA" = "1" ]; then
  runq results/queues/q_5090_3b_extra_base.txt Qwen/Qwen2.5-3B-Instruct
  echo "=== eval-only extras -> results/v2/evals_Qwen2.5-3B-Instruct.jsonl  $(date -Is)"
  "$PY" experiments/run_queue2.py results/queues/q_5090_3b_extra_evals.txt \
    results/v2/evals_Qwen2.5-3B-Instruct.jsonl --worker exp_worker_v2.py \
    --extra --model Qwen/Qwen2.5-3B-Instruct --eval-bs "$EVAL_BS"
  commit_stage "3B eval-only extras"
fi

# ---------------------------------------------------------------- push back
echo "=== summary"
for f in results/v2/runs_*.jsonl results/v2/evals_*.jsonl; do
  [ -f "$f" ] && echo "$f: $(wc -l < "$f") rows"
done
ls results/v2/*.failures.log 2>/dev/null && echo "(see failures logs above)"
cp "$LOG" "results/v2/run_5090_$(hostname).log"
git add results/v2
git diff --cached --quiet || git -c user.name="${GIT_AUTHOR_NAME:-khalil-5090}" \
  -c user.email="${GIT_AUTHOR_EMAIL:-khalil-5090@users.noreply.github.com}" commit -q -m "5090 results (final, $(hostname))"
if [ "${NO_PUSH:-0}" != "1" ]; then
  git pull --rebase --autostash origin main && git push origin HEAD:main && { echo "PUSHED to main"; exit 0; }
  echo "push to main failed; trying branch khalil-5090-results"
  git push origin HEAD:khalil-5090-results && { echo "PUSHED to branch khalil-5090-results"; exit 0; }
fi
tar czf "results_v2_$(hostname).tgz" results/v2 && echo "NOT PUSHED: send results_v2_$(hostname).tgz instead"
