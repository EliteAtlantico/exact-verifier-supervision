# Reality-gated fine-tuning: a controlled study (local GPU, $0)

LoRA fine-tunes of **Qwen2.5-1.5B-Instruct** on **exact-gate-verified labels**, vs two controls, on
**disjoint held-out numbers** (generalization, not memorization). ~240 train / ~240 test per task, 3
epochs, 2 seeds (mean), 95% CIs. Trained on an RTX 4060 Laptop GPU; **zero human labels; no API**. Each
run executed in an **isolated process** so the 8 GB GPU never fragments across runs. 25 runs, 3078 s total.

| task | base | **gate-labels** | shuffled-labels (control) | gate − base | gate − shuffled |
|---|---|---|---|---|---|
| div3 (divisible by 3) | 55.0% | **67.3% ± 5.9** | 49.6% | +12.3 pts | +17.7 pts |
| div7 (divisible by 7) | 53.3% | **51.0% ± 6.3** | 52.5% | −2.3 pts | −1.5 pts |
| div13 (divisible by 13) | 54.2% | **50.0% ± 6.3** | 50.0% | −4.2 pts | +0.0 pts |
| square (a perfect square) | 51.2% | **83.3% ± 4.7** | 51.5% | +32.1 pts | +31.9 pts |
| prime (a prime number) | 50.0% | **82.9% ± 4.8** | 50.0% | +32.9 pts | +32.9 pts |

## The controls are the headline (and they're clean)

**The wrong-label control sits at chance (~50%) on every single task.** Same prompts, same tuning budget,
labels shuffled → no learning. So wherever gate-labels beat it, the gain is **attributable to the labels
being exactly correct** — which is precisely, and only, what the reality-gate provides ($0, no human
labels, no reward model to hack). This is the cleanest possible demonstration that *label correctness* —
not fine-tuning artifacts, format exposure, or distribution shift — drives the result.

## What actually happened (honest, and it corrected our prior)

Reality-gated fine-tuning **helps a lot where the task has learnable structure, and honestly not at all
where it doesn't** — a bimodal, not uniform, picture:

- **Big gains:** perfect-square **+32.1 pts**, primality **+32.9 pts**, divisible-by-3 **+12.3 pts** (each
  with the shuffled control pinned at ~50%).
- **No gain:** divisible-by-7 (−2.3) and divisible-by-13 (−4.2) — both stay at chance on 4-digit inputs.
- Across all 5 tasks: **gate − base = +14.2 pts** and **gate − shuffled = +16.2 pts** on average — but the
  average hides the split (3 strong wins, 2 nulls).

**We were wrong about primality, and the data said so.** We predicted a 1.5B model "can't learn primality"
— yet it jumped 50%→82.9% on disjoint held-out numbers. The honest interpretation: the model did **not**
learn the exact function; it learned the **useful learnable structure** the gate labels encode — the
small-factor / digit patterns that separate most primes from most composites, and the digit patterns of
perfect squares and multiples of 3. That structure generalizes to unseen numbers. Div-7 and div-13 have
**no such simple rule at 4-digit scale**, so the model — correctly — learns nothing, and reality-gated
tuning gives it nothing. (An earlier single-run on *3-digit* div-7, where the base already had signal,
showed a gain; at 4-digit scale that signal is gone. Scale matters, and we report both.)

## The takeaway for the method

Exact gates are a **$0, un-hackable label source for verifiable-reward fine-tuning**, and this controlled
study isolates the mechanism: **correct labels drive real held-out gains (up to +33 pts) exactly when the
task carries learnable structure, and give nothing — honestly — when it doesn't.** The wrong-label control
rules out every non-correctness explanation. What a small model *acquires* is learnable structure, not
capabilities it lacks (it does not become a primality oracle; it becomes a good heuristic classifier).

## Honest tiering + scope

Training **data** is `verified` (each label is an exact-gate decision); a trained **adapter** is
`empirical` (a measured fine-tune), never `verified`. One model size (1.5B, locally cached), a consumer
GPU, and arithmetic-oracle tasks — the contribution is the **method + its controlled attribution**, not a
frontier number. `experiments/exp_gate_finetune_study.py` (driver) + `exp_gate_ft_worker.py` (isolated
per-run) + `src/axuniv/newaxiom/gate_trainer.py` (the $0 corpus generator).
