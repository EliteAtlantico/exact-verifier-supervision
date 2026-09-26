# Reality-gated LoRA fine-tune on this computer (measured)

A LoRA adapter on **Qwen2.5-1.5B-Instruct** trained with **only exact-gate-verified labels** (the oracle: is n divisible by 7?), on this machine's GPU (RTX 4060, 8.6 GB). Zero human labels; held-out numbers are **disjoint** from training numbers, so this tests generalization.

| | held-out accuracy |
|---|---|
| base Qwen2.5-1.5B-Instruct | **83.9%** |
| + reality-gated LoRA (210 train, 3 epochs) | **92.9%** |
| delta | **+8.9 pts** |

- Trained on this computer in **101 s** on the RTX 4060 GPU. $0 (no API).
- Labels are 100%% correct by construction (an exact gate) -- the data is `verified`; the trained adapter is tier **`empirical`** (a measured fine-tune), never `verified`.
- Adapter saved: `C:\Users\alexa\axiomatic-universe-may27clean\results\gate_lora_adapter`.
- Honest read: this is a real reality-gated fine-tune of a real LLM. Whether a 1.5B model can *generalize* a math rule from gate labels is exactly what the disjoint held-out delta measures -- reported as-is, not spun.