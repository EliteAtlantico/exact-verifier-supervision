# Reality-gated fine-tuning on LOGIC / proof-forming (local GPU, $0)

Does the exact-gate label source help on **logical reasoning**, not just arithmetic? LoRA fine-tunes of **Qwen2.5-1.5B-Instruct** on **Z3-verified** labels, vs two controls. The gate is `newaxiom.contradiction` (Z3): an argument `P1..Pn |- C` is **valid iff `{P1,...,Pn, not C}` is UNSAT**; a statement set is **consistent iff SAT**. Held-out test shares argument FORMS but never the same (form, atom-assignment) instance, so the model must learn the **logical form**, not memorize a prompt. 240 train / 240 test per task, 3 epochs, 2 seeds (mean), 95% CIs. RTX 4060 Laptop GPU; zero human labels; no API; each run isolated so the 8 GB GPU never fragments.

| task | base | **gate-labels** | shuffled (control) | gate − base | gate − shuffled |
|---|---|---|---|---|---|
| valid (argument validity / proof-forming) | 66.7% | **99.4% ± 1.0** | 50.0% | +32.7 pts | +49.4 pts |
| consist (joint consistency (SAT)) | 50.0% | **99.8% ± 0.6** | 48.5% | +49.8 pts | +51.3 pts |

## The read

- **Same reality-gate, new domain.** Arithmetic labels came from `n %% k == 0` / primality gates; here the label source is a **Z3 entailment/SAT gate** -- exactly the engine New Axiom already uses to find contradictions. It supervises **logical validity** at $0 with no human labels and nothing to hack.
- **The shuffled-label control isolates correctness.** Same prompts, same tuning budget, WRONG labels -> no learning. Wherever gate-labels beat it, the gain is attributable to the labels being **exactly correct** (the gate's contribution), not to fine-tuning artifacts or format exposure.
- **On learnable logical form** reality-gated tuning moves held-out accuracy **+41.2 pts vs base** and **+50.3 pts vs the shuffled control** on average across the two tasks.
- Honest scope: a 1.5B model learns the **surface form** of valid vs invalid inference (modus-ponens shape = valid, affirming-the-consequent shape = invalid), not a general theorem prover. The gate supplies perfect labels; it does not grant capabilities the small model structurally lacks.

## Why this is the interesting test (proof-forming)

Validity is **entailment** -- the same relation a proof establishes. Labeling it needs a decision procedure that is *exactly right every time*, which is what an exact gate is and a reward model is not. This is the training-side mirror of the project's **backpropagator** (`bidirectional_proof.search_backward` / `theorem_generator.proof_pursuit`): that component searches *backward* from a target to its premises; this study trains a model to *recognize* when such a backward chain exists. Both are gated by the same exact logic.

## Honest tiering

Training **data** is `verified` (each label is an exact Z3 decision); a trained **adapter** is `empirical` (a measured fine-tune), never `verified`. `experiments/exp_logic_dataset.py` (Z3 label generator) + `exp_logic_ft_worker.py` (isolated run) + `exp_logic_finetune_study.py` (this driver).

Total wall-clock: 1328 s across 10 isolated runs.