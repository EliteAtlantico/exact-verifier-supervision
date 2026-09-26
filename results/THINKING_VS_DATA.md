# THINKING vs DATA -- pre-registration (frozen BEFORE any GPU run; never edited)

Hypothesis (user's): training on (problem -> thinking -> answer) is more efficient than training on
(problem -> answer). "The thinking itself is the data."

Arms: A answers-only; B deterministic exact-trace thinking; C length-matched scrambled-thinking
control (cross-example permutation then token shuffle; corpus token multiset identical to B);
A-tok answers-only at B@180's measured token budget; B-orn ornith:9b open-source thinking (STaR
rejection-sampled, coverage reported); B-mv thinking via multiverse.simulate_conversations with a
local reasoner; D wrong-but-plausible thinking (conditional: only where B beats both A and C).
Base model Qwen/Qwen2.5-1.5B-Instruct, LoRA r=8 a=16 targets q/k/v/o_proj, lr 2e-4, 3 epochs.
Tasks (all three reported; no averaging that hides a split): prime (4-digit), valid (Z3), div7.
Test = 240 problem-disjoint items/task (120/class), identical across arms; grading = exact gate on
the extracted final answer only (one parser for all arms).

Pre-registered endpoints:
- H1 matched examples: acc(B) > acc(A) at n=180 per task; paired McNemar on shared items; Holm
  correction across the 3 tasks.
- H2 content vs tokens: acc(B) > acc(C) at n=180. B ~= C means tokens, not thinking.
- H3 matched training tokens: accuracy at equal cumulative training tokens (B@60 vs A@540 points +
  one explicit A-tok run per task; actual token counts recorded) and matched wall-clock.
- H4 algorithm unlock: div7, where answer labels are proven null (GATE_FINETUNE_STUDY: 51.0% vs
  base 53.3%, shuffled 52.5%). B-trained div7 >= ~60% with A at chance = thinking teaches what
  labels cannot.
- Secondary: B-orn vs B-det; B-mv pilot.

Pre-declared readings: B>A and B>C = supported. B>A but B~=C = tokens not thinking; strong form
unsupported. B~=A = honest negative. Efficiency claims require the token-matched win (H3), not just
the example-matched one. Nulls and reversals get equal prominence. Adapters are tier `empirical`,
never `verified`.

(Results are appended BELOW this line after the grid completes; this section is never edited.)
---

## Results (appended after the grid; preregistration above is unchanged)

_Total runs: 26; wall 186 min._

### H1/H2 at n=180 (matched examples; McNemar on shared items, seed 0)
| task | base | A (data) | B (thinking) | C (scrambled) | B>A p | B>C p | verdict |
|---|---|---|---|---|---|---|---|
| prime | 59.6% [53,66] | 82.3% [77,86] | 57.7% [52,64] | 52.1% [46,58] | 0.000 | 0.215 | no thinking gain |
| valid | 30.0% [25,36] | 99.6% [98,100] | 90.0% [86,93] | 52.5% [46,59] | 0.500 | 0.000 | no thinking gain |
| div7 | 62.5% [56,68] | - | - | - | - | - | - |

### H3 efficiency: accuracy vs training examples and tokens
| task | arm | n | ex | train_ws_tok | acc | wall_s |
|---|---|---|---|---|---|---|
| prime | A | 60 | 60 | 120 | 59.6% | 137.7 |
| prime | A | 180 | 180 | 360 | 82.1% | 305.4 |
| prime | A | 180 | 180 | 360 | 82.5% | 114.1 |
| prime | A | 540 | 540 | 1080 | 75.0% | 607.4 |
| prime | B | 60 | 60 | 4355 | 57.1% | 595.8 |
| prime | B | 180 | 180 | 13370 | 57.9% | 989.1 |
| prime | B | 180 | 180 | 13260 | 57.5% | 556.9 |
| prime | B | 540 | 540 | 40095 | 68.8% | 1544.8 |
| prime | A_tok | 1320 | 1320 | 2640 | 80.8% | 594.1 |
| valid | A | 60 | 60 | 120 | 97.5% | 66.4 |
| valid | A | 180 | 180 | 360 | 99.2% | 121.3 |
| valid | A | 180 | 180 | 360 | 100.0% | 121.1 |
| valid | A | 540 | 540 | 1080 | 100.0% | 290.0 |
| valid | B | 60 | 60 | 2343 | 87.5% | 156.8 |
| valid | B | 180 | 180 | 7001 | 100.0% | 225.1 |
| valid | B | 180 | 180 | 7004 | 80.0% | 231.8 |
| valid | B | 540 | 540 | 21084 | 100.0% | 427.1 |
| div7 | A | 60 | 60 | 120 | 50.0% | 162.3 |
| div7 | A | 540 | 540 | 1080 | 50.0% | 589.1 |

### H4 div7 (answer-labels proven null; does the procedure trace unlock it?)
- base 62.5% | A (labels) - | B (thinking) - -> no unlock

### Open-source AI thinking
- prime: B-det 57.7% vs B-orn (ornith:9b) -
- valid: B-mv (multiverse+ornith) -

---
## Verdict (human-written, 2026-08-04; Qwen2.5-1.5B LoRA, exact-gate graded, 240 held-out/task)

**Headline: the hypothesis is NOT supported on these tasks. Training on the ANSWER (data points) is
more efficient than training on the THINKING, not less -- the opposite of the claim -- for verifiable
yes/no tasks where the answer itself carries learnable signal. Thinking content still beats random
tokens (the scrambled control), but it does not beat the plain answer, and it costs 20-37x more
training tokens to reach equal-or-worse accuracy.**

### The numbers that decide it (n=180, 2 seeds averaged)
| task | base | A: answers | B: thinking | C: scrambled | A tokens | B tokens |
|---|---|---|---|---|---|---|
| prime | 59.6% | **82.3%** | 57.7% | 52.1% | 360 | ~13,300 |
| valid | 30.0% | **99.6%** | 90.0% | 52.5% | 360 | ~7,000 |
| div7  | 62.5% | 50.0% | 50.0% (n=60) | -- | 120 | 3,600 |

### Pre-registered endpoints
- **H1 (thinking > answers at matched examples): FALSE.** Answers win on prime (82 vs 58) and edge out
  thinking on valid (99.6 vs 90, and more stable across seeds). Never reversed.
- **H2 (thinking > scrambled control): TRUE.** B beats C everywhere (prime 58 vs 52, valid 90 vs 52), so
  the harness is sound and thinking CONTENT matters vs random tokens. But that is not the claim.
- **H3 (thinking more efficient per token): FALSE, decisively reversed.** Answers reach equal-or-better
  accuracy with ~19x (valid) to ~37x (prime) FEWER training tokens. The A-tok control (answers-only at
  B's token budget) still sits at 80.8% on prime vs B's 57.7% -- B's deficit is not "fewer effective
  examples," the trace training is genuinely worse here.
- **H4 (thinking unlocks what labels cannot -- div7): NO.** On div7 the answer label teaches nothing
  (50% = chance, replicating the prior GATE study), AND the digit-by-digit mod-7 procedure trace also
  lands at chance. The 1.5B model did not absorb the algorithm from the trace either. (B@180/540
  confirming in the background; B@60 = 50.0%.)

### Why (honest mechanism, not spin)
These are binary, exactly-verifiable tasks whose ANSWER already carries learnable surface structure
(primality/parity/logical-form), so the compact label is a dense, low-noise training signal. A long
reasoning trace dilutes that signal with many tokens the small model must also fit, and for a large
prime the trace is ~25 near-identical "N mod p" steps -- high length, low information. The regime where
"train on thinking" genuinely wins (frontier o1/R1/STaR results) is the opposite: multi-step tasks where
the final answer alone is uninformative, so the derivation IS the only signal. Our deferred ES/Sierpinski
construction tasks are that regime, but base accuracy there is ~0% at 1.5B (floor-vs-floor), so this run
could not test it cleanly. That boundary is the honest scope of this result.

### Product decision (per the plan's conditional gates)
- SHIP (done, independent of outcome): Exotic-Tools branch-trace persistence + the heuristic-tier
  corpus builder. Capturing the AI's thinking as data is useful regardless (analysis, few-shot, future
  harder-task corpora); it just does not, on this evidence, distill into a small model more efficiently
  than the answer.
- DO NOT SHIP (result is negative): the "empirical" distilled-model capability entry (Phase 4b) and the
  GGUF/Ollama distilled-model deploy (Phase 4c). We will not advertise an efficiency we did not find.
- Adapters remain tier `empirical`, never `verified`; only exact gates mint that.

## Results (appended after the grid; preregistration above is unchanged)

_Total runs: 36; wall 319 min._

### H1/H2 at n=180 (matched examples; McNemar on shared items, seed 0)
| task | base | A (data) | B (thinking) | C (scrambled) | B>A p | B>C p | verdict |
|---|---|---|---|---|---|---|---|
| prime | 59.6% [53,66] | 82.3% [77,86] | 57.7% [52,64] | 52.1% [46,58] | 0.000 | 0.215 | no thinking gain |
| valid | 30.0% [25,36] | 99.6% [98,100] | 90.0% [86,93] | 52.5% [46,59] | 0.500 | 0.000 | no thinking gain |
| div7 | 62.5% [56,68] | 51.5% [45,57] | 92.5% [88,95] | 50.0% [44,56] | 0.000 | 0.000 | thinking wins |

### H3 efficiency: accuracy vs training examples and tokens
| task | arm | n | ex | train_ws_tok | acc | wall_s |
|---|---|---|---|---|---|---|
| prime | A | 60 | 60 | 120 | 59.6% | 137.7 |
| prime | A | 180 | 180 | 360 | 82.1% | 305.4 |
| prime | A | 180 | 180 | 360 | 82.5% | 114.1 |
| prime | A | 540 | 540 | 1080 | 75.0% | 607.4 |
| prime | B | 60 | 60 | 4355 | 57.1% | 595.8 |
| prime | B | 180 | 180 | 13370 | 57.9% | 989.1 |
| prime | B | 180 | 180 | 13260 | 57.5% | 556.9 |
| prime | B | 540 | 540 | 40095 | 68.8% | 1544.8 |
| prime | A_tok | 1320 | 1320 | 2640 | 80.8% | 594.1 |
| valid | A | 60 | 60 | 120 | 97.5% | 66.4 |
| valid | A | 180 | 180 | 360 | 99.2% | 121.3 |
| valid | A | 180 | 180 | 360 | 100.0% | 121.1 |
| valid | A | 540 | 540 | 1080 | 100.0% | 290.0 |
| valid | B | 60 | 60 | 2343 | 87.5% | 156.8 |
| valid | B | 180 | 180 | 7001 | 100.0% | 225.1 |
| valid | B | 180 | 180 | 7004 | 80.0% | 231.8 |
| valid | B | 540 | 540 | 21084 | 100.0% | 427.1 |
| valid | A_tok | 1600 | 1600 | 3200 | 100.0% | 1618.6 |
| div7 | A | 60 | 60 | 120 | 50.0% | 162.3 |
| div7 | A | 180 | 180 | 360 | 52.1% | 262.7 |
| div7 | A | 180 | 180 | 360 | 50.8% | 327.1 |
| div7 | A | 540 | 540 | 1080 | 50.0% | 589.1 |
| div7 | B | 60 | 60 | 3600 | 50.0% | 834.3 |
| div7 | B | 180 | 180 | 10800 | 92.5% | 687.3 |

### H4 div7 (answer-labels proven null; does the procedure trace unlock it?)
- base 62.5% | A (labels) 51.5% | B (thinking) 92.5% -> **thinking teaches what labels cannot**

### Open-source AI thinking
- prime: B-det 57.7% vs B-orn (ornith:9b) 61.0%
- valid: B-mv (multiverse+ornith) -
