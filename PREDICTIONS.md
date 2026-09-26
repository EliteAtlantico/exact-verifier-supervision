# Preregistered predictions and decision rules (frozen before any run listed here)

Committed 2026-09-25 ~20:35 Toronto, before any of the runs below were started. The commit hash of this file is
the preregistration. Earlier results (results/THINKING_VS_DATA.md, frozen 2026-08-04, and its later staged
results block) are prior data, reported as such.

## Provenance of the prior div7 result (stated in the paper)

The 2026-08-04 verdict "H4 (a procedure trace unlocks div7): NO" was written from the n = 60 trace run alone
(B@60 = 50.0%). The preregistered main cell, B@180 seed 0, finished afterwards at 92.5% with the same worker
(no code change). The paper reports the premature verdict and the later result.

## Claim under test

Exact labels teach shortcuts; exact traces teach short programs. With labels from an exact verifier:
- answer-only supervision (arm A) recovers roughly what a simple probe on the answer surface recovers;
- trace supervision (arm B: procedure trace, then answer) succeeds when the per-step trace accuracy p,
  compounded over k steps, stays high (predicted accuracy about p^k), and fails when the procedure is long.

## Decision rules

- **S1 (go/no-go, before any other new run):** div7 B@180 seeds 1 and 2 each reach >= 80% accuracy, with
  B - A >= 20 percentage points and McNemar p < 0.01 against A of the same seed. If S1 fails, the paper is not
  submitted in this form.
- **S2:** at least 2 of {div3, div11, div13} (4-digit numbers) give B - A >= 15 pp with the same sign in both seeds.
- **S3:** div2 A >= 95% (the answer is surface-learnable from the last digit).
- **S4:** the div7 sign (B > A) holds for Qwen2.5-3B-Instruct.
- **S5 (falsifiable length test):** on 6-digit div7, B's accuracy is within 10 pp of p^6, where p is B's per-step
  accuracy measured on 4-digit div7 generations.
- **S6 (shortcut):** A on prime agrees more with the rule "odd and no factor <= 7" than with the true label on
  hard negatives (odd composites with no factor <= 7), where A's accuracy drops below 60%.

Definitions: a "win" is a difference of >= 10 pp with both seeds agreeing in sign; "neither" is both arms at or
below majority class + 8 pp. McNemar tests per seed, Holm correction across tasks within each hypothesis family,
alpha = 0.05. Every cell is reported, including those that go against the claim.

## Arms

base (zero-shot), A (answers), B (trace then answer), C (length-matched scrambled trace), B' (answer then trace,
div7 only), plus eval-only controls: zero-shot chain-of-thought prompt and 4-shot trace demonstrations on the base
model. Model: Qwen2.5-1.5B-Instruct (laptop), Qwen2.5-3B-Instruct and others where the RTX 5090 allows. LoRA r = 8,
alpha 16, 3 epochs, lr 2e-4, bf16, greedy decoding, n = 180 training examples, 240 problem-disjoint test items.
