# Preregistered predictions and decision rules (frozen before any run listed here)

Committed 2026-09-25 20:29:47 Toronto (commit b0370f5), before any of the runs below were started. The commit hash of this file is
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

## Decision log

- 2026-09-25 21:09:47 Toronto (commit fcaa05d) — **S1 PASS.** div7 B@180 seed 1 = 90.0% vs A 50.8% (+39.2 pp, McNemar p = 2e-18);
  seed 2 = 82.5% vs A 47.9% (+34.6 pp, p = 6.8e-13). Runs in results/thinking_vs_data/runs_L1.jsonl, produced by
  the unchanged worker experiments/exp_thinking_ft_worker.py. The paper proceeds; the remaining tests run on
  experiments/exp_worker_v2.py.
- 2026-09-25 21:26:10 Toronto (commit 27ba807; the div7_6d B seed-0 generation file was written at 21:27:30) — **S5 clarification, recorded before either S5 cell finished** (div7_6d B seed 0 was
  mid-training; the 4-digit-adapter transfer eval had not started). Primary S5 cell = arm B trained and tested on
  the 6-digit task (div7_6d), seeds 0 and 1, compared with p^6 where p = per-step accuracy of 4-digit div7 B
  generations (seed 0: p = 0.955, p^6 = 0.758). Secondary = the 4-digit B adapter evaluated on 6-digit inputs.
  Also reported, labelled post hoc: the fraction of fully correct 6-digit traces vs p^6, and answer accuracy vs
  p^6 + (1 - p^6) * g, where g is the answer-correct rate when the trace is wrong (4-digit: g = 0.561), since a
  wrong trace still yields the right yes/no answer about half the time.
- 2026-09-25 21:29:52 Toronto (commit 59e3989) — **S5 (primary) FAILED as preregistered**: 6-digit div7 B seed 0 = 96.2% vs p^6 = 75.8%
  (+20.4 pp > 10 pp tolerance). Its own per-step accuracy is 98.8% (vs 95.5% for the 4-digit model), so p is not
  a fixed property that transfers between training sets.
- 2026-09-25 21:29:52 Toronto (commit 59e3989; div11 B seed 0 started after 21:30:58) — **S7 (new hypothesis, preregistered before any of its cells ran):** per-step trace
  accuracy p is set by the number of supervised transitions per entry of the (remainder, digit) table,
  m = k * n / (10 * d) (k steps per trace, n training examples, 10d table entries). Observed so far: div7 n=180
  m=10.3 -> p=0.955; div7_6d n=180 m=15.4 -> p=0.988; div13 n=180 m=5.5 -> p=0.566. Predictions:
  (a) div13 B n=540 seed 0 (m=16.6): p >= 0.95 and answer accuracy >= 85%;
  (b) div7 B n=270 seed 0 (4-digit, m=15.4): p >= 0.97;
  (c) div7 B n=90 seed 0 (m=5.1): p <= 0.80 and answer accuracy <= 75%;
  (d) div11 B n=180 (m=6.5, not yet run): 0.566 < p < 0.955;
  (e) div3 B n=180 (m=24) and div2 B n=180 (m=36): p >= 0.97.
  S7 passes if at least 4 of (a)-(e) hold; every cell is reported either way.

Note (21:35): the clock times in the entries above were first written as estimates; they have been replaced by the
commit timestamps, which are the authoritative record.
- 2026-09-25 21:38:05 Toronto (commit 1163562) **S8 and S9, preregistered before their cells ran** (div11 B seed 0 = 99.2% had finished;
  none of the cells below had started):
  S8 (grounding control): div7 arm D (a fluent, correct long-division trace of a DIFFERENT number with the same
  label) at n = 180, seeds 0 and 1: accuracy <= 60% in both seeds, i.e. within 10 pp of A, far below B.
  S9 (dose, step difficulty fixed): div13 B at n = 360, seed 0 (m = 11.1 transitions per table entry, close to
  div7's 10.3 at n = 180): accuracy >= 80%; div13 A at n = 360 seed 0 stays <= 60%.
- 2026-09-25 21:46:44 Toronto (commit 8c0cfff) **S10, preregistered before any arm-S cell ran.** Arm S = self-generated traces filtered by the
  exact verifier on the final answer only (STaR / rejection-sampling fine-tuning, K = 4 samples per item at
  temperature 0.7 from the base model's chain-of-thought prompt; one kept completion per item; same 180 items and
  LoRA as A/B). Prediction: outcome-only verification does not supply correct steps, so on div7 arm S does not
  unlock the task: div7 S accuracy <= 65% in seeds 0 and 1, and fewer than half of the kept div7 traces have all
  remainders correct. On valid, where answers are surface-learnable, S >= 90%. Prime S is reported without a
  prediction.
- 2026-09-26 (time = this commit) **S10 (primary) FAILED as preregistered on its accuracy clause.** 1.5B div7 arm S seed 0 =
  77.9% (> 65%), so the rule "div7 S <= 65% in seeds 0 and 1" cannot hold whatever seed 1 gives. The trace clause holds:
  36.5% of the 159 kept div7 traces have every remainder correct (< half). For reference, the 1.5B base model with the
  chain-of-thought prompt scores 67.1% and arm A about 50%. Secondary cells point the same way: 3B div7 S 97.5% / 96.2%
  (seeds 0/1; 13.9% / 14.5% of kept traces fully correct), 7B div7 S 82.1% (26.7%). Outcome-only filtering kept traces
  whose steps are mostly wrong and still lifted answer accuracy above arm A at every size. Seed 1, prime S and valid S
  at 1.5B are still running and will be reported.
- 2026-09-26 (time = this commit) **Wording note (no change to any rule or verdict).** The seed-0 div7 B@180 value reported as
  "reproduced by a re-run" (92.5%) came from a deterministic re-execution of the same cell with the same seed and code;
  the paper now calls it that. The per-step accuracy written p in this file is written q in the paper, to keep it apart
  from McNemar p-values.
- 2026-09-26 (time = this commit) **S11 (second model family), preregistered before any of its cells ran.** Model:
  meta-llama/Llama-3.2-3B-Instruct if the RTX 5090 machine can download it, otherwise HuggingFaceTB/SmolLM2-1.7B-Instruct;
  the script records which one ran (results/v2/second_family_model.txt) before its first cell. Same data, prompts, LoRA
  and n = 180 as the Qwen cells (queue results/queues/q5_family.txt). Predictions: (a, primary) div7 B - A >= 20 pp in
  seeds 0 and 1; (b) div7 D (correct trace of a different number) within 10 pp of A in seed 0. S11 passes if (a) and
  (b) hold. prime and valid A/B and div7 base are reported without a prediction.
