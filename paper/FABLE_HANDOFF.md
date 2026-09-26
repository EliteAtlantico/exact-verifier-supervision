# Hand-off from the second reviewer pass (branch `fable/review-pass`, worktree only)

Written 2026-09-26 ~02:30 Toronto against main at d55a557. Nothing on `main` was touched; `paper/main.tex`
was not edited. Everything here is a proposal for the writer to merge, in order of expected effect on the
score. Each item is self-contained; take any subset.

## 1. A story figure (ready: `paper/fig_story.pdf`, script `paper/exp/make_fig_story.py`)

Three panels, built only from committed run records (`results/v2/runs_*.jsonl`, `evals_*.jsonl`,
`thinking_vs_data/runs*.jsonl`, and the saved generations), same style as `make_figures.py`:

- (a) **Erasure and retention, div7**: zero-shot / A / B / S at 1.5B, 3B, 7B, per-seed points with the mean.
  This is the paper's most striking and best-replicated result (two seeds of A at 3B and 7B, two seeds of S
  at 3B) and it is currently a table (Table 5) plus prose in Section 5.
- (b) **The primality shortcut**: arm A's accuracy on the 198 test items where "last digit is 1, 3, 7 or 9"
  gives the right answer versus the 42 composites where it does not, for every A cell with saved
  generations (1.5B s0/s2, 3B s0/s1, 7B s0/s1), B for contrast. A is at 96.5–100% on the first group and
  0–12% on the second at 1.5B and 3B; the one exception is 7B seed 0, the collapsed seed (answers "No").
  This is the cleanest picture of "exact answers teach the rule, not the function" in the paper.
- (c) **What the trace must be, div7 at 1.5B**: zero-shot, CoT prompt, 4-shot, A, C, D, B', S, B.
  Every control sits at chance except S (77.9) and B (82.5–92.5). This replaces the argument that
  "C alone does not show that the trace's information is what helps" with a picture where D and B' carry it.

Suggested placement (page-neutral): make it Figure 1 in the introduction, keep `fig_map` where the dose
paragraph references it (Section 4.2), and move Table 5 (scale) to Appendix A; every number in Table 5
that the text cites is already a macro, and panel (a) shows the div7 column.

Suggested caption (numbers from the run records; swap in macros if the writer prefers):

> Figure 1: (a) Divisibility by 7: zero-shot accuracy, answer-only fine-tuning (A), trace fine-tuning (B)
> and fine-tuning on the model's own verifier-filtered traces (S) at 1.5B, 3B and 7B; points are seeds,
> bars means. Where the base model already divides (3B: 94.2%, 7B: 82.5%), A drops it to chance
> (3B 54.6/50.0, 7B 49.2/52.9) and S keeps it (3B 97.5/96.2, 7B 82.1). (b) Primality: accuracy of A and B
> on the 198 test items where the rule "last digit is 1, 3, 7 or 9" is right (circles) and on the 42
> composites where it is wrong (crosses); at 1.5B and 3B, A scores 96.5–100% on the first group and 0–12%
> on the second; 7B seed 0 is the collapsed seed that answers "No". (c) Controls on div7 at 1.5B: a
> zero-shot CoT prompt, 4-shot trace demonstrations, scrambled traces (C), a correct trace of a different
> number (D) and the trace placed after the answer (B') all stay at chance; only the input's own trace,
> before the answer, teaches the task (S 77.9; B 82.5–92.5).

Caveat for the writer: panel (a) uses the 1.5B zero-shot value 62.5 from `thinking_vs_data/runs.jsonl`
(the paper's `\divsevenBase`); check that this is the cell the text cites.

## 2. Abstract: lead with the finding, cut the numbers by half (330 -> ~230 words)

The current abstract opens with a definition and reads as a results list; the reviewers who scored 3 said
the central claim was thin, and the abstract does not tell them which claim is central. Proposed text
(macros where the paper has them):

> Exact verifiers (primality tests, SAT solvers, unit tests) label data without noise and supply the rewards
> of verifiable-reward post-training. Because every label is exact, a difference between two training
> regimes on the same inputs is a difference in the supervision format. We fine-tune Qwen2.5 models (0.5B to
> 7B; main size 1.5B) with LoRA on exactly labeled decision tasks and compare supervision by the answer alone
> with a deterministic procedure trace followed by the answer. Three findings. Exact answers teach shortcuts:
> on 4-digit primality the answers follow "the last digit is 1, 3, 7 or 9" on \primeAagreeLastDigitLo--\primeAagreeLastDigitHi\%
> of test items, and on divisibility by 7 or 13 answer-only training stays at chance at every training size.
> Exact answers erase computation: at 3B and 7B, where the zero-shot model already divides by 7
> (\threeBdivsevenBase\% and \sevenBdivsevenBase\%), answer-only fine-tuning drops it to chance, while
> fine-tuning on the model's own verifier-filtered traces keeps it (\threeBdivsevenS\%). Exact traces teach
> steps seen often enough: within a task, per-step accuracy follows how often each step type is supervised,
> a relation we logged in timestamped commits before the runs; it did not transfer across tasks whose steps
> differ in difficulty. Controls show that the trace must be the input's own computation, written before the
> answer: a scrambled trace, a correct trace of another number, and a trace placed after the answer all stay
> at chance. Three of ten logged predictions failed and are reported. Verified labels do not by themselves
> determine what a model learns; the supervision format and the base model's own computation do.

## 3. Title: three lines, not four

Option A (recommended, keeps every claim):
`Perfect Labels for Free? Exact Answers Teach Shortcuts and Erase Computation; Exact Traces Teach Steps`
Option B: `Perfect Labels for Free? What Exact Answers and Exact Traces Teach a Language Model`

## 4. Contributions: three bullets, one sentence each

Bullet 2 is currently a 90-word sentence with four parentheticals. Proposed:

- Answer-only fine-tuning on exact labels learns the surface probe's function (primality at every size;
  chance on div7/div13) and, at 3B and 7B, erases the base model's own division; fine-tuning on the
  model's verifier-filtered traces keeps it (Sections 4.1, 5).
- Trace supervision teaches the procedure once each step type is supervised often enough: timestamped dose
  predictions held within tasks and failed across tasks, exposing step difficulty as a confound of
  supervision frequency (Section 4.2).
- Controls and exact audits: the trace must be the input's own computation, placed before the answer;
  trace-trained models also write shortcuts behind unfaithful steps, which the audits catch (Sections 4.3, 5);
  all failed predictions are reported (Table 4).

## 5. Related work: one sentence that names the difference from the closest prior work

Reviewers asked what is new relative to Lee et al. (2024), Prystawski et al. (2023) and Kim et al. (2025).
Add after the scratchpad sentence in "Intermediate steps as supervision":

> Those studies train from scratch or prompt; we fine-tune pretrained models on labels that are exact by
> construction, which lets us separate the supervision format from label noise, and it exposes an effect
> they cannot see: answer-only fine-tuning removing a computation the base model already performs.

## 6. Reorder the results so the headline comes first

Section 4 currently runs shortcut -> traces -> controls, and erasure (the strongest result) is in Section 5
"Scale". Suggested order: 4.1 answers teach the surface (shortcut + erasure, pulling the two-seed 3B/7B
paragraph up from Section 5), 4.2 traces teach steps (dose, S7/S9, S5), 4.3 controls (D, B', prompts, S).
Section 5 keeps the size table (or its appendix pointer) and the 3B dose replication. The panel's "central
claim is thin" reading comes from S7 being presented as the headline; erasure and shortcut are multi-seed,
multi-size results and should carry the paper.

## 7. Small things noticed while reading (no change to any number)

- Abstract and Section 1 both say "with the same drop at 7B"; the 7B A numbers are 49.2/52.9 against a
  base of 82.5, so the drop is 30 points against 40 at 3B. "the same" is fine; "a comparable drop" is safer.
- Table 4, S10 row, 7B column shows "div7 82.1 [1]"; the 7B S value equals the 7B base (82.5), which the
  text calls "level with the base". A reviewer reading the table alone sees a number with no anchor; add
  "(base 82.5)" if space allows.
- Section 4.2, div11 paragraph: "10 ≡ −1 (mod 11) makes its step a subtraction" is the right explanation;
  consider saying the same for div3 (10 ≡ 1) so the pattern is explicit: easy steps (add/subtract) reach
  q ≈ 1 at low m; hard steps (multiply) need more.
- Limitations lists "S10: \verdictSTen" twice in two places with different detail; fine, but one could go.
- The AI-use statement is good; keep it exactly as is (ICLR 2027 checks for it).

## 8. Tailscale / Khalil's 5090 (for the user, not the paper)

His node advertises Tailscale SSH host keys, ping works, but port 22 times out and `tailscale ssh` closes
immediately: his tailnet's access policy does not grant SSH to a user the node is shared with. He needs to
either add an SSH grant for `vicola.lexo@gmail.com` (or `autogroup:shared`) in his tailnet's Access
Controls, or run plain OpenSSH on another port with our public key in `~kc/.ssh/authorized_keys` and allow
that port for shared users. Until then the packages run through him, as before.
