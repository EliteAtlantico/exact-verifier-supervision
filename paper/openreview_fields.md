# OpenReview fields for the ICLR 2027 upload (edit the title/abstract on the form to match main.tex before 07:59 Toronto)

**Title.** Perfect Labels for Free? What Exact Answers and Exact Traces Teach a Small Language Model

**TL;DR.** With labels from exact verifiers, answer-only fine-tuning learns surface shortcuts and, at 3B and 7B, removes from the answers a division the base model already performs, while procedure traces teach the steps within a task once they are supervised often enough; timestamped predictions, controls and exact audits, with every failed prediction reported.

**Abstract.** (keep identical to the abstract in main.tex; regenerate after the final build)

Exact verifiers (primality tests, SAT solvers, unit tests) label data without noise and supply the rewards of verifiable-reward post-training. Because every label is exact, a difference between two training regimes on the same inputs is a difference in the supervision format. We fine-tune Qwen2.5 models (0.5B to 7B; main size 1.5B, three seeds) with LoRA on exactly labeled decision tasks and compare supervision by the answer alone with a deterministic procedure trace followed by the answer. Exact answers teach shortcuts: on 4-digit primality the answers follow "the last digit is 1, 3, 7 or 9" on 99.2-100.0% of test items, and on divisibility by 7 or 13 answer-only training stays at chance or constant at every size and configuration we ran (n<=540, including a logged sweep of eight non-collapsing configurations), while the same recipe learns divisibility by 3. Exact answers remove computation from the answers: at 3B and 7B the zero-shot model already divides by 7; answer-only fine-tuning drops it to chance in both seeds and the model stops writing a chain of thought even when asked, while fine-tuning on its own verifier-filtered traces keeps the division (3B, two seeds; 7B, one); whether the capability survives is untested. Exact traces teach steps within a task: accuracy rises with the number of traces and with how often each table entry was supervised, a count-based prediction across tasks failed because steps differ in difficulty, and only the input's own trace, written before the answer, helps. A second model family replicates every direction but misses one preset threshold. Predictions were logged in timestamped commits before their cells ran; four of twelve failed, among them every cross-task, cross-length and cross-family prediction, and all are reported.

**Keywords.** verifiable rewards; exact verifiers; supervision format; chain-of-thought fine-tuning; shortcut learning; process vs outcome supervision; preregistration; small language models; arithmetic reasoning; LoRA

**Primary area.** foundation or frontier models, including LLMs (alternative: learning theory / interpretability; a reviewer-friendly second choice is "reasoning" if the form offers it).

**Reciprocal reviewer.** The qualifying author (a main-track paper at a listed venue before the abstract deadline) must be registered as a reviewer on the form.

**Statements.** The AI-use statement is in the manuscript (required by ICLR 2027) and must also be entered on the form; reproducibility and ethics statements are in the manuscript.

**Supplementary.** PerfectLabels_ICLR2027_supplementary.zip (anonymized repository snapshot), produced by scripts/make_supplementary.py.
