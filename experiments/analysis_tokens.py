"""analysis_tokens.py -- real-tokenizer checks (CPU only, own process, no numpy import).

usage:  python experiments/analysis_tokens.py
writes: results/analysis/tokens.json

1. Does Qwen2.5-1.5B-Instruct's tokenizer split 4-digit (and 6-digit) numbers into single digits, in the
   exact prompt context the models see?
2. Training tokens per task x arm (every pool: A, B, C, D, Bprime, ...) x seed with the real tokenizer over the EXACT completions
   the worker trains on (replica of balanced_sample; loss-bearing ids = tok(completion + eos) as in
   build_example; the chat-templated prompt ids are masked with -100 and reported separately).
   The whitespace-token count of the same draw is compared with the train_tokens recorded in the runs,
   which validates the sampling replica.

Loads the tokenizer from the local HF cache (HF_HUB_OFFLINE=1). CUDA is hidden from this process.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import analysis_common as C  # noqa: E402

C.fix_sys_path()
from transformers import AutoTokenizer  # noqa: E402

N = 180
SEEDS = (0, 1, 2)
EPOCHS = 3


def main():
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL)
    out = {"generated_by": "experiments/analysis_tokens.py", "model": C.DEFAULT_MODEL,
           "tokenizer_class": type(tok).__name__, "digit_check": {}, "train_tokens": {}}

    # ------------------------------------------------------------ 1. digit tokenization
    samples = [1000, 1617, 2740, 7055, 9929, 9999, 123456, 997003]
    per = {}
    all_single = True
    for n in samples:
        contexts = {"bare": str(n), "space": f" {n}",
                    "prompt": f"Is {n} a prime number? End your reply with 'Answer: Yes' or 'Answer: No'."}
        entry = {}
        for cname, text in contexts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            pieces = tok.convert_ids_to_tokens(ids)
            num_pieces = [p for p in pieces if any(ch.isdigit() for ch in p)]
            entry[cname] = {"pieces": pieces if cname != "prompt" else num_pieces,
                            "digit_pieces": num_pieces,
                            "all_single_digit": all(len(p.replace("Ġ", "")) == 1 for p in num_pieces)}
            all_single &= entry[cname]["all_single_digit"]
        per[str(n)] = entry
    # exhaustive over all 4-digit numbers in the bare and prompt-embedded forms
    bad = []
    for n in range(1000, 10000):
        for text in (str(n), f"Is {n} a prime number?", f"Is {n} divisible by 7?"):
            a = text.index(str(n))
            b = a + len(str(n))
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            spans = [(s0, s1) for s0, s1 in enc["offset_mapping"] if s1 > a and s0 < b and s1 > s0]
            # each token overlapping the number must cover exactly one of its digits
            if len(spans) != len(str(n)) or any(s1 - s0 != 1 or s0 < a or s1 > b for s0, s1 in spans):
                bad.append(text)
    out["digit_check"] = {"examples": per, "all_examples_single_digit": all_single,
                          "exhaustive_4digit_contexts_checked": 9000 * 3,
                          "exhaustive_4digit_violations": bad[:20], "n_violations": len(bad),
                          "verdict": ("every 4-digit number tokenizes to 4 single-digit tokens" if not bad
                                      else f"{len(bad)} contexts do NOT split into single digits")}

    # ------------------------------------------------------------ 2. training tokens per arm
    runs, _ = C.load_runs()
    recorded = defaultdict(dict)
    for r in runs:
        if r["n"] == N and r["model"] == C.DEFAULT_MODEL:
            recorded[(r["task"], r["arm"])][r["seed"]] = r.get("train_tokens")
    tasks = C.load_tasks()
    for t, td in tasks.items():
        pools = td.get("pools") or {}
        if not pools:
            continue
        tt = {}
        ids_by_seed = {}
        for arm in sorted(pools, key=lambda a: (a != "A", a)):
            seeds = {}
            for s in SEEDS:
                picked = C.balanced_sample(pools[arm], N, s)
                comp = [len(tok(e["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]) for e in picked]
                prompt = [len(tok(tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False,
                                                           add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                          for e in picked]
                ws = sum(len(e["completion"].split()) for e in picked)
                rec = recorded.get((t, arm), {}).get(s)
                seeds[str(s)] = {"completion_tokens_incl_eos": sum(comp), "per_example_mean": sum(comp) / len(comp),
                                 "per_example_min": min(comp), "per_example_max": max(comp),
                                 "prompt_tokens_masked": sum(prompt), "whitespace_tokens": ws,
                                 "tokens_per_ws_token": sum(comp) / ws if ws else None,
                                 "loss_tokens_x_epochs": EPOCHS * sum(comp),
                                 "recorded_train_tokens_ws": rec,
                                 "ws_matches_recorded_run": (None if rec is None else rec == ws)}
                ids_by_seed.setdefault(s, {})[arm] = [e["id"] for e in picked]
            tt[arm] = {"seeds": seeds,
                       "mean_completion_tokens": statistics.mean(v["completion_tokens_incl_eos"] for v in seeds.values())}
        base = tt.get("A", {}).get("mean_completion_tokens")
        for arm in tt:
            tt[arm]["ratio_vs_A"] = (tt[arm]["mean_completion_tokens"] / base) if base else None
        same = {str(s): len({tuple(v) for v in d.values()}) == 1 for s, d in ids_by_seed.items()}
        out["train_tokens"][t] = {"arms": tt, "same_items_across_arms_per_seed": same}
    C.ensure_out()
    json.dump(out, open(os.path.join(C.OUT, "tokens.json"), "w", encoding="utf-8"), indent=1)
    print("digit check:", out["digit_check"]["verdict"])
    for t, d in out["train_tokens"].items():
        print(t, {a: (round(v["mean_completion_tokens"]), round(v["ratio_vs_A"], 1)) for a, v in d["arms"].items()},
              "same items:", d["same_items_across_arms_per_seed"],
              "ws==recorded:", {a: [x["ws_matches_recorded_run"] for x in v["seeds"].values()] for a, v in d["arms"].items()})


if __name__ == "__main__":
    main()
