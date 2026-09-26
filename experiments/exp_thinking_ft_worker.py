"""exp_thinking_ft_worker.py -- ONE thinking-vs-data fine-tune run in an isolated process.

usage:  python exp_thinking_ft_worker.py <task> <arm> <n> <seed>
        task in {prime, valid, div7}
        arm  in {base, A, B, C, D, A_tok, B_orn, B_mv}
        n    total balanced train examples (0 for base)
prints: RESULT {json}  (task, arm, n, seed, acc, n_test, bits, train_tokens, train_s, eval_s)

Loads prompt/completion pools from results/thinking_vs_data/datasets.json (+ ornith_traces.jsonl /
mv_traces.jsonl for the AI-thinking arms). Trains a LoRA on the arm's completions; evaluates by
GENERATING (arms with traces answer AFTER thinking) and extracting the final answer with ONE parser;
grades against the stored exact-gate label. Emits a per-item correctness bitmap (hex) for paired
McNemar, plus actual training whitespace-token count and wall times.

Windows/torch landmines (encoded here; do NOT copy the fp16 line from exp_gate_ft_worker.py):
  bfloat16 (fp16 -> NaN loss), low_cpu_mem_usage=False + attn_implementation="eager" (else
  meta-device SIGSEGV), no numpy/axuniv import (else segfault), one subprocess per run,
  expandable_segments (8GB fragmentation).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "thinking_vs_data", "datasets.json")
# A thinking trace for a large PRIME checks ~25 divisors up to isqrt, which tokenizes to ~350 tokens;
# the answer must fit AFTER that or the model is truncated before "Answer:" and eval reads noise. These
# caps clear the longest honest trace per task (answers-only arms hit EOS long before the cap, so the
# generous ceiling only costs wall-clock on the rambly base runs).
MAXNEW = {"prime": 512, "valid": 320, "div7": 176}
EVAL_BS = 8
ANS_RE = re.compile(r"Answer:\s*(Yes|No)", re.I)
STANDALONE = re.compile(r"\b(Yes|No)\b", re.I)


def extract(text: str):
    """The ONE answer extractor used for every arm."""
    m = ANS_RE.findall(text)
    if m:
        return m[-1].capitalize()
    m2 = STANDALONE.search(text)
    if m2:
        return m2.group(1).capitalize()
    return "?"


def load_pool(task, arm):
    d = json.load(open(DATA, encoding="utf-8"))["tasks"][task]
    if arm in ("A", "B", "C", "D"):
        return d["pools"][arm], d["test"]
    if arm == "A_tok":
        return d["pools"]["A"], d["test"]                 # answers-only; n is scaled by the driver
    if arm == "B_orn":
        path = os.path.join(HERE, "..", "results", "thinking_vs_data", "ornith_traces.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
        return rows, d["test"]
    if arm == "B_mv":
        path = os.path.join(HERE, "..", "results", "thinking_vs_data", "mv_traces.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
        return rows, d["test"]
    return d["pools"]["A"], d["test"]                     # base ignores the pool


def balanced_sample(pool, n, seed):
    """n total, balanced by the completion's final answer, seeded."""
    def ans_of(e):
        return extract(e["completion"])
    yes = [e for e in pool if ans_of(e) == "Yes"]
    no = [e for e in pool if ans_of(e) == "No"]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    half = n // 2
    picked = yes[:half] + no[:half]
    rng.shuffle(picked)
    return picked


def build_example(tok, prompt, completion):
    msgs = [{"role": "user", "content": prompt}]
    pre_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre = tok(pre_text, add_special_tokens=False)["input_ids"]
    ans_ids = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
    return pre + ans_ids, [-100] * len(pre) + ans_ids


@torch.no_grad()
def evaluate(model, tok, test, task):
    model.eval()
    prompts = [t["prompt"] for t in test]
    golds = [t["label"] for t in test]
    bits = []
    ok = 0
    tok.padding_side = "left"
    for i in range(0, len(prompts), EVAL_BS):
        batch = prompts[i:i + EVAL_BS]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in batch]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=MAXNEW[task], do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(batch)):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            pred = extract(txt)
            good = int(pred == golds[i + j])
            bits.append(good)
            ok += good
    return ok / max(1, len(test)), bits


def bits_to_hex(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (1 if b else 0)
    return format(v, "x")


def main():
    task, arm, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pool, test = load_pool(task, arm)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, low_cpu_mem_usage=False, attn_implementation="eager").to(DEV)

    train_tokens = 0
    t_train = 0.0
    if arm != "base":
        if arm in ("B_orn", "B_mv"):
            picked = balanced_sample(pool, min(n, (len(pool) // 2) * 2), seed) if pool else []
            if not picked:
                print("RESULT " + json.dumps({"task": task, "arm": arm, "n": n, "seed": seed,
                      "acc": None, "n_test": len(test), "skip": "no AI traces available"}))
                return
        else:
            picked = balanced_sample(pool, n, seed)
        train_tokens = sum(len(e["completion"].split()) for e in picked)
        torch.manual_seed(seed)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        exs = [build_example(tok, e["prompt"], e["completion"]) for e in picked]
        model.train()
        rng = random.Random(seed)
        t0 = time.time()
        for _ep in range(EPOCHS):
            rng.shuffle(exs)
            for ids, labels in exs:
                out = model(input_ids=torch.tensor([ids], device=DEV),
                            labels=torch.tensor([labels], device=DEV))
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        t_train = time.time() - t0

    t1 = time.time()
    acc, bits = evaluate(model, tok, test, task)
    t_eval = time.time() - t1
    print("RESULT " + json.dumps({
        "task": task, "arm": arm, "n": (len(exs) if arm != "base" else 0), "seed": seed,
        "acc": round(acc, 4), "n_test": len(test), "bits": bits_to_hex(bits),
        "train_tokens": train_tokens, "train_s": round(t_train, 1), "eval_s": round(t_eval, 1)}))


if __name__ == "__main__":
    main()
