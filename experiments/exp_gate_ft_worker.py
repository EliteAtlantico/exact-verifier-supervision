"""exp_gate_ft_worker.py -- ONE reality-gated fine-tune run in an isolated process (so the GPU is fully
released on exit; this is the pattern that avoids the multi-run OOM on an 8 GB laptop GPU).

usage:  python exp_gate_ft_worker.py <task> <condition> <seed>
        condition in {base, gate, shuffled}
prints: RESULT {json}   (task, condition, seed, acc, n_test)
"""
from __future__ import annotations

import os
import sys
import json
import math
import random

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
N_TRAIN_PER_CLASS = 120
N_TEST_PER_CLASS = 120


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


TASKS = {
    "div3":  ("Is {n} divisible by 3? Reply with only 'Yes' or 'No'.", lambda n: n % 3 == 0, (100, 9999)),
    "div7":  ("Is {n} divisible by 7? Reply with only 'Yes' or 'No'.", lambda n: n % 7 == 0, (100, 9999)),
    "div13": ("Is {n} divisible by 13? Reply with only 'Yes' or 'No'.", lambda n: n % 13 == 0, (100, 9999)),
    "square": ("Is {n} a perfect square? Reply with only 'Yes' or 'No'.", _is_square, (100, 250000)),
    "prime": ("Is {n} a prime number? Reply with only 'Yes' or 'No'.", _is_prime, (100, 9999)),
}


def collect(oracle, lo, hi):
    rng = random.Random(999)
    need = N_TEST_PER_CLASS + N_TRAIN_PER_CLASS + 40
    pos, neg, seen = [], [], set()
    tries = 0
    while (len(pos) < need or len(neg) < need) and tries < 8_000_000:
        n = rng.randint(lo, hi); tries += 1
        if n in seen:
            continue
        seen.add(n)
        (pos if oracle(n) else neg).append(n)
    return pos, neg


def split_pools(pos, neg):
    test = [(x, "Yes") for x in pos[:N_TEST_PER_CLASS]] + [(x, "No") for x in neg[:N_TEST_PER_CLASS]]
    random.Random(100).shuffle(test)
    return test, pos[N_TEST_PER_CLASS:], neg[N_TEST_PER_CLASS:]


def train_for_seed(pool_pos, pool_neg, seed):
    rng = random.Random(seed)
    p = rng.sample(pool_pos, min(N_TRAIN_PER_CLASS, len(pool_pos)))
    q = rng.sample(pool_neg, min(N_TRAIN_PER_CLASS, len(pool_neg)))
    tr = [(x, "Yes") for x in p] + [(x, "No") for x in q]
    rng.shuffle(tr)
    return tr


def build_example(tok, template, n, ans):
    msgs = [{"role": "user", "content": template.format(n=n)}]
    pre_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre = tok(pre_text, add_special_tokens=False)["input_ids"]
    ans_ids = tok(ans + tok.eos_token, add_special_tokens=False)["input_ids"]
    return pre + ans_ids, [-100] * len(pre) + ans_ids


@torch.no_grad()
def evaluate(model, tok, template, data):
    model.eval()
    ok = 0
    for n, ans in data:
        msgs = [{"role": "user", "content": template.format(n=n)}]
        enc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(enc, add_special_tokens=False, return_tensors="pt").to(DEV)
        out = model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        pred = "Yes" if txt.startswith("y") else ("No" if txt.startswith("n") else "?")
        ok += int(pred == ans)
    return ok / max(1, len(data))


def main():
    task, cond, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    template, oracle, (lo, hi) = TASKS[task]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pos, neg = collect(oracle, lo, hi)
    test, pool_pos, pool_neg = split_pools(pos, neg)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)

    if cond == "base":
        acc = evaluate(model, tok, template, test)
    else:
        torch.manual_seed(seed)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        tr = train_for_seed(pool_pos, pool_neg, seed)
        if cond == "shuffled":                          # wrong-label control
            rng = random.Random(1000 + seed)
            labels = [a for _n, a in tr]; rng.shuffle(labels)
            tr = [(tr[i][0], labels[i]) for i in range(len(tr))]
        exs = [build_example(tok, template, n, a) for n, a in tr]
        model.train()
        rng = random.Random(seed)
        for _ep in range(EPOCHS):
            rng.shuffle(exs)
            for ids, labels in exs:
                out = model(input_ids=torch.tensor([ids], device=DEV),
                            labels=torch.tensor([labels], device=DEV))
                out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        acc = evaluate(model, tok, template, test)

    print("RESULT " + json.dumps({"task": task, "condition": cond, "seed": seed,
                                  "acc": acc, "n_test": len(test)}))


if __name__ == "__main__":
    main()
