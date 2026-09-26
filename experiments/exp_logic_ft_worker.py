"""exp_logic_ft_worker.py -- ONE reality-gated LOGIC fine-tune run in an isolated process (GPU released on
exit; the pattern that avoids multi-run OOM on an 8 GB laptop GPU).

Reads gate-verified (prompt,label) pairs from results/logic_dataset.json (built by exp_logic_dataset.py,
labels = exact Z3 decisions). Same conditions as the arithmetic study:

  base      -- no fine-tune, evaluate the base model
  gate      -- LoRA fine-tune on the EXACT Z3 labels
  shuffled  -- LoRA fine-tune on the SAME prompts with labels shuffled (wrong-label control)

usage:  python exp_logic_ft_worker.py <task> <condition> <seed>   (task in {valid, consist})
prints: RESULT {json}
"""
from __future__ import annotations

import os
import sys
import json
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
N_TRAIN = 240                      # 120/class after the dataset's own balance
DATA = os.path.join(os.path.dirname(__file__), "..", "results", "logic_dataset.json")


def load_task(task):
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    return d[task]["train"], d[task]["test"]


def build_example(tok, prompt, ans):
    msgs = [{"role": "user", "content": prompt}]
    pre_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre = tok(pre_text, add_special_tokens=False)["input_ids"]
    ans_ids = tok(ans + tok.eos_token, add_special_tokens=False)["input_ids"]
    return pre + ans_ids, [-100] * len(pre) + ans_ids


@torch.no_grad()
def evaluate(model, tok, data):
    model.eval()
    ok = 0
    for ex in data:
        msgs = [{"role": "user", "content": ex["prompt"]}]
        enc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(enc, add_special_tokens=False, return_tensors="pt").to(DEV)
        out = model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        pred = "Yes" if txt.startswith("y") else ("No" if txt.startswith("n") else "?")
        ok += int(pred == ex["label"])
    return ok / max(1, len(data))


def main():
    task, cond, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    train_pool, test = load_task(task)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)

    if cond == "base":
        acc = evaluate(model, tok, test)
    else:
        torch.manual_seed(seed)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        rng = random.Random(seed)
        tr = rng.sample(train_pool, min(N_TRAIN, len(train_pool)))
        tr = [(ex["prompt"], ex["label"]) for ex in tr]
        if cond == "shuffled":                          # wrong-label control
            r2 = random.Random(1000 + seed)
            labels = [a for _p, a in tr]; r2.shuffle(labels)
            tr = [(tr[i][0], labels[i]) for i in range(len(tr))]
        exs = [build_example(tok, p, a) for p, a in tr]
        model.train()
        for _ep in range(EPOCHS):
            rng.shuffle(exs)
            for ids, labels in exs:
                out = model(input_ids=torch.tensor([ids], device=DEV),
                            labels=torch.tensor([labels], device=DEV))
                out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        acc = evaluate(model, tok, test)

    print("RESULT " + json.dumps({"task": task, "condition": cond, "seed": seed,
                                  "acc": acc, "n_test": len(test)}))


if __name__ == "__main__":
    main()
