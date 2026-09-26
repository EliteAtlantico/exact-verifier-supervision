"""exp_gate_finetune.py -- REALITY-GATED LoRA fine-tune of a real LLM on THIS computer.

Trains a LoRA adapter on Qwen2.5-1.5B-Instruct using ONLY exact-gate-verified labels (zero human labels),
then measures held-out accuracy before vs after. The task is decided by an EXACT oracle (divisibility),
so every training label is 100% correct by construction -- this is the reality-gated fine-tune.

Honest design:
  * held-out numbers are DISJOINT from training numbers -> the eval tests GENERALIZATION, not memorization.
  * the trained adapter is tier 'empirical' (a measured fine-tune), never 'verified'.
  * $0 (no API); GPU compute only. Deterministic label source (n % k == 0).

Run:  PYTHONPATH=src py experiments/exp_gate_finetune.py
"""
from __future__ import annotations

import os
import sys
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DIV = 7                       # the exact oracle: "is n divisible by 7?"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)


def prompt_for(n: int) -> str:
    return "Is %d divisible by 7? Reply with only 'Yes' or 'No'." % n


def gate_label(n: int) -> str:
    return "Yes" if (n % DIV == 0) else "No"     # the exact gate (100%-correct label)


def make_split():
    yes = [n for n in range(100, 1000) if n % DIV == 0]
    no = [n for n in range(100, 1000) if n % DIV != 0]
    random.shuffle(yes); random.shuffle(no)
    n_te = 28
    train = [(n, gate_label(n)) for n in yes[n_te:n_te + 110] + no[n_te:n_te + 110]]
    test = [(n, gate_label(n)) for n in yes[:n_te] + no[:n_te]]        # DISJOINT numbers
    random.shuffle(train); random.shuffle(test)
    return train, test


def build_example(tok, n, ans):
    msgs = [{"role": "user", "content": prompt_for(n)}]
    pre_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre = tok(pre_text, add_special_tokens=False)["input_ids"]
    ans_ids = tok(ans + tok.eos_token, add_special_tokens=False)["input_ids"]
    full = pre + ans_ids
    labels = [-100] * len(pre) + ans_ids                              # loss only on the answer
    return full, labels


@torch.no_grad()
def evaluate(model, tok, data):
    model.eval()
    correct = 0
    for n, ans in data:
        msgs = [{"role": "user", "content": prompt_for(n)}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=3, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][plen:], skip_special_tokens=True).strip().lower()
        pred = "Yes" if txt.startswith("y") else ("No" if txt.startswith("n") else "?")
        correct += int(pred == ans)
    return correct / max(1, len(data))


def main():
    t0 = time.time()
    print("device:", DEV, "| model:", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)

    train, test = make_split()
    print("train=%d (disjoint) test=%d | task: divisible-by-%d (exact-gate labels)" % (len(train), len(test), DIV))

    base_acc = evaluate(model, tok, test)
    print("BASE held-out accuracy: %.1f%%" % (100 * base_acc))

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    examples = [build_example(tok, n, a) for n, a in train]
    EPOCHS = 3
    model.train()
    step = 0
    for ep in range(EPOCHS):
        random.shuffle(examples)
        tot = 0.0
        for ids, labels in examples:
            x = torch.tensor([ids], device=DEV)
            y = torch.tensor([labels], device=DEV)
            out = model(input_ids=x, labels=y)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += out.loss.item(); step += 1
        print("  epoch %d/%d  avg loss %.4f" % (ep + 1, EPOCHS, tot / len(examples)))

    lora_acc = evaluate(model, tok, test)
    print("LoRA held-out accuracy: %.1f%%  (base was %.1f%%)  | %+.1f pts"
          % (100 * lora_acc, 100 * base_acc, 100 * (lora_acc - base_acc)))
    out_dir = ROOT / "results" / "gate_lora_adapter"
    try:
        model.save_pretrained(str(out_dir))
        saved = str(out_dir)
    except Exception as e:  # noqa: BLE001
        saved = "save failed: %s" % str(e)[:80]

    dt = time.time() - t0
    doc = ROOT / "results" / "GATE_FINETUNE.md"
    doc.write_text("\n".join([
        "# Reality-gated LoRA fine-tune on this computer (measured)",
        "",
        "A LoRA adapter on **Qwen2.5-1.5B-Instruct** trained with **only exact-gate-verified labels** "
        "(the oracle: is n divisible by 7?), on this machine's GPU (RTX 4060, 8.6 GB). Zero human labels; "
        "held-out numbers are **disjoint** from training numbers, so this tests generalization.",
        "",
        "| | held-out accuracy |",
        "|---|---|",
        "| base Qwen2.5-1.5B-Instruct | **%.1f%%** |" % (100 * base_acc),
        "| + reality-gated LoRA (%d train, %d epochs) | **%.1f%%** |" % (len(train), EPOCHS, 100 * lora_acc),
        "| delta | **%+.1f pts** |" % (100 * (lora_acc - base_acc)),
        "",
        "- Trained on this computer in **%.0f s** on the RTX 4060 GPU. $0 (no API)." % dt,
        "- Labels are 100%% correct by construction (an exact gate) -- the data is `verified`; the trained "
        "adapter is tier **`empirical`** (a measured fine-tune), never `verified`.",
        "- Adapter saved: `%s`." % saved,
        "- Honest read: this is a real reality-gated fine-tune of a real LLM. Whether a 1.5B model can "
        "*generalize* a math rule from gate labels is exactly what the disjoint held-out delta measures -- "
        "reported as-is, not spun.",
    ]), encoding="utf-8")
    print("wrote %s  (%.0f s total)" % (doc, dt))


if __name__ == "__main__":
    main()
