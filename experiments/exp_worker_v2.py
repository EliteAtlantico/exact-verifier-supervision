"""exp_worker_v2.py -- ONE v2 fine-tune(+eval) or eval-only run in an isolated process.

Copy of exp_thinking_ft_worker.py; training/eval semantics unchanged (bf16 on CUDA, eager attention,
low_cpu_mem_usage=False, no numpy import, same env vars, LoRA r=8/a=16 q/k/v/o, lr 2e-4, 3 epochs, batch 1,
prompt tokens masked with -100, greedy decoding, left padding, ONE answer extractor). Additions only:

usage:  python exp_worker_v2.py <task> <arm> <n> <seed> [options]
  --model HF_ID|PATH        default Qwen/Qwen2.5-1.5B-Instruct
  --data PATH               default results/v2/datasets_v2.json
  --eval-task TASK          evaluate on this task's test set (default: <task>), e.g. prime_hard, div7_6d
  --eval-only DIR|none|auto no training. none = base model (arm reported as "base", n 0);
                            auto = results/adapters/<model_short>/<task>_<arm>_<n>_<seed>/; DIR = that adapter
  --prompt-mode plain|cot|fewshot4
                            cot appends "Think step by step, then give the final line as 'Answer: Yes' or
                            'Answer: No'."; fewshot4 prepends the 4 fixed B demos datasets_v2["fewshot4"][<task>]
                            as chat turns (demos come from the positional <task>'s train pool)
  --eval-bs N               eval batch size (default 8 = old behavior)
  --max-new N               override the generation cap (default MAXNEW[eval_task]; cot: max(that, 512))
  --repetition-penalty X    override generation_config (default: model's own, as in the old worker)
  --eval-limit N            evaluate only the first N test items (smoke tests; 0 = all)
  --save-dir DIR            generations root, default results/v2/gens
  --adapter-root DIR        adapters root, default results/adapters
  --no-save-adapter         skip saving the trained LoRA adapter
Relative paths resolve against the repo root, so the worker runs from any cwd (Linux or Windows).

outputs: RESULT {json} on stdout (old fields + model, prompt_mode, eval_task, req_n, eval_only, adapter,
  max_new, eval_bs, n_hit_cap, gens_path, adapter_path);
  <save-dir>/<model_short>/<name>.jsonl  one row per test item: id, gold, pred, correct, gen, n_gen_tokens, hit_cap;
  <adapter-root>/<model_short>/<task>_<arm>_<n>_<seed>/  LoRA adapter + train_meta.json (training runs only).
  name = <task>_<arm>_<n>_<seed>[__on-<eval_task>][__<prompt_mode>][__reeval]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATA = REPO / "results" / "v2" / "datasets_v2.json"
OLD_TV = REPO / "results" / "thinking_vs_data"
# Old caps (prime/valid/div7) kept verbatim. New caps = ceil16(max(1.5*L, L+48)) where L is the longest B
# completion in Qwen2.5 tokens incl. EOS (div7: L=115 -> 176, the old value): div2/div3 L=115, div11/div13
# L=127, div7_6d L=155. prime_hard shares prime's cap (same trace family, isqrt <= 99).
MAXNEW = {"prime": 512, "valid": 320, "div7": 176,
          "div2": 176, "div3": 176, "div11": 192, "div13": 192, "div7_6d": 240, "prime_hard": 512}
COT_MIN_NEW = 512
COT_SUFFIX = " Think step by step, then give the final line as 'Answer: Yes' or 'Answer: No'."
ANS_RE = re.compile(r"Answer:\s*(Yes|No)", re.I)
STANDALONE = re.compile(r"\b(Yes|No)\b", re.I)


def extract(text: str):
    """The ONE answer extractor used for every arm (unchanged)."""
    m = ANS_RE.findall(text)
    if m:
        return m[-1].capitalize()
    m2 = STANDALONE.search(text)
    if m2:
        return m2.group(1).capitalize()
    return "?"


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def model_short(model: str) -> str:
    return Path(model.rstrip("/\\")).name


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="v2 LoRA fine-tune + eval worker (one cell)")
    ap.add_argument("task")
    ap.add_argument("arm")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--eval-task", default=None)
    ap.add_argument("--eval-only", default=None, metavar="DIR|none|auto")
    ap.add_argument("--prompt-mode", default="plain", choices=["plain", "cot", "fewshot4"])
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--save-dir", default="results/v2/gens")
    ap.add_argument("--adapter-root", default="results/adapters")
    ap.add_argument("--no-save-adapter", action="store_true")
    a = ap.parse_args(argv)
    a.eval_task = a.eval_task or a.task
    if a.eval_only is not None and a.eval_only.lower() == "none":
        a.eval_only = "none"
        a.arm, a.n = "base", 0
    return a


def cell_name(a) -> str:
    name = f"{a.task}_{a.arm}_{a.n}_{a.seed}"
    if a.eval_task != a.task:
        name += f"__on-{a.eval_task}"
    if a.prompt_mode != "plain":
        name += f"__{a.prompt_mode}"
    if a.eval_only not in (None, "none"):
        name += "__reeval"
    return name


def adapter_dir_for(a) -> Path:
    return repo_path(a.adapter_root) / model_short(a.model) / f"{a.task}_{a.arm}_{a.n}_{a.seed}"


def load_pool(data, task, arm):
    """Same arm semantics as the old worker; arms beyond A-D (e.g. Bprime) come straight from pools."""
    d = data["tasks"][task]
    pools = d.get("pools", {})
    if arm == "A_tok":
        return pools["A"]                                  # answers-only; n is scaled by the driver
    if arm == "B_orn":
        path = OLD_TV / "ornith_traces.jsonl"
        return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    if arm == "B_mv":
        path = OLD_TV / "mv_traces.jsonl"
        return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    if arm == "base":
        return pools.get("A", [])                          # base ignores the pool
    if arm not in pools:
        raise SystemExit(f"arm {arm!r} not in {task} pools {sorted(pools)}")
    return pools[arm]


def balanced_sample(pool, n, seed):
    """n total, balanced by the completion's final answer, seeded (unchanged)."""
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


def eval_messages(prompt, mode, demos):
    if mode == "cot":
        return [{"role": "user", "content": prompt + COT_SUFFIX}]
    msgs = []
    if mode == "fewshot4":
        for dm in demos:
            msgs += [{"role": "user", "content": dm["prompt"]}, {"role": "assistant", "content": dm["completion"]}]
    return msgs + [{"role": "user", "content": prompt}]


@torch.no_grad()
def evaluate(model, tok, test, max_new, eval_bs, mode, demos, rep_pen):
    model.eval()
    prompts = [t["prompt"] for t in test]
    golds = [t["label"] for t in test]
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stop_ids |= set(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
    stop_ids.discard(None)
    gen_kw = {"max_new_tokens": max_new, "do_sample": False, "pad_token_id": tok.eos_token_id}
    if rep_pen is not None:
        gen_kw["repetition_penalty"] = rep_pen
    bits, rows = [], []
    ok = 0
    tok.padding_side = "left"
    for i in range(0, len(prompts), eval_bs):
        batch = prompts[i:i + eval_bs]
        texts = [tok.apply_chat_template(eval_messages(p, mode, demos), tokenize=False,
                                         add_generation_prompt=True) for p in batch]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, **gen_kw)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j in range(len(batch)):
            ids = gen[j].tolist()
            n_gen = next((k for k, t in enumerate(ids) if t in stop_ids), None)
            hit_cap = n_gen is None
            txt = tok.decode(gen[j], skip_special_tokens=True)
            pred = extract(txt)
            good = int(pred == golds[i + j])
            bits.append(good)
            ok += good
            rows.append({"id": test[i + j]["id"], "gold": golds[i + j], "pred": pred, "correct": good,
                         "gen": txt, "n_gen_tokens": len(ids) if hit_cap else n_gen, "hit_cap": hit_cap})
    return ok / max(1, len(test)), bits, rows


def bits_to_hex(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (1 if b else 0)
    return format(v, "x")


def main(argv=None):
    a = parse_args(argv)
    task, arm, n, seed = a.task, a.arm, a.n, a.seed
    data_path = repo_path(a.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if a.eval_task not in data["tasks"]:
        raise SystemExit(f"eval task {a.eval_task!r} not in {data_path}")
    test = data["tasks"][a.eval_task]["test"]
    if a.eval_limit:
        test = test[:a.eval_limit]
    demos = []
    if a.prompt_mode == "fewshot4":
        by_id = {e["id"]: e for e in data["tasks"][task]["pools"]["B"]}
        demos = [by_id[i] for i in data["fewshot4"][task]]
    max_new = a.max_new or (max(MAXNEW[a.eval_task], COT_MIN_NEW) if a.prompt_mode == "cot"
                            else MAXNEW[a.eval_task])

    adapter_src = None
    if a.eval_only not in (None, "none"):
        adapter_src = adapter_dir_for(a) if a.eval_only == "auto" else repo_path(a.eval_only)
        meta_p = adapter_src / "train_meta.json"
        if meta_p.exists():                                  # refuse to mislabel a cell
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            want = {"task": task, "arm": arm, "req_n": n, "seed": seed, "model": a.model}
            bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
            if bad:
                raise SystemExit(f"adapter {adapter_src} does not match this cell: {bad}")
        pool = []
    else:
        pool = load_pool(data, task, arm) if arm != "base" else []

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, low_cpu_mem_usage=False, attn_implementation="eager").to(DEV)

    train_tokens = 0
    t_train = 0.0
    exs = []
    adapter_path = None
    if adapter_src is not None:
        model = PeftModel.from_pretrained(model, str(adapter_src))
        adapter_path = str(adapter_src)
    elif arm != "base" and a.eval_only is None:
        if arm in ("B_orn", "B_mv"):
            picked = balanced_sample(pool, min(n, (len(pool) // 2) * 2), seed) if pool else []
            if not picked:
                print("RESULT " + json.dumps({"task": task, "arm": arm, "n": n, "seed": seed,
                      "acc": None, "n_test": len(test), "skip": "no AI traces available",
                      "model": a.model, "prompt_mode": a.prompt_mode, "eval_task": a.eval_task}))
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
        if not a.no_save_adapter:
            ad = adapter_dir_for(a)
            ad.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ad))
            (ad / "train_meta.json").write_text(json.dumps({
                "task": task, "arm": arm, "req_n": n, "n": len(exs), "seed": seed, "model": a.model,
                "data": str(a.data), "train_tokens": train_tokens, "train_s": round(t_train, 1),
                "picked_ids": [e["id"] for e in picked]}), encoding="utf-8")
            adapter_path = str(ad)

    t1 = time.time()
    acc, bits, rows = evaluate(model, tok, test, max_new, a.eval_bs, a.prompt_mode, demos,
                               a.repetition_penalty)
    t_eval = time.time() - t1

    gdir = repo_path(a.save_dir) / model_short(a.model)
    gdir.mkdir(parents=True, exist_ok=True)
    gpath = gdir / f"{cell_name(a)}.jsonl"
    with open(gpath, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    trained = a.eval_only is None and arm != "base"
    print("RESULT " + json.dumps({
        "task": task, "arm": arm, "n": (len(exs) if trained else (n if adapter_src else 0)), "seed": seed,
        "acc": round(acc, 4), "n_test": len(test), "bits": bits_to_hex(bits),
        "train_tokens": train_tokens, "train_s": round(t_train, 1), "eval_s": round(t_eval, 1),
        "req_n": n, "model": a.model, "prompt_mode": a.prompt_mode, "eval_task": a.eval_task,
        "eval_only": a.eval_only is not None, "adapter": adapter_path, "max_new": max_new,
        "eval_bs": a.eval_bs, "repetition_penalty": a.repetition_penalty,
        "n_hit_cap": sum(r["hit_cap"] for r in rows), "eval_limit": a.eval_limit,
        "gens_path": str(gpath.relative_to(REPO)) if gpath.is_relative_to(REPO) else str(gpath)}))


if __name__ == "__main__":
    main()
