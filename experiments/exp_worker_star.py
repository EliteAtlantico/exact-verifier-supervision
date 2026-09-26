"""exp_worker_star.py -- arm S: self-generated, verifier-filtered traces (STaR / rejection-sampling FT).

usage:  python exp_worker_star.py <task> S <n> <seed> [--model ..] [--data ..] [--eval-bs 8] [--save-dir ..]
                                  [--adapter-root ..] [--k 4] [--temperature 0.7] [--top-p 0.95]
                                  [--sample-bs 8] [--samples-dir results/v2/star_samples] [--no-save-adapter]

Procedure (one process, one model instance -- memory-safe on 8 GB):
  1. the n training problems = exp_worker_v2.balanced_sample(pool A, n, seed): the SAME problems A/B/C train on;
  2. the BASE model samples K completions per problem with v2's cot prompt (prompt + COT_SUFFIX), temperature
     0.7, top_p 0.95, top_k off, max_new = MAXNEW[task]; prompts batched (--sample-bs prompts x K sequences);
  3. EXACT gate: keep a completion iff v2.extract(completion) == gold label and it ended with a stop token
     (a completion truncated at the cap has no final line; counted in n_trunc_rejected);
  4. fresh LoRA (v2 config, same torch.manual_seed(seed) before get_peft_model, 3 epochs, lr 2e-4, batch 1) on
     the FIRST kept completion per problem, trained as (PLAIN task prompt -> completion), loss on completion only
     -- the same prompt B trains with, so S and B differ only in who wrote the trace;
  5. eval exactly like arm B: v2.evaluate(plain prompt, greedy, MAXNEW[task]) on the task's test set.
Reported n = requested problems (matched-example budget, same as A/B); n_kept = problems actually trained on.
For div tasks every kept trace is audited: remainder / quotient claims it makes ("X mod d = r", "(10*a + b) mod d
= r", "X divided by d ... remainder r", "X / d = q") are checked exactly; claims about a prefix of N are checked
against step_truth. kept_trace_correct_frac = fraction of kept traces with >= 1 parsed claim and all correct.

outputs: RESULT {json} (v2 fields + keep_rate, item_keep_rate, n_kept, n_kept_yes/no, kept_trace_correct_frac,
  kept_trace_parsed_frac, n_trunc_rejected, sample_s, sampling params); gens as v2
  (<save-dir>/<model_short>/<task>_S_<n>_<seed>.jsonl); all K samples + gate/audit per problem in
  <samples-dir>/<model_short>/<task>_S_<n>_<seed>.jsonl; adapter + train_meta.json as v2.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import exp_worker_v2 as v2          # env vars, torch, DEV, extract, balanced_sample, build_example, evaluate
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DIV_D = {"div2": 2, "div3": 3, "div7": 7, "div11": 11, "div13": 13, "div7_6d": 7}
_N = r"(\d[\d,]*)"
CLAIM_RES = [  # (regex, kind)
    (re.compile(r"\(\s*10\s*\*\s*(\d+)\s*\+\s*(\d)\s*\)\s*(?:mod|%)\s*(\d+)\s*(?:=|is|equals)\s*(\d+)", re.I), "step"),
    (re.compile(_N + r"\s*(?:mod|%)\s*(\d+)\s*(?:=|is|equals)\s*(\d+)", re.I), "mod"),
    (re.compile(_N + r"\s*(?:divided by|÷|/)\s*(\d+)[^.\n]{0,40}?remainder(?:\s+(?:of|is))?\s*(\d+)", re.I), "rem"),
    (re.compile(_N + r"\s*(?:divided by|÷|/)\s*(\d+)\s*(?:=|is|equals)\s*(\d+(?:\.\d+)?)(?![\d.])"
                r"(?!\s*(?:with\s+(?:a\s+)?)?(?:remainder|r\b))", re.I), "quot"),
]


def _int(s):
    return int(s.replace(",", ""))


def audit_trace(text, n, d, rems):
    """Exact audit of the remainder/quotient claims a div-task trace makes. Returns dict."""
    claims, seen = [], set()
    s = str(n)
    step_i = 0
    for rx, kind in CLAIM_RES:
        for m in rx.finditer(text):
            if m.span() in seen:
                continue
            seen.add(m.span())
            if kind == "step":                  # running-remainder step i: check the whole step vs step_truth
                r0, dig, dd, r = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
                if dd != d:
                    continue
                i = step_i
                step_i += 1
                ok = (i < len(rems) and dig == s[i] and r0 == (rems[i - 1] if i else 0) and r == rems[i])
                claims.append({"kind": "step", "i": i, "a": 10 * r0 + int(dig), "claim": str(r), "ok": bool(ok)})
                continue
            else:
                a, dd = _int(m.group(1)), int(m.group(2))
                r = m.group(3)
            if dd != d:
                continue
            if kind == "quot":
                q = float(r)
                ok = (a == d * int(q)) if q.is_integer() else abs(a / d - q) < 0.01
            else:
                r = int(r)
                a_s = str(a)
                if s.startswith(a_s) and 0 < len(a_s) <= len(rems):
                    ok = r == rems[len(a_s) - 1]            # claim about a prefix of N: check vs step_truth
                else:
                    ok = a % d == r
            claims.append({"kind": kind, "a": a, "claim": str(r), "ok": bool(ok)})
    return {"n_claims": len(claims), "all_ok": bool(claims) and all(c["ok"] for c in claims),
            "claims": claims[:40]}


def gate(text, gold, hit_cap):
    """The exact verifier: same extractor as eval; truncated completions have no final line -> reject."""
    return (not hit_cap) and v2.extract(text) == gold


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="arm S: STaR / rejection-sampling fine-tune + eval (one cell)")
    ap.add_argument("task")
    ap.add_argument("arm")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--model", default=v2.DEFAULT_MODEL)
    ap.add_argument("--data", default=str(v2.DEFAULT_DATA))
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--save-dir", default="results/v2/gens")
    ap.add_argument("--adapter-root", default="results/adapters")
    ap.add_argument("--samples-dir", default="results/v2/star_samples")
    ap.add_argument("--no-save-adapter", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sample-bs", type=int, default=8, help="prompts per sampling call (x K sequences)")
    ap.add_argument("--eval-limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.arm != "S":
        ap.error("exp_worker_star.py only runs arm S")
    # fields v2.cell_name / v2.adapter_dir_for expect
    a.eval_task, a.prompt_mode, a.eval_only = a.task, "plain", None
    return a


@torch.no_grad()
def sample_completions(model, tok, items, a, max_new):
    model.eval()
    tok.padding_side = "left"
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    stop_ids |= set(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
    stop_ids.discard(None)
    out_rows = []
    for i in range(0, len(items), a.sample_bs):
        batch = items[i:i + a.sample_bs]
        texts = [tok.apply_chat_template(v2.eval_messages(e["prompt"], "cot", []), tokenize=False,
                                         add_generation_prompt=True) for e in batch]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(v2.DEV)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=a.temperature,
                             top_p=a.top_p, top_k=0, num_return_sequences=a.k, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        del out, enc
        for j, e in enumerate(batch):
            samples = []
            for s in range(a.k):
                ids = gen[j * a.k + s]
                hit_cap = not any(t in stop_ids for t in ids)
                samples.append({"text": tok.decode(ids, skip_special_tokens=True).strip(), "hit_cap": hit_cap})
            out_rows.append(samples)
    if v2.DEV == "cuda":
        torch.cuda.empty_cache()
    return out_rows


def main(argv=None):
    a = parse_args(argv)
    task, n, seed = a.task, a.n, a.seed
    data_path = v2.repo_path(a.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    tdata = data["tasks"][task]
    test = tdata["test"][:a.eval_limit] if a.eval_limit else tdata["test"]
    picked = v2.balanced_sample(tdata["pools"]["A"], n, seed)          # same problems as A/B/C at this seed
    for e in picked:
        e["gold"] = v2.extract(e["completion"])
    max_new = v2.MAXNEW[task]
    d = DIV_D.get(task)
    step_truth = tdata.get("step_truth", {})

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if v2.DEV == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, low_cpu_mem_usage=False, attn_implementation="eager").to(v2.DEV)

    # ---- 1-3: sample K per problem with the cot prompt, exact gate, audit ----
    torch.manual_seed(seed)
    t0 = time.time()
    samples = sample_completions(model, tok, picked, a, max_new)
    t_sample = time.time() - t0
    kept, rows = [], []
    n_pass = n_trunc = 0
    audits = []
    for e, ss in zip(picked, samples):
        first = None
        for s in ss:
            s["pass"] = bool(gate(s["text"], e["gold"], s["hit_cap"]))
            n_pass += s["pass"]
            n_trunc += int(s["hit_cap"] and v2.extract(s["text"]) == e["gold"])
            if s["pass"] and first is None:
                first = s
        audit = None
        if first is not None:
            kept.append({"id": e["id"], "prompt": e["prompt"], "completion": first["text"], "gold": e["gold"]})
            if d is not None:
                st = step_truth.get(e["id"])
                num = st["n"] if st else int(e["id"].rsplit("-", 1)[1])
                rems = st["rems"] if st else [int(str(num)[:i + 1]) % d for i in range(len(str(num)))]
                audit = audit_trace(first["text"], num, d, rems)
                audits.append(audit)
        rows.append({"id": e["id"], "gold": e["gold"], "samples": ss,
                     "kept_index": ss.index(first) if first is not None else None, "audit": audit})

    sdir = v2.repo_path(a.samples_dir) / v2.model_short(a.model)
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{task}_S_{n}_{seed}.jsonl"
    with open(spath, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    base_fields = {
        "task": task, "arm": "S", "n": n, "seed": seed, "req_n": n, "model": a.model, "prompt_mode": "plain",
        "eval_task": task, "eval_only": False, "sample_prompt_mode": "cot", "k": a.k,
        "temperature": a.temperature, "top_p": a.top_p, "top_k": 0, "sample_max_new": max_new,
        "sample_repetition_penalty": getattr(model.generation_config, "repetition_penalty", None),
        "keep_rate": round(n_pass / max(1, len(picked) * a.k), 4),
        "item_keep_rate": round(len(kept) / max(1, len(picked)), 4), "n_kept": len(kept),
        "n_kept_yes": sum(k["gold"] == "Yes" for k in kept), "n_kept_no": sum(k["gold"] == "No" for k in kept),
        "n_trunc_rejected": n_trunc,
        "kept_trace_correct_frac": (round(sum(x["all_ok"] for x in audits) / len(audits), 4)
                                    if d is not None and audits else None),
        "kept_trace_parsed_frac": (round(sum(x["n_claims"] > 0 for x in audits) / len(audits), 4)
                                   if d is not None and audits else None),
        "sample_s": round(t_sample, 1), "samples_path": str(spath.relative_to(v2.REPO))
        if spath.is_relative_to(v2.REPO) else str(spath)}
    if not kept:
        print("RESULT " + json.dumps({**base_fields, "acc": None, "n_test": len(test),
                                      "skip": "no completion passed the exact gate"}))
        return

    # ---- 4: fresh LoRA on the first kept completion per problem (same model instance, weights untouched) ----
    train_tokens = sum(len(k["completion"].split()) for k in kept)
    torch.manual_seed(seed)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    exs = [v2.build_example(tok, k["prompt"], k["completion"]) for k in kept]
    model.train()
    rng = random.Random(seed)
    t1 = time.time()
    for _ep in range(v2.EPOCHS):
        rng.shuffle(exs)
        for ids, labels in exs:
            out = model(input_ids=torch.tensor([ids], device=v2.DEV), labels=torch.tensor([labels], device=v2.DEV))
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    t_train = time.time() - t1
    del opt
    if v2.DEV == "cuda":
        torch.cuda.empty_cache()
    adapter_path = None
    if not a.no_save_adapter:
        ad = v2.adapter_dir_for(a)
        ad.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ad))
        (ad / "train_meta.json").write_text(json.dumps({
            "task": task, "arm": "S", "req_n": n, "n": n, "n_kept": len(kept), "seed": seed, "model": a.model,
            "data": str(a.data), "train_tokens": train_tokens, "train_s": round(t_train, 1),
            "picked_ids": [e["id"] for e in picked], "kept_ids": [k["id"] for k in kept]}), encoding="utf-8")
        adapter_path = str(ad)

    # ---- 5: eval exactly like arm B ----
    t2 = time.time()
    acc, bits, grows = v2.evaluate(model, tok, test, max_new, a.eval_bs, "plain", [], None)
    t_eval = time.time() - t2
    gdir = v2.repo_path(a.save_dir) / v2.model_short(a.model)
    gdir.mkdir(parents=True, exist_ok=True)
    gpath = gdir / f"{v2.cell_name(a)}.jsonl"
    with open(gpath, "w", encoding="utf-8") as fh:
        for r in grows:
            fh.write(json.dumps(r) + "\n")
    print("RESULT " + json.dumps({
        **base_fields, "acc": round(acc, 4), "n_test": len(test), "bits": v2.bits_to_hex(bits),
        "train_tokens": train_tokens, "train_s": round(t_train, 1), "eval_s": round(t_eval, 1),
        "adapter": adapter_path, "max_new": max_new, "eval_bs": a.eval_bs, "repetition_penalty": None,
        "n_hit_cap": sum(r["hit_cap"] for r in grows), "eval_limit": a.eval_limit,
        "gens_path": str(gpath.relative_to(v2.REPO)) if gpath.is_relative_to(v2.REPO) else str(gpath)}))


if __name__ == "__main__":
    main()
