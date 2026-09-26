"""exp_thinking_ornith_gen.py -- Phase 0b: OPEN-SOURCE AI THINKING via local ornith:9b ($0).

Two lanes, both STaR-style rejection sampling (keep only traces whose extracted final answer agrees
with the exact gate label; coverage reported honestly):

  B-orn : for the PRIME task, ask ornith:9b (Ollama, OpenAI-compatible endpoint) to think step by
          step then answer. <=MAX_SAMPLES generations at T=0.7 per problem; the first gate-agreeing
          trace is kept verbatim (its own words). Coverage = fraction of problems with a kept trace.
  B-mv  : for the VALID task, thinking is generated through the REAL Multiverse machinery
          (axuniv.newaxiom.multiverse.simulate_conversations) with a ~15-line local reasoner shim
          over Ollama -- the product's own branch structure as a thinking generator.

Resumable: appends results/thinking_vs_data/{ornith_traces,mv_traces}.jsonl, skips done ids.
Ends by releasing VRAM (keep_alive: 0) so the LoRA workers can have the GPU.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "thinking_vs_data")
DATA = os.path.join(OUT_DIR, "datasets.json")
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("ORNITH_MODEL", "ornith:9b")
MAX_SAMPLES = 3
N_ORN_PRIME = 260            # pool to sample so >=180 covered examples survive rejection
N_MV_VALID = 90              # pool for the n=60 B-mv pilot
ANS_RE = re.compile(r"Answer:\s*(Yes|No)", re.I)
THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.I)

SYS = ("Think through the problem step by step, showing your reasoning, then give the final answer. "
       "You MUST end your reply with exactly 'Answer: Yes' or 'Answer: No'.")


def _chat(prompt: str, *, temperature: float = 0.7, keep_alive="10m", max_tokens: int = 700) -> str:
    body = json.dumps({"model": MODEL, "stream": False, "keep_alive": keep_alive,
                       "options": {"temperature": temperature, "num_predict": max_tokens},
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.loads(r.read()).get("message") or {}).get("content") or ""


def _extract(reply: str):
    """(thinking_text, answer) from an ornith reply; ornith natively emits <think> blocks."""
    m = ANS_RE.findall(reply)
    ans = m[-1].capitalize() if m else None
    thinks = THINK_RE.findall(reply)
    if thinks:
        thinking = " ".join(t.strip() for t in thinks)
    else:                                              # no think block: everything before the answer line
        thinking = ANS_RE.split(reply)[0].strip()
    thinking = re.sub(r"\s+", " ", thinking)[:2400]
    return thinking, ans


def _done_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                ids.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    return ids


def gen_orn(tasks):
    """B-orn: ornith thinking for prime (STaR)."""
    path = os.path.join(OUT_DIR, "ornith_traces.jsonl")
    done = _done_ids(path)
    pool = tasks["prime"]["pools"]["B"][:N_ORN_PRIME]
    labels = {e["id"]: e["completion"].rstrip().rsplit("Answer: ", 1)[-1] for e in pool}
    kept = len(done)
    tried = 0
    with open(path, "a", encoding="utf-8") as f:
        for e in pool:
            if e["id"] in done:
                continue
            tried += 1
            gold = labels[e["id"]]
            rec = None
            for s in range(MAX_SAMPLES):
                try:
                    reply = _chat(e["prompt"], temperature=0.7 if s else 0.3)
                except Exception as exc:  # noqa: BLE001
                    print("  ollama error:", str(exc)[:80])
                    time.sleep(3)
                    continue
                thinking, ans = _extract(reply)
                if ans == gold and len(thinking.split()) >= 10:
                    rec = {"id": e["id"], "prompt": e["prompt"],
                           "completion": thinking + f"\nAnswer: {gold}",
                           "samples_used": s + 1, "source": MODEL}
                    break
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if tried % 10 == 0:
                print(f"  orn prime: tried={tried} kept={kept}")
    total = len(pool)
    print(f"B-orn coverage: {kept}/{total} = {kept / max(1, total):.2f}")
    return kept


class _OllamaShim:
    """Minimal reasoner contract for multiverse.simulate_conversations ($0 local)."""

    def available(self):
        try:
            urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5)
            return True
        except Exception:  # noqa: BLE001
            return False

    def role_answer(self, role, message, system=""):
        try:
            return (_chat(message + "\n\n" + "You MUST end with 'Answer: Yes' or 'Answer: No'.",
                          temperature=0.6, max_tokens=400), MODEL)
        except Exception:  # noqa: BLE001
            return ("", "")

    def answer(self, message, system=""):
        try:
            return _chat(message, temperature=0.4, max_tokens=400)
        except Exception:  # noqa: BLE001
            return ""


def gen_mv(tasks):
    """B-mv: thinking through the real Multiverse branch machinery (k=2, turns=1), STaR-filtered."""
    from axuniv.newaxiom import multiverse as mv
    path = os.path.join(OUT_DIR, "mv_traces.jsonl")
    done = _done_ids(path)
    pool = tasks["valid"]["pools"]["B"][:N_MV_VALID]
    labels = {e["id"]: e["completion"].rstrip().rsplit("Answer: ", 1)[-1] for e in pool}
    shim = _OllamaShim()
    kept = len(done)
    with open(path, "a", encoding="utf-8") as f:
        for i, e in enumerate(pool):
            if e["id"] in done:
                continue
            gold = labels[e["id"]]
            try:
                res = mv.simulate_conversations(shim, e["prompt"], k=2, turns=1)
            except Exception as exc:  # noqa: BLE001
                print("  mv error:", str(exc)[:80])
                continue
            best = None
            for b in (res.get("branches") or []):
                text = " ".join(t.get("text", "") for t in b.get("transcript") or [])
                thinking, ans = _extract(text)
                if ans == gold and len(thinking.split()) >= 8:
                    best = thinking
                    break
            if best:
                f.write(json.dumps({"id": e["id"], "prompt": e["prompt"],
                                    "completion": best + f"\nAnswer: {gold}",
                                    "source": f"multiverse+{MODEL}"}, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if (i + 1) % 10 == 0:
                print(f"  mv valid: {i + 1}/{len(pool)} kept={kept}")
    print(f"B-mv coverage: {kept}/{len(pool)} = {kept / max(1, len(pool)):.2f}")
    return kept


def main():
    tasks = json.load(open(DATA, encoding="utf-8"))["tasks"]
    t0 = time.time()
    gen_orn(tasks)
    gen_mv(tasks)
    try:                                                # release VRAM for the LoRA workers
        _chat("done", keep_alive=0, max_tokens=1)
    except Exception:  # noqa: BLE001
        pass
    print(f"DONE in {time.time() - t0:.0f}s; VRAM released (keep_alive 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
