#!/usr/bin/env python
"""fig_story.pdf: the paper's three findings in one figure, from the committed run records.

(a) Erasure and retention on div7: zero-shot base, answers-only A, trace B, and self-generated
    verifier-filtered S at 1.5B, 3B and 7B (points: seeds; bars: mean). Answer-only fine-tuning pulls
    a model that already divides (3B, 7B) to chance; its own filtered traces keep the computation.
(b) The primality shortcut: arm A's accuracy on test items where the rule "last digit is 1, 3, 7
    or 9" gives the right answer versus items where it does not (composites ending in 1, 3, 7, 9),
    for every A cell with saved generations; B and base for contrast.
(c) What the trace must be (div7, 1.5B): base, zero-shot CoT prompt, 4-shot demonstrations,
    A, B, C (scrambled), D (correct trace of a different number), B' (answer then trace), S.

usage: python paper/exp/make_fig_story.py   (writes paper/fig_story.pdf)
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PAPER = os.path.join(ROOT, "paper")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"base": "#8a8984", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "Bprime": "#4a3aa7",
       "D": "#b9a11c", "S": "#c2185b", "cot": "#8a8984", "fewshot4": "#8a8984"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif"})
MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]
SHORT = {"Qwen2.5-0.5B-Instruct": "0.5B", "Qwen2.5-1.5B-Instruct": "1.5B", "Qwen2.5-3B-Instruct": "3B",
         "Qwen2.5-7B-Instruct": "7B"}


def load_runs():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "v2", "runs_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            r["model"] = r.get("model", "").split("/")[-1]
            rows.append(r)
    for f in glob.glob(os.path.join(ROOT, "results", "v2", "evals_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            r["model"] = r.get("model", "").split("/")[-1]
            r.setdefault("arm", "base")
            r.setdefault("n", 0)
            rows.append(r)
    for f in ("runs.jsonl", "runs_L1.jsonl"):
        for l in open(os.path.join(ROOT, "results", "thinking_vs_data", f), encoding="utf-8"):
            r = json.loads(l)
            r.setdefault("model", "Qwen2.5-1.5B-Instruct")
            r["model"] = r["model"].split("/")[-1]
            rows.append(r)
    seen, out = set(), []
    for r in rows:
        key = (r["task"], r["arm"], r["n"], r["seed"], r["model"], r.get("prompt_mode") or "plain",
               bool(r.get("eval_only")), r.get("eval_task") or r["task"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def cells(runs, task, arm, model, n=180, prompt="plain"):
    return sorted([r for r in runs if r["task"] == task and r["arm"] == arm and r["model"] == model
                   and (r.get("eval_task") or task) == task and (r.get("prompt_mode") or "plain") == prompt
                   and (r["n"] == n or arm == "base")], key=lambda r: r["seed"])


def dots(ax, x, accs, color, w=0.18):
    if not accs:
        return
    m = 100 * sum(accs) / len(accs)
    ax.plot([x - w, x + w], [m, m], color=color, lw=2.2, solid_capstyle="round", zorder=3)
    for i, a in enumerate(accs):
        ax.scatter([x + (i - (len(accs) - 1) / 2) * 0.09], [100 * a], s=14, color=color, edgecolor="white",
                   lw=0.6, zorder=4)


def panel_a(ax, runs):
    arms = ["base", "A", "B", "S"]
    labels = {"base": "zero-shot", "A": "A answers", "B": "B trace", "S": "S own traces"}
    for gi, model in enumerate(MODELS):
        for ai, arm in enumerate(arms):
            accs = [r["acc"] for r in cells(runs, "div7", arm, model)]
            dots(ax, gi * 5 + ai, accs, COL[arm])
        ax.text(gi * 5 + 1.5, 103, SHORT[model], ha="center", fontsize=8, color=INK)
    ax.axhline(50, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.set_xticks([gi * 5 + ai for gi in range(3) for ai in range(4)])
    ax.set_xticklabels([labels[a] for _ in range(3) for a in arms], rotation=60, ha="right", fontsize=6.3)
    ax.set_ylim(40, 108)
    ax.set_ylabel("div7 accuracy (%)")
    ax.set_title("(a) erasure and retention (div7)", fontsize=7.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def split_prime(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    cons, incons = [], []
    for r in rows:
        n = int(r["id"].split("-")[-1])
        rule_ans = "Yes" if n % 10 in (1, 3, 7, 9) else "No"
        ok = str(r["correct"]) in ("1", "True")
        (cons if rule_ans == r["gold"] else incons).append(ok)
    return 100 * sum(cons) / len(cons), 100 * sum(incons) / len(incons), len(cons), len(incons)


def panel_b(ax):
    order = [("Qwen2.5-1.5B-Instruct", "A"), ("Qwen2.5-3B-Instruct", "A"), ("Qwen2.5-7B-Instruct", "A"),
             ("Qwen2.5-1.5B-Instruct", "B"), ("Qwen2.5-3B-Instruct", "B"), ("Qwen2.5-7B-Instruct", "B")]
    xt, xl = [], []
    for i, (model, arm) in enumerate(order):
        paths = sorted(glob.glob(os.path.join(ROOT, "results", "v2", "gens", model, f"prime_{arm}_180_*.jsonl")))
        paths = [p for p in paths if "reeval" not in p]
        if not paths:
            continue
        for j, p in enumerate(paths):
            c, ic, nc, nic = split_prime(p)
            off = (j - (len(paths) - 1) / 2) * 0.12
            ax.scatter([i + off], [c], s=16, marker="o", color=COL[arm], edgecolor="white", lw=0.6, zorder=4)
            ax.scatter([i + off], [ic], s=18, marker="X", color=COL[arm], edgecolor="white", lw=0.4, zorder=4)
            ax.plot([i + off, i + off], [ic, c], color=COL[arm], lw=0.8, alpha=0.5, zorder=2)
        xt.append(i)
        xl.append(f"{arm} {SHORT[model]}")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=6.3)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("primality accuracy (%)")
    ax.set_title("(b) the primality shortcut", fontsize=7.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    h = [Line2D([], [], marker="o", color=MUTED, lw=0, markersize=4, label="rule correct (198 items)"),
         Line2D([], [], marker="X", color=MUTED, lw=0, markersize=5, label="rule wrong (42 composites)")]
    ax.legend(handles=h, loc="lower left", frameon=False, fontsize=6, handletextpad=0.3, borderaxespad=0.2)


def panel_c(ax, runs, model="Qwen2.5-1.5B-Instruct"):
    items = [("base", "plain", "zero-shot"), ("base", "cot", "CoT prompt"), ("base", "fewshot4", "4-shot traces"),
             ("A", "plain", "A answers"), ("C", "plain", "C scrambled"), ("D", "plain", "D other number"),
             ("Bprime", "plain", "B$'$ answer first"), ("S", "plain", "S own traces"), ("B", "plain", "B trace")]
    xl = []
    for i, (arm, prompt, lab) in enumerate(items):
        accs = [r["acc"] for r in cells(runs, "div7", arm, model, prompt=prompt)]
        dots(ax, i, accs, COL[arm] if prompt == "plain" else COL["base"])
        xl.append(lab)
    ax.axhline(50, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(xl, rotation=60, ha="right", fontsize=6.3)
    ax.set_ylim(40, 108)
    ax.set_ylabel("div7 accuracy (%), 1.5B")
    ax.set_title("(c) what the trace must be (div7, 1.5B)", fontsize=7.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def main():
    runs = load_runs()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.35, 0.85, 1.05]})
    panel_a(axes[0], runs)
    panel_b(axes[1])
    panel_c(axes[2], runs)
    fig.tight_layout(w_pad=1.2)
    out = os.path.join(PAPER, "fig_story.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    # numbers used in the caption
    for model in MODELS:
        for arm in ("base", "A", "B", "S"):
            print(SHORT[model], arm, [round(100 * r["acc"], 1) for r in cells(runs, "div7", arm, model)])
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "v2", "gens", "*", "prime_[AB]_180_*.jsonl"))):
        if "reeval" not in p:
            print(os.path.relpath(p, ROOT), [round(x, 1) for x in split_prime(p)[:2]])


if __name__ == "__main__":
    main()
