"""make_figures.py -- paper/fig_map.pdf and paper/fig_arms.pdf from the same files as make_numbers.py.

fig_map : (a) arm B on every divisibility cell with saved generations (Qwen2.5-1.5B): answer accuracy and
          per-step accuracy p against m = k n / (10 d) (results/analysis/transitions.json); hollow = observed
          before S7 was registered, filled = cells predicted in advance by S7 or S9. (b) arm A accuracy against
          the fair surface probe (CV-selected; one-hot features only on div2/div3/div11).
fig_arms: accuracy by arm per task, pooled over seeds with a 95% Wilson interval, per-seed dots; one row
          of panels per model that has runs. Missing cells are simply absent.
usage: python paper/exp/make_figures.py
"""
from __future__ import annotations

import math
import os
import sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_numbers as MN  # noqa: E402  (loads runs, tasks, gens, analysis JSON; writes nothing on import)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PAPER = MN.PAPER
N = MN.MAIN_N
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARM_COLOR = {"base": "#8a8984", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "Bprime": "#4a3aa7"}
ARM_LABEL = {"base": "base", "A": "A answers", "B": "B trace+answer", "C": "C scrambled",
             "Bprime": "B$'$ answer+trace"}
WIN_STYLE = {"A": ("#2a78d6", "o", "A wins"), "B": ("#eb6834", "s", "B wins"),
             "both": ("#1baf7a", "^", "both work"), "neither": ("#8a8984", "X", "neither"),
             "mixed": ("#b9b8b2", "D", "unresolved")}
TASK_ORDER = ["div2", "div3", "div7", "div11", "div13", "div7_6d", "prime", "valid"]
TASK_LABEL = {"div7_6d": "div7 (6-digit)"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif"})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def winner(task, model=MN.M15):
    """PREDICTIONS.md: a win is >= 10 pp with all paired seeds agreeing; 'neither' = both <= majority + 8 pp."""
    common = [s for s in MN.seeds(task, "A", N, model) if MN.run(task, "B", N, s, model)]
    if not common:
        return None
    d = [MN.run(task, "B", N, s, model)["acc"] - MN.run(task, "A", N, s, model)["acc"] for s in common]
    a, b = MN.acc_mean(task, "A", N, model), MN.acc_mean(task, "B", N, model)
    if all(x >= 0.10 for x in d):
        return "B"
    if all(x <= -0.10 for x in d):
        return "A"
    if a <= 0.58 and b <= 0.58:
        return "neither"
    if a > 0.58 and b > 0.58:
        return "both"
    return "mixed"


def predicted_trace(task, model=MN.M15):
    """p^k with p = p_cond from results/analysis/steps.json (pooled over seeds); falls back to the local step
    accuracy recomputed from saved arm-B generations; None when no generations exist."""
    cells = MN.steps_cells(task, model=model, eval_task=task)
    if cells:
        p = MN.pooled(cells, "p_cond")
        k = cells[0]["k"]
        if p is not None:
            return p ** k
    rows = []
    for (label, arm, n, seed, m), g in MN.GENS.items():
        if label == task and arm == "B" and n == N and m == MN.C.short_model(model):
            rows += g["rows"]
    if not rows:
        return None
    if task.startswith("div"):
        d = MN.TASKS[task]["meta"].get("d", 7) if MN.TASKS[task]["meta"] else 7
        k = len(rows[0]["id"].split("-")[1])
        p, _full = MN.div_step_accuracy(rows, d)
        return None if p is None else p ** k
    if task == "prime":
        p = MN.prime_step_accuracy(rows)
        if p is None:
            return None
        ks = []
        for x in rows:
            n = int(x["id"].split("-")[1])
            r, k = math.isqrt(n), 0
            for q in MN.C.SMALL_PRIMES:
                if q > r:
                    break
                k += 1
                if n % q == 0:
                    break
            ks.append(k)
        return mean(p ** k for k in ks)
    return None


def fig_map():
    """(a) arm B on div tasks: accuracy and per-step p against m = k n / (10 d) (transitions.json, analysis_steps.py),
    hollow = observed before S7 was registered, filled = predicted in advance (S7 / S9 cells);
    (b) arm A accuracy against the fair surface probe."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.6, 2.55), gridspec_kw={"width_ratios": [1.25, 1]})
    rows = MN.trans_rows()
    for r in rows:
        key = (r["task"], r["n"])
        pre = key in MN.PRE_S7_CELLS
        pred = key in MN.S7_CELLS or key in MN.S9_CELLS
        for val, col, mk in ((r["accuracy"], ARM_COLOR["B"], "o"), (r["p"], ARM_COLOR["Bprime"], "s")):
            ax.scatter([r["m"]], [100 * val], s=30, marker=mk, facecolor=col if pred else "white",
                       edgecolor=col if (pre or pred) else "#b9b8b2", linewidth=1.3, zorder=3)
        lab = TASK_LABEL.get(r["task"], r["task"]) + ("" if r["n"] == N else " n=%d" % r["n"])
        ax.annotate(lab, (r["m"], 100 * min(r["accuracy"], r["p"])), xytext=(4, -9), textcoords="offset points",
                    fontsize=6.5, color=INK)
    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.set_xlim(2, 45)
    ax.set_ylim(40, 103)
    ax.set_xlabel("transitions per table entry $m = kn/(10d)$")
    ax.set_ylabel("%")
    ax.set_title("(a) arm B, divisibility tasks", fontsize=8, loc="left")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    handles = [Line2D([], [], marker="o", ls="", color=ARM_COLOR["B"], markersize=5, label="B accuracy"),
               Line2D([], [], marker="s", ls="", color=ARM_COLOR["Bprime"], markersize=5, label="per-step $p$"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, markersize=5,
                      label="observed before S7"),
               Line2D([], [], marker="o", ls="", color=MUTED, markersize=5, label="predicted in advance")]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    for task in TASK_ORDER:
        x = MN.fair_probe(task)
        a = MN.acc_mean(task, "A", N)
        if x is None or a is None:
            continue
        bx.scatter([100 * x], [100 * a], s=30, color=ARM_COLOR["A"], edgecolor="white", lw=0.8, zorder=3)
        if not task.startswith("div"):
            bx.annotate(TASK_LABEL.get(task, task), (100 * x, 100 * a), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=INK)
    divs = [t for t in TASK_ORDER if t.startswith("div") and MN.fair_probe(t) is not None and MN.acc_mean(t, "A", N)]
    if divs:
        bx.annotate("divisibility tasks (%d)" % len(divs), (52, 50), xytext=(58, 43), fontsize=6.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    bx.plot([40, 102], [40, 102], color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    bx.set_xlim(40, 104)
    bx.set_ylim(40, 104)
    bx.set_xlabel("surface probe accuracy (%)")
    bx.set_ylabel("arm A accuracy (%)")
    bx.set_title("(b) arm A vs. surface probe", fontsize=8, loc="left")
    bx.grid(color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, "fig_map.pdf"))
    plt.close(fig)


def fig_arms():
    models = [m for m in MN.MODELS.values() if any(k[4] == m for k in MN.IDX)]
    arms = ["base", "A", "B", "C", "Bprime"]
    fig, axes = plt.subplots(len(models), 1, figsize=(5.5, 1.9 * len(models) + 0.35), squeeze=False)
    for row, model in enumerate(models):
        ax = axes[row][0]
        tasks = [t for t in TASK_ORDER if any(MN.seeds(t, a, N if a != "base" else 0, model) for a in arms)]
        width = 0.16
        for i, task in enumerate(tasks):
            present = [a for a in arms if MN.seeds(task, a, N if a != "base" else 0, model)]
            for j, arm in enumerate(present):
                n = N if arm != "base" else 0
                rs = [MN.run(task, arm, n, s, model) for s in MN.seeds(task, arm, n, model)]
                k, tot = sum(r["k"] for r in rs), sum(r["n_test"] for r in rs)
                lo, hi = wilson(k, tot)
                xpos = i + (j - (len(present) - 1) / 2) * width
                ax.plot([xpos, xpos], [100 * lo, 100 * hi], color=ARM_COLOR[arm], lw=2, solid_capstyle="round",
                        zorder=2)
                ax.scatter([xpos], [100 * k / tot], s=18, color=ARM_COLOR[arm], edgecolor="white", lw=0.8,
                           zorder=3)
                ax.scatter([xpos + 0.05] * len(rs), [100 * r["acc"] for r in rs], s=6, color=ARM_COLOR[arm],
                           alpha=0.55, lw=0, zorder=3)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks])
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.set_ylim(0, 102)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(MN.C.short_model(model), fontsize=8, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    shown = [a for a in arms if any(k[1] == a and k[4] in models and k[2] in (0, N) for k in MN.IDX)]
    handles = [Line2D([], [], marker="o", color=ARM_COLOR[a], lw=2, markersize=4, label=ARM_LABEL[a])
               for a in shown]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(PAPER, "fig_arms.pdf"))
    plt.close(fig)


def main():
    fig_map()
    fig_arms()
    print("wrote paper/fig_map.pdf, paper/fig_arms.pdf")


if __name__ == "__main__":
    main()
