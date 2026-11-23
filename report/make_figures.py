"""Figures for Section 4 of the report, drawn from the saved outputs and Q-tables.

    python report/make_figures.py            # summary figure only (reads outputs/*.json)
    python report/make_figures.py --maps     # also the policy maps; trains the 15 Q-tables they
                                             # need (~4 min on 15 cores) unless figures/tables.pkl
                                             # exists, and checks them against policy_analysis.json

Colors follow the report's TikZ palette (teal / rust / ink), validated for colour-vision
deficiency with the dataviz checks: series #0B8F7C, #C0603F; policy fills add #D9A13A.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"

INK = "#182F30"        # primary text (TikZ "ink")
MUTED = "#4F6362"      # secondary text
AXIS = "#B9C3C2"       # hairline axes
GRID = "#E3E8E7"       # hairline grid
TEAL = "#0B8F7C"       # slot 1
RUST = "#C0603F"       # slot 2
GOLD = "#D9A13A"       # third policy fill (double)
T95 = 2.776            # t quantile, 4 degrees of freedom

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["cmss10", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "cmss10",
    "mathtext.it": "cmss10",
    "mathtext.bf": "cmss10",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
})


def _ci(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), T95 * statistics.stdev(xs) / math.sqrt(len(xs))


def _ladder(name: str) -> tuple[dict, dict]:
    data = json.loads((ROOT / "outputs" / name).read_text())
    runs, refs = {}, {}
    for r in data["runs"]:
        runs.setdefault(r["stage"], {})[r["seed"]] = r["final"]
    for r in data["references"]:
        refs.setdefault(r["policy"], {})[r["seed"]] = r["final"]
    return runs, refs


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, pad=5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def summary_figure() -> None:
    """(a) value of the count vs shoe size; (b) bust rate and (c) cost vs bust penalty."""
    gain = {"count": [], "count_double": []}
    for name in ("q_table_ladder.json", "q_table_ladder_2deck.json", "q_table_ladder_1deck.json"):
        runs, refs = _ladder(name)
        gain["count"].append(_ci([runs["count"][s]["mean_return"]
                                  - refs["basic_strategy"][s]["mean_return"] for s in range(5)]))
        gain["count_double"].append(_ci([runs["count_double"][s]["mean_return"]
                                         - refs["basic_strategy_double"][s]["mean_return"]
                                         for s in range(5)]))
    runs, _ = _ladder("q_table_ladder.json")
    stages = ["double", "bust_0.1", "bust_0.25", "bust_0.5"]
    penalties = [0.0, 0.1, 0.25, 0.5]
    bust = [100 * statistics.mean(runs[k][s]["bust_rate"] for s in range(5)) for k in stages]
    lost = [(0.0, 0.0)] + [_ci([runs["double"][s]["mean_return"] - runs[k][s]["mean_return"]
                                for s in range(5)]) for k in stages[1:]]
    exact = json.loads((ROOT / "outputs" / "policy_analysis.json").read_text())
    predicted = [0.0] + [row["cost_per_hand"] for row in exact["bust_penalty"]["exact_infinite_deck"]]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.05), gridspec_kw={"width_ratios": [1.12, 1, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.86, wspace=0.42)

    ax = axes[0]
    x = [0, 1, 2]
    for key, color, label in (("count", TEAL, "+ count"),
                              ("count_double", RUST, "+ count + double")):
        m = [v[0] * 1000 for v in gain[key]]
        e = [v[1] * 1000 for v in gain[key]]
        ax.errorbar(x, m, yerr=e, color=color, lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8,
                    capsize=2, elinewidth=0.9, label=label, zorder=3)
        ax.annotate(f"$+{m[-1]:.1f}$", (x[-1], m[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_xticks(x, ["6 decks", "2 decks", "1 deck"])
    ax.set_xlim(-0.3, 2.55)
    ax.set_ylim(-0.4, 3.5)
    ax.set_ylabel(r"Gain over basic strategy ($\times 10^{-3}$)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    _style(ax, "(a) The count pays off as the shoe shrinks")

    ax = axes[1]
    ax.plot(penalties, bust, color=TEAL, lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8, zorder=3)
    for i in (0, 3):
        ax.annotate(f"{bust[i]:.1f}%", (penalties[i], bust[i]), xytext=(7, 1) if i == 0 else (-6, 7),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xticks(penalties, ["0", "0.1", "0.25", "0.5"])
    ax.set_xlim(-0.03, 0.6)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Bust penalty $p$")
    ax.set_ylabel("Bust rate (%)")
    _style(ax, "(b) The penalty cuts busts")

    ax = axes[2]
    ax.plot(penalties, [p * 1000 for p in predicted], color=INK, lw=1.0, ls=(0, (3, 2)),
            marker="o", ms=6.5, mfc="white", mec=INK, mew=0.9, label="Exact prediction", zorder=2)
    ax.errorbar(penalties[1:], [v[0] * 1000 for v in lost[1:]], yerr=[v[1] * 1000 for v in lost[1:]],
                color=TEAL, ls="none", marker="o", ms=3.8, mec=TEAL, mew=0.5, capsize=2,
                elinewidth=0.9, label="Measured", zorder=3)
    ax.set_xticks(penalties, ["0", "0.1", "0.25", "0.5"])
    ax.set_xlim(-0.03, 0.6)
    ax.set_ylim(0, 34)
    ax.set_xlabel("Bust penalty $p$")
    ax.set_ylabel(r"True return lost ($\times 10^{-3}$)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    _style(ax, "(c) ...at the predicted cost")

    OUT.mkdir(exist_ok=True)
    _save(fig, "summary")


# ---------------------------------------------------------------- policy maps
ACTION_COLOR = None  # filled in by _policy_imports (needs the repo's action ids)


def _policy_imports():
    sys.path.insert(0, str(ROOT))
    import policy_analysis as pa  # noqa: E402
    global ACTION_COLOR
    ACTION_COLOR = {pa.STICK: TEAL, pa.HIT: RUST, pa.DOUBLE: GOLD}
    return pa


def _tint(hex_color: str, keep: float) -> tuple[float, float, float]:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - keep * (1 - c) for c in (r, g, b))


def _draw_grid(ax, pa, grid, totals, title, compare=None, fade_unchanged=False, show_y=True):
    """One policy map. compare: grid to box differences against; fade_unchanged: emphasis mode."""
    for i, row in enumerate(grid):
        for j, action in enumerate(row):
            differs = compare is not None and compare[i][j] != action
            faded = fade_unchanged and not differs
            color = ACTION_COLOR[action]
            face = _tint(color, 0.28) if faded else color
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", lw=0.8))
            text = INK if action == pa.DOUBLE or faded else "white"
            ax.text(j + 0.5, i + 0.53, pa.ACTION_NAME[action], ha="center", va="center",
                    fontsize=6.0 if not differs else 6.4, color=MUTED if faded else text,
                    fontweight="bold" if differs else None)
            if differs:
                ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False,
                                       edgecolor=INK, lw=1.1, zorder=4))
    ax.set_xlim(0, len(pa.UPCARDS))
    ax.set_ylim(len(totals), 0)
    ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(len(pa.UPCARDS))],
                  ["A" if u == 1 else str(u) for u in pa.UPCARDS])
    ax.set_yticks([i + 0.5 for i in range(len(totals))], [str(t) for t in totals] if show_y else [])
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", color=INK, pad=4)


def _legend(fig, pa, y, extra=None):
    handles = [Patch(facecolor=ACTION_COLOR[a], edgecolor="none", label=n)
               for a, n in ((pa.STICK, "Stand (S)"), (pa.HIT, "Hit (H)"), (pa.DOUBLE, "Double (D)"))]
    if extra:
        handles.append(Patch(facecolor="white", edgecolor=INK, lw=1.3, label=extra))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               frameon=False, handlelength=1.1, handleheight=1.1, columnspacing=1.6)


HARD = list(range(8, 18))      # hard 5-7 always hit, 18+ always stand
SOFT = list(range(13, 20))     # soft 20 always stands
LATER = list(range(12, 18))    # below 12 a hit cannot bust


CELL = 0.19   # inches per policy cell, shared by every policy map


def _grid_figure(rows: list[int], labelled: list[bool], cell: float = CELL):
    """Panels of 10 columns with the same cell size, top-aligned; returns fig, axes."""
    CELL = cell  # noqa: N806
    left, ylab, gap, right, top, bottom = 0.42, 0.24, 0.12, 0.06, 0.26, 0.66
    widths = [10 * CELL for _ in rows]
    W = left + sum(widths) + sum(gap + (ylab if lab else 0) for lab in labelled[1:]) + right
    H = top + max(rows) * CELL + bottom
    fig = plt.figure(figsize=(W, H))
    axes, x = [], left
    for i, (n, lab) in enumerate(zip(rows, labelled)):
        if i:
            x += gap + (ylab if lab else 0)
        axes.append(fig.add_axes([x / W, 1 - (top + n * CELL) / H, 10 * CELL / W, n * CELL / H]))
        x += 10 * CELL
    fig.text(left / W + (x - left) / W / 2, 0.36 / H, "Dealer upcard", ha="center", color=INK)
    fig.text(0.07 / W, 1 - (top + max(rows) * CELL / 2) / H, "Player total", rotation=90,
             va="center", ha="left", color=INK)
    return fig, axes


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def policy_figure(pa, tables) -> None:
    """Learned 6-deck double table vs basic strategy: first decision (hard, soft), later (hard)."""
    stage = pa.STAGE["double"]
    basic = pa.basic_strategy(allow_double=True)
    panels = ((False, HARD, True, "Hard, first decision"), (True, SOFT, True, "Soft, first decision"),
              (False, LATER, False, "Hard, later decisions"))
    fig, axes = _grid_figure([len(p[1]) for p in panels], [True] * 3, cell=0.172)
    for ax, (soft, totals, first, name) in zip(axes, panels):
        learned = pa.policy_grid(stage, tables, soft, totals=totals, first=first)
        ref = [[basic(pa.decision(t, u, soft, first=first)) for u in pa.UPCARDS] for t in totals]
        _draw_grid(ax, pa, learned, totals, name, compare=ref)
    # the rows left out must hold no difference
    hidden = ((False, [5, 6, 7, 18, 19, 20], True), (True, [20], True), (False, [18, 19, 20], False))
    for soft, totals, first in hidden:
        learned = pa.policy_grid(stage, tables, soft, totals=totals, first=first)
        ref = [[basic(pa.decision(t, u, soft, first=first)) for u in pa.UPCARDS] for t in totals]
        assert learned == ref, (soft, totals, first)
    _legend(fig, pa, 0.0, extra="Differs from basic strategy")
    _save(fig, "policy_vs_basic")


def count_figure(pa, tables) -> None:
    stage = pa.STAGE["count_double"]
    grids = {c: pa.policy_grid(stage, tables, False, count=c, totals=HARD) for c in (-2, 0, 2)}
    fig, axes = _grid_figure([len(HARD)] * 3, [True, False, False])
    _draw_grid(axes[0], pa, grids[-2], HARD, "True count $-2$: more hits",
               compare=grids[0], fade_unchanged=True)
    _draw_grid(axes[1], pa, grids[0], HARD, "True count $0$", show_y=False)
    _draw_grid(axes[2], pa, grids[2], HARD, "True count $+2$: more stands, doubles",
               compare=grids[0], fade_unchanged=True, show_y=False)
    _legend(fig, pa, 0.0, extra="Differs from count 0")
    _save(fig, "count_deviations")


def penalty_figure(pa, true_tables, penalty_tables) -> None:
    plain_stage, pen_stage = pa.STAGE["double"], pa.STAGE["bust_0.1"]
    fig, axes = _grid_figure([len(HARD), len(LATER)], [True, True])
    for ax, first, totals, name in ((axes[0], True, HARD, "First decision"),
                                    (axes[1], False, LATER, "Later decisions (3+ cards)")):
        plain = pa.policy_grid(plain_stage, true_tables, False, totals=totals, first=first)
        pen = pa.policy_grid(pen_stage, penalty_tables, False, totals=totals, first=first)
        _draw_grid(ax, pa, pen, totals, name, compare=plain)
    _legend(fig, pa, 0.0, extra="Changed by the penalty")
    _save(fig, "bust_penalty_policy")


def train_tables(pa) -> dict:
    """The tables behind the maps: same stages, seeds and settings as policy_analysis.py."""
    from concurrent.futures import ProcessPoolExecutor
    groups = {"policy": ("double", 6), "count": ("count_double", 1), "bust": ("bust_0.1", 6)}
    jobs = [(name, seed, 30_000_000, decks, pa.EPSILON, pa.ALPHA_POWER)
            for name, decks in groups.values() for seed in range(5)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        runs = list(pool.map(pa._train, jobs))
    return {group: [r["table"] for r in runs if r["name"] == name and r["decks"] == decks]
            for group, (name, decks) in groups.items()}


def check_against_json(pa, tables) -> None:
    """The retrained tables must reproduce the saved analysis exactly."""
    saved = json.loads((ROOT / "outputs" / "policy_analysis.json").read_text())
    dis = pa.compare_with_basic(pa.STAGE["double"], tables["policy"])["disagreements"]
    got = sorted((d["kind"], d["total"], d["upcard"], d["learned"]) for d in dis)
    want = sorted((d["kind"], d["total"], d["upcard"], d["learned"])
                  for d in saved["policy_vs_basic_6deck"]["disagreements"])
    assert got == want, (got, want)
    stage = pa.STAGE["count_double"]
    n = sum(1 for soft, totals in ((False, pa.HARD_TOTALS), (True, pa.SOFT_TOTALS))
            for t in totals for u in pa.UPCARDS
            if len({pa.majority_action(stage, tables["count"], pa.decision(t, u, soft, b))
                    for b in range(-2, 3)}) > 1)
    assert n == len(saved["count_deviations_1deck"]), n
    print("retrained tables match policy_analysis.json:", len(got), "disagreements,", n, "count cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", action="store_true", help="also draw the policy maps")
    parser.add_argument("--tables", type=Path, default=OUT / "tables.pkl",
                        help="cache of the trained Q-tables (created if missing)")
    args = parser.parse_args()
    summary_figure()
    if args.maps:
        pa = _policy_imports()
        if not args.tables.exists():
            args.tables.write_bytes(pickle.dumps(train_tables(pa)))
        tables = pickle.loads(args.tables.read_bytes())
        check_against_json(pa, tables)
        policy_figure(pa, tables["policy"])
        count_figure(pa, tables["count"])
        penalty_figure(pa, tables["policy"], tables["bust"])
    print("figures written to", OUT)
