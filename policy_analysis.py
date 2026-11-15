"""Qualitative and sensitivity analysis of learned Q-tables.

1. Policy charts: the learned first-decision action for every (total, upcard), next to
   basic strategy, with disagreements marked.
2. Count deviations: states where the count-aware table picks a different action
   depending on the Hi-Lo true-count bucket.
3. Hyperparameter sensitivity: final return for several epsilon / step-size exponents.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from blackjack_shoe import DOUBLE, Observation
from evaluate import EVAL_SEED, basic_strategy, evaluate_policy
from q_table import STAGES, Stage, greedy_policy, train_q_table
from traditional_baselines import HIT, STICK

STAGE = {stage.name: stage for stage in STAGES}
UPCARDS = list(range(2, 11)) + [1]
HARD_TOTALS = list(range(5, 21))
SOFT_TOTALS = list(range(13, 21))
BUCKETS = (-2, -1, 0, 1, 2)
ACTION_NAME = {STICK: "S", HIT: "H", DOUBLE: "D"}
COLORS = ListedColormap(["#4c78a8", "#e45756", "#54a24b"])  # stick, hit, double


def first_decision(total: int, upcard: int, soft: bool, count: float = 0.0) -> Observation:
    return Observation(total, upcard, soft, 2, True, count)


def _train(job: tuple[str, int, int, int | None, float, float]) -> dict:
    name, seed, hands, decks, epsilon, alpha_power = job
    stage = STAGE[name]
    table = train_q_table(
        stage, hands=hands, seed=seed, num_decks=decks, epsilon=epsilon, alpha_power=alpha_power
    )
    return {"name": name, "seed": seed, "decks": decks, "eps": epsilon, "alpha": alpha_power,
            "table": table}


def _evaluate(job: tuple[dict, int]) -> dict:
    run, eval_hands = job
    stage = STAGE[run["name"]]
    result = evaluate_policy(
        greedy_policy(stage, run["table"]),
        hands=eval_hands,
        seed=EVAL_SEED + run["seed"],
        num_decks=run["decks"],
        allow_double=stage.allow_double,
    )
    return {key: run[key] for key in ("name", "seed", "decks", "eps", "alpha")} | result


def majority_action(stage: Stage, tables: list[dict], observation: Observation) -> int:
    votes = Counter(greedy_policy(stage, table)(observation) for table in tables)
    return votes.most_common(1)[0][0]


def policy_grid(stage: Stage, tables: list[dict], soft: bool, count: float = 0.0):
    totals = SOFT_TOTALS if soft else HARD_TOTALS
    return [[majority_action(stage, tables, first_decision(t, u, soft, count)) for u in UPCARDS]
            for t in totals]


def basic_grid(soft: bool):
    policy = basic_strategy(allow_double=True)
    totals = SOFT_TOTALS if soft else HARD_TOTALS
    return [[policy(first_decision(t, u, soft)) for u in UPCARDS] for t in totals]


def _draw(ax, grid, title, totals, compare=None):
    ax.imshow(grid, cmap=COLORS, vmin=0, vmax=2, aspect="auto")
    for i, row in enumerate(grid):
        for j, action in enumerate(row):
            differs = compare is not None and compare[i][j] != action
            ax.text(j, i, ACTION_NAME[action], ha="center", va="center", fontsize=8,
                    color="black" if differs else "white", fontweight="bold" if differs else None)
            if differs:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2))
    ax.set_xticks(range(len(UPCARDS)), ["A" if u == 1 else str(u) for u in UPCARDS])
    ax.set_yticks(range(len(totals)), totals)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Player total")
    ax.set_title(title, fontsize=10)


def policy_figure(stage: Stage, tables: list[dict], path: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), height_ratios=[2, 1])
    disagreements = {}
    for row, soft in enumerate((False, True)):
        kind = "soft" if soft else "hard"
        totals = SOFT_TOTALS if soft else HARD_TOTALS
        learned, basic = policy_grid(stage, tables, soft), basic_grid(soft)
        _draw(axes[row][0], basic, f"Basic strategy ({kind})", totals)
        _draw(axes[row][1], learned, f"Learned Q-table ({kind}); boxed = differs", totals, basic)
        disagreements[kind] = [
            {"total": totals[i], "upcard": UPCARDS[j],
             "basic": ACTION_NAME[basic[i][j]], "learned": ACTION_NAME[learned[i][j]]}
            for i in range(len(totals)) for j in range(len(UPCARDS)) if basic[i][j] != learned[i][j]
        ]
    fig.suptitle("First decision, S = stick, H = hit, D = double (majority over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return disagreements


def count_deviations(stage: Stage, tables: list[dict], path: Path) -> list[dict]:
    rows = []
    for soft, totals in ((False, HARD_TOTALS), (True, SOFT_TOTALS)):
        for total in totals:
            for upcard in UPCARDS:
                actions = [majority_action(stage, tables, first_decision(total, upcard, soft, b))
                           for b in BUCKETS]
                if len(set(actions)) > 1:
                    rows.append({"total": total, "upcard": "A" if upcard == 1 else upcard,
                                 "soft": soft,
                                 "by_count": {b: ACTION_NAME[a] for b, a in zip(BUCKETS, actions)}})
    grids = [policy_grid(stage, tables, False, b) for b in (-2, 0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, b, grid in zip(axes, (-2, 0, 2), grids):
        _draw(ax, grid, f"Hard totals, true count {b:+d}", HARD_TOTALS,
              None if b == 0 else grids[1])
    fig.suptitle("Count-aware Q-table: boxed cells change relative to count 0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def sensitivity_figure(results: list[dict], path: Path) -> list[dict]:
    summary = []
    for key in sorted({(r["eps"], r["alpha"]) for r in results}):
        values = [r["mean_return"] for r in results if (r["eps"], r["alpha"]) == key]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        summary.append({"epsilon": key[0], "alpha_power": key[1], "mean_return": mean,
                        "seed_range": spread, "seeds": len(values)})
    fig, ax = plt.subplots(figsize=(7, 4))
    for alpha in sorted({s["alpha_power"] for s in summary}):
        points = [s for s in summary if s["alpha_power"] == alpha]
        ax.errorbar([s["epsilon"] for s in points], [s["mean_return"] for s in points],
                    yerr=[s["seed_range"] / 2 for s in points], marker="o", capsize=3,
                    label=f"step size = visits^-{alpha}")
    ax.set_xlabel("Exploration epsilon")
    ax.set_ylabel("Mean return / hand (true reward)")
    ax.set_title("Sensitivity: '+ double down', 6 decks (bars = half seed range)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hands", type=int, default=10_000_000)
    parser.add_argument("--sens-hands", type=int, default=3_000_000)
    parser.add_argument("--eval-hands", type=int, default=2_000_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    seeds = range(args.seeds)

    policy_jobs = [("double", s, args.hands, 6, 0.3, 0.85) for s in seeds]
    count_jobs = [("count_double", s, args.hands, 1, 0.3, 0.85) for s in seeds]
    sens_jobs = [("double", s, args.sens_hands, 6, eps, alpha)
                 for eps in (0.1, 0.3, 0.5) for alpha in (0.6, 0.85, 1.0) for s in seeds]

    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(_train, policy_jobs + count_jobs + sens_jobs))
        policy_runs = runs[: len(policy_jobs)]
        count_runs = runs[len(policy_jobs): len(policy_jobs) + len(count_jobs)]
        sens_runs = runs[len(policy_jobs) + len(count_jobs):]
        sens_results = list(pool.map(_evaluate, [(r, args.eval_hands) for r in sens_runs]))

    args.out.mkdir(exist_ok=True)
    report = {
        "settings": vars(args) | {"out": str(args.out)},
        "policy_vs_basic_6deck": policy_figure(
            STAGE["double"], [r["table"] for r in policy_runs], args.out / "policy_vs_basic.png"),
        "count_deviations_1deck": count_deviations(
            STAGE["count_double"], [r["table"] for r in count_runs],
            args.out / "count_deviations.png"),
        "sensitivity": sensitivity_figure(sens_results, args.out / "sensitivity.png"),
    }
    (args.out / "policy_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "count_deviations_1deck"}, indent=2))
    print(f"{len(report['count_deviations_1deck'])} states change action with the count")


if __name__ == "__main__":
    main()
