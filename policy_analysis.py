"""Qualitative and sensitivity analysis of learned Q-tables.

1. Learned policy vs basic strategy, with value gaps for each disagreement.
2. Count deviations with buckets clipped at +/-2 and +/-3.
3. Bust penalty: learned policy changes and the exact infinite-deck cost.
4. Hyperparameter sensitivity.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from blackjack_shoe import DOUBLE, BlackjackShoe, Observation, count_bucket
from evaluate import EVAL_SEED, basic_strategy, double_value, evaluate_policy
from q_table import ALPHA_POWER, EPSILON, STAGES, Stage, greedy_policy, train_q_table
from traditional_baselines import (
    CARD_PROBABILITIES,
    HIT,
    STATES,
    STICK,
    BlackjackModel,
    State,
    _add_card,
    policy_iteration,
)
from traditional_baselines import evaluate_policy as evaluate_on_gym

STAGE = {stage.name: stage for stage in STAGES}
UPCARDS = list(range(2, 11)) + [1]
HARD_TOTALS = list(range(5, 21))
LATER_HARD_TOTALS = list(range(12, 21))  # below 12 a hit cannot bust
SOFT_TOTALS = list(range(13, 21))
ACTION_NAME = {STICK: "S", HIT: "H", DOUBLE: "D"}
COLORS = ListedColormap(["#4c78a8", "#e45756", "#54a24b"])  # stick, hit, double
PENALTIES = (0.1, 0.25, 0.5)


def decision(total: int, upcard: int, soft: bool, count: float = 0.0,
             first: bool = True) -> Observation:
    return Observation(total, upcard, soft, 2 if first else 3, first, count)


def _upcard_name(upcard: int) -> int | str:
    return "A" if upcard == 1 else upcard


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


# --- Exact infinite-deck quantities -------------------------------------------------

@lru_cache(maxsize=None)
def _model() -> BlackjackModel:
    return BlackjackModel()


@lru_cache(maxsize=None)
def _optimal_values() -> dict[State, float]:
    return policy_iteration(_model()).values


@lru_cache(maxsize=None)
def deal_probabilities() -> dict[State, float]:
    probabilities: defaultdict[State, float] = defaultdict(float)
    for first, p_first in enumerate(CARD_PROBABILITIES, start=1):
        for second, p_second in enumerate(CARD_PROBABILITIES, start=1):
            total, soft = _add_card(0, False, first)
            total, soft = _add_card(total, soft, second)
            for upcard, p_upcard in enumerate(CARD_PROBABILITIES, start=1):
                probabilities[(total, upcard, soft)] += p_first * p_second * p_upcard
    return dict(probabilities)


def exact_first_values(state: State) -> dict[int, float]:
    model, values = _model(), _optimal_values()
    return {STICK: model.action_value(state, STICK, values),
            HIT: model.action_value(state, HIT, values),
            DOUBLE: double_value(model, state)}


def _bust_probability(state: State) -> float:
    total, _, soft = state
    return sum(p for card, p in enumerate(CARD_PROBABILITIES, start=1)
               if _add_card(total, soft, card)[0] > 21)


def _hit_value(state: State, values: dict[State, float], penalty: float) -> float:
    total, upcard, soft = state
    result = 0.0
    for card, probability in enumerate(CARD_PROBABILITIES, start=1):
        next_total, next_soft = _add_card(total, soft, card)
        result += probability * (-1.0 - penalty if next_total > 21
                                 else values[(next_total, upcard, next_soft)])
    return result


def _iterate(update) -> dict[State, float]:
    values = {state: 0.0 for state in STATES}
    while True:
        delta = 0.0
        for state in STATES:
            new = update(state, values)
            delta = max(delta, abs(new - values[state]))
            values[state] = new
        if delta < 1e-12:
            return values


def solve_with_bust_penalty(penalty: float) -> tuple[dict[State, int], dict[State, int]]:
    """Exact infinite-deck optimum for reward - penalty * bust: (first, later) policies."""
    stick = {state: _model().action_value(state, STICK, {}) for state in STATES}
    values = _iterate(lambda s, v: max(stick[s], _hit_value(s, v, penalty)))
    later = {s: HIT if _hit_value(s, values, penalty) > stick[s] else STICK for s in STATES}
    first = {}
    for state in STATES:
        hit = _hit_value(state, values, penalty)
        double = double_value(_model(), state) - penalty * _bust_probability(state)
        first[state] = (DOUBLE if double > max(stick[state], hit)
                        else HIT if hit > stick[state] else STICK)
    return first, later


def exact_return(first: dict[State, int], later: dict[State, int]) -> float:
    """Exact infinite-deck return per hand on the true reward."""
    model = _model()
    values = _iterate(lambda s, v: model.action_value(s, later[s], v))
    result = 0.0
    for state, probability in deal_probabilities().items():
        action = first[state]
        value = (double_value(model, state) if action == DOUBLE
                 else model.action_value(state, action, values))
        result += probability * value
    return result


def _reachable(state: State) -> bool:
    total, _, soft = state
    return total >= 12 if soft else total <= 20


def exact_bust_penalty(penalties=PENALTIES) -> list[dict]:
    true_first, true_later = solve_with_bust_penalty(0.0)
    true_return = exact_return(true_first, true_later)
    results = []
    for penalty in penalties:
        first, later = solve_with_bust_penalty(penalty)
        changes = [
            {"decision": kind, "total": s[0], "upcard": _upcard_name(s[1]), "soft": s[2],
             "true_reward": ACTION_NAME[old[s]], "penalty": ACTION_NAME[new[s]]}
            for kind, old, new in (("first", true_first, first), ("later", true_later, later))
            for s in STATES if _reachable(s) and old[s] != new[s]
        ]
        results.append({"penalty": penalty,
                        "cost_per_hand": round(true_return - exact_return(first, later), 5),
                        "changes": changes})
    return results


# --- Learned tables -----------------------------------------------------------------

def majority_action(stage: Stage, tables: list[dict], observation: Observation) -> int:
    votes = Counter(greedy_policy(stage, table)(observation) for table in tables)
    return votes.most_common(1)[0][0]


def policy_grid(stage: Stage, tables: list[dict], soft: bool, count: float = 0.0,
                totals: list[int] | None = None, first: bool = True):
    totals = totals or (SOFT_TOTALS if soft else HARD_TOTALS)
    return [[majority_action(stage, tables, decision(t, u, soft, count, first)) for u in UPCARDS]
            for t in totals]


def basic_grid(soft: bool):
    policy = basic_strategy(allow_double=True)
    totals = SOFT_TOTALS if soft else HARD_TOTALS
    return [[policy(decision(t, u, soft)) for u in UPCARDS] for t in totals]


def explain_disagreement(stage: Stage, tables: list[dict], observation: Observation,
                         basic: int, learned: int) -> dict:
    """q_gap = Q(learned) - Q(basic) per seed; exact_gap = exact V(basic) - V(learned)."""
    key = (observation.total, observation.upcard, observation.soft)
    state = stage.state(observation)
    gaps = [table[state][learned] - table[state][basic] for table in tables if state in table]
    votes = sum(greedy_policy(stage, table)(observation) == learned for table in tables)
    exact = exact_first_values(key)
    exact_gap = exact[basic] - exact[learned]
    probability = deal_probabilities()[key]
    return {
        "seeds_choosing_learned": f"{votes}/{len(tables)}",
        "q_gap_by_seed": [round(gap, 4) for gap in gaps],
        "exact_gap": round(exact_gap, 4),
        "deal_probability": round(probability, 5),
        "cost_per_hand": round(probability * exact_gap, 6),
    }


def compare_with_basic(stage: Stage, tables: list[dict]) -> dict:
    rows = []
    for soft in (False, True):
        totals = SOFT_TOTALS if soft else HARD_TOTALS
        learned, basic = policy_grid(stage, tables, soft), basic_grid(soft)
        for i, total in enumerate(totals):
            for j, upcard in enumerate(UPCARDS):
                if basic[i][j] != learned[i][j]:
                    rows.append({"kind": "soft" if soft else "hard", "total": total,
                                 "upcard": _upcard_name(upcard),
                                 "basic": ACTION_NAME[basic[i][j]],
                                 "learned": ACTION_NAME[learned[i][j]]}
                                | explain_disagreement(stage, tables,
                                                       decision(total, upcard, soft),
                                                       basic[i][j], learned[i][j]))
    return {"cells_compared": len(UPCARDS) * (len(HARD_TOTALS) + len(SOFT_TOTALS)),
            "disagreements": rows,
            "predicted_cost_per_hand": round(sum(row["cost_per_hand"] for row in rows), 6)}


def q_error_by_action(stage: Stage, tables: list[dict]) -> dict[str, float]:
    """Deal-weighted mean of learned minus exact Q at first decisions (infinite deck)."""
    error: defaultdict[int, float] = defaultdict(float)
    weight: defaultdict[int, float] = defaultdict(float)
    for key, probability in deal_probabilities().items():
        if key[0] > 20:
            continue
        exact = exact_first_values(key)
        state = stage.state(decision(*key))
        for table in tables:
            if state in table:
                for action in (STICK, HIT, DOUBLE):
                    error[action] += probability * (table[state][action] - exact[action])
                    weight[action] += probability
    return {ACTION_NAME[a]: round(error[a] / weight[a], 5) for a in (STICK, HIT, DOUBLE)}


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


def policy_figure(stage: Stage, tables: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), height_ratios=[2, 1])
    for row, soft in enumerate((False, True)):
        kind = "soft" if soft else "hard"
        totals = SOFT_TOTALS if soft else HARD_TOTALS
        learned, basic = policy_grid(stage, tables, soft), basic_grid(soft)
        _draw(axes[row][0], basic, f"Basic strategy ({kind})", totals)
        _draw(axes[row][1], learned, f"Learned Q-table ({kind}); boxed = differs", totals, basic)
    fig.suptitle("First decision, S = stick, H = hit, D = double (majority over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def count_deviations(stage: Stage, tables: list[dict], path: Path) -> list[dict]:
    limit = stage.count_limit
    buckets = range(-limit, limit + 1)
    rows = []
    for soft, totals in ((False, HARD_TOTALS), (True, SOFT_TOTALS)):
        for total in totals:
            for upcard in UPCARDS:
                actions = [majority_action(stage, tables, decision(total, upcard, soft, b))
                           for b in buckets]
                if len(set(actions)) > 1:
                    rows.append({"total": total, "upcard": _upcard_name(upcard), "soft": soft,
                                 "by_count": {b: ACTION_NAME[a] for b, a in zip(buckets, actions)}})
    panels = (-limit, 0, limit)
    grids = [policy_grid(stage, tables, False, b) for b in panels]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, b, grid in zip(axes, panels, grids):
        _draw(ax, grid, f"Hard totals, true count {b:+d}", HARD_TOTALS,
              None if b == 0 else grids[1])
    fig.suptitle(f"Count-aware Q-table, buckets clipped at ±{limit}: "
                 "boxed cells change relative to count 0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def bucket_comparison(rows_2: list[dict], rows_3: list[dict]) -> dict:
    """Cells where the +/-3 table plays differently at 3 than at 2 (merged under +/-2)."""
    beyond = [
        {"total": row["total"], "upcard": row["upcard"], "soft": row["soft"],
         "side": side, "at_2": row["by_count"][2 * sign], "at_3": row["by_count"][3 * sign]}
        for row in rows_3
        for side, sign in (("+", 1), ("-", -1))
        if row["by_count"][2 * sign] != row["by_count"][3 * sign]
    ]
    return {"cells_varying_with_count": {"buckets_2": len(rows_2), "buckets_3": len(rows_3)},
            "changes_beyond_2": beyond}


def validation(hands: int) -> dict:
    """Infinite-deck simulator vs the exact model and Gymnasium Blackjack-v1."""
    solved = policy_iteration(_model())
    exact_plain = sum(p * solved.values[s] for s, p in deal_probabilities().items())
    plain = evaluate_policy(basic_strategy(), hands=hands, num_decks=None, seed=EVAL_SEED)
    double = evaluate_policy(basic_strategy(allow_double=True), hands=hands, num_decks=None,
                             allow_double=True, seed=EVAL_SEED)
    episodes = hands // 2
    gym = evaluate_on_gym(solved.policy, episodes=episodes, seed=EVAL_SEED)
    gym_variance = gym["win_rate"] + gym["loss_rate"] - gym["mean_return"] ** 2
    return {
        "basic_strategy": {"exact": exact_plain, "simulated": plain["mean_return"],
                           "simulated_ci95": plain["ci95"], "gymnasium": gym["mean_return"],
                           "gymnasium_ci95": 1.96 * (gym_variance / episodes) ** 0.5},
        "basic_strategy_double": {"exact": exact_return(*solve_with_bust_penalty(0.0)),
                                  "simulated": double["mean_return"],
                                  "simulated_ci95": double["ci95"]},
    }


def count_shares(decks: int, hands: int) -> dict:
    """Share of hands dealt at each |true count| bucket (players stand; counts unclipped)."""
    env = BlackjackShoe(num_decks=decks, seed=EVAL_SEED)
    buckets = Counter()
    for _ in range(hands):
        buckets[min(abs(count_bucket(env.reset().true_count, 99)), 3)] += 1
        env.step(STICK)
    return {"decks": decks, "abs_0": buckets[0] / hands, "abs_1": buckets[1] / hands,
            "abs_2": buckets[2] / hands, "abs_3_or_more": buckets[3] / hands}


def penalty_differences(true_tables: list[dict], penalty_stage: Stage,
                        penalty_tables: list[dict], path: Path) -> list[dict]:
    true_stage = STAGE["double"]
    rows = []
    grids = {}
    for first in (True, False):
        for soft in (False, True):
            totals = SOFT_TOTALS if soft else (HARD_TOTALS if first else LATER_HARD_TOTALS)
            plain = policy_grid(true_stage, true_tables, soft, totals=totals, first=first)
            penal = policy_grid(penalty_stage, penalty_tables, soft, totals=totals, first=first)
            grids[(first, soft)] = (totals, plain, penal)
            for i, total in enumerate(totals):
                for j, upcard in enumerate(UPCARDS):
                    if plain[i][j] != penal[i][j]:
                        rows.append({"decision": "first" if first else "later",
                                     "total": total, "upcard": _upcard_name(upcard), "soft": soft,
                                     "true_reward": ACTION_NAME[plain[i][j]],
                                     "penalty": ACTION_NAME[penal[i][j]]})
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), height_ratios=[16, 9])
    for row, first in enumerate((True, False)):
        totals, plain, penal = grids[(first, False)]
        when = "first decision" if first else "later decisions (3+ cards)"
        _draw(axes[row][0], plain, f"True reward, hard, {when}", totals)
        _draw(axes[row][1], penal, f"Bust penalty {penalty_stage.bust_penalty}, hard, {when}",
              totals, plain)
    fig.suptitle("Bust penalty vs true reward (6 decks, + double): boxed = changed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def sensitivity_figure(results: list[dict], path: Path) -> list[dict]:
    summary = []
    for key in sorted({(r["eps"], r["alpha"]) for r in results}):
        values = [r["mean_return"] for r in results if (r["eps"], r["alpha"]) == key]
        summary.append({"epsilon": key[0], "alpha_power": key[1],
                        "mean_return": statistics.fmean(values),
                        "seed_range": max(values) - min(values), "seeds": len(values)})
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
    parser.add_argument("--hands", type=int, default=30_000_000)
    parser.add_argument("--sens-hands", type=int, default=3_000_000)
    parser.add_argument("--eval-hands", type=int, default=2_000_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.seeds)

    def jobs(name: str, decks: int | None) -> list[tuple]:
        return [(name, s, args.hands, decks, EPSILON, ALPHA_POWER) for s in seeds]

    groups = {
        "policy": jobs("double", 6),
        "infinite": jobs("double", None),
        "count": jobs("count_double", 1),
        "count_3": jobs("count_double_3", 1),
        "bust": jobs("bust_0.1", 6),
        "sensitivity": [("double", s, args.sens_hands, 6, eps, alpha)
                        for eps in (0.1, 0.3, 0.5) for alpha in (0.6, 0.85, 1.0) for s in seeds],
    }
    with ProcessPoolExecutor(args.workers) as pool:
        runs = iter(pool.map(_train, [job for group in groups.values() for job in group]))
        tables = {name: [next(runs) for _ in group] for name, group in groups.items()}
        sens_results = list(pool.map(_evaluate,
                                     [(r, args.eval_hands) for r in tables.pop("sensitivity")]))
    tables = {name: [run["table"] for run in group] for name, group in tables.items()}

    args.out.mkdir(exist_ok=True)
    double = STAGE["double"]
    policy_figure(double, tables["policy"], args.out / "policy_vs_basic.png")
    count_2 = count_deviations(STAGE["count_double"], tables["count"],
                               args.out / "count_deviations.png")
    count_3 = count_deviations(STAGE["count_double_3"], tables["count_3"],
                               args.out / "count_deviations_buckets3.png")
    report = {
        "settings": vars(args) | {"out": str(args.out)},
        "validation_infinite_deck": validation(args.eval_hands),
        "policy_vs_basic_6deck": compare_with_basic(double, tables["policy"]),
        "policy_vs_basic_infinite_deck": compare_with_basic(double, tables["infinite"])
        | {"q_error_by_action": q_error_by_action(double, tables["infinite"])},
        "count_deviations_1deck": count_2,
        "count_deviations_1deck_buckets3": count_3,
        "bucket_comparison_1deck": bucket_comparison(count_2, count_3),
        "true_count_shares": [count_shares(d, args.eval_hands) for d in (1, 2, 6)],
        "bust_penalty": {
            "learned_6deck": penalty_differences(tables["policy"], STAGE["bust_0.1"],
                                                 tables["bust"],
                                                 args.out / "bust_penalty_policy.png"),
            "exact_infinite_deck": exact_bust_penalty(),
        },
        "sensitivity": sensitivity_figure(sens_results, args.out / "sensitivity.png"),
    }
    (args.out / "policy_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
                      ("policy_vs_basic_6deck", "policy_vs_basic_infinite_deck",
                       "bucket_comparison_1deck")}, indent=1))
    print(f"count deviations: {len(count_2)} cells (±2), {len(count_3)} cells (±3)")
    print(f"bust penalty 0.1 changes {len(report['bust_penalty']['learned_6deck'])} cells")
    for row in report["bust_penalty"]["exact_infinite_deck"]:
        print(f"exact penalty {row['penalty']}: {len(row['changes'])} cells, "
              f"cost {row['cost_per_hand']:+.5f} per hand")


if __name__ == "__main__":
    main()
