"""Tabular Q-learning ladder: each stage changes one thing about a simpler stage."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Hashable

from blackjack_shoe import Observation, BlackjackShoe, count_bucket, legal_actions
from evaluate import (
    EVAL_SEED,
    Policy,
    basic_strategy,
    evaluate_policy,
    hit_below_17,
    random_policy,
)
from traditional_baselines import STICK

QTable = dict[Hashable, list[float]]
EPSILON = 0.3
ALPHA_POWER = 0.85
CURVE_BUDGETS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000)


@dataclass(frozen=True)
class Stage:
    """A state representation, an action set, and a training-only reward change."""

    name: str
    label: str
    use_num_cards: bool = False
    use_count: bool = False
    allow_double: bool = False
    bust_penalty: float = 0.0

    def state(self, observation: Observation) -> tuple:
        key: tuple = (observation.total, observation.upcard, observation.soft)
        if self.use_num_cards:
            key += (min(observation.num_cards, 4),)
        if self.use_count:
            key += (count_bucket(observation.true_count),)
        if self.allow_double:
            key += (observation.can_double,)  # Double is legal only in these states.
        return key


STAGES = (
    Stage("baseline", "Baseline: total, upcard, usable ace"),
    Stage("num_cards", "+ number of cards", use_num_cards=True),
    Stage("count", "+ Hi-Lo true count", use_count=True),
    Stage("double", "+ double down", allow_double=True),
    Stage("count_double", "+ Hi-Lo count + double down", use_count=True, allow_double=True),
    Stage("bust_0.1", "+ double, bust penalty 0.1", allow_double=True, bust_penalty=0.1),
    Stage("bust_0.25", "+ double, bust penalty 0.25", allow_double=True, bust_penalty=0.25),
    Stage("bust_0.5", "+ double, bust penalty 0.5", allow_double=True, bust_penalty=0.5),
)


def train_q_table(
    stage: Stage,
    *,
    hands: int,
    seed: int,
    num_decks: int | None = 6,
    epsilon: float = EPSILON,
    alpha_power: float = ALPHA_POWER,
    checkpoints: tuple[int, ...] = (),
    on_checkpoint: Callable[[int, QTable], None] | None = None,
) -> QTable:
    """Q-learning with gamma=1; a state-action's step size is visits**-alpha_power."""
    rng = random.Random(seed)
    env = BlackjackShoe(num_decks=num_decks, allow_double=stage.allow_double, seed=seed + 1)
    q_values: QTable = {}
    visits: dict[Hashable, list[int]] = {}
    pending = sorted(checkpoints, reverse=True)
    for hand in range(1, hands + 1):
        observation = env.reset()
        done = False
        while not done:
            state = stage.state(observation)
            if state not in q_values:
                q_values[state] = [0.0, 0.0, 0.0]
                visits[state] = [0, 0, 0]
            values = q_values[state]
            actions = legal_actions(observation)
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(actions, key=values.__getitem__)
            observation, reward, done = env.step(action)
            target = reward
            if done:
                target -= stage.bust_penalty * env.busted
            else:
                next_values = q_values.get(stage.state(observation), (0.0, 0.0, 0.0))
                target += max(next_values[a] for a in legal_actions(observation))
            visits[state][action] += 1
            values[action] += (target - values[action]) / visits[state][action] ** alpha_power
        if pending and hand == pending[-1]:
            pending.pop()
            if on_checkpoint:
                on_checkpoint(hand, q_values)
    return q_values


def greedy_policy(stage: Stage, q_values: QTable) -> Policy:
    """Freeze a table; states never visited in training fall back to stick."""

    def policy(observation: Observation) -> int:
        values = q_values.get(stage.state(observation))
        if values is None:
            return STICK
        return max(legal_actions(observation), key=values.__getitem__)

    return policy


def _run(job: tuple[Stage, int, int, int, int, int | None]) -> dict:
    stage, seed, hands, eval_hands, curve_hands, num_decks = job
    curve: list[dict] = []

    def evaluate(q_values: QTable, eval_size: int) -> dict:
        return evaluate_policy(
            greedy_policy(stage, q_values),
            hands=eval_size,
            seed=EVAL_SEED + seed,
            num_decks=num_decks,
            allow_double=stage.allow_double,
        )

    q_values = train_q_table(
        stage,
        hands=hands,
        seed=seed,
        num_decks=num_decks,
        checkpoints=tuple(budget for budget in CURVE_BUDGETS if budget <= hands),
        on_checkpoint=lambda hand, table: curve.append(
            {"train_hands": hand, "mean_return": evaluate(table, curve_hands)["mean_return"]}
        ),
    )
    return {
        "stage": stage.name,
        "seed": seed,
        "states": len(q_values),
        "final": evaluate(q_values, eval_hands),
        "curve": curve,
    }


def _reference(job: tuple[str, int, int, int | None]) -> dict:
    name, seed, hands, num_decks = job
    allow_double = name == "basic_strategy_double"
    policy = {
        "random": random_policy,
        "hit_below_17": lambda: hit_below_17,
        "basic_strategy": basic_strategy,
        "basic_strategy_double": lambda: basic_strategy(allow_double=True),
    }[name]()
    result = evaluate_policy(
        policy,
        hands=hands,
        seed=EVAL_SEED + seed,
        num_decks=num_decks,
        allow_double=allow_double,
    )
    return {"policy": name, "seed": seed, "final": result}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=30_000_000)
    parser.add_argument("--eval-hands", type=int, default=10_000_000)
    parser.add_argument("--curve-hands", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--decks", type=int, default=6, help="0 means an infinite deck")
    parser.add_argument("--stages", nargs="*", help="stage names; default is all")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--output", type=Path, default=Path("outputs/q_table_ladder.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    num_decks = args.decks or None
    stages = [stage for stage in STAGES if not args.stages or stage.name in args.stages]
    jobs = [
        (stage, seed, args.hands, args.eval_hands, args.curve_hands, num_decks)
        for stage in stages
        for seed in range(args.seeds)
    ]
    reference_jobs = [
        (name, seed, args.eval_hands, num_decks)
        for name in ("random", "hit_below_17", "basic_strategy", "basic_strategy_double")
        for seed in range(args.seeds)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(_run, jobs))
        references = list(pool.map(_reference, reference_jobs))
    report = {
        "settings": {
            "num_decks": num_decks,
            "penetration": 0.75,
            "train_hands": args.hands,
            "eval_hands": args.eval_hands,
            "curve_hands": args.curve_hands,
            "train_seeds": args.seeds,
            "eval_seeds": f"{EVAL_SEED} + training seed",
            "epsilon": EPSILON,
            "alpha_power": ALPHA_POWER,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "stages": [asdict(stage) for stage in stages],
        "references": references,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} in {report['settings']['seconds']}s")


if __name__ == "__main__":
    main()
