"""One evaluation protocol for every policy: same shoe, same seed, true reward only."""

from __future__ import annotations

import math
import random
from typing import Callable

from blackjack_shoe import DOUBLE, BlackjackShoe, Observation, legal_actions
from traditional_baselines import HIT, STATES, STICK, BlackjackModel, State, policy_iteration

Policy = Callable[[Observation], int]
EVAL_SEED = 2_000_000


def evaluate_policy(
    policy: Policy,
    *,
    hands: int = 1_000_000,
    seed: int = EVAL_SEED,
    num_decks: int | None = 6,
    allow_double: bool = False,
) -> dict[str, float | int]:
    """Play a frozen policy and report per-hand return with a 95% confidence interval."""
    if hands < 2:
        raise ValueError("hands must be at least 2")
    env = BlackjackShoe(num_decks=num_decks, allow_double=allow_double, seed=seed)
    total = squared = wins = draws = busts = doubles = 0.0
    for _ in range(hands):
        observation = env.reset()
        done = False
        while not done:
            action = policy(observation)
            doubles += action == DOUBLE
            observation, reward, done = env.step(action)
        total += reward
        squared += reward * reward
        wins += reward > 0
        draws += reward == 0
        busts += env.busted
    mean = total / hands
    variance = (squared - hands * mean * mean) / (hands - 1)
    return {
        "hands": hands,
        "mean_return": mean,
        "ci95": 1.96 * math.sqrt(variance / hands),
        "win_rate": wins / hands,
        "draw_rate": draws / hands,
        "loss_rate": 1 - (wins + draws) / hands,
        "bust_rate": busts / hands,
        "double_rate": doubles / hands,
    }


def random_policy(seed: int = 0) -> Policy:
    rng = random.Random(seed)
    return lambda observation: rng.choice(legal_actions(observation))


def hit_below_17(observation: Observation) -> int:
    return HIT if observation.total < 17 else STICK


def double_value(model: BlackjackModel, state: State) -> float:
    """Exact infinite-deck value of doubling: twice one card, then stick."""
    value = 0.0
    for probability, next_state, reward, terminated in model.transition_distribution(state, HIT):
        stick_value = reward if terminated else model.action_value(next_state, STICK, {})
        value += 2 * probability * stick_value
    return value


def basic_strategy(*, allow_double: bool = False) -> Policy:
    """Infinite-deck optimal play from policy iteration; it ignores the count."""
    model = BlackjackModel()
    solved = policy_iteration(model)
    first_action = dict(solved.policy)
    if allow_double:
        for state in STATES:
            best_other = max(
                model.action_value(state, action, solved.values) for action in (STICK, HIT)
            )
            if double_value(model, state) > best_other:
                first_action[state] = DOUBLE

    def policy(observation: Observation) -> int:
        state = (observation.total, observation.upcard, observation.soft)
        return first_action[state] if observation.can_double else solved.policy[state]

    return policy
