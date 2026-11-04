"""Model-based policy iteration and tabular Q-learning for Blackjack-v1."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from env_setup import make_blackjack_env

State: TypeAlias = tuple[int, int, bool]
Action: TypeAlias = int
STICK = 0
HIT = 1
CARD_PROBABILITIES = tuple([1 / 13] * 9 + [4 / 13])
STATES = tuple(
    (player_sum, dealer_showing, usable_ace)
    for player_sum in range(4, 22)
    for dealer_showing in range(1, 11)
    for usable_ace in (False, True)
)


def _add_card(total: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    if usable_ace:
        total += card
        if total > 21:
            return total - 10, False
        return total, True
    if card == 1 and total + 11 <= 21:
        return total + 11, True
    return total + card, False


class BlackjackModel:
    """Exact infinite-deck model matching Blackjack-v1 with natural=False, sab=False."""

    def __init__(self) -> None:
        self._dealer_cache: dict[tuple[int, bool], dict[int, float]] = {}

    def dealer_distribution(self, dealer_showing: int) -> dict[int, float]:
        """Return dealer final-score probabilities; score 0 represents a bust."""
        if dealer_showing not in range(1, 11):
            raise ValueError(f"dealer_showing must be 1..10, got {dealer_showing}")
        outcomes: defaultdict[int, float] = defaultdict(float)
        for hole_card, probability in enumerate(CARD_PROBABILITIES, start=1):
            total, usable = _add_card(0, False, dealer_showing)
            total, usable = _add_card(total, usable, hole_card)
            for score, conditional_probability in self._dealer_finish(total, usable).items():
                outcomes[score] += probability * conditional_probability
        return dict(outcomes)

    def _dealer_finish(self, total: int, usable_ace: bool) -> dict[int, float]:
        key = (total, usable_ace)
        if key in self._dealer_cache:
            return self._dealer_cache[key]
        if total >= 17:
            return {0 if total > 21 else total: 1.0}
        outcomes: defaultdict[int, float] = defaultdict(float)
        for card, probability in enumerate(CARD_PROBABILITIES, start=1):
            next_total, next_usable = _add_card(total, usable_ace, card)
            for score, conditional_probability in self._dealer_finish(
                next_total, next_usable
            ).items():
                outcomes[score] += probability * conditional_probability
        result = dict(outcomes)
        self._dealer_cache[key] = result
        return result

    def action_value(self, state: State, action: Action, values: dict[State, float]) -> float:
        """Expected return for an action, marginalizing the dealer's hidden card."""
        player_sum, dealer_showing, usable_ace = state
        if action == STICK:
            return sum(
                probability
                * (float(player_sum > dealer_score) - float(player_sum < dealer_score))
                for dealer_score, probability in self.dealer_distribution(dealer_showing).items()
            )
        if action != HIT:
            raise ValueError(f"Action must be STICK (0) or HIT (1), got {action}")

        result = 0.0
        for card, probability in enumerate(CARD_PROBABILITIES, start=1):
            next_sum, next_usable_ace = _add_card(player_sum, usable_ace, card)
            if next_sum > 21:
                result -= probability
            else:
                result += probability * values.get(
                    (next_sum, dealer_showing, next_usable_ace), 0.0
                )
        return result

    def transition_distribution(
        self, state: State, action: Action
    ) -> list[tuple[float, State | None, float, bool]]:
        """List (probability, next decision state, reward, terminated) outcomes."""
        player_sum, dealer_showing, usable_ace = state
        if action == STICK:
            return [
                (probability, None,
                 float(player_sum > score) - float(player_sum < score), True)
                for score, probability in self.dealer_distribution(dealer_showing).items()
            ]
        if action != HIT:
            raise ValueError(f"Action must be STICK (0) or HIT (1), got {action}")
        transitions = []
        for card, probability in enumerate(CARD_PROBABILITIES, start=1):
            next_sum, next_usable = _add_card(player_sum, usable_ace, card)
            busted = next_sum > 21
            transitions.append(
                (
                    probability,
                    None if busted else (next_sum, dealer_showing, next_usable),
                    -1.0 if busted else 0.0,
                    busted,
                )
            )
        return transitions


@dataclass(frozen=True)
class PolicyIterationResult:
    policy: dict[State, Action]
    values: dict[State, float]
    iterations: int


def policy_iteration(
    model: BlackjackModel | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> PolicyIterationResult:
    """Solve the known model by iterative policy evaluation and greedy improvement."""
    model = model or BlackjackModel()
    policy = {state: STICK for state in STATES}
    values = {state: 0.0 for state in STATES}

    for iteration in range(1, max_iterations + 1):
        while True:
            delta = 0.0
            for state in STATES:
                updated = model.action_value(state, policy[state], values)
                delta = max(delta, abs(updated - values[state]))
                values[state] = updated
            if delta < tolerance:
                break

        stable = True
        for state in STATES:
            stick_value = model.action_value(state, STICK, values)
            hit_value = model.action_value(state, HIT, values)
            best_action = HIT if hit_value > stick_value else STICK
            if best_action != policy[state]:
                policy[state] = best_action
                stable = False
        if stable:
            return PolicyIterationResult(policy, values, iteration)
    raise RuntimeError(f"Policy iteration did not converge in {max_iterations} iterations")


def train_q_learning(
    *,
    episodes: int = 100_000,
    seed: int = 0,
    learning_rate: float = 0.1,
    discount_factor: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> dict[State, tuple[float, float]]:
    """Train tabular Q-learning in the configured Gymnasium environment."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    rng = random.Random(seed)
    q_values: defaultdict[State, list[float]] = defaultdict(lambda: [0.0, 0.0])
    with make_blackjack_env() as env:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode / max(
                episodes - 1, 1
            )
            while True:
                action = (
                    rng.randrange(2)
                    if rng.random() < epsilon
                    else int(q_values[state][HIT] > q_values[state][STICK])
                )
                next_state, reward, terminated, truncated, _ = env.step(action)
                target = reward
                if not (terminated or truncated):
                    target += discount_factor * max(q_values[next_state])
                q_values[state][action] += learning_rate * (
                    target - q_values[state][action]
                )
                if terminated or truncated:
                    break
                state = next_state
    return {state: (values[STICK], values[HIT]) for state, values in q_values.items()}


def evaluate_policy(
    policy: dict[State, Action],
    *,
    episodes: int = 10_000,
    seed: int = 1_000_000,
) -> dict[str, float | int]:
    """Evaluate a frozen policy on freshly seeded environment episodes."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    returns = []
    with make_blackjack_env() as env:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            while True:
                action = policy.get(state, STICK)
                state, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    returns.append(reward)
                    break
    return {
        "episodes": episodes,
        "mean_return": sum(returns) / episodes,
        "win_rate": sum(reward == 1 for reward in returns) / episodes,
        "draw_rate": sum(reward == 0 for reward in returns) / episodes,
        "loss_rate": sum(reward == -1 for reward in returns) / episodes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-episodes", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/phase2_baselines.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = BlackjackModel()
    result = policy_iteration(model)
    q_values = train_q_learning(episodes=args.q_episodes, seed=args.seed)
    q_policy = {state: int(hit > stick) for state, (stick, hit) in q_values.items()}
    report = {
        "environment": "Blackjack-v1 (natural=False, sab=False)",
        "training_seed": args.seed,
        "policy_iteration_iterations": result.iterations,
        "policy_iteration": evaluate_policy(
            result.policy, episodes=args.eval_episodes, seed=args.seed + 1_000_000
        ),
        "q_learning": evaluate_policy(
            q_policy,
            episodes=args.eval_episodes,
            seed=args.seed + 2_000_000,
        ),
        "q_learning_training_episodes": args.q_episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
