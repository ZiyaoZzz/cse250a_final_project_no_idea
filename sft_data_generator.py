"""Generate policy-iteration labels from real Blackjack-v1 environment play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from env_setup import make_blackjack_env
from traditional_baselines import HIT, STICK, PolicyIterationResult, policy_iteration

ACTION_NAMES = {STICK: "Stick", HIT: "Hit"}
STATE_FIELDS = {"player_sum", "dealer_showing", "usable_ace"}


def _write_example(
    output: TextIO, *, episode: int, step: int, state: tuple[int, int, bool], action: int
) -> None:
    if action not in ACTION_NAMES:
        raise ValueError(f"Unexpected expert action: {action}")
    player_sum, dealer_showing, usable_ace = state
    example = {
        "episode": episode,
        "step": step,
        "state": {
            "player_sum": player_sum,
            "dealer_showing": dealer_showing,
            "usable_ace": bool(usable_ace),
        },
        "action": ACTION_NAMES[action],
        "action_id": action,
    }
    output.write(json.dumps(example, separators=(",", ":")) + "\n")


def generate_expert_data(
    output_path: Path,
    *,
    episodes: int = 100_000,
    seed: int = 2026,
    solution: PolicyIterationResult | None = None,
) -> int:
    """Write one decision-time state/action label per Gymnasium action."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    solution = solution or policy_iteration()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        with make_blackjack_env() as env:
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                step = 0
                while True:
                    player_sum, dealer_showing, usable_ace = observation
                    state = (int(player_sum), int(dealer_showing), bool(usable_ace))
                    if not 4 <= state[0] <= 21 or not 1 <= state[1] <= 10:
                        raise RuntimeError(f"Unexpected Gymnasium observation: {observation!r}")
                    action = solution.policy[state]
                    _write_example(
                        output, episode=episode, step=step, state=state, action=action
                    )
                    count += 1
                    observation, _, terminated, truncated, _ = env.step(action)
                    step += 1
                    if terminated or truncated:
                        break
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/expert_blackjack.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = generate_expert_data(args.output, episodes=args.episodes, seed=args.seed)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": args.episodes,
                "examples": examples,
                "seed": args.seed,
                "environment": "Blackjack-v1 (natural=False, sab=False)",
                "fields": ["episode", "step", "state", "action", "action_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
