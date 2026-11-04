"""Configured Gymnasium environment for Blackjack-v1."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class BlackjackEnvironment:
    """Small lifecycle wrapper around the project's fixed Blackjack variant."""

    def __init__(self, seed: int | None = None) -> None:
        self._env = gym.make("Blackjack-v1", natural=False, sab=False)
        self._seed = seed

    def reset(self, *, seed: int | None = None) -> tuple[tuple[int, int, bool], dict[str, Any]]:
        """Start a hand, optionally setting its random seed."""
        effective_seed = self._seed if seed is None else seed
        self._seed = None
        observation, info = self._env.reset(seed=effective_seed)
        return observation, info

    def step(
        self, action: int
    ) -> tuple[tuple[int, int, bool], float, bool, bool, dict[str, Any]]:
        """Apply one Blackjack action (0=stick, 1=hit)."""
        if action not in (0, 1):
            raise ValueError(f"Blackjack action must be 0 (stick) or 1 (hit), got {action!r}")
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> BlackjackEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_blackjack_env(seed: int | None = None) -> BlackjackEnvironment:
    """Create Blackjack-v1 with the project's required reward rules."""
    return BlackjackEnvironment(seed=seed)
