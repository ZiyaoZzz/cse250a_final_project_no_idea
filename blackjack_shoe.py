"""Blackjack-v1 rules on a finite shoe, with a Hi-Lo count and optional double down.

Blackjack-v1 draws every card independently from an infinite deck, so a card count
carries no information there and the environment has no double-down action. This
environment keeps the v1 rules (dealer stands on all 17s, naturals pay 1:1, the
dealer does not draw after a player bust) and changes only those two things.
``num_decks=None`` restores the infinite deck, which is how it is checked against
Blackjack-v1.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

from traditional_baselines import HIT, STICK, _add_card

DOUBLE = 2
CARDS_PER_DECK = 52
ONE_DECK = tuple(list(range(1, 10)) * 4 + [10] * 16)
INFINITE_DECK = tuple(list(range(1, 10)) + [10] * 4)


def hi_lo_value(card: int) -> int:
    """Hi-Lo tag: low cards leaving the shoe are good for the player."""
    if 2 <= card <= 6:
        return 1
    return -1 if card in (1, 10) else 0


class Observation(NamedTuple):
    """Everything a player can see at a decision; never the dealer's hole card."""

    total: int
    upcard: int
    soft: bool
    num_cards: int
    can_double: bool
    true_count: float


class BlackjackShoe:
    """One-player Blackjack played hand after hand from a persistent shoe."""

    def __init__(
        self,
        *,
        num_decks: int | None = 6,
        penetration: float = 0.75,
        allow_double: bool = False,
        seed: int = 0,
    ) -> None:
        if num_decks is not None and num_decks < 1:
            raise ValueError("num_decks must be at least 1, or None for an infinite deck")
        if not 0 < penetration < 1:
            raise ValueError("penetration must be in (0, 1)")
        self.num_decks = num_decks
        self.allow_double = allow_double
        self.busted = False
        self._rng = random.Random(seed)
        self._cut_card = 0 if num_decks is None else round(
            num_decks * CARDS_PER_DECK * (1 - penetration)
        )
        self._cards: list[int] = []
        self._running_count = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self._running_count = 0
        if self.num_decks is not None:
            self._cards = list(ONE_DECK) * self.num_decks
            self._rng.shuffle(self._cards)

    def _draw(self) -> int:
        if self.num_decks is None:
            return self._rng.choice(INFINITE_DECK)
        if not self._cards:  # Only reachable by a freak hand past the cut card.
            self._shuffle()
        return self._cards.pop()

    def _draw_seen(self) -> int:
        card = self._draw()
        self._running_count += hi_lo_value(card)
        return card

    def _true_count(self) -> float:
        """Running count per unseen deck; the hole card is still unseen."""
        if self.num_decks is None:
            return 0.0
        return self._running_count * CARDS_PER_DECK / (len(self._cards) + 1)

    def _observe(self) -> Observation:
        return Observation(
            self._total,
            self._upcard,
            self._soft,
            self._num_cards,
            self.allow_double and self._num_cards == 2,
            self._true_count(),
        )

    def reset(self) -> Observation:
        """Deal the next hand, reshuffling first if the cut card has been reached."""
        if self.num_decks is not None and len(self._cards) <= self._cut_card:
            self._shuffle()
        self.busted = False
        self._bet = 1.0
        self._total, self._soft = _add_card(0, False, self._draw_seen())
        self._total, self._soft = _add_card(self._total, self._soft, self._draw_seen())
        self._num_cards = 2
        self._upcard = self._draw_seen()
        self._hole_card = self._draw()
        return self._observe()

    def _hit_player(self) -> None:
        self._total, self._soft = _add_card(self._total, self._soft, self._draw_seen())
        self._num_cards += 1
        self.busted = self._total > 21

    def _finish(self) -> tuple[None, float, bool]:
        """Reveal the hole card, play out the dealer unless busted, and settle."""
        self._running_count += hi_lo_value(self._hole_card)
        if self.busted:
            return None, -self._bet, True
        dealer_total, dealer_soft = _add_card(0, False, self._upcard)
        dealer_total, dealer_soft = _add_card(dealer_total, dealer_soft, self._hole_card)
        while dealer_total < 17:
            dealer_total, dealer_soft = _add_card(
                dealer_total, dealer_soft, self._draw_seen()
            )
        dealer_score = 0 if dealer_total > 21 else dealer_total
        outcome = float(self._total > dealer_score) - float(self._total < dealer_score)
        return None, self._bet * outcome, True

    def step(self, action: int) -> tuple[Observation | None, float, bool]:
        """Apply stick, hit, or double; returns (observation, reward, done)."""
        if action == STICK:
            return self._finish()
        if action == HIT:
            self._hit_player()
            return self._finish() if self.busted else (self._observe(), 0.0, False)
        if action == DOUBLE:
            if not (self.allow_double and self._num_cards == 2):
                raise ValueError("double is only legal on the first two cards")
            self._bet = 2.0
            self._hit_player()
            return self._finish()
        raise ValueError(f"unknown action {action!r}")


def legal_actions(observation: Observation) -> tuple[int, ...]:
    return (STICK, HIT, DOUBLE) if observation.can_double else (STICK, HIT)


def count_bucket(true_count: float, limit: int = 2) -> int:
    """Round a true count to the nearest integer and clip it to +/- limit."""
    return max(-limit, min(limit, math.floor(true_count + 0.5)))
