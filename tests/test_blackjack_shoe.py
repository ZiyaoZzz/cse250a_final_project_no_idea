import unittest

from blackjack_shoe import (
    DOUBLE,
    ONE_DECK,
    BlackjackShoe,
    count_bucket,
    hi_lo_value,
    legal_actions,
)
from traditional_baselines import HIT, STICK


def stacked_shoe(deal_order, *, allow_double=False):
    """A shoe that deals deal_order first: player, player, upcard, hole, then draws."""
    env = BlackjackShoe(num_decks=1, allow_double=allow_double, seed=0)
    env._cards = [5] * 30 + list(reversed(deal_order))
    return env


class CountTests(unittest.TestCase):
    def test_hi_lo_is_balanced_over_a_deck(self):
        self.assertEqual(sum(hi_lo_value(card) for card in ONE_DECK), 0)

    def test_hole_card_is_hidden_from_the_count_until_the_hand_ends(self):
        env = stacked_shoe([10, 9, 9, 5, 7])  # Hole card 5 would add +1 if it leaked.
        observation = env.reset()
        self.assertEqual(env._running_count, -1)
        self.assertLess(observation.true_count, 0)
        env.step(STICK)
        self.assertEqual(env._running_count, 0)

    def test_reshuffle_at_the_cut_card_resets_the_count(self):
        env = BlackjackShoe(num_decks=1, seed=0)
        env._cards = env._cards[:10]
        env._running_count = 7
        env.reset()
        self.assertEqual(len(env._cards), 48)
        self.assertLessEqual(abs(env._running_count), 3)

    def test_count_bucket_rounds_and_clips(self):
        self.assertEqual(count_bucket(-9.0), -2)
        self.assertEqual(count_bucket(0.4), 0)
        self.assertEqual(count_bucket(0.6), 1)
        self.assertEqual(count_bucket(9.0), 2)

    def test_infinite_deck_has_no_count(self):
        env = BlackjackShoe(num_decks=None, seed=0)
        self.assertEqual(env.reset().true_count, 0.0)


class RuleTests(unittest.TestCase):
    def test_dealer_stands_on_soft_17(self):
        env = stacked_shoe([10, 8, 1, 6])
        env.reset()
        self.assertEqual(env.step(STICK), (None, 1.0, True))

    def test_natural_pays_even_money(self):
        env = stacked_shoe([1, 10, 10, 7])
        observation = env.reset()
        self.assertEqual((observation.total, observation.soft), (21, True))
        self.assertEqual(env.step(STICK)[1], 1.0)

    def test_dealer_does_not_draw_after_a_player_bust(self):
        env = stacked_shoe([10, 6, 5, 5, 10])
        env.reset()
        cards_before = len(env._cards)
        _, reward, done = env.step(HIT)
        self.assertEqual((reward, done, env.busted), (-1.0, True, True))
        self.assertEqual(len(env._cards), cards_before - 1)


class DoubleTests(unittest.TestCase):
    def test_double_pays_twice_on_a_win_and_a_bust(self):
        win = stacked_shoe([5, 6, 10, 7, 10], allow_double=True)
        win.reset()
        self.assertEqual(win.step(DOUBLE), (None, 2.0, True))

        bust = stacked_shoe([10, 6, 10, 7, 10], allow_double=True)
        bust.reset()
        self.assertEqual(bust.step(DOUBLE), (None, -2.0, True))
        self.assertTrue(bust.busted)

    def test_double_is_only_legal_on_the_first_decision(self):
        env = stacked_shoe([2, 3, 10, 7, 2], allow_double=True)
        first = env.reset()
        self.assertIn(DOUBLE, legal_actions(first))
        second, _, _ = env.step(HIT)
        self.assertNotIn(DOUBLE, legal_actions(second))
        with self.assertRaises(ValueError):
            env.step(DOUBLE)

    def test_double_is_rejected_when_disabled(self):
        env = stacked_shoe([5, 6, 10, 7, 10])
        self.assertNotIn(DOUBLE, legal_actions(env.reset()))
        with self.assertRaises(ValueError):
            env.step(DOUBLE)


if __name__ == "__main__":
    unittest.main()
