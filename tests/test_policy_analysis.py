import unittest

from blackjack_shoe import Observation
from evaluate import basic_strategy
from policy_analysis import deal_probabilities, exact_return, solve_with_bust_penalty
from traditional_baselines import STATES, policy_iteration


class ExactBustPenaltyTests(unittest.TestCase):
    def test_zero_penalty_is_basic_strategy(self):
        first, later = solve_with_bust_penalty(0.0)
        self.assertEqual(later, policy_iteration().policy)
        basic = basic_strategy(allow_double=True)
        for state in STATES:
            self.assertEqual(first[state], basic(Observation(*state, 2, True, 0.0)))

    def test_penalty_lowers_the_true_return(self):
        best = exact_return(*solve_with_bust_penalty(0.0))
        self.assertLess(exact_return(*solve_with_bust_penalty(0.1)), best)

    def test_deal_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(deal_probabilities().values()), 1.0)


if __name__ == "__main__":
    unittest.main()
