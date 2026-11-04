import unittest

from traditional_baselines import (
    HIT,
    STICK,
    BlackjackModel,
    policy_iteration,
)


class BlackjackModelTests(unittest.TestCase):
    def test_dealer_distribution_is_normalized_and_uses_bust_score_zero(self):
        model = BlackjackModel()
        distribution = model.dealer_distribution(10)

        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertIn(0, distribution)
        self.assertTrue(all(score == 0 or 17 <= score <= 21 for score in distribution))

    def test_hit_transitions_match_card_probabilities_and_terminal_reward(self):
        model = BlackjackModel()
        transitions = model.transition_distribution((20, 10, False), HIT)

        self.assertAlmostEqual(sum(item[0] for item in transitions), 1.0)
        self.assertAlmostEqual(
            sum(
                prob
                for prob, _, reward, done in transitions
                if done and reward == -1
            ),
            12 / 13,
        )

    def test_policy_iteration_returns_legal_actions_and_sticks_at_21(self):
        result = policy_iteration()

        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(result.policy[(21, 10, False)], STICK)
        self.assertTrue(set(result.policy.values()).issubset({STICK, HIT}))
