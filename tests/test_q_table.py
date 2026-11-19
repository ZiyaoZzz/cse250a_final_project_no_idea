import unittest

from blackjack_shoe import DOUBLE, Observation
from evaluate import basic_strategy, double_value, evaluate_policy, hit_below_17, random_policy
from q_table import STAGES, Stage, greedy_policy, train_q_table
from traditional_baselines import (
    HIT,
    STICK,
    BlackjackModel,
    _add_card,
    evaluate_policy as evaluate_on_gym,
    policy_iteration,
)

STAGE = {stage.name: stage for stage in STAGES}
CARD_PROBABILITY = {card: (4 / 13 if card == 10 else 1 / 13) for card in range(1, 11)}


def exact_initial_value(value_of_state):
    """Average a state value over the infinite-deck initial deal."""
    value = 0.0
    for first, first_probability in CARD_PROBABILITY.items():
        for second, second_probability in CARD_PROBABILITY.items():
            total, soft = _add_card(0, False, first)
            total, soft = _add_card(total, soft, second)
            for upcard, upcard_probability in CARD_PROBABILITY.items():
                probability = first_probability * second_probability * upcard_probability
                value += probability * value_of_state((total, upcard, soft))
    return value


class EnvironmentAgreementTests(unittest.TestCase):
    def test_infinite_deck_matches_the_exact_value_and_blackjack_v1(self):
        solved = policy_iteration()
        exact = exact_initial_value(solved.values.__getitem__)

        ours = evaluate_policy(basic_strategy(), hands=400_000, num_decks=None, seed=11)
        self.assertLess(abs(ours["mean_return"] - exact), 1.5 * ours["ci95"])

        gym = evaluate_on_gym(solved.policy, episodes=40_000, seed=11)
        self.assertLess(abs(gym["mean_return"] - exact), 0.015)

    def test_double_matches_the_exact_value(self):
        model = BlackjackModel()
        solved = policy_iteration(model)
        exact = exact_initial_value(
            lambda state: max(solved.values[state], double_value(model, state))
        )
        result = evaluate_policy(
            basic_strategy(allow_double=True),
            hands=400_000,
            num_decks=None,
            allow_double=True,
            seed=12,
        )
        self.assertLess(abs(result["mean_return"] - exact), 1.5 * result["ci95"])
        self.assertGreater(result["double_rate"], 0.0)


class StateTests(unittest.TestCase):
    def test_each_stage_only_adds_its_own_features(self):
        observation = Observation(16, 10, False, 3, False, 1.6)
        self.assertEqual(STAGE["baseline"].state(observation), (16, 10, False))
        self.assertEqual(STAGE["num_cards"].state(observation), (16, 10, False, 3))
        self.assertEqual(STAGE["count"].state(observation), (16, 10, False, 2))
        self.assertEqual(STAGE["double"].state(observation), (16, 10, False, False))
        self.assertEqual(STAGE["count_double"].state(observation), (16, 10, False, 2, False))

    def test_count_limit_sets_the_outer_bucket(self):
        observation = Observation(16, 10, False, 3, False, 2.7)
        self.assertEqual(STAGE["count"].state(observation), (16, 10, False, 2))
        self.assertEqual(STAGE["count_3"].state(observation), (16, 10, False, 3))
        self.assertEqual(STAGE["count_double_3"].state(observation._replace(true_count=-9.0)),
                         (16, 10, False, -3, False))


class TrainingTests(unittest.TestCase):
    def test_table_learns_obvious_decisions_and_beats_random(self):
        stage = STAGE["baseline"]
        q_values = train_q_table(stage, hands=100_000, seed=0)
        policy = greedy_policy(stage, q_values)

        self.assertEqual(policy(Observation(20, 6, False, 2, False, 0.0)), STICK)
        self.assertEqual(policy(Observation(5, 10, False, 2, False, 0.0)), HIT)
        learned = evaluate_policy(policy, hands=50_000)["mean_return"]
        self.assertGreater(learned, evaluate_policy(random_policy(), hands=50_000)["mean_return"])

    def test_double_stage_never_doubles_after_the_first_decision(self):
        stage = STAGE["double"]
        q_values = train_q_table(stage, hands=20_000, seed=0)
        policy = greedy_policy(stage, q_values)
        self.assertNotEqual(policy(Observation(11, 6, False, 3, False, 0.0)), DOUBLE)

    def test_bust_penalty_only_changes_training_targets(self):
        stage = Stage("hit_only", "hit only", bust_penalty=0.5)
        penalised = train_q_table(stage, hands=5_000, seed=0)
        plain = train_q_table(Stage("plain", "plain"), hands=5_000, seed=0)
        self.assertLess(penalised[(20, 10, False)][HIT], plain[(20, 10, False)][HIT])

        policy = greedy_policy(stage, penalised)
        self.assertGreaterEqual(evaluate_policy(policy, hands=2_000)["mean_return"], -1.0)

    def test_unvisited_states_fall_back_to_stick(self):
        policy = greedy_policy(STAGE["baseline"], {})
        self.assertEqual(policy(Observation(12, 2, False, 2, False, 0.0)), STICK)

    def test_hit_below_17_rule(self):
        self.assertEqual(hit_below_17(Observation(16, 2, False, 2, False, 0.0)), HIT)
        self.assertEqual(hit_below_17(Observation(17, 2, True, 2, False, 0.0)), STICK)


class ShapingTests(unittest.TestCase):
    def test_potentials_are_zero_at_terminal_and_defined_everywhere(self):
        from q_table import potential

        for kind in ("heuristic", "value"):
            self.assertEqual(potential(kind, None), 0.0)
            for total in range(4, 22):
                for upcard in range(1, 11):
                    obs = Observation(total, upcard, total >= 12, 2, True, 0.0)
                    self.assertTrue(-2.0 <= potential(kind, obs) <= 2.0)

    def test_value_shaping_still_learns_basic_strategy_quality(self):
        stage = STAGE["shaped_value"]
        table = train_q_table(stage, hands=200_000, seed=3, num_decks=None)
        result = evaluate_policy(greedy_policy(stage, table), hands=100_000, num_decks=None,
                                 allow_double=True, seed=5)
        self.assertGreater(result["mean_return"], -0.07)


if __name__ == "__main__":
    unittest.main()
