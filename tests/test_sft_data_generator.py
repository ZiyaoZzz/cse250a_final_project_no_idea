import json
import tempfile
import unittest
from pathlib import Path

from sft_data_generator import STATE_FIELDS, generate_expert_data


class ExpertDataTests(unittest.TestCase):
    def test_examples_are_sampled_state_action_pairs_without_hidden_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.jsonl"
            count = generate_expert_data(path, episodes=20, seed=7)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, len(rows))
        self.assertGreaterEqual(count, 20)
        for row in rows:
            self.assertEqual(set(row), {"episode", "step", "state", "action", "action_id"})
            self.assertEqual(set(row["state"]), STATE_FIELDS)
            self.assertIn(row["action"], {"Hit", "Stick"})
            self.assertIn(row["action_id"], {0, 1})
            self.assertEqual(row["action_id"], {"Stick": 0, "Hit": 1}[row["action"]])
            self.assertTrue(4 <= row["state"]["player_sum"] <= 21)
            self.assertTrue(1 <= row["state"]["dealer_showing"] <= 10)


if __name__ == "__main__":
    unittest.main()
