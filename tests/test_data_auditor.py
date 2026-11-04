import tempfile
import unittest
from pathlib import Path

from data_auditor import audit_csv


class AuditCsvTests(unittest.TestCase):
    def audit_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            path.write_text(text, encoding="utf-8")
            return audit_csv(path)

    def test_outcome_only_data_is_not_transition_data(self):
        report = self.audit_text("game_id,winloss\n1,Win\n")

        self.assertEqual(report.status, "outcome_only")
        self.assertTrue(report.outcome_only)
        self.assertIn("must not be used as transition data", report.notes[0])

    def test_terminal_action_history_is_not_mislabeled_as_outcome_only(self):
        report = self.audit_text(
            'initial_hand,actions_taken,win\n"[10, 7]","[\'S\']",1\n'
        )

        self.assertEqual(report.status, "terminal_with_action_history")
        self.assertFalse(report.outcome_only)
        self.assertEqual(report.action_history_columns, ["actions_taken"])

    def test_transition_schema_is_detected_and_hidden_card_is_flagged(self):
        report = self.audit_text(
            "player_sum,dealer_showing,usable_ace,action,next_player_sum,"
            "next_dealer_showing,next_usable_ace,reward,terminated,hole_card\n"
            "15,10,False,hit,19,10,False,0,False,7\n"
        )

        self.assertEqual(report.status, "transition_candidate")
        self.assertEqual(report.missing_fields, [])
        self.assertEqual(report.hidden_card_columns, ["hole_card"])
        self.assertEqual(report.rows_inspected, 1)

    def test_incomplete_schema_is_reported(self):
        report = self.audit_text("player_sum,action\n15,hit\n")

        self.assertEqual(report.status, "incomplete_schema")
        self.assertIn("state_dealer_showing", report.missing_fields)

    def test_malformed_rows_are_reported(self):
        report = self.audit_text(
            "player_sum,dealer_showing,usable_ace,action,next_player_sum,"
            "next_dealer_showing,next_usable_ace,reward,terminated\n"
            "15,10,False,hit\n"
        )

        self.assertEqual(report.status, "malformed_rows")
        self.assertEqual(report.row_width_errors, 1)


if __name__ == "__main__":
    unittest.main()
