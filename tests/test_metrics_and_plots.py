import unittest

from metrics_and_plots import merge_reports, results_table


def _report(stages, seconds=1.0, eval_hands=100, offset=0.0):
    return {
        "settings": {"eval_hands": eval_hands, "seconds": seconds},
        "stages": [{"name": name, "label": name} for name in stages],
        "references": [],
        "runs": [{"stage": name, "seed": seed, "states": 1,
                  "final": {"mean_return": offset + i / 10 + seed / 1000, "win_rate": 0.4,
                            "bust_rate": 0.2, "double_rate": 0.0}}
                 for i, name in enumerate(stages) for seed in range(3)],
    }


class MergeTests(unittest.TestCase):
    def test_merge_keeps_first_copy_and_pairs_by_seed(self):
        merged = merge_reports([_report(["baseline", "count"]),
                                _report(["count", "count_3"], seconds=9.0, offset=1.0)])
        self.assertEqual([s["name"] for s in merged["stages"]], ["baseline", "count", "count_3"])
        self.assertEqual(len(merged["runs"]), 9)
        table = results_table(merged, ["count", "count_3"], [("count_3", "count")])
        self.assertIn("| count_3 − (count) | +1.0000 ± 0.0000 |", table)

    def test_merge_rejects_different_settings(self):
        with self.assertRaises(ValueError):
            merge_reports([_report(["baseline"]), _report(["count"], eval_hands=5)])


if __name__ == "__main__":
    unittest.main()
