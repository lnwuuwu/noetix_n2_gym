import csv
import os
import tempfile
import unittest

from humanoid.scripts.select_faststair_checkpoint import (
    absolute_gate,
    aggregate,
    load_evaluation,
    relative_gate,
)


FIELDS = (
    "terrain_level",
    "completion_rate",
    "fall_rate",
    "path_failure_rate",
    "mean_max_lateral_deviation_m",
    "mean_max_yaw_deviation_rad",
    "mean_command_error_m_s",
    "mean_action_rate_rms",
    "mean_action_accel_rms",
    "mean_faststair_planner_valid_fraction",
    "mean_faststair_foothold_error_m",
    "mean_faststair_edge_margin_m",
)


def capable_rows():
    rows = {}
    for level in range(5):
        rows[level] = {
            "terrain_level": float(level),
            "completion_rate": 0.94 if level >= 3 else 0.80,
            "fall_rate": 0.02,
            "path_failure_rate": 0.02,
            "mean_max_lateral_deviation_m": 0.08,
            "mean_max_yaw_deviation_rad": 0.16,
            "mean_command_error_m_s": 0.05,
            "mean_action_rate_rms": 0.60,
            "mean_action_accel_rms": 0.50,
            "mean_faststair_planner_valid_fraction": 0.98,
            "mean_faststair_foothold_error_m": 0.04,
            "mean_faststair_edge_margin_m": 0.02,
        }
    return rows


class FastStairSelectorTests(unittest.TestCase):
    def test_capable_policy_passes_absolute_gate(self):
        self.assertEqual(absolute_gate(capable_rows()), [])

    def test_unsafe_high_stair_policy_is_rejected(self):
        rows = capable_rows()
        rows[4]["completion_rate"] = 0.70
        rows[4]["fall_rate"] = 0.20
        reasons = absolute_gate(rows)
        self.assertTrue(any("10 cm completion" in reason for reason in reasons))
        self.assertTrue(any("10 cm fall" in reason for reason in reasons))

    def test_missing_planner_signal_is_rejected(self):
        rows = capable_rows()
        for row in rows.values():
            row["mean_faststair_planner_valid_fraction"] = 0.0
            row["mean_faststair_edge_margin_m"] = 0.0
        reasons = absolute_gate(rows)
        self.assertTrue(any("planner validity" in reason for reason in reasons))
        self.assertTrue(any("edge margin" in reason for reason in reasons))

    def test_relative_gate_protects_baseline_completion(self):
        baseline = capable_rows()
        candidate = capable_rows()
        candidate[4]["completion_rate"] = 0.85
        reasons = relative_gate(baseline, candidate)
        self.assertTrue(any("10 cm completion" in reason for reason in reasons))

    def test_csv_loader_and_weighted_aggregate(self):
        rows = capable_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "evaluation.csv")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for level in range(5):
                    writer.writerow(rows[level])
            loaded = load_evaluation(path)
        metrics = aggregate(loaded)
        self.assertAlmostEqual(metrics["planner_valid"], 0.98)
        self.assertGreater(metrics["completion"], 0.80)


if __name__ == "__main__":
    unittest.main()
