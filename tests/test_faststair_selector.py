import csv
import os
import tempfile
import unittest

from humanoid.scripts.select_faststair_checkpoint import (
    absolute_gate,
    aggregate,
    load_evaluation,
    preflight_gate,
    relative_gate,
    stage_gate,
)


FIELDS = (
    "terrain_level",
    "success_rate",
    "completion_rate",
    "curriculum_completion_rate",
    "first_step_rate",
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
    "mean_phase_contact_match",
    "mean_sagittal_foot_phase_match",
    "mean_alternating_tread_count",
    "mean_alternating_tread_rate",
    "mean_repeated_lead_rate",
    "mean_same_tread_join_rate",
    "mean_paired_lead_advances",
    "mean_paired_trailing_joins",
    "mean_paired_sequence_rate",
    "mean_paired_join_coverage",
    "mean_paired_premature_rate",
    "mean_paired_lead_switch_rate",
    "mean_left_tread_advances",
    "mean_right_tread_advances",
    "mean_left_swing_length_m",
    "mean_right_swing_length_m",
    "mean_left_swing_action_accel_rms",
    "mean_left_swing_roll_rate_rms",
    "mean_actual_sole_support_fraction",
)


def capable_rows():
    rows = {}
    for level in range(5):
        rows[level] = {
            "terrain_level": float(level),
            "success_rate": 0.75,
            "completion_rate": 0.94 if level >= 3 else 0.80,
            "curriculum_completion_rate": 0.70,
            "first_step_rate": 0.98,
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
            "mean_phase_contact_match": 0.76,
            "mean_sagittal_foot_phase_match": 0.80,
            "mean_alternating_tread_count": 0.2,
            "mean_alternating_tread_rate": 0.03,
            "mean_repeated_lead_rate": 0.45,
            "mean_same_tread_join_rate": 0.45,
            "mean_paired_lead_advances": 4.5,
            "mean_paired_trailing_joins": 4.0,
            "mean_paired_sequence_rate": 0.90,
            "mean_paired_join_coverage": 0.85,
            "mean_paired_premature_rate": 0.05,
            "mean_paired_lead_switch_rate": 0.02,
            "mean_left_tread_advances": 0.2,
            "mean_right_tread_advances": 4.5,
            "mean_left_swing_length_m": 0.17,
            "mean_right_swing_length_m": 0.28,
            "mean_left_swing_action_accel_rms": 0.50,
            "mean_left_swing_roll_rate_rms": 0.50,
            "mean_actual_sole_support_fraction": 0.90,
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

    def test_stage_one_gate_rejects_standing_policy(self):
        rows = capable_rows()
        rows[0]["completion_rate"] = 0.0
        rows[0]["first_step_rate"] = 0.0
        rows[0]["mean_faststair_planner_valid_fraction"] = 0.0
        reasons = stage_gate(rows, 1)
        self.assertTrue(any("completion" in reason for reason in reasons))
        self.assertTrue(any("first-step" in reason for reason in reasons))
        self.assertTrue(any("planner validity" in reason for reason in reasons))

    def test_stage_one_and_two_capable_policy_passes(self):
        rows = capable_rows()
        self.assertEqual(stage_gate(rows, 1), [])
        self.assertEqual(stage_gate(rows, 2), [])

    def test_preflight_allows_small_task_difference_but_not_broken_migration(self):
        baseline = capable_rows()
        candidate = capable_rows()
        candidate[0]["completion_rate"] = 0.70
        candidate[0]["fall_rate"] = 0.10
        self.assertEqual(preflight_gate(candidate, baseline), [])

        candidate[0]["completion_rate"] = 0.40
        reasons = preflight_gate(candidate, baseline)
        self.assertTrue(any("2 cm completion" in reason for reason in reasons))

    def test_verified_lead_and_join_policy_can_be_promoted(self):
        rows = capable_rows()
        self.assertEqual(stage_gate(rows, 1), [])
        self.assertEqual(stage_gate(rows, 2), [])
        self.assertEqual(absolute_gate(rows), [])

    def test_unpaired_repeated_lead_shortcut_cannot_be_final_best(self):
        rows = capable_rows()
        for level in (3, 4):
            rows[level]["mean_paired_trailing_joins"] = 0.2
            rows[level]["mean_paired_sequence_rate"] = 0.30
            rows[level]["mean_paired_join_coverage"] = 0.05
            rows[level]["mean_paired_premature_rate"] = 0.80
        reasons = absolute_gate(rows)
        self.assertTrue(
            any("verified lead/join sequence" in reason for reason in reasons)
        )

    def test_natural_alternating_policy_is_still_accepted(self):
        rows = capable_rows()
        for row in rows.values():
            row.update(
                {
                    "mean_alternating_tread_count": 3.2,
                    "mean_alternating_tread_rate": 0.55,
                    "mean_repeated_lead_rate": 0.05,
                    "mean_same_tread_join_rate": 0.15,
                    "mean_paired_lead_advances": 0.0,
                    "mean_paired_trailing_joins": 0.0,
                    "mean_paired_sequence_rate": 0.0,
                    "mean_paired_join_coverage": 0.0,
                    "mean_paired_premature_rate": 0.0,
                    "mean_left_tread_advances": 2.8,
                    "mean_right_tread_advances": 2.7,
                    "mean_left_swing_length_m": 0.17,
                    "mean_right_swing_length_m": 0.18,
                }
            )
        self.assertEqual(absolute_gate(rows), [])

    def test_stage_two_requires_six_centimeter_capability(self):
        rows = capable_rows()
        rows[2]["completion_rate"] = 0.20
        rows[2]["first_step_rate"] = 0.40
        reasons = stage_gate(rows, 2)
        self.assertTrue(any("6 cm completion" in reason for reason in reasons))
        self.assertTrue(any("6 cm first-step" in reason for reason in reasons))

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
