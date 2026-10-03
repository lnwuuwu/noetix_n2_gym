import json
import tempfile
import unittest
from pathlib import Path

from sim2sim.parkour_evaluation import (
    best_evaluation,
    checkpoint_number,
    evaluation_record,
    evaluation_report,
    write_evaluation_report,
)


def _result(
        finished=False, clears=0, failures=0, progress=0.0,
        resets=0, sim_time=600.0):
    return {
        "campaign_finished": finished,
        "stage_clears": clears,
        "failures": failures,
        "best_progress_fraction": progress,
        "resets": resets,
        "sim_time": sim_time,
        "max_x": progress * 59.0,
    }


class ParkourEvaluationTest(unittest.TestCase):

    def test_checkpoint_is_extracted_from_export_name(self):
        self.assertEqual(checkpoint_number("policy_4200.pt"), 4200)
        self.assertEqual(
            checkpoint_number("/tmp/policy-3800_candidate.pt"), 3800)
        self.assertIsNone(checkpoint_number("policy_best_stable.pt"))

    def test_record_preserves_quantitative_result(self):
        result = _result(
            finished=True, clears=5, failures=0, progress=1.0,
            resets=0, sim_time=123.4)
        result["stage_failure_counts"] = {
            "up_stairs": 0, "down_stairs": 0}
        record = evaluation_record(
            "policy_4200.pt", result=result, wall_time_seconds=2.5)
        self.assertEqual(record["status"], "ok")
        self.assertTrue(record["success"])
        self.assertEqual(record["checkpoint"], 4200)
        self.assertEqual(record["stage_clears"], 5)
        self.assertEqual(
            record["stage_failure_counts"]["up_stairs"], 0)

    def test_ranking_is_success_then_stages_then_stability(self):
        incomplete = evaluation_record(
            "policy_3000.pt",
            result=_result(
                clears=4, failures=0, progress=0.95))
        unstable_success = evaluation_record(
            "policy_3400.pt",
            result=_result(
                finished=True, clears=5, failures=3, progress=1.0))
        stable_success = evaluation_record(
            "policy_4200.pt",
            result=_result(
                finished=True, clears=5, failures=0, progress=1.0,
                sim_time=180.0))
        self.assertIs(
            best_evaluation(
                [incomplete, unstable_success, stable_success]),
            stable_success)

    def test_fewer_failures_wins_within_the_same_stage(self):
        farther_but_unstable = evaluation_record(
            "policy_3800.pt",
            result=_result(
                clears=3, failures=2, progress=0.79))
        stable = evaluation_record(
            "policy_4000.pt",
            result=_result(
                clears=3, failures=0, progress=0.72))
        self.assertIs(
            best_evaluation([farther_but_unstable, stable]), stable)

    def test_errors_are_reported_but_never_selected(self):
        error = evaluation_record(
            "policy_3000.pt", error=FileNotFoundError("missing"))
        valid = evaluation_record(
            "policy_3400.pt", result=_result(progress=0.1))
        self.assertEqual(error["status"], "error")
        self.assertFalse(error["success"])
        self.assertIs(best_evaluation([error, valid]), valid)
        self.assertIsNone(best_evaluation([error]))

    def test_json_report_is_atomic_and_machine_readable(self):
        candidate = evaluation_record(
            "policy_4200.pt",
            result=_result(
                finished=True, clears=5, progress=1.0))
        report = evaluation_report(
            "n2_parkour_slow_stable.yaml", 600.0, [candidate],
            command_x=0.35,
            control_overrides={
                "action_filter_alpha": 1.0,
                "action_gain": 1.0,
                "kd_scale": 1.0,
            })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "summary.json"
            write_evaluation_report(output, report)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["candidate_count"], 1)
        self.assertEqual(loaded["successful_count"], 1)
        self.assertEqual(loaded["best_checkpoint"], 4200)
        self.assertEqual(
            loaded["control_overrides"]["action_gain"], 1.0)
        self.assertEqual(
            loaded["best_policy_path"], "policy_4200.pt")


if __name__ == "__main__":
    unittest.main()
