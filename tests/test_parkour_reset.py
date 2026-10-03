import unittest

from sim2sim.parkour_reset import (
    RESET_FALL,
    RESET_LOW_CLEARANCE,
    RESET_STAGE_CLEAR,
    RESET_STUCK,
    RESET_SUMMIT,
    RESET_TILT,
    choose_campaign_transition,
    choose_up_only_reset,
    update_success_hold,
    update_forward_progress,
    validate_course_mode,
)


class ParkourResetTest(unittest.TestCase):

    def test_route_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_course_mode("waypoints", True, True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_course_mode("waypoints", False, True, True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_course_mode("waypoints", True, False, True)

    def test_up_only_requires_waypoints(self):
        with self.assertRaisesRegex(ValueError, "requires goal_mode=waypoints"):
            validate_course_mode("carrot", False, True)
        with self.assertRaisesRegex(ValueError, "requires goal_mode=waypoints"):
            validate_course_mode("carrot", False, False, True)
        validate_course_mode("waypoints", False, True)
        validate_course_mode("waypoints", True, False)
        validate_course_mode("waypoints", False, False, True)

    def test_progress_anchor_moves_only_after_minimum_advance(self):
        self.assertEqual(
            update_forward_progress(0.04, 0.0, 1.0, 2.0, 0.05),
            (0.0, 1.0))
        self.assertEqual(
            update_forward_progress(0.05, 0.0, 1.0, 2.0, 0.05),
            (0.05, 2.0))

    def test_progress_threshold_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            update_forward_progress(1.0, 0.0, 0.0, 1.0, 0.0)

    def test_summit_has_priority_over_failure_conditions(self):
        self.assertEqual(
            choose_up_only_reset(True, 0.1, 0.35, 20.0, 10.0),
            RESET_SUMMIT)

    def test_fall_has_priority_over_stuck(self):
        self.assertEqual(
            choose_up_only_reset(False, 0.2, 0.35, 20.0, 10.0),
            RESET_FALL)

    def test_stuck_timeout_can_be_disabled(self):
        self.assertEqual(
            choose_up_only_reset(False, 0.8, 0.35, 10.0, 10.0),
            RESET_STUCK)
        self.assertIsNone(
            choose_up_only_reset(False, 0.8, 0.35, 100.0, 0.0))

    def test_campaign_success_requires_continuous_hold(self):
        started, stable = update_success_hold(True, None, 10.0, 0.3)
        self.assertEqual(started, 10.0)
        self.assertFalse(stable)
        started, stable = update_success_hold(True, started, 10.3, 0.3)
        self.assertTrue(stable)
        started, stable = update_success_hold(False, started, 10.4, 0.3)
        self.assertIsNone(started)
        self.assertFalse(stable)

    def test_campaign_hold_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            update_success_hold(True, None, 0.0, -0.1)

    def test_campaign_clear_precedes_failures(self):
        self.assertEqual(
            choose_campaign_transition(
                True, 0.0, 0.35, 0.0, 0.35, 99.0, 10.0),
            RESET_STAGE_CLEAR)

    def test_campaign_reset_grace_suppresses_failures(self):
        self.assertIsNone(
            choose_campaign_transition(
                False, 0.0, 0.35, 0.0, 0.35, 99.0, 10.0,
                in_reset_grace=True))

    def test_campaign_failure_priorities_and_timeout(self):
        self.assertEqual(
            choose_campaign_transition(
                False, 0.2, 0.35, 1.0, 0.35, 20.0, 10.0),
            RESET_LOW_CLEARANCE)
        self.assertEqual(
            choose_campaign_transition(
                False, 0.8, 0.35, 0.2, 0.35, 20.0, 10.0),
            RESET_TILT)
        self.assertEqual(
            choose_campaign_transition(
                False, 0.8, 0.35, 1.0, 0.35, 20.0, 10.0),
            RESET_STUCK)
        self.assertIsNone(
            choose_campaign_transition(
                False, 0.8, 0.35, 1.0, 0.35, 99.0, 0.0))


if __name__ == "__main__":
    unittest.main()
