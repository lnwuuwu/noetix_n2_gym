import math
import unittest

import numpy as np

from sim2sim.camera_follow import SmoothFollowCamera


class SmoothFollowCameraTest(unittest.TestCase):

    def test_first_update_snaps_to_full_xyz_target(self):
        camera = SmoothFollowCamera(
            lookahead=0.5, z_offset=-0.05, azimuth_offset=70.0)
        lookat, azimuth = camera.update(
            np.array([2.0, 1.0, 2.6]), yaw=0.0, dt=1.0 / 30.0)
        np.testing.assert_allclose(lookat, [2.5, 1.0, 2.55])
        self.assertAlmostEqual(azimuth, 70.0)

    def test_descent_smoothly_follows_base_z_without_a_floor_clamp(self):
        camera = SmoothFollowCamera(
            lookahead=0.0, z_offset=-0.05,
            position_tau=0.2, yaw_tau=0.2)
        first, _ = camera.update(
            np.array([0.0, 0.0, 1.0]), yaw=0.0, dt=0.1)
        second, _ = camera.update(
            np.array([0.2, 0.1, 0.3]), yaw=0.0, dt=0.1)
        self.assertGreater(second[0], first[0])
        self.assertGreater(second[1], first[1])
        self.assertLess(second[2], first[2])
        self.assertLess(second[2], 0.95)
        self.assertGreater(second[2], 0.25)

    def test_heading_wrap_uses_the_short_rotation(self):
        camera = SmoothFollowCamera(
            lookahead=0.0, azimuth_offset=0.0,
            position_tau=0.1, yaw_tau=0.1)
        _, first = camera.update(
            np.zeros(3), yaw=math.radians(179.0), dt=0.1)
        _, second = camera.update(
            np.zeros(3), yaw=math.radians(-179.0), dt=0.1)
        self.assertAlmostEqual(first, 179.0)
        self.assertLess(abs(second - first), 2.0)

    def test_large_reset_teleport_snaps_immediately(self):
        camera = SmoothFollowCamera(
            lookahead=0.0, z_offset=0.0, snap_distance=2.0)
        camera.update(np.array([10.0, 0.0, 2.0]), yaw=0.0, dt=0.03)
        lookat, _ = camera.update(
            np.array([0.0, 0.0, 0.8]), yaw=0.0, dt=0.03)
        np.testing.assert_allclose(lookat, [0.0, 0.0, 0.8])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "snap_distance"):
            SmoothFollowCamera(snap_distance=0.0)
        with self.assertRaisesRegex(ValueError, "time constants"):
            SmoothFollowCamera(position_tau=-0.1)


if __name__ == "__main__":
    unittest.main()
