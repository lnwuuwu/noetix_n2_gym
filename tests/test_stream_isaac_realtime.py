import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPT_PATH = os.path.join(
    REPOSITORY_ROOT, "humanoid", "scripts", "stream_isaac_parkour.py")


def _load_stream_module():
    isaacgym = types.ModuleType("isaacgym")
    isaacgym.gymapi = types.SimpleNamespace()
    isaacgym.gymutil = types.SimpleNamespace()

    envs = types.ModuleType("humanoid.envs")
    envs.__all__ = []
    task_registry_module = types.ModuleType("humanoid.utils.task_registry")
    task_registry_module.task_registry = object()
    mjpeg_module = types.ModuleType("sim2sim.mjpeg_stream")
    mjpeg_module.MJPEGStreamer = object

    stubs = {
        "isaacgym": isaacgym,
        "humanoid.envs": envs,
        "humanoid.utils.task_registry": task_registry_module,
        "sim2sim.mjpeg_stream": mjpeg_module,
    }
    spec = importlib.util.spec_from_file_location(
        "_stream_isaac_parkour_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self, now=0.0):
        self.now = float(now)
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeStartStreamer:
    def __init__(self, client=True, start=True):
        self.client = client
        self.start = start
        self.calls = []

    def wait_for_client(self):
        self.calls.append("client")
        return self.client

    def wait_for_start(self):
        self.calls.append("start")
        return self.start


class IsaacRealtimePacingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = _load_stream_module()

    def test_sleep_until_uses_short_interruptible_chunks(self):
        clock = FakeClock(now=3.0)
        self.stream._sleep_until(
            3.12, clock=clock.monotonic, sleeper=clock.sleep)

        self.assertAlmostEqual(clock.now, 3.12)
        self.assertGreater(len(clock.sleeps), 1)
        self.assertLessEqual(
            max(clock.sleeps), self.stream.MAX_PACING_SLEEP)

    def test_absolute_deadlines_do_not_accumulate_previous_overshoot(self):
        wall_start = 10.0
        sim_start = 2.0

        first = self.stream._realtime_deadline(
            wall_start, sim_start, sim_time=2.02)
        second = self.stream._realtime_deadline(
            wall_start, sim_start, sim_time=2.04)

        self.assertAlmostEqual(first, 10.02)
        self.assertAlmostEqual(second, 10.04)

    def test_measured_rtf_is_one_after_realtime_pacing(self):
        self.assertAlmostEqual(
            self.stream._measured_rtf(
                wall_start=10.0,
                sim_start=3.0,
                sim_time=4.5,
                wall_time=11.5),
            1.0)

    def test_camera_is_robot_right_and_follows_root_xyz(self):
        args = types.SimpleNamespace(
            camera_rear=2.6,
            camera_side=3.4,
            camera_up=2.8,
            camera_lookahead=0.8,
            camera_target_up=0.25,
        )
        yaw = 0.7
        root = np.asarray([
            4.0, 5.0, 1.9,
            0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0),
        ])
        position, target = self.stream._follow_camera_pose(root, args)
        forward = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.asarray([np.sin(yaw), -np.cos(yaw), 0.0])
        offset = position - root[:3]

        self.assertAlmostEqual(float(np.dot(offset, right)), 3.4)
        self.assertAlmostEqual(float(np.dot(offset, forward)), -2.6)
        self.assertAlmostEqual(position[2] - root[2], 2.8)
        self.assertAlmostEqual(target[2] - root[2], 0.25)

        translation = np.asarray([1.3, -0.4, -1.2])
        moved = root.copy()
        moved[:3] += translation
        moved_position, moved_target = self.stream._follow_camera_pose(
            moved, args)
        np.testing.assert_allclose(
            moved_position - position, translation, atol=1e-12)
        np.testing.assert_allclose(
            moved_target - target, translation, atol=1e-12)

    def test_camera_position_stays_inside_outer_safety_boundary(self):
        args = types.SimpleNamespace(
            camera_rear=2.6,
            camera_side=3.4,
            camera_up=2.8,
            camera_lookahead=0.8,
            camera_target_up=0.25,
        )
        root = np.asarray([1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0])
        position, _ = self.stream._follow_camera_pose(
            root, args, lateral_bounds=(0.35, 9.65))
        self.assertAlmostEqual(position[1], 0.35)

    def test_manual_start_waits_for_client_then_start_gate(self):
        streamer = FakeStartStreamer()
        args = types.SimpleNamespace(
            manual_start=True, start_immediately=False)

        self.assertTrue(
            self.stream._wait_for_playback_start(streamer, args))
        self.assertEqual(streamer.calls, ["client", "start"])

    def test_manual_start_never_reaches_gate_without_client(self):
        streamer = FakeStartStreamer(client=False)
        args = types.SimpleNamespace(
            manual_start=True, start_immediately=False)

        self.assertFalse(
            self.stream._wait_for_playback_start(streamer, args))
        self.assertEqual(streamer.calls, ["client"])

    def test_existing_start_modes_remain_compatible(self):
        browser_gated = FakeStartStreamer()
        args = types.SimpleNamespace(
            manual_start=False, start_immediately=False)
        self.assertTrue(self.stream._wait_for_playback_start(
            browser_gated, args))
        self.assertEqual(browser_gated.calls, ["client"])

        immediate = FakeStartStreamer()
        args.start_immediately = True
        self.assertTrue(self.stream._wait_for_playback_start(
            immediate, args))
        self.assertEqual(immediate.calls, [])

    def test_manual_and_immediate_start_are_mutually_exclusive(self):
        args = types.SimpleNamespace(
            task=self.stream.COURSE_TASK,
            stream_port=18081,
            camera_width=640,
            camera_height=360,
            stream_fps=25.0,
            jpeg_quality=85,
            camera_rear=2.6,
            camera_side=3.4,
            camera_up=2.8,
            camera_boundary_clearance=0.35,
            play_steps=0,
            manual_start=True,
            start_immediately=True,
        )

        with self.assertRaisesRegex(ValueError, "不能同时使用"):
            self.stream._validate_args(args)


if __name__ == "__main__":
    unittest.main()
