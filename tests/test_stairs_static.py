"""CPU-only checks for n2_stairs geometry, dimensions, registration, and YAML."""

import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_pure_stairs_module():
    path = ROOT / "humanoid" / "utils" / "stairs_terrain.py"
    spec = importlib.util.spec_from_file_location("n2_stairs_geometry_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pure_gait_guidance_module():
    return load_module(
        "n2_gait_guidance_test", "sim2sim/gait_guidance.py"
    )


def load_isaac_stage_gate_module():
    return load_module(
        "n2_isaac_stage_gate_test",
        "humanoid/scripts/isaac_stairs_stage_gate.py",
    )


def load_isaac_stability_tournament_module():
    return load_module(
        "n2_isaac_stability_tournament_test",
        "humanoid/scripts/isaac_stability_tournament.py",
    )


def load_pure_mujoco_eval_module():
    """Load the contact tracker without requiring MuJoCo or Isaac Gym."""
    previous_mujoco = sys.modules.get("mujoco")
    previous_sim2sim = sys.modules.get("sim2sim")
    fake_mujoco = types.ModuleType("mujoco")
    fake_sim2sim = types.ModuleType("sim2sim")
    for name in (
        "gait_phase_observations",
        "get_obs",
        "load_mujoco_model",
        "pd_control",
        "resolve_joint_layout",
        "stair_height_observations",
    ):
        setattr(fake_sim2sim, name, lambda *args, **kwargs: None)
    sys.modules["mujoco"] = fake_mujoco
    sys.modules["sim2sim"] = fake_sim2sim
    try:
        return load_module(
            "n2_mujoco_eval_test", "sim2sim/eval_stairs_mujoco.py"
        )
    finally:
        if previous_mujoco is None:
            sys.modules.pop("mujoco", None)
        else:
            sys.modules["mujoco"] = previous_mujoco
        if previous_sim2sim is None:
            sys.modules.pop("sim2sim", None)
        else:
            sys.modules["sim2sim"] = previous_sim2sim


def parse_tree(relative_path):
    path = ROOT / relative_path
    return ast.parse(path.read_text(), filename=str(path))


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def nested_class(tree, *names):
    body = tree.body
    current = None
    for name in names:
        current = next(
            node for node in body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        body = current.body
    return current


def literal_assignments(class_node):
    values = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


class StairGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = load_pure_stairs_module()

    def test_five_exact_monotonic_levels(self):
        for requested_height in (0.02, 0.04, 0.06, 0.08, 0.10):
            field, metadata = self.geometry.build_directional_stairs(
                terrain_length=5.0,
                terrain_width=3.0,
                horizontal_scale=0.05,
                vertical_scale=0.005,
                start_platform_length=1.15,
                step_width=0.30,
                step_height=requested_height,
                num_steps=6,
            )
            self.assertEqual(field.shape, (100, 60))
            self.assertTrue(np.all(np.diff(field[:, 30]) >= 0))
            self.assertTrue(np.all(field[:23] == 0))
            self.assertAlmostEqual(metadata["step_height"], requested_height)
            self.assertAlmostEqual(metadata["step_width"], 0.30)
            self.assertAlmostEqual(metadata["top_height"], 6 * requested_height)
            self.assertGreater(metadata["success_x"], metadata["top_start_x"])
            self.assertLess(metadata["success_x"], 5.0)

    def test_actor_height_indices_match_meshgrid_flattening(self):
        indices = self.geometry.select_height_indices(
            [-0.20, 0.0, 0.25, 0.45, 0.65, 0.85, 1.05],
            [-0.24, 0.0, 0.24],
            [0.25, 0.45, 0.65, 0.85],
            [-0.24, 0.0, 0.24],
        )
        np.testing.assert_array_equal(indices, np.arange(6, 18))

    def test_analytic_sim2sim_heights_match_riser_boundaries(self):
        heights = self.geometry.terrain_height_at_x(
            np.asarray([0.59, 0.60, 0.89, 0.90, 2.40, 3.00]),
            start_x=0.60,
            step_width=0.30,
            step_height=0.06,
            num_steps=6,
        )
        np.testing.assert_allclose(heights, [0.0, 0.06, 0.06, 0.12, 0.36, 0.36])

    def test_tread_classifier_rejects_step_to_and_repeated_lead_gaits(self):
        def run_sequence(landings):
            previous_tread = 0
            previous_foot = -1
            previous_joined_tread = -1
            counts = np.zeros(5, dtype=np.int64)
            for tread, foot in landings:
                result = self.geometry.classify_tread_transition(
                    np.asarray([True]),
                    np.asarray([tread]),
                    np.asarray([foot]),
                    np.asarray([previous_tread]),
                    np.asarray([previous_foot]),
                    np.asarray([previous_joined_tread]),
                )
                flags = np.asarray([bool(value[0]) for value in result])
                counts += flags.astype(np.int64)
                if flags[0]:
                    previous_tread = tread
                    previous_foot = foot
                if flags[3]:
                    previous_joined_tread = tread
            return counts

        # advanced, alternating, repeated-lead, same-tread-join, skipped
        natural = run_sequence([(1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (6, 1)])
        np.testing.assert_array_equal(natural, [6, 6, 0, 0, 0])

        step_to = run_sequence(
            [
                (1, 0), (1, 1),
                (2, 0), (2, 1),
                (3, 0), (3, 1),
                (4, 0), (4, 1),
                (5, 0), (5, 1),
                (6, 0), (6, 1),
            ]
        )
        np.testing.assert_array_equal(step_to, [6, 1, 5, 6, 0])

        skipped = run_sequence([(1, 0), (3, 1)])
        np.testing.assert_array_equal(skipped, [2, 1, 0, 0, 1])

        # Normal alternating strides after reaching the top platform must
        # not be counted as an unlimited series of same-tread joins.
        top_platform_walk = run_sequence(
            [
                (1, 0), (2, 1), (3, 0),
                (4, 1), (5, 0), (6, 1),
                (6, 0), (6, 1), (6, 0), (6, 1),
            ]
        )
        np.testing.assert_array_equal(top_platform_walk, [6, 6, 0, 1, 0])

    def test_stable_contact_rising_edge_drives_one_landing(self):
        previous = np.asarray([[True, False]])

        # A high-speed raw impact is not stable and therefore is not a landing.
        impact = np.asarray([[True, False]])
        np.testing.assert_array_equal(
            self.geometry.stable_landing_mask(impact, previous),
            [[False, False]],
        )

        # The same raw contact becomes one landing when the right foot first
        # stabilizes while the left foot was already supporting the robot.
        settled = np.asarray([[True, True]])
        np.testing.assert_array_equal(
            self.geometry.stable_landing_mask(settled, previous),
            [[False, True]],
        )
        np.testing.assert_array_equal(
            self.geometry.stable_landing_mask(settled, settled),
            [[False, False]],
        )

        # Simultaneous stabilization and a landing without prior opposite
        # support are both rejected as hopping/unsupported impacts.
        unsupported = np.asarray([[False, False]])
        np.testing.assert_array_equal(
            self.geometry.stable_landing_mask(settled, unsupported),
            [[False, False]],
        )
        np.testing.assert_array_equal(
            self.geometry.stable_landing_mask(
                np.asarray([[False, True]]), unsupported
            ),
            [[False, False]],
        )

    def test_continuous_swing_trajectory_has_exact_endpoints(self):
        start = np.asarray([[0.0, 0.09, 0.065]])
        landing = np.asarray([[0.45, 0.09, 0.265]])
        arc = 0.05

        at_start = self.geometry.smooth_swing_trajectory(
            start, landing, np.asarray([0.0]), arc
        )
        at_mid = self.geometry.smooth_swing_trajectory(
            start, landing, np.asarray([0.5]), arc
        )
        at_end = self.geometry.smooth_swing_trajectory(
            start, landing, np.asarray([1.0]), arc
        )
        np.testing.assert_allclose(at_start, start)
        np.testing.assert_allclose(at_end, landing)
        # XY deliberately lags Z so N2's forward-extending toe clears the
        # riser before the ankle travels into the next tread.
        np.testing.assert_allclose(
            at_mid[:, :2],
            start[:, :2]
            + 0.337961498939682 * (landing - start)[:, :2],
        )
        expected_apex = np.maximum(
            0.5 * (start[:, 2] + landing[:, 2]) + arc,
            np.maximum(start[:, 2], landing[:, 2]),
        )
        np.testing.assert_allclose(at_mid[:, 2], expected_apex)
        # Two 10 cm treads end at 0.20 m plus the ankle-to-sole offset.
        self.assertAlmostEqual(float(at_end[0, 2]), 0.20 + 0.065)

        # Both the endpoint interpolation and clearance bump approach zero
        # vertical velocity at lift-off/touchdown instead of kicking the foot.
        epsilon = 1.0e-4
        just_after_start = self.geometry.smooth_swing_trajectory(
            start, landing, np.asarray([epsilon]), arc
        )
        just_before_end = self.geometry.smooth_swing_trajectory(
            start, landing, np.asarray([1.0 - epsilon]), arc
        )
        np.testing.assert_allclose(
            (just_after_start - at_start) / epsilon,
            np.zeros_like(start),
            atol=5.0e-3,
        )
        np.testing.assert_allclose(
            (at_end - just_before_end) / epsilon,
            np.zeros_like(landing),
            atol=5.0e-3,
        )

    def test_delayed_horizontal_swing_clears_ten_centimeter_riser(self):
        step_width = 0.30
        step_height = 0.10
        ankle_offset = 0.045
        toe_extent = 0.113
        progress = np.linspace(0.0, 1.0, 10001)
        start = np.asarray([[0.0, 0.09, ankle_offset]])
        landing = np.asarray(
            [[step_width, 0.09, step_height + ankle_offset]]
        )
        trajectory = self.geometry.smooth_swing_trajectory(
            start,
            landing,
            progress,
            0.09,
            forward_delay=0.15,
            lift_end=0.35,
            descent_start=0.72,
        )
        first_crossing = np.flatnonzero(
            trajectory[:, 0] + toe_extent >= 0.5 * step_width
        )[0]
        pitch = 0.20
        toe_local_z = (
            -np.sin(pitch) * toe_extent
            + np.cos(pitch) * -0.039522
        )
        toe_clearance = (
            trajectory[first_crossing, 2]
            + toe_local_z
            - step_height
        )
        self.assertGreaterEqual(float(toe_clearance), 0.02)

        descent_index = int(0.72 * (len(progress) - 1))
        heel_x = trajectory[descent_index, 0] - 0.072176
        self.assertGreaterEqual(float(heel_x - 0.5 * step_width), 0.01)

    def test_three_stage_swing_holds_before_forward_motion(self):
        progress = np.linspace(0.0, 1.0, 1001)
        start = np.asarray([[0.0, 0.09, 0.045]])
        landing = np.asarray([[0.30, 0.09, 0.145]])
        trajectory = self.geometry.smooth_swing_trajectory(
            start, landing, progress, 0.09
        )
        delay_index = int(0.15 * (len(progress) - 1))
        np.testing.assert_allclose(
            trajectory[: delay_index + 1, :2],
            np.broadcast_to(start[:, :2], (delay_index + 1, 2)),
            atol=1.0e-12,
        )
        self.assertTrue(np.all(np.diff(trajectory[:, 0]) >= -1.0e-12))
        lift_end_index = int(0.35 * (len(progress) - 1))
        descent_index = int(0.72 * (len(progress) - 1))
        self.assertTrue(
            np.all(
                np.diff(trajectory[: lift_end_index + 1, 2]) >= -1.0e-12
            )
        )
        np.testing.assert_allclose(
            trajectory[lift_end_index:descent_index, 2],
            trajectory[lift_end_index, 2],
            atol=1.0e-12,
        )
        self.assertTrue(
            np.all(np.diff(trajectory[descent_index:, 2]) <= 1.0e-12)
        )

    def test_swing_support_uses_stable_contact_hysteresis(self):
        pending = np.asarray([[True, True], [True, True]])
        new_swing = np.asarray([[True, False], [False, False]])
        previous_valid = np.asarray([[False, True], [True, True]])
        # A raw sensor dropout can still be stable after release hysteresis;
        # only the genuinely lost right-side support invalidates its swing.
        opposite_stable = np.asarray([[True, True], [True, False]])
        np.testing.assert_array_equal(
            self.geometry.retained_swing_support_mask(
                pending,
                new_swing,
                previous_valid,
                opposite_stable,
                np.asarray([[False, False], [False, False]]),
            ),
            [[True, True], [True, False]],
        )

        # A true opposite-foot lift permanently invalidates that swing even
        # while stable-contact release hysteresis is still retaining support.
        np.testing.assert_array_equal(
            self.geometry.retained_swing_support_mask(
                np.asarray([[True, True]]),
                np.asarray([[False, False]]),
                np.asarray([[True, True]]),
                np.asarray([[True, True]]),
                np.asarray([[True, False]]),
            ),
            [[False, True]],
        )

    def test_true_airborne_rejects_dropout_and_horizontal_riser_force(self):
        force = np.asarray([[0.0, 20.0, 0.0, 0.0]])
        clearance = np.asarray([[0.0, 0.03, 0.03, 0.03]])
        np.testing.assert_array_equal(
            self.geometry.true_airborne_mask(
                force,
                clearance,
                5.0,
                0.015,
            ),
            [[False, False, True, True]],
        )

    def test_same_tread_support_excludes_approach_and_top(self):
        stable = np.asarray(
            [[True, True], [True, True], [True, True], [True, False]]
        )
        treads = np.asarray([[0, 0], [3, 3], [6, 6], [3, 3]])
        np.testing.assert_array_equal(
            self.geometry.same_tread_support_mask(stable, treads, 6),
            [False, True, False, False],
        )

    def test_stable_tread_advance_requires_latched_real_swing(self):
        stable = np.asarray(
            [[True, True], [True, True], [True, True], [True, True]]
        )
        geometry = np.asarray(
            [[True, False], [True, True], [True, True], [True, True]]
        )
        pending = np.asarray(
            [[True, False], [True, True], [False, False], [True, True]]
        )
        treads = np.asarray([[1, 1], [1, 2], [2, 2], [0, 1]])
        accepted = np.asarray([[0, 1], [0, 0], [1, 1], [1, 1]])
        np.testing.assert_array_equal(
            self.geometry.stable_tread_advance_mask(
                stable, geometry, pending, treads, accepted
            ),
            [
                [True, False],
                [True, True],
                [False, False],
                [False, False],
            ],
        )


class MujocoSim2SimTests(unittest.TestCase):
    def test_phase_swing_guidance_is_alternating_and_foot_level(self):
        guidance = load_pure_gait_guidance_module()
        cfg = {
            "reference_step_height_m": 0.02,
            "motion_start_phase": 0.06,
            "blend_ramp_fraction": 0.20,
            "hip_landing_offset_rad": 0.22,
            "hip_landing_height_gain": 1.25,
            "knee_clearance_offset_rad": 0.34,
            "knee_clearance_height_gain": 3.00,
            "knee_landing_offset_rad": 0.05,
            "knee_landing_height_gain": 0.75,
        }
        indices = np.asarray([[6, 7, 8], [15, 16, 17]])
        right, right_mask, right_foot, _, right_weight = (
            guidance.phase_swing_action_residual(
                0.25, 0.02, 0.25, 18, indices, cfg
            )
        )
        left, left_mask, left_foot, _, left_weight = (
            guidance.phase_swing_action_residual(
                0.75, 0.02, 0.25, 18, indices, cfg
            )
        )
        self.assertEqual(right_foot, 1)
        self.assertEqual(left_foot, 0)
        np.testing.assert_array_equal(
            np.flatnonzero(right_mask), indices[1]
        )
        np.testing.assert_array_equal(
            np.flatnonzero(left_mask), indices[0]
        )
        self.assertGreater(right_weight, 0.99)
        self.assertGreater(left_weight, 0.99)
        self.assertLess(right[15], 0.0)
        self.assertGreater(right[16], 0.0)
        self.assertAlmostEqual(
            float(np.sum(right[indices[1]])), 0.0, places=6
        )
        np.testing.assert_allclose(
            left[indices[0]], right[indices[1]], atol=1.0e-7
        )

    def test_guidance_adds_only_to_scheduled_sagittal_joints(self):
        guidance = load_pure_gait_guidance_module()
        policy = np.linspace(-0.5, 0.5, 18, dtype=np.float32)
        residual = np.zeros(18, dtype=np.float32)
        residual[[6, 7, 8]] = [-1.0, 1.0, 0.0]
        mask = np.zeros(18, dtype=bool)
        mask[[6, 7, 8]] = True
        assisted, scale = guidance.apply_swing_action_residual(
            policy, residual, mask, 0.40, 0.50
        )
        self.assertAlmostEqual(scale, 0.20)
        np.testing.assert_allclose(assisted[~mask], policy[~mask])
        np.testing.assert_allclose(
            assisted[mask],
            policy[mask] + 0.20 * residual[mask],
        )

    def test_guidance_requires_opposite_support_and_trailing_foot(self):
        guidance = load_pure_gait_guidance_module()
        self.assertTrue(
            guidance.support_guard_allows_residual(
                0, [0, 0], [False, False], [0, 0]
            )
        )
        self.assertFalse(
            guidance.support_guard_allows_residual(
                0, [0, -1], [False, False], [0, 0]
            )
        )
        self.assertFalse(
            guidance.support_guard_allows_residual(
                0, [0, 1], [False, True], [0, 1]
            )
        )
        self.assertFalse(
            guidance.support_guard_allows_residual(
                0, [1, 0], [False, False], [2, 1]
            )
        )

    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_pure_mujoco_eval_module()
        cls.inertia = load_module(
            "n2_urdf_inertia_test", "sim2sim/urdf_inertia.py"
        )

    def test_level_four_config_matches_deployable_policy(self):
        config = yaml.safe_load(
            (ROOT / "sim2sim/configs/n2_stairs_walk.yaml").read_text()
        )
        self.assertEqual(config["num_single_obs"], 82)
        self.assertEqual(config["frame_stack"], 5)
        self.assertEqual(config["num_obs"], 410)
        self.assertEqual(config["stairs"]["step_height"], 0.10)
        self.assertEqual(config["stairs"]["step_width"], 0.30)
        self.assertEqual(config["stairs"]["num_steps"], 6)
        self.assertEqual(config["cmd_init"], [0.18, 0.0, 0.0])
        self.assertEqual(config["integrator"], "implicitfast")
        self.assertEqual(config["mujoco_physics"]["joint_armature"], 0.0)
        self.assertEqual(config["mujoco_physics"]["joint_frictionloss"], 0.0)
        self.assertEqual(config["mujoco_physics"]["contact_dim"], 3)
        self.assertTrue(
            config["mujoco_physics"]["align_inertials_from_urdf"]
        )
        self.assertEqual(
            config["heading_stabilizer"]["yaw_observation_gain"], 1.0
        )
        self.assertEqual(config["heading_stabilizer"]["hip_yaw_kp"], 0.0)
        self.assertIn("numerical_failure", (
            ROOT / "sim2sim/eval_stairs_mujoco.py"
        ).read_text())

    def test_raw_isaac_checkpoint_actor_loads_without_isaacgym(self):
        widths = (410, 512, 256, 128, 18)
        state = {}
        for sequence_index, (input_dim, output_dim) in enumerate(
            zip(widths, widths[1:])
        ):
            module_index = 2 * sequence_index
            state["actor.{}.weight".format(module_index)] = torch.zeros(
                output_dim, input_dim
            )
            state["actor.{}.bias".format(module_index)] = torch.zeros(
                output_dim
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_123.pt"
            torch.save(
                {"model_state_dict": state, "iter": 123},
                path,
            )
            policy, metadata = self.evaluator.load_checkpoint_actor(
                str(path)
            )
        self.assertEqual(metadata["iteration"], 123)
        self.assertEqual(metadata["input_dim"], 410)
        self.assertEqual(metadata["output_dim"], 18)
        self.assertEqual(metadata["hidden_dims"], [512, 256, 128])
        self.assertEqual(tuple(policy(torch.zeros(1, 410)).shape), (1, 18))

    def test_raw_checkpoint_auto_selects_legacy_observation_layout(self):
        config = yaml.safe_load(
            (ROOT / "sim2sim/configs/n2_stairs_walk.yaml").read_text()
        )
        updated = self.evaluator.configure_observation_layout_for_policy(
            config, 375
        )
        self.assertEqual(updated["num_single_obs"], 75)
        self.assertEqual(updated["num_obs"], 375)
        self.assertFalse(updated["include_base_lin_vel"])
        self.assertNotIn("gait_phase", updated)
        self.assertNotIn("navigation_state", updated)

    def test_flat_diagnostic_moves_stairs_and_disables_initial_noise(self):
        config = yaml.safe_load(
            (ROOT / "sim2sim/configs/n2_stairs_walk.yaml").read_text()
        )
        args = types.SimpleNamespace(
            policy_path=None,
            step_height=None,
            duration=None,
            command_speed=None,
            stair_start_x=100.0,
            initial_joint_noise=0.0,
            initial_lateral_noise=0.0,
            phase_offset=None,
            physics_preset=None,
        )
        updated = self.evaluator._apply_cli_overrides(config, args)
        self.assertEqual(updated["stairs"]["start_x"], 100.0)
        self.assertAlmostEqual(updated["validation"]["success_x"], 102.0)
        self.assertEqual(updated["validation"]["initial_joint_noise"], 0.0)
        self.assertEqual(updated["validation"]["initial_lateral_noise"], 0.0)

    def test_mujoco_route_diagnostic_overrides_are_explicit(self):
        config = yaml.safe_load(
            (ROOT / "sim2sim/configs/n2_stairs_walk.yaml").read_text()
        )
        args = types.SimpleNamespace(
            policy_path=None,
            step_height=None,
            duration=None,
            command_speed=None,
            stair_start_x=None,
            initial_joint_noise=None,
            initial_lateral_noise=None,
            corridor_half_width=0.70,
            corridor_yaw_limit=0.80,
            path_violation_dwell_s=1.25,
            phase_offset=None,
            physics_preset=None,
        )
        updated = self.evaluator._apply_cli_overrides(config, args)
        validation = updated["validation"]
        self.assertEqual(validation["corridor_half_width"], 0.70)
        self.assertEqual(validation["corridor_yaw_limit"], 0.80)
        self.assertEqual(validation["path_violation_dwell_s"], 1.25)

    def test_aggregation_does_not_double_prefix_mean_metrics(self):
        config = {
            "stairs": {"start_x": 0.60, "step_height": 0.10},
            "cmd_init": [0.18, 0.0, 0.0],
            "mujoco_physics": {"preset": "test"},
        }
        result = {
            "success": 0.0,
            "completion": 0.0,
            "fall": 0.0,
            "path_failure": 1.0,
            "numerical_failure": 0.0,
            "mean_forward_speed_m_s": 0.17,
        }
        summary = self.evaluator.aggregate_results(
            [result], config, "policy.pt"
        )
        self.assertEqual(summary["mean_forward_speed_m_s"], 0.17)
        self.assertEqual(summary["inertial_source"], "mjcf")
        self.assertNotIn("mean_mean_forward_speed_m_s", summary)

    def test_physics_presets_isolate_joint_contact_and_self_collision(self):
        presets = self.evaluator.PHYSICS_PRESETS
        self.assertEqual(presets["legacy_mjcf"]["joint_armature"], 0.01)
        self.assertFalse(
            presets["legacy_mjcf"]["disable_self_collisions"]
        )
        self.assertFalse(
            presets["legacy_mjcf"]["align_inertials_from_urdf"]
        )
        self.assertTrue(
            presets["legacy_urdf_inertias"]["align_inertials_from_urdf"]
        )
        self.assertTrue(
            presets["legacy_no_self"]["disable_self_collisions"]
        )
        self.assertEqual(presets["hybrid"]["contact_priority"], 1)
        self.assertEqual(presets["isaac_aligned"]["joint_armature"], 0.0)
        self.assertEqual(
            self.evaluator.PHASE_SWEEP_OFFSETS,
            (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
        )
        self.assertIn(
            "yaw_obs_2_hip_030", self.evaluator.STABILIZATION_PRESETS
        )

    def test_heading_stabilizer_is_bounded_and_has_expected_sign(self):
        config = {
            "heading_stabilizer": {
                "hip_yaw_kp": 0.30,
                "hip_yaw_kd": 0.06,
                "max_hip_yaw_offset": 0.18,
            }
        }
        self.assertAlmostEqual(
            self.evaluator.heading_stabilizer_offset(0.2, 0.1, config),
            0.066,
        )
        self.assertEqual(
            self.evaluator.heading_stabilizer_offset(2.0, 1.0, config),
            0.18,
        )
        self.assertEqual(
            self.evaluator.heading_stabilizer_offset(-2.0, -1.0, config),
            -0.18,
        )

    def test_contact_phase_reset_schedules_the_opposite_next_swing(self):
        frequency = 0.30
        elapsed = 3.7
        left_offset = (
            self.evaluator.contact_synchronized_phase_offset(
                elapsed, frequency, 0
            )
        )
        right_offset = (
            self.evaluator.contact_synchronized_phase_offset(
                elapsed, frequency, 1
            )
        )
        self.assertAlmostEqual(
            (left_offset + elapsed * frequency) % 1.0,
            0.0,
        )
        self.assertAlmostEqual(
            (right_offset + elapsed * frequency) % 1.0,
            0.5,
        )
        with self.assertRaises(ValueError):
            self.evaluator.contact_synchronized_phase_offset(
                elapsed, frequency, -1
            )

    def test_mjcf_inertias_are_rebuilt_from_training_urdf(self):
        mjcf_root = ET.parse(
            ROOT / "resources/robots/N2/mjcf/n2_18dof.xml"
        ).getroot()
        urdf_path = ROOT / "resources/robots/N2/urdf/N2.urdf"
        updated = self.inertia.align_mjcf_inertials_from_urdf(
            mjcf_root, str(urdf_path)
        )
        self.assertEqual(len(updated), 19)

        ankle = mjcf_root.find(
            ".//body[@name='L_leg_ankle_link']/inertial"
        )
        ankle_values = np.fromstring(
            ankle.attrib["fullinertia"], sep=" "
        )
        np.testing.assert_allclose(
            ankle_values[:3], [0.01, 0.01, 0.01], rtol=0.0, atol=1e-12
        )
        self.assertNotIn("diaginertia", ankle.attrib)
        self.assertNotIn("quat", ankle.attrib)

        shoulder = mjcf_root.find(
            ".//body[@name='R_arm_shoulder_yaw_Link']/inertial"
        )
        shoulder_values = np.fromstring(
            shoulder.attrib["fullinertia"], sep=" "
        )
        np.testing.assert_allclose(
            shoulder_values[:3], [0.01, 0.01, 0.01], rtol=0.0, atol=1e-12
        )

        elbow = mjcf_root.find(
            ".//body[@name='L_arm_elbow_Link']/inertial"
        )
        self.assertAlmostEqual(float(elbow.attrib["mass"]), 0.421265)
        np.testing.assert_allclose(
            np.fromstring(elbow.attrib["pos"], sep=" "),
            [-0.003975, 0.000743355, -0.0898509],
            rtol=0.0,
            atol=1e-7,
        )

        urdf_root = ET.parse(urdf_path).getroot()
        urdf_mass = sum(
            float(link.find("inertial/mass").attrib["value"])
            for link in urdf_root.findall("link")
        )
        mjcf_mass = sum(
            float(body.find("inertial").attrib["mass"])
            for body in mjcf_root.findall("./worldbody//body")
        )
        self.assertAlmostEqual(mjcf_mass, urdf_mass, places=9)

    def test_contact_tracker_accepts_true_alternating_stairs(self):
        tracker = self.evaluator.GaitTracker(
            0.02,
            {"step_height": 0.10, "num_steps": 6},
            {"stable_contact_s": 0.04, "stable_release_s": 0.06},
        )

        def update(raw_tread, site_z):
            tracker.update(
                np.asarray(raw_tread),
                np.asarray(site_z),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=bool),
            )

        update([0, 0], [0.045, 0.045])
        update([0, 0], [0.045, 0.045])
        for tread, foot in (
            (1, 0),
            (2, 1),
            (3, 0),
            (4, 1),
            (5, 0),
            (6, 1),
        ):
            raw = tracker.stable_tread.copy()
            raw[foot] = -1
            site_z = tracker.accepted_tread * 0.10 + 0.145
            for _ in range(3):
                update(raw, site_z)
            raw[foot] = tread
            site_z[foot] = tread * 0.10 + 0.045
            for _ in range(2):
                update(raw, site_z)

        summary = tracker.summary()
        self.assertEqual(summary["alternating_tread_count"], 6)
        self.assertEqual(summary["alternating_tread_rate"], 1.0)
        self.assertEqual(summary["same_tread_join_rate"], 0.0)
        self.assertEqual(summary["left_tread_advances"], 3)
        self.assertEqual(summary["right_tread_advances"], 3)

    def test_headless_imports_do_not_open_x_server(self):
        tree = parse_tree("sim2sim/sim2sim.py")
        top_level_imports = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported = []
        for node in top_level_imports:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            else:
                imported.append(node.module or "")
        imported = "\n".join(imported)
        self.assertNotIn("mujoco_viewer", imported)
        self.assertNotIn("pynput", imported)


class StairConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_tree = parse_tree("humanoid/envs/n2/n2_stairs_config.py")

    def test_observation_dimensions(self):
        env = literal_assignments(nested_class(self.config_tree, "N2StairsCfg", "env"))
        terrain = literal_assignments(
            nested_class(self.config_tree, "N2StairsCfg", "terrain")
        )
        actor_heights = len(terrain["actor_measured_points_x"]) * len(
            terrain["actor_measured_points_y"]
        )
        critic_heights = len(terrain["measured_points_x"]) * len(
            terrain["measured_points_y"]
        )
        self.assertEqual(env["num_single_obs"], 63 + actor_heights)
        self.assertEqual(env["frame_stack"] * env["num_single_obs"], 375)
        # 63 proprio + base velocity/payload/randomization/contact + full scan.
        expected_critic = 63 + 3 + 1 + 1 + 1 + 18 + 18 + 18 + 2 + critic_heights
        self.assertEqual(env["num_privileged_obs"], expected_critic)
        self.assertEqual(env["num_actions"], 18)
        self.assertEqual(env["stall_timeout_s"], 6.0)
        self.assertEqual(env["progress_epsilon"], 0.04)
        self.assertEqual(env["flight_grace_s"], 0.5)
        self.assertEqual(env["max_double_flight_s"], 0.08)

    def test_original_n2_discrete_terrain_mix_is_audited(self):
        n2_tree = parse_tree("humanoid/envs/n2/n2_config.py")
        terrain = literal_assignments(
            nested_class(n2_tree, "N2_18DofCfg", "terrain")
        )
        self.assertTrue(terrain["curriculum"])
        cumulative = np.cumsum(terrain["terrain_proportions"])
        counts = [0] * (len(cumulative) + 1)
        for column in range(terrain["num_cols"]):
            choice = column / terrain["num_cols"] + 0.001
            category = next(
                (index for index, threshold in enumerate(cumulative) if choice < threshold),
                len(cumulative),
            )
            counts[category] += 1
        # rough, obstacles, random-uniform, up slope, down slope,
        # upstairs, downstairs, and the unallocated flat tail.
        self.assertEqual(counts, [1, 0, 1, 0, 1, 4, 2, 1])

    def test_config_imports_and_instantiates_without_isaacgym(self):
        def package(name):
            module = types.ModuleType(name)
            module.__path__ = []
            return module

        injected = {
            "humanoid.envs": package("humanoid.envs"),
            "humanoid.envs.base": package("humanoid.envs.base"),
            "humanoid.envs.n2": package("humanoid.envs.n2"),
        }
        previous = {name: sys.modules.get(name) for name in injected}
        sys.modules.update(injected)
        loaded_names = []
        try:
            modules = [
                ("humanoid.envs.base.base_config", "humanoid/envs/base/base_config.py"),
                (
                    "humanoid.envs.base.legged_robot_config",
                    "humanoid/envs/base/legged_robot_config.py",
                ),
                ("humanoid.envs.n2.n2_config", "humanoid/envs/n2/n2_config.py"),
                (
                    "humanoid.envs.n2.n2_stairs_config",
                    "humanoid/envs/n2/n2_stairs_config.py",
                ),
            ]
            for module_name, path in modules:
                load_module(module_name, path)
                loaded_names.append(module_name)
            stairs_module = sys.modules["humanoid.envs.n2.n2_stairs_config"]
            n2_module = sys.modules["humanoid.envs.n2.n2_config"]
            cfg = stairs_module.N2StairsCfg()
            robust_cfg = stairs_module.N2StairsRobustCfg()
            walk_cfg = stairs_module.N2StairsWalkCfg()
            faststair_cfg = stairs_module.N2FastStairCfg()
            train_cfg = stairs_module.N2StairsCfgPPO()
            walk_train_cfg = stairs_module.N2StairsWalkCfgPPO()
            faststair_train_cfg = stairs_module.N2FastStairCfgPPO()
            self.assertEqual(cfg.env.num_observations, 375)
            self.assertEqual(cfg.env.num_privileged_obs, 146)
            self.assertTrue(cfg.asset.use_foot_force_sensors)
            self.assertFalse(n2_module.N2_18DofCfg().asset.use_foot_force_sensors)
            self.assertEqual(robust_cfg.terrain.max_init_terrain_level, 4)
            self.assertEqual(walk_cfg.env.num_single_obs, 82)
            self.assertEqual(walk_cfg.env.num_observations, 410)
            self.assertEqual(walk_cfg.env.num_privileged_obs, 153)
            self.assertTrue(walk_cfg.env.include_gait_phase)
            self.assertTrue(walk_cfg.env.include_base_lin_vel)
            self.assertTrue(walk_cfg.env.include_navigation_state)
            self.assertTrue(walk_cfg.env.enforce_walk_gait)
            self.assertEqual(walk_cfg.env.gait_frequency_start, 1.25)
            self.assertEqual(walk_cfg.env.gait_frequency_start_gain, 0.75)
            self.assertEqual(
                walk_cfg.env.gait_frequency_start_reference_speed, 0.12
            )
            self.assertEqual(
                walk_cfg.env.gait_frequency_transition_steps, 800 * 24
            )
            legacy_frequency_018 = walk_cfg.env.gait_frequency_start + (
                walk_cfg.env.gait_frequency_start_gain
                * (
                    0.18
                    - walk_cfg.env.gait_frequency_start_reference_speed
                )
            )
            self.assertAlmostEqual(legacy_frequency_018, 1.295, places=6)
            target_frequency_018 = walk_cfg.env.gait_frequency + (
                walk_cfg.env.gait_frequency_gain
                * (0.18 - walk_cfg.env.gait_reference_speed)
            )
            self.assertAlmostEqual(target_frequency_018, 0.30, places=6)
            self.assertLessEqual(walk_cfg.env.corridor_half_width, 0.30)
            self.assertLessEqual(walk_cfg.env.corridor_yaw_limit, 0.40)
            self.assertLess(
                walk_cfg.env.success_max_lateral_deviation,
                walk_cfg.env.corridor_half_width,
            )
            self.assertLess(
                walk_cfg.env.success_max_yaw_deviation,
                walk_cfg.env.corridor_yaw_limit,
            )
            self.assertGreaterEqual(
                walk_cfg.env.success_min_phase_contact_match, 0.70
            )
            permanent_double_support_match = (
                0.5 + 0.5 * walk_cfg.env.double_support_ratio
            )
            self.assertGreater(
                walk_cfg.env.success_min_phase_contact_match,
                permanent_double_support_match,
            )
            self.assertLess(
                walk_cfg.env.success_min_phase_contact_match, 0.80
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_double_flight_fraction, 0.08
            )
            self.assertGreater(
                walk_cfg.env.completion_dwell_s,
                walk_cfg.env.top_dwell_s,
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_mean_speed_bias, 0.05
            )
            self.assertGreaterEqual(
                walk_cfg.env.success_min_alternating_tread_count, 4
            )
            self.assertGreaterEqual(
                walk_cfg.env.success_min_alternating_tread_rate, 0.75
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_same_tread_join_rate, 0.20
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_skipped_tread_rate, 0.20
            )
            self.assertLess(
                walk_cfg.env.curriculum_min_alternating_tread_rate,
                walk_cfg.env.success_min_alternating_tread_rate,
            )
            self.assertGreater(
                walk_cfg.env.curriculum_max_same_tread_join_rate,
                walk_cfg.env.success_max_same_tread_join_rate,
            )
            self.assertEqual(walk_cfg.terrain.curriculum_successes, 1)
            self.assertEqual(walk_cfg.terrain.curriculum_failures, 2)
            self.assertLess(
                walk_cfg.env.max_sagittal_foot_separation,
                walk_cfg.env.success_max_sagittal_foot_separation,
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_sagittal_foot_separation, 0.44
            )
            landing_knee = (
                walk_cfg.env.swing_knee_landing_target
                + 0.10 * walk_cfg.env.swing_knee_landing_height_gain
            )
            peak_knee = (
                walk_cfg.env.swing_knee_peak_target
                + 0.10 * walk_cfg.env.swing_knee_peak_height_gain
            )
            self.assertLess(landing_knee, peak_knee)
            self.assertLessEqual(
                peak_knee, walk_cfg.env.swing_knee_max_target
            )
            self.assertGreaterEqual(walk_cfg.env.double_support_ratio, 0.25)
            self.assertLessEqual(walk_cfg.env.double_support_ratio, 0.40)
            self.assertGreater(
                walk_cfg.env.stable_contact_max_horizontal_speed, 0.0
            )
            self.assertLessEqual(
                walk_cfg.env.stable_contact_max_horizontal_speed, 0.20
            )
            self.assertGreater(
                walk_cfg.env.stable_contact_min_vertical_ratio, 0.0
            )
            self.assertLessEqual(
                walk_cfg.env.stable_contact_min_vertical_ratio, 1.0
            )
            self.assertGreater(
                walk_cfg.env.stable_contact_confirmation_s, 0.0
            )
            self.assertGreaterEqual(
                walk_cfg.env.stable_contact_release_s,
                walk_cfg.env.stable_contact_confirmation_s,
            )
            self.assertGreater(
                walk_cfg.env.stable_landing_height_tolerance, 0.0
            )
            self.assertGreater(
                walk_cfg.env.stable_landing_height_tolerance_ratio, 0.0
            )
            self.assertLess(
                walk_cfg.env.stable_landing_height_tolerance_ratio, 0.5
            )
            self.assertGreaterEqual(
                walk_cfg.env.nominal_foot_surface_offset, 0.035
            )
            self.assertLessEqual(
                walk_cfg.env.nominal_foot_surface_offset, 0.055
            )
            self.assertGreater(
                walk_cfg.env.stable_landing_tread_margin, 0.0
            )
            self.assertLess(
                2.0 * walk_cfg.env.stable_landing_tread_margin,
                walk_cfg.terrain.step_width,
            )
            self.assertGreater(
                walk_cfg.env.swing_trajectory_arc_base, 0.0
            )
            self.assertGreater(
                walk_cfg.env.swing_trajectory_arc_height_gain, 0.0
            )
            for normalizer in (
                walk_cfg.env.swing_trajectory_x_normalizer,
                walk_cfg.env.swing_trajectory_y_normalizer,
                walk_cfg.env.swing_trajectory_z_normalizer,
            ):
                self.assertGreater(normalizer, 0.0)
            self.assertGreater(walk_cfg.env.swing_timeout_ratio, 1.0)
            self.assertGreaterEqual(walk_cfg.env.swing_timeout_margin_s, 0.0)
            self.assertGreaterEqual(walk_cfg.env.late_swing_progress, 0.5)
            self.assertLess(walk_cfg.env.late_swing_progress, 1.0)
            self.assertNotIn("knee", walk_cfg.asset.penalize_contacts_on)
            for contact_name in ("hip", "shoulder", "elbow", "hand"):
                self.assertIn(contact_name, walk_cfg.asset.penalize_contacts_on)
            self.assertGreater(walk_cfg.env.arm_swing_amplitude, 0.0)
            self.assertAlmostEqual(
                walk_cfg.env.sagittal_foot_phase_amplitude, 0.26
            )
            self.assertEqual(train_cfg.runner.experiment_name, "n2_stairs")
            self.assertEqual(
                walk_train_cfg.runner.experiment_name, "n2_stairs_walk"
            )
            self.assertTrue(faststair_cfg.env.enable_faststair_planner)
            self.assertTrue(
                faststair_cfg.env.include_faststair_planner_privileged
            )
            self.assertTrue(
                faststair_cfg.env.faststair_follow_physical_swing
            )
            self.assertEqual(faststair_cfg.env.num_single_obs, 115)
            self.assertEqual(faststair_cfg.env.num_observations, 575)
            self.assertEqual(faststair_cfg.env.num_privileged_obs, 217)
            self.assertEqual(
                len(faststair_cfg.terrain.actor_measured_points_x)
                * len(faststair_cfg.terrain.actor_measured_points_y),
                45,
            )
            self.assertEqual(
                len(faststair_cfg.terrain.measured_points_x)
                * len(faststair_cfg.terrain.measured_points_y),
                77,
            )
            self.assertFalse(faststair_cfg.terrain.curriculum)
            self.assertEqual(
                faststair_cfg.terrain.level_mix,
                [0, 0, 1, 1, 2, 3, 4, 4],
            )
            self.assertEqual(
                len(faststair_cfg.env.faststair_candidate_x_offsets)
                * len(faststair_cfg.env.faststair_candidate_y_offsets),
                35,
            )
            self.assertEqual(
                faststair_cfg.env.success_min_alternating_tread_count,
                0,
            )
            self.assertEqual(
                faststair_train_cfg.runner.experiment_name,
                "n2_faststair",
            )
            self.assertEqual(
                faststair_train_cfg.algorithm.schedule,
                "adaptive",
            )
            with (
                ROOT / "sim2sim" / "configs" / "n2_stairs_walk.yaml"
            ).open() as stream:
                walk_yaml = yaml.safe_load(stream)
            self.assertEqual(
                walk_yaml["gait_phase"]["frequency"],
                walk_cfg.env.gait_frequency,
            )
            self.assertEqual(
                walk_yaml["gait_phase"]["frequency_gain"],
                walk_cfg.env.gait_frequency_gain,
            )
            self.assertEqual(
                walk_yaml["gait_phase"]["reference_speed"],
                walk_cfg.env.gait_reference_speed,
            )
            self.assertEqual(
                walk_yaml["navigation_state"]["lateral_scale"],
                walk_cfg.env.lateral_position_obs_scale,
            )
            self.assertEqual(
                walk_yaml["navigation_state"]["yaw_scale"],
                walk_cfg.env.yaw_error_obs_scale,
            )
            self.assertFalse(train_cfg.runner.init_at_random_ep_len)

            def to_plain_dict(obj):
                if not hasattr(obj, "__dict__"):
                    return obj
                result = {}
                for key in dir(obj):
                    if key.startswith("_"):
                        continue
                    value = getattr(obj, key)
                    if isinstance(value, list):
                        result[key] = [to_plain_dict(item) for item in value]
                    else:
                        result[key] = to_plain_dict(value)
                return result

            encoded = json.dumps(
                {"environment": to_plain_dict(cfg), "training": to_plain_dict(train_cfg)}
            )
            decoded_actions = json.loads(encoded)["environment"]["env"][
                "num_actions"
            ]
            self.assertEqual(decoded_actions, 18)
        finally:
            for name in loaded_names:
                sys.modules.pop(name, None)
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_terrain_is_only_upstairs_and_starts_at_two_cm(self):
        terrain = literal_assignments(
            nested_class(self.config_tree, "N2StairsCfg", "terrain")
        )
        self.assertTrue(terrain["curriculum"])
        self.assertEqual(terrain["num_rows"], 5)
        self.assertEqual(terrain["max_init_terrain_level"], 0)
        self.assertEqual(terrain["step_heights"], [0.02, 0.04, 0.06, 0.08, 0.10])
        self.assertEqual(terrain["terrain_proportions"], [0, 0, 0, 0, 0, 1, 0])
        self.assertAlmostEqual(sum(terrain["terrain_proportions"]), 1.0)
        self.assertLess(
            terrain["slope_treshold"],
            terrain["step_heights"][0] / terrain["horizontal_scale"],
        )
        self.assertGreater(terrain["start_platform_length"] - terrain["spawn_x"], 0.5)

    def test_expected_policy_order_matches_established_mjcf_tree_order(self):
        asset = literal_assignments(
            nested_class(self.config_tree, "N2StairsCfg", "asset")
        )
        xml_root = ET.parse(
            ROOT / "resources" / "robots" / "N2" / "mjcf" / "n2_18dof.xml"
        ).getroot()
        mjcf_tree_order = [
            joint.attrib["name"]
            for joint in xml_root.find("worldbody").findall(".//joint")
        ]
        urdf_root = ET.parse(
            ROOT / "resources" / "robots" / "N2" / "urdf" / "N2.urdf"
        ).getroot()
        urdf_order = [
            joint.attrib["name"]
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]
        self.assertEqual(asset["expected_dof_order"], mjcf_tree_order)
        self.assertCountEqual(asset["expected_dof_order"], urdf_order)
        self.assertNotEqual(mjcf_tree_order, urdf_order)

    def test_every_nonzero_reward_scale_has_an_implementation(self):
        scales = literal_assignments(
            nested_class(self.config_tree, "N2StairsCfg", "rewards", "scales")
        )
        implemented = set()
        for relative_path in (
            "humanoid/envs/base/legged_robot.py",
            "humanoid/envs/n2/n2_env.py",
            "humanoid/envs/n2/n2_stairs_env.py",
        ):
            tree = parse_tree(relative_path)
            implemented.update(
                node.name[len("_reward_"):]
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("_reward_")
            )
        missing = sorted(
            name for name, value in scales.items()
            if value != 0 and name not in implemented
        )
        self.assertEqual(missing, [])

        walk_scales = literal_assignments(
            nested_class(
                self.config_tree,
                "N2StairsWalkCfg",
                "rewards",
                "scales",
            )
        )
        walk_env = literal_assignments(
            nested_class(self.config_tree, "N2StairsWalkCfg", "env")
        )
        walk_missing = sorted(
            name for name, value in walk_scales.items()
            if value != 0 and name not in implemented
        )
        self.assertEqual(walk_missing, [])
        faststair_scales = literal_assignments(
            nested_class(
                self.config_tree,
                "N2FastStairCfg",
                "rewards",
                "scales",
            )
        )
        faststair_missing = sorted(
            name for name, value in faststair_scales.items()
            if value != 0 and name not in implemented
        )
        self.assertEqual(faststair_missing, [])
        self.assertGreater(faststair_scales["faststair_foothold"], 0.0)
        self.assertLess(
            faststair_scales["faststair_foothold_error"], 0.0
        )
        self.assertEqual(
            faststair_scales["stairs_alternating_tread"], 0.0
        )
        self.assertEqual(faststair_scales["stairs_repeated_lead"], 0.0)
        self.assertEqual(faststair_scales["stairs_same_tread_join"], 0.0)
        for required_reward in (
            "stairs_overspeed",
            "stairs_command_speed_error",
            "stairs_phase_contact",
            "stairs_phase_contact_mismatch",
            "stairs_sagittal_foot_phase",
            "stairs_sagittal_foot_phase_error",
            "stairs_next_tread_target",
            "stairs_next_tread_target_error",
            "stairs_foothold_lateral",
            "stairs_foothold_lateral_error",
            "stairs_completion",
            "stairs_curriculum_completion",
            "stairs_heading_alignment",
            "stairs_leg_alignment",
            "stairs_feet_yaw",
            "stairs_alternating_tread",
            "stairs_repeated_lead",
            "stairs_same_tread_join",
            "stairs_skipped_tread",
            "stairs_overstride",
            "stairs_swing_knee_flexion",
            "stairs_swing_knee_deficit",
            "stairs_arm_swing",
            "stairs_swing_trajectory",
            "stairs_swing_trajectory_error",
            "stairs_swing_timeout",
            "stairs_same_tread_support",
            "stairs_lower_leg_collision",
            "stairs_foot_riser_collision",
            "stairs_forward_pitch",
            "stairs_base_behind_support",
            "stairs_foot_pitch",
        ):
            self.assertIn(required_reward, walk_scales)

        self.assertGreaterEqual(walk_scales["stairs_alternating_tread"], 10.0)
        self.assertLessEqual(walk_scales["stairs_repeated_lead"], -8.0)
        self.assertLessEqual(walk_scales["stairs_same_tread_join"], -10.0)
        self.assertLessEqual(walk_scales["stairs_skipped_tread"], -5.0)
        self.assertLessEqual(walk_scales["stairs_overstride"], -8.0)
        self.assertLess(walk_scales["stairs_swing_knee_deficit"], 0.0)
        self.assertGreater(walk_scales["stairs_sagittal_foot_phase"], 0.0)
        self.assertLess(walk_scales["stairs_sagittal_foot_phase_error"], 0.0)
        self.assertEqual(walk_scales["stairs_next_tread_target"], 0.0)
        self.assertEqual(walk_scales["stairs_next_tread_target_error"], 0.0)
        self.assertEqual(walk_scales["stairs_foothold_lateral"], 0.0)
        self.assertEqual(walk_scales["stairs_foothold_lateral_error"], 0.0)
        self.assertEqual(walk_scales["stairs_swing_clearance"], 0.0)
        self.assertGreater(walk_scales["stairs_swing_trajectory"], 0.0)
        self.assertLess(walk_scales["stairs_swing_trajectory_error"], 0.0)
        for penalty in (
            "stairs_swing_timeout",
            "stairs_same_tread_support",
            "stairs_lower_leg_collision",
            "stairs_foot_riser_collision",
            "stairs_base_behind_support",
            "stairs_foot_pitch",
        ):
            self.assertLess(walk_scales[penalty], 0.0)
        self.assertGreater(walk_scales["stairs_forward_pitch"], 0.0)
        self.assertGreater(walk_scales["stairs_completion"], 0.0)
        self.assertGreater(
            walk_scales["stairs_curriculum_completion"],
            walk_scales["stairs_completion"],
        )
        self.assertGreater(
            walk_scales["stairs_success"],
            walk_scales["stairs_curriculum_completion"],
        )
        self.assertLess(
            walk_env["next_tread_target_start_phase"],
            walk_env["next_tread_target_full_phase"],
        )

        # Sparse events must cancel the base framework's unconditional dt
        # scaling, while retaining bounded configured magnitudes.
        self.assertEqual(scales["stairs_vertical_progress"], 1.0)
        self.assertEqual(scales["stairs_foot_step_progress"], 1.0)
        self.assertEqual(scales["stairs_double_flight"], -4.0)
        self.assertEqual(scales["stairs_single_support"], 0.80)
        self.assertEqual(scales["stairs_success"], 25.0)
        self.assertEqual(scales["termination"], -10.0)
        stairs_source = (
            ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        ).read_text()
        self.assertGreaterEqual(stairs_source.count("/ self.dt"), 5)
        self.assertIn("self.progress_checkpoint", stairs_source)
        self.assertIn("self.foot_force_sensor_forces", stairs_source)
        self.assertIn("stable_tread_advance_mask", stairs_source)
        self.assertIn("self.last_stable_contacts", stairs_source)
        self.assertIn("self.double_flight_time", stairs_source)
        self.assertIn("stairs_first_step_rate", stairs_source)
        self.assertIn("classify_tread_transition", stairs_source)
        self.assertIn("self.last_advanced_foot", stairs_source)
        self.assertIn("natural_step_sequence", stairs_source)
        self.assertIn("gait_frequency_transition_step", stairs_source)
        self.assertIn("_scheduled_gait_frequency", stairs_source)
        self.assertIn("gait_frequency_transition_active", stairs_source)
        self.assertIn("Gait frequency transition: step", stairs_source)
        self.assertIn(
            "-amplitude * torch.cos(2.0 * torch.pi * phase)",
            stairs_source,
        )
        self.assertIn("self.stair_start_x[levels, types]", stairs_source)
        self.assertIn("self.last_advanced_tread + 1", stairs_source)
        self.assertIn("smooth_swing_trajectory", stairs_source)
        self.assertIn("self.swing_start_pos", stairs_source)
        self.assertIn("self.swing_elapsed_time", stairs_source)
        self.assertIn("def _next_tread_swing_state", stairs_source)
        self.assertIn("actually_airborne", stairs_source)
        self.assertIn("opposite_supported", stairs_source)
        self.assertIn("def _swing_trajectory_state", stairs_source)
        self.assertIn("def _lower_leg_collision_per_foot", stairs_source)
        self.assertIn("def _foot_riser_collision_per_foot", stairs_source)
        self.assertIn("self.lower_leg_indices", stairs_source)
        self.assertIn('"L_leg_knee_link"', stairs_source)
        self.assertIn('"R_leg_knee_link"', stairs_source)
        self.assertIn("self.stable_contacts[env_ids] = False", stairs_source)
        self.assertIn(
            "self.stable_contact_candidate_time[env_ids] = 0.0",
            stairs_source,
        )
        self.assertIn(
            "self.stable_contact_loss_time[env_ids] = 0.0",
            stairs_source,
        )
        self.assertIn("self.accepted_foot_tread[env_ids] = 0", stairs_source)
        self.assertIn(
            "self.swing_opposite_support_valid[env_ids] = False",
            stairs_source,
        )
        self.assertIn(
            "self.swing_true_airborne_seen[env_ids] = False",
            stairs_source,
        )
        self.assertIn("self.swing_start_valid[env_ids] = False", stairs_source)
        self.assertIn("self.swing_elapsed_time[env_ids] = 0.0", stairs_source)
        self.assertIn("self.swing_pending_time[env_ids] = 0.0", stairs_source)
        self.assertIn("true_airborne_started", stairs_source)
        self.assertIn(
            "(self.swing_pending_time - allowed)", stairs_source
        )
        for diagnostic in (
            "stairs_same_tread_support_fraction",
            "stairs_lower_leg_collision_fraction",
            "stairs_foot_riser_collision_fraction",
            "stairs_swing_timeout_fraction",
            "stairs_mean_base_behind_support",
            "stairs_mean_foot_pitch_error",
            "stairs_max_swing_duration",
            "stairs_left_tread_advances",
            "stairs_right_tread_advances",
        ):
            self.assertIn(diagnostic, stairs_source)
        self.assertIn("self.completion_buf.float() / self.dt", stairs_source)
        self.assertIn(
            "self.curriculum_completion_buf.float() / self.dt",
            stairs_source,
        )
        self.assertIn("success_max_mean_speed_bias", stairs_source)
        self.assertIn(
            "success = self.curriculum_completion_buf[env_ids] & valid",
            stairs_source,
        )
        self.assertIn('"stairs_completion_rate"', stairs_source)
        self.assertIn('"stairs_curriculum_pass_rate"', stairs_source)
        self.assertIn(
            "self.tread_advance_count + self.same_tread_join_count",
            stairs_source,
        )
        train_source = (
            ROOT / "humanoid" / "scripts" / "train.py"
        ).read_text()
        self.assertIn("--reset_optimizer", train_source)
        self.assertIn("--command_speed", train_source)
        self.assertIn("--learning_rate", train_source)
        self.assertIn("--fixed_learning_rate", train_source)
        self.assertIn("--action_noise_std", train_source)
        self.assertIn("load_optimizer=not args.reset_optimizer", train_source)
        self.assertIn('param_group["lr"] = learning_rate', train_source)
        self.assertIn('ppo_runner.alg.schedule = "fixed"', train_source)
        self.assertIn('ppo_runner.alg_cfg["schedule"] = "fixed"', train_source)
        self.assertIn("policy.std.fill_(action_noise_std)", train_source)
        self.assertIn("refusing to silently train from zero", train_source)
        registry_source = (
            ROOT / "humanoid" / "utils" / "task_registry.py"
        ).read_text()
        self.assertIn("Checkpoint iteration mismatch", registry_source)
        self.assertIn("Verified checkpoint iteration", registry_source)

    def test_tasks_are_registered(self):
        registration = (ROOT / "humanoid" / "envs" / "__init__.py").read_text()
        for original_task in ('"n2"', '"n2_10dof"', '"n2_mimic"'):
            self.assertIn(original_task, registration)
        self.assertIn('"n2_stairs"', registration)
        self.assertIn('"n2_stairs_robust"', registration)
        self.assertIn('"n2_stairs_walk"', registration)
        self.assertIn('"n2_faststair"', registration)
        self.assertIn("N2StairsEnv", registration)

    def test_faststair_launcher_is_guarded_and_from_scratch(self):
        launcher = (
            ROOT / "humanoid" / "scripts" / "run_faststair_n2.sh"
        ).read_text()
        self.assertIn("--task=n2_faststair", launcher)
        self.assertIn("warm_start=False", launcher)
        self.assertIn("schedule=adaptive", launcher)
        self.assertNotIn("--fixed_learning_rate", launcher)
        self.assertIn("STAGE1_MIX=", launcher)
        self.assertIn("STAGE2_MIX=", launcher)
        self.assertIn("STAGE3_MIX=", launcher)
        self.assertIn("select_faststair_checkpoint.py", launcher)
        self.assertIn("holdout_candidate.csv", launcher)
        self.assertIn("holdout_baseline.csv", launcher)
        self.assertIn("FASTSTAIR_HOLDOUT_APPROVED=True", launcher)
        self.assertIn("FASTSTAIR_HOLDOUT_APPROVED=False", launcher)
        self.assertIn("model_screen_best.pt", launcher)
        self.assertIn("selected_checkpoint.txt", launcher)
        self.assertIn("stability_selected_s*/model_best.pt", launcher)

    def test_play_has_no_one_meter_per_second_override(self):
        play_source = (ROOT / "humanoid" / "scripts" / "play.py").read_text()
        self.assertNotIn("env.commands[:,0] = 1.0", play_source)
        self.assertIn("--command_speed", play_source)
        self.assertIn("camera_offset = env.env_origins[0]", play_source)
        eval_source = (
            ROOT / "humanoid" / "scripts" / "eval_stairs.py"
        ).read_text()
        self.assertIn("env_cfg.env.test = False", eval_source)
        self.assertIn('"n2_stairs_walk"', eval_source)
        for metric in (
            "completion_rate",
            "curriculum_completion_rate",
            "mean_forward_speed_m_s",
            "mean_command_error_m_s",
            "mean_phase_contact_match",
            "mean_sagittal_foot_phase_match",
            "mean_double_flight_fraction",
            "mean_max_lateral_deviation_m",
            "mean_max_yaw_deviation_rad",
            "path_failure_rate",
            "mean_alternating_tread_count",
            "mean_alternating_tread_rate",
            "mean_repeated_lead_rate",
            "mean_same_tread_join_rate",
            "mean_skipped_tread_rate",
            "mean_max_sagittal_foot_separation_m",
            "mean_swing_knee_flexion_rad",
            "mean_arm_swing_match",
            "mean_gait_frequency_hz",
            "mean_same_tread_support_fraction",
            "mean_lower_leg_collision_fraction",
            "mean_foot_riser_collision_fraction",
            "mean_swing_timeout_fraction",
            "mean_base_behind_support",
            "mean_foot_pitch_error",
            "mean_max_swing_duration",
            "mean_left_tread_advances",
            "mean_right_tread_advances",
        ):
            self.assertIn(metric, eval_source)


class Sim2SimConsistencyTests(unittest.TestCase):
    def test_stairs_yaml_parses_and_matches_policy_shape(self):
        path = ROOT / "sim2sim" / "configs" / "n2_stairs.yaml"
        with path.open() as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["num_actions"], 18)
        self.assertEqual(config["num_single_obs"], 75)
        self.assertEqual(config["num_obs"], 75 * config["frame_stack"])
        self.assertEqual(config["clip_observations"], 18.0)
        self.assertEqual(config["clip_actions"], 18.0)
        self.assertEqual(
            len(config["height_measurements"]["points_x"])
            * len(config["height_measurements"]["points_y"]),
            12,
        )
        self.assertAlmostEqual(
            config["simulation_dt"] * config["control_decimation"], 0.02
        )

    def test_phase_walk_yaml_matches_strict_actor_layout(self):
        path = ROOT / "sim2sim" / "configs" / "n2_stairs_walk.yaml"
        with path.open() as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["num_actions"], 18)
        self.assertEqual(config["num_single_obs"], 82)
        self.assertEqual(config["num_obs"], 410)
        self.assertEqual(config["num_obs"], 82 * config["frame_stack"])
        self.assertIn("logs/n2_stairs_walk/", config["policy_path"])
        self.assertTrue(config["include_base_lin_vel"])
        self.assertEqual(config["gait_phase"]["frequency"], 0.20)
        self.assertAlmostEqual(
            config["gait_phase"]["frequency_gain"], 1.6666667
        )
        self.assertEqual(config["navigation_state"]["lateral_scale"], 2.0)
        self.assertEqual(config["navigation_state"]["yaw_scale"], 1.0)
        self.assertEqual(
            len(config["height_measurements"]["points_x"])
            * len(config["height_measurements"]["points_y"]),
            12,
        )

    def test_mujoco_mapping_covers_isaac_policy_joint_order(self):
        with (ROOT / "sim2sim" / "configs" / "n2_stairs.yaml").open() as stream:
            config = yaml.safe_load(stream)
        with (
            ROOT / "sim2sim" / "configs" / "n2_stairs_walk.yaml"
        ).open() as stream:
            walk_config = yaml.safe_load(stream)
        for key in (
            "joint_order",
            "torque_limits",
            "kps",
            "kds",
            "default_angles",
        ):
            self.assertEqual(walk_config[key], config[key])
        urdf_root = ET.parse(
            ROOT / "resources" / "robots" / "N2" / "urdf" / "N2.urdf"
        ).getroot()
        urdf_order = [
            joint.attrib["name"]
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]
        urdf_efforts = {
            joint.attrib["name"]: float(joint.find("limit").attrib["effort"])
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        }
        xml_root = ET.parse(
            ROOT / "resources" / "robots" / "N2" / "mjcf" / "n2_18dof.xml"
        ).getroot()
        mjcf_tree_order = [
            joint.attrib["name"]
            for joint in xml_root.find("worldbody").findall(".//joint")
        ]
        actuator_order = [
            actuator.attrib["joint"]
            for actuator in xml_root.findall("./actuator/*")
        ]
        self.assertEqual(config["joint_order"], mjcf_tree_order)
        self.assertCountEqual(actuator_order, config["joint_order"])
        self.assertEqual(
            config["torque_limits"],
            [urdf_efforts[name] for name in config["joint_order"]],
        )
        for key in ("kps", "kds", "default_angles"):
            self.assertEqual(len(config[key]), len(actuator_order))

        n2_tree = parse_tree("humanoid/envs/n2/n2_config.py")
        defaults = literal_assignments(
            nested_class(n2_tree, "N2_18DofCfg", "init_state")
        )["default_joint_angles"]
        control = literal_assignments(
            nested_class(n2_tree, "N2_18DofCfg", "control")
        )
        self.assertEqual(
            config["default_angles"],
            [defaults[name] for name in config["joint_order"]],
        )
        for yaml_key, config_key in (("kps", "stiffness"), ("kds", "damping")):
            expected = []
            for joint_name in config["joint_order"]:
                matching = [
                    value
                    for pattern, value in control[config_key].items()
                    if pattern in joint_name
                ]
                self.assertEqual(len(matching), 1)
                expected.append(matching[0])
            self.assertEqual(config[yaml_key], expected)

    def test_legacy_sim2sim_configs_use_named_actuator_mappings(self):
        cases = (
            ("n2_18dof.yaml", "n2_18dof.xml"),
            ("n2_10dof.yaml", "N2_10dof.xml"),
        )
        for yaml_name, xml_name in cases:
            with (ROOT / "sim2sim" / "configs" / yaml_name).open() as stream:
                config = yaml.safe_load(stream)
            xml_root = ET.parse(
                ROOT / "resources" / "robots" / "N2" / "mjcf" / xml_name
            ).getroot()
            actuator_joints = [
                actuator.attrib["joint"]
                for actuator in xml_root.findall("./actuator/*")
            ]
            self.assertCountEqual(config["joint_order"], actuator_joints)
            self.assertEqual(len(config["joint_order"]), config["num_actions"])
            for vector_name in ("kps", "kds", "default_angles", "torque_limits"):
                self.assertEqual(len(config[vector_name]), config["num_actions"])
            self.assertEqual(
                config["num_single_obs"] * config["frame_stack"],
                config["num_obs"],
            )


class SourceCompatibilityTests(unittest.TestCase):
    def test_modified_python_sources_compile_without_importing_isaacgym(self):
        paths = [
            "humanoid/envs/base/base_task.py",
            "humanoid/envs/n2/n2_stairs_config.py",
            "humanoid/envs/n2/n2_stairs_env.py",
            "humanoid/algo/ppo/ppo.py",
            "humanoid/scripts/eval_stairs.py",
            "humanoid/scripts/play.py",
            "humanoid/scripts/stream_stairs.py",
            "humanoid/scripts/train.py",
            "humanoid/utils/stairs_terrain.py",
            "sim2sim/eval_stairs_mujoco.py",
            "sim2sim/compare_isaac_checkpoints_mujoco.py",
            "sim2sim/sim2sim.py",
        ]
        for relative_path in paths:
            path = ROOT / relative_path
            compile(path.read_text(), str(path), "exec")

    def test_browser_stream_uses_headless_camera_sensor(self):
        base_source = (
            ROOT / "humanoid" / "envs" / "base" / "base_task.py"
        ).read_text()
        stream_source = (
            ROOT / "humanoid" / "scripts" / "stream_stairs.py"
        ).read_text()
        self.assertIn('getattr(cfg.env, "enable_camera_sensors", False)', base_source)
        self.assertIn("if self.headless and not self.enable_camera_sensors", base_source)
        self.assertIn("create_camera_sensor", base_source)
        self.assertIn("env_cfg.env.enable_camera_sensors = True", stream_source)
        self.assertIn("args.headless = True", stream_source)
        self.assertIn("args.num_envs = 1", stream_source)
        self.assertIn("render_all_camera_sensors", stream_source)
        self.assertIn("get_camera_image", stream_source)
        self.assertIn("camera_offset = env.env_origins[0]", stream_source)
        self.assertIn('ThreadingHTTPServer(("127.0.0.1", args.stream_port)', stream_source)

    def test_ppo_storage_and_checkpoint_iteration_fixes_are_present(self):
        ppo_source = (ROOT / "humanoid" / "algo" / "ppo" / "ppo.py").read_text()
        helpers_source = (ROOT / "humanoid" / "utils" / "helpers.py").read_text()
        runner_source = (
            ROOT / "humanoid" / "algo" / "ppo" / "on_policy_runner.py"
        ).read_text()
        self.assertIn("device=self.device", ppo_source)
        self.assertIn("env_cfg.seed = args.seed", helpers_source)
        self.assertIn("self.current_learning_iteration = it + 1", runner_source)
        self.assertIn("Perf/collection_fps", runner_source)
        self.assertIn('saved_dict["env_state"]', runner_source)
        stairs_env_source = (
            ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        ).read_text()
        self.assertIn("def get_checkpoint_state", stairs_env_source)
        self.assertIn("def load_checkpoint_state", stairs_env_source)
        self.assertIn('state.get("task_name")', stairs_env_source)
        registry_source = (
            ROOT / "humanoid" / "utils" / "task_registry.py"
        ).read_text()
        self.assertIn("env_cfg.env.task_name = name", registry_source)
        base_source = (
            ROOT / "humanoid" / "envs" / "base" / "legged_robot.py"
        ).read_text()
        self.assertIn("create_asset_force_sensor", base_source)
        self.assertIn("enable_forward_dynamics_forces = False", base_source)
        sim2sim_source = (ROOT / "sim2sim" / "sim2sim.py").read_text()
        self.assertIn("seen_ground_plane", sim2sim_source)
        self.assertIn("def gait_phase_observations", sim2sim_source)
        self.assertIn('config.get("navigation_state")', sim2sim_source)
        self.assertIn('config.get("include_base_lin_vel", False)', sim2sim_source)

    def test_guarded_isaac_polish_contract_is_explicit(self):
        train_source = (
            ROOT / "humanoid" / "scripts" / "train.py"
        ).read_text()
        ppo_source = (
            ROOT / "humanoid" / "algo" / "ppo" / "ppo.py"
        ).read_text()
        stairs_source = (
            ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        ).read_text()
        launcher = (
            ROOT / "humanoid" / "scripts"
            / "run_isaac_stairs_polish.sh"
        ).read_text()

        for option in (
            "--actor_reference_loss_coeff",
            "--actor_head_only",
            "--freeze_action_noise",
            "--symmetry_loss_coeff",
        ):
            self.assertIn(option, train_source)
            self.assertIn(option, launcher)
        self.assertIn("def set_actor_reference", ppo_source)
        self.assertIn('"actor_reference": mean_actor_reference_loss', ppo_source)
        self.assertIn("def set_symmetry_config", ppo_source)
        self.assertIn("def mirror_observations", stairs_source)
        self.assertIn("def mirror_actions", stairs_source)
        self.assertIn("--fixed_terrain_level=4", launcher)
        self.assertIn("--command_speed=0.18", launcher)
        self.assertIn('PILOT_ITERATIONS="${N2_PILOT_ITERATIONS:-50}"', launcher)
        self.assertIn('LEARNING_RATE="${N2_LEARNING_RATE:-5e-6}"', launcher)
        self.assertIn(
            "baseline|smoke|pilot|status|log|stop|candidate|compare|view",
            launcher,
        )
        self.assertIn("resolve_view_checkpoint", launcher)
        self.assertIn("stream_checkpoint", launcher)
        self.assertIn("humanoid/scripts/stream_stairs.py", launcher)
        self.assertIn("--terrain_level=4", launcher)
        self.assertIn("--stream_port=${STREAM_PORT}", launcher)

    def test_isaac_stability_training_is_continuous_and_guarded(self):
        launcher = (
            ROOT / "humanoid" / "scripts"
            / "run_isaac_stability_curriculum.sh"
        ).read_text()
        train_source = (
            ROOT / "humanoid" / "scripts" / "train.py"
        ).read_text()
        eval_source = (
            ROOT / "humanoid" / "scripts" / "eval_stairs.py"
        ).read_text()
        stairs_source = (
            ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        ).read_text()

        self.assertIn(
            'TERRAIN_MIX="${N2_STABILITY_TERRAIN_MIX:-0,1,2,3,4,4,4,4}"',
            launcher,
        )
        self.assertIn(
            'TRAIN_ITERATIONS="${N2_STABILITY_TRAIN_ITERATIONS:-600}"',
            launcher,
        )
        self.assertIn(
            'STAGE_ITERATIONS="${N2_STABILITY_STAGE_ITERATIONS:-200}"',
            launcher,
        )
        self.assertIn(
            'CHECKPOINT_INTERVAL="${N2_STABILITY_CHECKPOINT_INTERVAL:-25}"',
            launcher,
        )
        self.assertIn("--terrain_level_mix=${TERRAIN_MIX}", launcher)
        self.assertIn(
            'ACTOR_LAYERS="${N2_STABILITY_ACTOR_LAYERS:-4}"',
            launcher,
        )
        self.assertIn("--actor_trainable_layers=${ACTOR_LAYERS}", launcher)
        self.assertIn(
            'RESIDUAL_POLICY="${N2_STABILITY_RESIDUAL_POLICY:-True}"',
            launcher,
        )
        self.assertIn("--residual_policy", launcher)
        self.assertIn("--residual_hidden_dims=${RESIDUAL_HIDDEN_DIMS}", launcher)
        self.assertIn("--residual_action_scales=${RESIDUAL_ACTION_SCALES}", launcher)
        self.assertIn("--residual_l2_coeff=${RESIDUAL_L2_COEFF}", launcher)
        self.assertIn("stairs_alternating_tread=0", launcher)
        self.assertIn("stairs_repeated_lead=0", launcher)
        self.assertIn("stairs_same_tread_join=0", launcher)
        self.assertIn(
            "--reward_scale_overrides=${REWARD_OVERRIDES}", launcher
        )
        self.assertIn("--save_interval=${checkpoint_interval}", launcher)
        self.assertIn("isaac_stability_tournament.py", launcher)
        self.assertIn("ISAAC_STABILITY_CONTINUOUS_TRAIN", launcher)
        self.assertIn("ISAAC_STABILITY_STAGE_START", launcher)
        self.assertIn("ISAAC_STABILITY_STAGE_STOP", launcher)
        self.assertIn("ISAAC_STABILITY_SCREEN", launcher)
        self.assertIn("ISAAC_STABILITY_HOLDOUT", launcher)
        self.assertIn("ISAAC_STABILITY_BEST_UPDATE", launcher)
        self.assertNotIn("ISAAC_STABILITY_MICRO_TRAIN", launcher)
        self.assertNotIn("local profiles=", launcher)
        self.assertNotIn("for level in 0 1 2 3 4", launcher)
        self.assertNotIn("isaac_stairs_stage_gate.py", launcher)
        self.assertIn("N2_ISAAC_STABILITY_CHECKPOINT=", launcher)
        self.assertIn("N2_ISAAC_STABILITY_IMPROVED=", launcher)
        self.assertIn("N2_ISAAC_STABILITY_APPROVED=", launcher)
        self.assertIn("stability_selected_s*/model_best.pt", launcher)
        self.assertIn("newest guarded stability-selected model", launcher)
        self.assertIn("N2_STABILITY_INIT_CHECKPOINT", launcher)
        self.assertIn("N2_STABILITY_SOURCE_APPROVED=True", launcher)
        self.assertIn("N2_STABILITY_TRAIN_ITERATIONS=100", launcher)
        self.assertIn("require_approved_selection", launcher)
        self.assertIn("run_reflection_preflight", launcher)
        self.assertIn("ISAAC_STABILITY_CORRECTION_ABORT", launcher)
        self.assertIn("N2_STABILITY_ACTOR_LAYERS=4", launcher)
        self.assertIn("N2_STABILITY_POLICY_LOSS_SCALE=0.0", launcher)
        self.assertIn("N2_STABILITY_SYMMETRIZE_REFERENCE=True", launcher)
        self.assertIn("--symmetrize_actor_reference", launcher)
        self.assertIn("stairs_left_drift=-4", launcher)
        self.assertIn("stairs_lateral_excursion=-3", launcher)
        self.assertIn("stairs_terminal_lateral=-2", launcher)
        self.assertIn("stairs_foot_crossover=-6", launcher)
        self.assertIn("stairs_foot_lane_error=-4", launcher)
        self.assertIn("stairs_single_support_stability=-2", launcher)
        self.assertIn("stairs_right_support_stability=-3", launcher)
        self.assertIn("stairs_right_stride_excess=-2", launcher)
        self.assertIn(
            "stairs_right_stride_excess_continuous=-3", launcher
        )
        self.assertIn(
            'SELECTION_MODE="${N2_ISAAC_STABILITY_SELECTION_MODE:-targeted}"',
            launcher,
        )
        self.assertIn("N2_ISAAC_STABILITY_BEST=", launcher)
        self.assertIn(
            "bash humanoid/scripts/run_isaac_stairs_polish.sh view",
            launcher,
        )
        self.assertNotIn(
            'INIT_CHECKPOINT="${N2_INIT_CHECKPOINT:-}"', launcher
        )
        for option in (
            "--actor_trainable_layers",
            "--residual_policy",
            "--residual_hidden_dims",
            "--residual_action_scales",
            "--residual_l2_coeff",
            "--reward_scale_overrides",
            "--observation_noise_level",
            "--terrain_level_mix",
            "--save_interval",
        ):
            self.assertIn(option, train_source)
        for metric in (
            "mean_action_rate_rms",
            "mean_action_accel_rms",
            "mean_final_lateral_position_m",
            "mean_left_swing_length_m",
            "mean_right_swing_length_m",
            "mean_left_foot_inward_error_m",
            "mean_right_foot_inward_error_m",
            "mean_left_swing_action_accel_rms",
            "mean_right_swing_action_accel_rms",
            "mean_left_swing_roll_rate_rms",
            "mean_right_swing_roll_rate_rms",
            "mean_actor_symmetry_error_rms",
        ):
            self.assertIn(metric, eval_source)
        self.assertIn(
            "def _reward_stairs_stride_symmetry", stairs_source
        )
        self.assertIn(
            "def _reward_stairs_right_stride_excess", stairs_source
        )
        self.assertIn(
            "def _reward_stairs_right_stride_excess_continuous",
            stairs_source,
        )
        self.assertIn("def _reward_stairs_left_drift", stairs_source)
        self.assertIn("def _reward_stairs_lateral_excursion", stairs_source)
        self.assertIn("def _reward_stairs_terminal_lateral", stairs_source)
        self.assertIn("self.swing_displacement_event", stairs_source)
        self.assertIn("stance_knee_velocity", stairs_source)
        self.assertIn("stance_hip_roll_velocity", stairs_source)
        self.assertIn("level_mix", stairs_source)

        train_tree = ast.parse(train_source)
        mix_parser = next(
            node for node in train_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "parse_terrain_level_mix"
        )
        parser_module = ast.fix_missing_locations(
            ast.Module(body=[mix_parser], type_ignores=[])
        )
        namespace = {}
        exec(compile(parser_module, "train.py", "exec"), namespace)
        self.assertEqual(
            namespace["parse_terrain_level_mix"]("0,1,2,3,4,4,4,4"),
            [0, 1, 2, 3, 4, 4, 4, 4],
        )

    def test_stride_correction_is_touchdown_aligned_and_bounded(self):
        source_path = ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        environment = nested_class(tree, "N2StairsEnv")
        wanted = {
            "_swing_progress_state",
            "_reward_stairs_stride_symmetry",
            "_reward_stairs_right_stride_excess",
            "_reward_stairs_right_stride_excess_continuous",
        }
        methods = [
            node for node in environment.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        harness = ast.ClassDef(
            name="StrideHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[harness], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, str(source_path), "exec"), namespace)

        instance = namespace["StrideHarness"]()
        instance.cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(
                stride_symmetry_deadband=0.015,
                right_stride_excess_deadband=0.015,
                right_stride_reference_floor=0.16,
                right_stride_excess_start_phase=0.35,
            ),
            terrain=types.SimpleNamespace(step_width=0.30),
        )
        instance.dt = 0.02
        instance.swing_displacement_valid = torch.ones(
            2, 2, dtype=torch.bool
        )
        instance.swing_displacement_event = torch.tensor(
            [[False, True], [False, True]]
        )
        instance.last_swing_forward_displacement = torch.tensor(
            [[0.10, 0.22], [0.10, 0.112]]
        )
        instance.root_states = torch.zeros(2, 13)
        instance.root_states[:, 7] = 0.18
        instance.projected_gravity = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
        )
        instance.feet_pos = torch.zeros(2, 2, 3)
        instance.swing_start_pos = torch.zeros(2, 2, 3)
        instance.feet_pos[:, 1, 0] = torch.tensor([0.22, 0.17])
        instance.swing_active = torch.tensor(
            [[False, True], [False, True]]
        )
        instance.swing_start_valid = instance.swing_active.clone()
        instance.swing_elapsed_time = torch.ones(2, 2)
        instance._nominal_swing_duration = lambda: torch.ones(2)

        symmetric = instance._reward_stairs_stride_symmetry()
        right_excess = instance._reward_stairs_right_stride_excess()
        continuous = (
            instance._reward_stairs_right_stride_excess_continuous()
        )
        expected = ((0.12 - 0.015) / 0.30) ** 2 / 0.02
        expected_continuous = ((0.22 - 0.16 - 0.015) / 0.30) ** 2
        self.assertAlmostEqual(symmetric[0].item(), expected, places=5)
        self.assertAlmostEqual(right_excess[0].item(), expected, places=5)
        self.assertAlmostEqual(
            continuous[0].item(), expected_continuous, places=5
        )
        self.assertEqual(symmetric[1].item(), 0.0)
        self.assertEqual(right_excess[1].item(), 0.0)
        self.assertEqual(continuous[1].item(), 0.0)

        # A stale displacement must not keep penalising unrelated timesteps,
        # and the asymmetric correction applies only on right touchdown.
        instance.swing_displacement_event.zero_()
        self.assertTrue(
            torch.equal(
                instance._reward_stairs_stride_symmetry(),
                torch.zeros(2),
            )
        )
        instance.swing_displacement_event[0, 0] = True
        self.assertEqual(
            instance._reward_stairs_right_stride_excess()[0].item(), 0.0
        )
        instance.swing_active[0, 1] = False
        self.assertEqual(
            instance._reward_stairs_right_stride_excess_continuous()[0].item(),
            0.0,
        )

    def test_lateral_rewards_target_left_drift_and_episode_excursion(self):
        source_path = ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        environment = nested_class(tree, "N2StairsEnv")
        wanted = {
            "_reward_stairs_left_drift",
            "_reward_stairs_lateral_excursion",
            "_reward_stairs_terminal_lateral",
        }
        methods = [
            node for node in environment.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        harness = ast.ClassDef(
            name="LateralHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[harness], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, str(source_path), "exec"), namespace)

        instance = namespace["LateralHarness"]()
        instance.cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(
                left_drift_deadband=0.015,
                lateral_error_normalizer=0.10,
                lateral_excursion_normalizer=0.10,
                terminal_lateral_normalizer=0.10,
            )
        )
        instance.dt = 0.02
        instance.root_states = torch.zeros(2, 13)
        instance.root_states[:, 1] = torch.tensor([0.065, -0.065])
        instance.root_states[:, 7] = 0.18
        instance.env_origins = torch.zeros(2, 3)
        instance.projected_gravity = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
        )
        instance.lateral_excursion_delta = torch.tensor([0.01, 0.01])
        instance.left_lateral_excursion_delta = torch.tensor([0.01, 0.00])
        instance.reset_buf = torch.tensor([False, True])

        left_drift = instance._reward_stairs_left_drift()
        excursion = instance._reward_stairs_lateral_excursion()
        terminal = instance._reward_stairs_terminal_lateral()
        self.assertAlmostEqual(left_drift[0].item(), 0.25, places=5)
        self.assertEqual(left_drift[1].item(), 0.0)
        self.assertAlmostEqual(excursion[0].item(), 10.0, places=5)
        self.assertAlmostEqual(excursion[1].item(), 5.0, places=5)
        self.assertEqual(terminal[0].item(), 0.0)
        self.assertAlmostEqual(terminal[1].item(), 21.125, places=4)

    def test_right_support_stability_targets_the_stance_leg(self):
        source_path = ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        environment = nested_class(tree, "N2StairsEnv")
        wanted = {
            "_single_support_stability_state",
            "_reward_stairs_single_support_stability",
            "_reward_stairs_right_support_stability",
        }
        methods = [
            node for node in environment.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        harness = ast.ClassDef(
            name="SupportHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[harness], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, str(source_path), "exec"), namespace)

        instance = namespace["SupportHarness"]()
        instance.cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(
                single_support_roll_rate_scale=0.2,
                single_support_lateral_position_scale=0.5,
                single_support_lateral_position_normalizer=0.10,
                single_support_lateral_velocity_scale=0.5,
                single_support_vertical_velocity_scale=0.2,
                single_support_stance_knee_velocity_scale=0.04,
                single_support_stance_hip_roll_velocity_scale=0.08,
                single_support_action_rate_scale=0.08,
                single_support_action_accel_scale=0.04,
            )
        )
        instance.stable_contacts = torch.tensor(
            [[False, True], [True, False]]
        )
        instance.contacts = instance.stable_contacts.clone()
        instance.projected_gravity = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
        )
        instance.base_ang_vel = torch.zeros(2, 3)
        instance.base_lin_vel = torch.zeros(2, 3)
        instance.root_states = torch.zeros(2, 13)
        instance.root_states[:, 7] = 0.18
        instance.env_origins = torch.zeros(2, 3)
        instance.knee_dof_idxs = [7, 16]
        instance.hip_roll_dof_idxs = [5, 14]
        instance.support_leg_dof_idxs = torch.tensor(
            [[4, 5, 6, 7, 8], [13, 14, 15, 16, 17]]
        )
        instance.dof_vel = torch.zeros(2, 18)
        instance.dof_vel[0, 16] = 2.0
        instance.dof_vel[0, 14] = 1.0
        instance.actions = torch.zeros(2, 18)
        instance.last_actions = torch.zeros(2, 18)
        instance.last_last_actions = torch.zeros(2, 18)
        instance.actions[0, 13:18] = 0.5

        both = instance._reward_stairs_single_support_stability()
        right_only = instance._reward_stairs_right_support_stability()
        self.assertGreater(both[0].item(), 0.20)
        self.assertEqual(both[1].item(), 0.0)
        self.assertAlmostEqual(right_only[0].item(), both[0].item())
        self.assertEqual(right_only[1].item(), 0.0)

    def test_isaac_stability_tournament_handles_episode_resolution(self):
        tournament = load_isaac_stability_tournament_module()

        def row(**overrides):
            values = {
                "episodes": 64.0,
                "completion_rate": 0.984375,
                "fall_rate": 0.0,
                "path_failure_rate": 0.0,
                "mean_command_error_m_s": 0.04,
                "mean_action_rate_rms": 0.70,
                "mean_action_accel_rms": 0.50,
                "mean_left_swing_length_m": 0.10,
                "mean_right_swing_length_m": 0.19,
                "mean_final_lateral_position_m": 0.03,
                "mean_max_lateral_deviation_m": 0.09,
                "mean_max_yaw_deviation_rad": 0.20,
                "mean_double_flight_fraction": 0.04,
                "mean_left_foot_inward_error_m": 0.004,
                "mean_right_foot_inward_error_m": 0.012,
                "mean_left_foot_lateral_position_m": 0.09,
                "mean_right_foot_lateral_position_m": -0.09,
                "mean_right_swing_action_rate_rms": 0.68,
                "mean_right_swing_action_accel_rms": 0.48,
                "mean_left_swing_action_rate_rms": 0.74,
                "mean_left_swing_action_accel_rms": 0.55,
                "mean_right_swing_roll_rate_rms": 0.10,
                "mean_left_swing_roll_rate_rms": 0.16,
                "mean_right_swing_lateral_velocity_rms": 0.04,
                "mean_left_swing_lateral_velocity_rms": 0.07,
                "mean_actor_symmetry_error_rms": 0.05,
            }
            values.update(overrides)
            return values

        baseline = {level: row() for level in range(5)}
        candidate = {
            level: row(
                mean_action_rate_rms=0.64,
                mean_action_accel_rms=0.44,
                mean_right_swing_length_m=0.17,
                mean_final_lateral_position_m=0.02,
                mean_max_lateral_deviation_m=0.08,
                mean_right_foot_inward_error_m=0.008,
                mean_left_swing_action_rate_rms=0.68,
                mean_left_swing_action_accel_rms=0.48,
                mean_left_swing_roll_rate_rms=0.12,
                mean_left_swing_lateral_velocity_rms=0.05,
                mean_actor_symmetry_error_rms=0.035,
            )
            for level in range(5)
        }
        # Two episodes out of 64 is 3.125%, so a literal 3% threshold would
        # reject a candidate based only on evaluation quantization.
        candidate[4]["completion_rate"] = 0.953125
        accepted = tournament.compare(baseline, candidate, episodes=64)
        self.assertTrue(accepted["eligible"], accepted["reasons"])
        self.assertAlmostEqual(
            accepted["high_level_episode_tolerance"], 2.0 / 64.0
        )

        unsafe = {
            level: dict(values) for level, values in candidate.items()
        }
        unsafe[4]["completion_rate"] = 0.75
        rejected = tournament.compare(baseline, unsafe, episodes=64)
        self.assertFalse(rejected["eligible"])
        self.assertTrue(
            any(
                "10 cm completion regressed" in reason
                for reason in rejected["reasons"]
            )
        )

        inward_regression = {
            level: dict(values) for level, values in candidate.items()
        }
        for values in inward_regression.values():
            values["mean_right_foot_inward_error_m"] = 0.020
        rejected = tournament.compare(
            baseline, inward_regression, episodes=64
        )
        self.assertFalse(rejected["eligible"])
        self.assertTrue(
            any(
                "combined foot inward error increased" in reason
                for reason in rejected["reasons"]
            )
        )

        lane_shift = {
            level: dict(values) for level, values in candidate.items()
        }
        for values in lane_shift.values():
            values["mean_left_foot_lateral_position_m"] = 0.01
            values["mean_right_foot_lateral_position_m"] = -0.17
        rejected = tournament.compare(baseline, lane_shift, episodes=64)
        self.assertFalse(rejected["eligible"])
        self.assertTrue(
            any(
                "foot-lane center error increased" in reason
                for reason in rejected["reasons"]
            )
        )

        # A good aggregate score must not hide a regression in the exact
        # 10 cm defects seen in the recorded rollout.
        aggregate_trap = {
            level: dict(values) for level, values in candidate.items()
        }
        aggregate_trap[4].update(
            mean_max_lateral_deviation_m=0.094,
            mean_final_lateral_position_m=0.036,
            mean_right_swing_length_m=0.205,
            mean_left_swing_action_rate_rms=0.76,
            mean_left_swing_action_accel_rms=0.58,
            mean_left_swing_roll_rate_rms=0.17,
            mean_left_swing_lateral_velocity_rms=0.08,
        )
        rejected = tournament.compare(
            baseline, aggregate_trap, episodes=64
        )
        self.assertFalse(rejected["eligible"])
        self.assertFalse(rejected["hard_safe"])
        self.assertTrue(
            any(
                reason.startswith("10 cm")
                for reason in rejected["hard_reasons"]
            )
        )
        self.assertFalse(
            rejected["target_group_improvements"]["lateral"]
        )

        # Correcting only the feet's common lane offset must not count as a
        # lateral-body improvement when pelvis excursion and final translation
        # both get worse.
        lane_biased_baseline = {
            level: dict(values) for level, values in baseline.items()
        }
        lane_only_candidate = {
            level: dict(values) for level, values in candidate.items()
        }
        for values in lane_biased_baseline.values():
            values["mean_left_foot_lateral_position_m"] = 0.03
            values["mean_right_foot_lateral_position_m"] = -0.15
        for values in lane_only_candidate.values():
            values["mean_max_lateral_deviation_m"] = 0.095
            values["mean_final_lateral_position_m"] = 0.035
        lane_only = tournament.compare(
            lane_biased_baseline, lane_only_candidate, episodes=64
        )
        self.assertGreater(
            lane_only["target_deltas"]["foot_lane_center"], 0.0
        )
        self.assertLess(lane_only["target_deltas"]["max_lateral"], 0.0)
        self.assertLess(lane_only["target_deltas"]["signed_lateral"], 0.0)
        self.assertFalse(
            lane_only["target_group_improvements"]["lateral"]
        )

        # A tiny isolated style trade-off may use the balanced fallback only
        # when the hard completion/fall/path gates and net style gain pass.
        fallback_candidate = {
            level: dict(values) for level, values in candidate.items()
        }
        for values in fallback_candidate.values():
            values["mean_max_yaw_deviation_rad"] = 0.235
        with mock.patch.dict(
            "os.environ",
            {"N2_ISAAC_STABILITY_SELECTION_MODE": "balanced"},
        ):
            fallback = tournament.compare(
                baseline, fallback_candidate, episodes=64
            )
        self.assertFalse(fallback["eligible"])
        self.assertTrue(fallback["hard_safe"])
        self.assertTrue(fallback["fallback_eligible"])
        self.assertTrue(
            any(
                "weighted yaw deviation increased" in reason
                for reason in fallback["soft_reasons"]
            )
        )

    def test_isaac_stability_gate_accepts_improvement_and_blocks_regression(self):
        gate = load_isaac_stage_gate_module()

        def summary(**overrides):
            values = {
                "completion_rate": 0.94,
                "fall_rate": 0.04,
                "path_failure_rate": 0.02,
                "mean_max_lateral_deviation_m": 0.08,
                "mean_max_yaw_deviation_rad": 0.18,
                "mean_final_lateral_position_m": -0.03,
                "mean_left_swing_length_m": 0.29,
                "mean_right_swing_length_m": 0.25,
                "mean_action_rate_rms": 0.08,
                "mean_action_accel_rms": 0.05,
                "mean_double_flight_fraction": 0.03,
                "mean_command_error_m_s": 0.03,
            }
            values.update(overrides)
            return values

        baseline_stage = summary()
        baseline_high = summary(completion_rate=0.93, fall_rate=0.05)
        candidate_stage = summary(
            completion_rate=0.96,
            fall_rate=0.02,
            mean_max_lateral_deviation_m=0.06,
            mean_final_lateral_position_m=-0.01,
            mean_right_swing_length_m=0.28,
            mean_action_rate_rms=0.06,
            mean_action_accel_rms=0.04,
        )
        candidate_high = summary(
            completion_rate=0.94,
            fall_rate=0.04,
            mean_max_lateral_deviation_m=0.075,
            mean_right_swing_length_m=0.27,
            mean_action_rate_rms=0.07,
            mean_action_accel_rms=0.045,
        )
        accepted = gate.decide(
            baseline_stage,
            candidate_stage,
            baseline_high,
            candidate_high,
            2,
        )
        self.assertTrue(accepted["accepted"], accepted["reasons"])

        regressed_high = dict(candidate_high)
        regressed_high["completion_rate"] = 0.70
        rejected = gate.decide(
            baseline_stage,
            candidate_stage,
            baseline_high,
            regressed_high,
            2,
        )
        self.assertFalse(rejected["accepted"])
        self.assertTrue(
            any("10 cm completion regressed" in reason for reason in rejected["reasons"])
        )

    def test_isaac_actor_mirror_is_an_involution(self):
        source_path = ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        environment = nested_class(tree, "N2StairsEnv")
        method_names = {
            "_build_mirror_layout",
            "mirror_actions",
            "mirror_observations",
        }
        methods = [
            node for node in environment.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ]
        harness = ast.ClassDef(
            name="MirrorHarness",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[harness], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, str(source_path), "exec"), namespace)

        config_tree = parse_tree("humanoid/envs/n2/n2_stairs_config.py")
        joint_order = literal_assignments(
            nested_class(config_tree, "N2StairsCfg", "asset")
        )["expected_dof_order"]
        instance = namespace["MirrorHarness"]()
        instance.num_actions = 18
        instance.include_gait_phase = True
        instance.include_base_lin_vel = True
        instance.include_navigation_state = True
        instance.cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(
                num_single_obs=82,
                num_observations=410,
                frame_stack=5,
            ),
            terrain=types.SimpleNamespace(
                actor_measured_points_x=[0.25, 0.45, 0.65, 0.85],
                actor_measured_points_y=[-0.24, 0.0, 0.24],
            ),
        )
        instance._build_mirror_layout(joint_order)

        # These signs are physical, not merely an involution.  URDF FK maps
        # both elbow coordinates with the same sign while hip roll changes
        # sign under sagittal reflection.
        self.assertEqual(
            instance.mirror_action_sign[joint_order.index(
                "L_arm_elbow_joint"
            )].item(),
            1.0,
        )
        self.assertEqual(
            instance.mirror_action_sign[joint_order.index(
                "R_arm_elbow_joint"
            )].item(),
            1.0,
        )
        self.assertEqual(
            instance.mirror_action_sign[joint_order.index(
                "L_leg_hip_roll_joint"
            )].item(),
            -1.0,
        )

        actions = torch.randn(7, 18)
        observations = torch.randn(7, 410)
        self.assertTrue(
            torch.equal(
                instance.mirror_actions(instance.mirror_actions(actions)),
                actions,
            )
        )
        self.assertTrue(
            torch.equal(
                instance.mirror_observations(
                    instance.mirror_observations(observations)
                ),
                observations,
            )
        )

        instance.include_residual_targets = True
        instance.residual_target_obs_dim = 15
        instance.cfg.env.num_observations = 425
        instance._build_mirror_layout(joint_order)
        residual_observations = torch.randn(7, 425)
        self.assertTrue(
            torch.equal(
                instance.mirror_observations(
                    instance.mirror_observations(residual_observations)
                ),
                residual_observations,
            )
        )

    def test_right_foot_inward_metric_uses_verified_world_y_sign(self):
        source_path = ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        environment = nested_class(tree, "N2StairsEnv")
        method = next(
            node for node in environment.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_foot_crossover_state"
        )
        harness = ast.ClassDef(
            name="LateralHarness",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[harness], type_ignores=[])
        )
        namespace = {"torch": torch}
        exec(compile(module, str(source_path), "exec"), namespace)

        instance = namespace["LateralHarness"]()
        instance.cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(
                foothold_min_half_width=0.055,
                foothold_crossover_normalizer=0.055,
                first_tread_target_activation_distance=0.30,
            )
        )
        instance.feet_pos = torch.tensor(
            [[[0.0, 0.080, 0.0], [0.0, -0.020, 0.0]]]
        )
        instance.env_origins = torch.zeros(1, 3)
        instance.terrain_levels = torch.zeros(1, dtype=torch.long)
        instance.terrain_types = torch.zeros(1, dtype=torch.long)
        instance.stair_start_x = torch.tensor([[0.60]])
        instance.root_states = torch.zeros(1, 13)
        instance.root_states[:, 0] = 0.40
        instance.root_states[:, 7] = 0.18
        instance.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]])

        relative_y, inward, normalized, active = (
            instance._foot_crossover_state()
        )
        torch.testing.assert_close(
            relative_y, torch.tensor([[0.080, -0.020]])
        )
        torch.testing.assert_close(
            inward, torch.tensor([[0.0, 0.035]])
        )
        torch.testing.assert_close(
            normalized, torch.tensor([[0.0, 0.035 / 0.055]])
        )
        self.assertTrue(bool(active.item()))

    def test_actor_reference_is_an_independent_frozen_teacher(self):
        from humanoid.algo.ppo.actor_critic import ActorCritic
        from humanoid.algo.ppo.ppo import PPO

        policy = ActorCritic(
            4,
            3,
            2,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
        )
        algorithm = PPO(policy, device="cpu")
        algorithm.set_actor_reference(0.25)
        reference_before = {
            name: value.detach().clone()
            for name, value in algorithm.actor_reference.state_dict().items()
        }
        with torch.no_grad():
            policy.actor[-1].weight.add_(1.0)
        self.assertEqual(algorithm.actor_reference_loss_coeff, 0.25)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in algorithm.actor_reference.parameters()
            )
        )
        for name, value in algorithm.actor_reference.state_dict().items():
            self.assertTrue(torch.equal(value, reference_before[name]))

    def test_residual_policy_starts_exactly_at_the_base_and_is_bounded(self):
        from humanoid.algo.ppo.actor_critic import (
            ActorCritic,
            ResidualActorCritic,
        )

        base = ActorCritic(
            5,
            3,
            6,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            init_noise_std=0.1,
        )
        residual = ResidualActorCritic(
            8,
            3,
            6,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            residual_base_obs_dim=5,
            residual_hidden_dims=[7],
            residual_action_indices=[1, 4],
            residual_action_scales=[0.10, 0.20],
            init_noise_std=0.1,
        )
        residual.load_state_dict(base.state_dict())
        observations = torch.randn(11, 8)
        with torch.no_grad():
            expected = base.actor(observations[:, :5])
            actual = residual.act_inference(observations)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in residual.actor.parameters()
            )
        )
        untouched = [0, 2, 3, 5]
        residual.update_distribution(observations)
        self.assertTrue(torch.equal(
            residual.action_std[:, untouched],
            torch.full_like(
                residual.action_std[:, untouched],
                residual.residual_frozen_action_std.item(),
            ),
        ))
        self.assertTrue(torch.equal(
            residual.action_std[:, [1, 4]],
            torch.full_like(residual.action_std[:, [1, 4]], 0.1),
        ))

        with torch.no_grad():
            residual.residual_actor[-1].bias.fill_(100.0)
            correction = residual.residual_correction(observations)
        self.assertTrue(torch.equal(
            correction[:, untouched],
            torch.zeros_like(correction[:, untouched]),
        ))
        self.assertTrue(
            torch.all(correction[:, 1].abs() <= 0.10 + 1.0e-7)
        )
        self.assertTrue(
            torch.all(correction[:, 4].abs() <= 0.20 + 1.0e-7)
        )

    def test_residual_checkpoint_metadata_round_trip(self):
        import tempfile

        from humanoid.algo.ppo.actor_critic import ResidualActorCritic
        from humanoid.utils.residual_policy import (
            residual_metadata_from_checkpoint,
        )

        policy = ResidualActorCritic(
            8,
            3,
            6,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            residual_base_obs_dim=5,
            residual_hidden_dims=[7],
            residual_action_indices=[1, 4],
            residual_action_scales=[0.10, 0.20],
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "policy_metadata": policy.checkpoint_metadata(),
                    "iter": 1,
                },
                stream.name,
            )
            metadata = residual_metadata_from_checkpoint(stream.name)
        self.assertEqual(metadata["class_name"], "ResidualActorCritic")
        self.assertEqual(metadata["residual_base_obs_dim"], 5)
        self.assertEqual(metadata["residual_observation_dim"], 8)
        self.assertEqual(metadata["residual_hidden_dims"], [7])
        self.assertEqual(metadata["residual_action_indices"], [1, 4])

    def test_symmetrized_actor_reference_target_is_equivariant(self):
        from humanoid.algo.ppo.actor_critic import ActorCritic
        from humanoid.algo.ppo.ppo import PPO

        class MirrorEnvironment:
            @staticmethod
            def mirror_observations(observations):
                return observations[:, [1, 0, 3, 2]]

            @staticmethod
            def mirror_actions(actions):
                return actions[:, [1, 0]]

        policy = ActorCritic(
            4,
            3,
            2,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
        )
        algorithm = PPO(policy, device="cpu")
        mirror = MirrorEnvironment()
        algorithm.set_actor_reference(0.25, symmetry_env=mirror)

        observations = torch.randn(11, 4)
        target = algorithm._actor_reference_target(observations)
        mirrored_target = algorithm._actor_reference_target(
            mirror.mirror_observations(observations)
        )
        self.assertTrue(
            torch.allclose(
                mirrored_target,
                mirror.mirror_actions(target),
                atol=1.0e-7,
                rtol=1.0e-6,
            )
        )

    def test_inference_reflection_blend_is_exact_at_one_half(self):
        from humanoid.utils.policy_symmetry import (
            make_reflection_blended_policy,
        )

        class MirrorEnvironment:
            @staticmethod
            def mirror_observations(observations):
                return observations[:, [1, 0, 3, 2]]

            @staticmethod
            def mirror_actions(actions):
                return actions[:, [1, 0]]

        weight = torch.tensor(
            [[1.0, -0.5], [0.3, 1.2], [-0.7, 0.2], [0.9, -1.0]]
        )

        def asymmetric_policy(observations):
            return observations @ weight

        mirror = MirrorEnvironment()
        policy = make_reflection_blended_policy(
            asymmetric_policy, mirror, 0.5
        )
        observations = torch.randn(13, 4)
        torch.testing.assert_close(
            policy(mirror.mirror_observations(observations)),
            mirror.mirror_actions(policy(observations)),
        )

    def test_randomized_surface_properties_stay_two_dimensional(self):
        base_source = (
            ROOT / "humanoid" / "envs" / "base" / "legged_robot.py"
        ).read_text()
        self.assertGreaterEqual(
            base_source.count(
                "torch.randint(0, num_buckets, (self.num_envs,))"
            ),
            2,
        )
        self.assertGreaterEqual(
            base_source.count("torch.randint(0, num_buckets, (len(env_ids),))"),
            2,
        )
        stairs_source = (
            ROOT / "humanoid" / "envs" / "n2" / "n2_stairs_env.py"
        ).read_text()
        self.assertIn("def _reshape_critic_feature", stairs_source)
        self.assertIn('("friction", self.friction_coeffs, 1)', stairs_source)
        self.assertIn(
            '("foot_contacts", self.contacts, len(self.feet_indices))',
            stairs_source,
        )

    def test_latest_checkpoint_resolution_ignores_report_directories(self):
        fake_isaacgym = types.ModuleType("isaacgym")
        fake_isaacgym.gymapi = types.ModuleType("isaacgym.gymapi")
        fake_isaacgym.gymutil = types.ModuleType("isaacgym.gymutil")
        previous = sys.modules.get("isaacgym")
        sys.modules["isaacgym"] = fake_isaacgym
        try:
            helpers = load_module(
                "n2_helpers_checkpoint_test", "humanoid/utils/helpers.py"
            )
            with tempfile.TemporaryDirectory() as root:
                run = Path(root) / "run_a"
                reports = Path(root) / "evaluations"
                run.mkdir()
                reports.mkdir()
                (run / "model_9.pt").touch()
                (run / "model_100.pt").touch()
                (run / "model_best.pt").touch()
                (reports / "latest.json").touch()
                resolved = helpers.get_load_path(root, load_run=-1, checkpoint=-1)
                self.assertEqual(Path(resolved), run / "model_100.pt")
        finally:
            sys.modules.pop("n2_helpers_checkpoint_test", None)
            if previous is None:
                sys.modules.pop("isaacgym", None)
            else:
                sys.modules["isaacgym"] = previous

    def test_native_mujoco_training_contract_is_fixed_and_complete(self):
        path = ROOT / "sim2sim" / "configs" / "n2_stairs_walk.yaml"
        with path.open() as stream:
            config = yaml.safe_load(stream)
        training = config["mujoco_training"]
        curriculum = training["curriculum"]
        self.assertEqual(config["stairs"]["step_height"], 0.10)
        self.assertEqual(config["stairs"]["num_steps"], 6)
        self.assertEqual(config["num_obs"], 410)
        self.assertEqual(curriculum["target_steps"], list(range(7)))
        self.assertEqual(len(curriculum["target_x_m"]), 7)
        self.assertEqual(curriculum["target_steps"][-1], 6)
        self.assertEqual(
            curriculum["physical_step_heights_m"],
            [
                0.02,
                0.025,
                0.03,
                0.035,
                0.04,
                0.045,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
            ],
        )
        self.assertEqual(curriculum["successes_before_promotion"], 2)
        self.assertEqual(
            curriculum["physical_promotion"]["consecutive_evaluations"],
            2,
        )
        self.assertFalse(
            curriculum["physical_promotion"]["require_gait_quality"]
        )
        self.assertTrue(
            curriculum["physical_promotion"][
                "reset_optimizer_on_promotion"
            ]
        )
        self.assertEqual(
            curriculum["physical_promotion"][
                "max_speed_error_m_s_by_height"
            ],
            [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.09,
                0.09,
                0.09,
                0.08,
                0.08,
                0.08,
            ],
        )
        self.assertGreater(
            curriculum["physical_promotion"]["max_speed_error_m_s"],
            0.0,
        )
        self.assertEqual(
            curriculum["gait_promotion_min_target_steps"], 2
        )
        self.assertFalse(
            curriculum["require_gait_for_logical_promotion"]
        )
        self.assertGreaterEqual(
            curriculum["gait_promotion_min_alternating_tread_rate"],
            0.50,
        )
        self.assertLessEqual(
            curriculum["gait_promotion_max_same_tread_join_rate"],
            0.25,
        )
        self.assertGreater(
            curriculum["checkpoint_gate"]["min_completion_rate"], 0.0
        )
        self.assertGreaterEqual(
            curriculum["checkpoint_gate"]["min_completion_rate"], 0.70
        )
        self.assertFalse(
            curriculum["checkpoint_gate"]["require_natural_gait"]
        )
        self.assertLess(
            curriculum["checkpoint_gate"]["max_fall_rate"], 1.0
        )
        self.assertGreater(curriculum["path_violation_dwell_s"], 0.0)
        self.assertGreater(
            curriculum["natural_min_alternating_tread_rate"],
            curriculum["natural_max_same_tread_join_rate"],
        )
        self.assertLess(
            training["training_corridor_half_width_m"],
            training["adaptation_corridor_half_width_m"],
        )
        self.assertLess(
            training["training_corridor_yaw_limit_rad"],
            training["adaptation_corridor_yaw_limit_rad"],
        )
        required_rewards = {
            "tracking_speed",
            "overspeed",
            "forward_progress",
            "directed_progress",
            "vertical_progress",
            "heading_alignment",
            "heading_error",
            "lateral_error",
            "lateral_velocity",
            "yaw_rate",
            "forward_pitch",
            "base_behind_support",
            "foot_pitch",
            "phase_contact",
            "phase_contact_mismatch",
            "sagittal_foot_phase",
            "sagittal_foot_phase_error",
            "single_support",
            "swing_knee",
            "swing_knee_deficit",
            "swing_clearance",
            "swing_clearance_deficit",
            "swing_trajectory",
            "swing_trajectory_error",
            "next_tread_target",
            "foothold_lateral",
            "expected_swing_liftoff",
            "expected_swing_delay",
            "wrong_foot_swing",
            "arm_swing",
            "foot_riser_collision",
            "lower_leg_collision",
            "tread_advance",
            "alternating_tread",
            "repeated_lead",
            "same_tread_join",
            "completion",
            "unnatural_completion",
            "gait_completion",
            "gait_failure",
            "natural_completion",
            "fall",
            "path_failure",
            "stall",
        }
        self.assertTrue(required_rewards.issubset(training["reward_scales"]))
        reward_scales = training["reward_scales"]
        regression_guard = training["regression_guard"]
        self.assertEqual(
            regression_guard["promotion_grace_evaluations"], 4
        )
        self.assertEqual(
            regression_guard["consecutive_evaluations"], 2
        )
        self.assertLessEqual(
            regression_guard["absolute_max_completion_rate"], 0.40
        )
        self.assertGreaterEqual(
            regression_guard["absolute_min_fall_rate"], 0.50
        )
        self.assertGreater(
            reward_scales["completion"]
            + reward_scales["unnatural_completion"],
            reward_scales["fall"],
        )
        self.assertGreater(
            reward_scales["completion"]
            + reward_scales["gait_completion"],
            reward_scales["completion"]
            + reward_scales["unnatural_completion"],
        )
        self.assertGreater(
            reward_scales["completion"]
            + reward_scales["natural_completion"],
            reward_scales["completion"]
            + reward_scales["gait_completion"],
        )
        self.assertGreater(
            reward_scales["gait_failure"], reward_scales["fall"]
        )
        self.assertGreater(
            reward_scales["stall"], reward_scales["fall"]
        )

        env_source = (
            ROOT / "sim2sim" / "mujoco_stairs_env.py"
        ).read_text()
        self.assertIn(
            "self.num_privileged_obs = self.num_obs + 3", env_source
        )
        self.assertIn("def mirror_observations", env_source)
        self.assertIn("def mirror_actions", env_source)
        self.assertIn("scaled *= self.dt", env_source)
        self.assertIn("self.path_violation_steps", env_source)
        self.assertIn("def _advance_curriculum", env_source)
        self.assertIn("def _apply_physical_step_height", env_source)
        self.assertIn("def update_physical_curriculum", env_source)
        self.assertIn(
            "def _curriculum_gait_gate_passed", env_source
        )
        self.assertIn("def _swing_reference_state", env_source)
        self.assertIn("smooth_swing_trajectory", env_source)
        self.assertIn(
            '"completion": float(curriculum_completed[env_id])',
            env_source,
        )
        self.assertIn(
            "def _synchronize_phase_after_advance", env_source
        )
        self.assertIn("NATIVE_MUJOCO_HEIGHT_PROMOTION", env_source)
        self.assertIn(
            "NATIVE_MUJOCO_HEIGHT_PROMOTION {:.3f}m -> {:.3f}m",
            env_source,
        )
        self.assertIn("target_contact_reached", env_source)
        self.assertIn('"version": 13', env_source)
        self.assertIn(
            "4, 5, 6, 7, 8, 9, 10, 11, 12, 13", env_source
        )
        self.assertIn("LEGACY_PHYSICAL_STEP_HEIGHTS", env_source)
        self.assertIn('"physical_step_height_m"', env_source)
        self.assertIn(
            "MuJoCo checkpoint physical height is missing", env_source
        )
        self.assertIn('"gait_completion"', env_source)
        self.assertIn('"gait_failure"', env_source)
        self.assertIn("mujoco_gait_failure_rate", env_source)
        self.assertIn(
            "def _curriculum_micro_gait_gate_active", env_source
        )
        self.assertIn(
            'event_delta["advance"]\n                    '
            '> event_delta["alternating"]',
            env_source,
        )
        self.assertIn('"scheduled_active"', env_source)
        self.assertIn("scheduled_clearance_base_m", env_source)
        self.assertIn("actual_contacts[opposite_foot]", env_source)
        self.assertIn('"wrong_foot_swing"', env_source)
        self.assertIn("def set_gait_guidance", env_source)
        self.assertIn("def _prepare_executed_actions", env_source)
        self.assertIn("self.executed_actions", env_source)
        self.assertIn("support_guard_allows_residual", env_source)
        self.assertIn("tracker.stable_tread", env_source)
        self.assertNotIn('"gait_guide_match"', env_source)
        guidance = training["gait_guidance"]
        self.assertFalse(guidance["enabled"])
        self.assertGreater(guidance["max_assistance_scale"], 0.0)
        self.assertLessEqual(guidance["max_assistance_scale"], 0.50)
        self.assertGreater(guidance["fade_iterations"], 0)
        self.assertGreater(
            guidance["post_fade_stagnation_evaluations"], 0
        )
        self.assertFalse(training["randomize_gait_phase"])
        self.assertGreaterEqual(
            guidance["activation_distance_m"],
            config["stairs"]["start_x"],
        )
        self.assertIn("-16.0 * float(state[\"yaw\"] ** 2)", env_source)
        self.assertTrue(config["gait_phase"]["contact_phase_reset"])

    def test_native_mujoco_trainer_has_curriculum_and_robust_selection(self):
        source = (
            ROOT / "sim2sim" / "train_stairs_mujoco.py"
        ).read_text()
        self.assertIn("set_actor_trunk_trainable", source)
        self.assertIn("freeze_actor_iterations", source)
        self.assertIn("critic and Adam reset", source)
        self.assertIn("selection_score", source)
        self.assertIn("physical_promotion_readiness", source)
        self.assertIn("checkpoint_gate_passed", source)
        self.assertIn("promotion_evaluation_seed", source)
        self.assertIn("robust_checkpoint_tournament", source)
        self.assertIn("NATIVE_MUJOCO_CURRICULUM", source)
        self.assertIn("NATIVE_MUJOCO_HEIGHT_GATE", source)
        self.assertIn("NATIVE_MUJOCO_HEIGHT_OPTIMIZER_RESET", source)
        self.assertIn(
            "reset_optimizer_after_height_promotion", source
        )
        self.assertIn("NATIVE_MUJOCO_BEST_REJECT", source)
        self.assertIn("NATIVE_MUJOCO_TOURNAMENT", source)
        self.assertIn("NATIVE_MUJOCO_ROBUST_BEST", source)
        self.assertIn("NATIVE_MUJOCO_PROGRESS_BEST", source)
        self.assertIn("NATIVE_MUJOCO_BASELINE", source)
        self.assertIn("NATIVE_MUJOCO_BASELINE_REJECTED", source)
        self.assertIn("baseline_unusable", source)
        self.assertIn("absolute_regression", source)
        self.assertIn("NATIVE_MUJOCO_EARLY_STOP", source)
        self.assertIn("gait_guidance_schedule", source)
        self.assertIn("env.set_gait_guidance", source)
        self.assertIn("reason=post_guidance_", source)
        self.assertIn('"--gait_guide_scale"', source)
        self.assertIn('"0.0"', source)
        self.assertIn("NATIVE_MUJOCO_STAGE_BEST", source)
        self.assertIn('"fixed" if args.fixed_learning_rate', source)
        self.assertIn('"model_progress_best.pt"', source)
        self.assertIn('"model_stage_best.pt"', source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("model_interrupted.pt", source)
        self.assertIn("NATIVE_MUJOCO_CHECKPOINT=NONE", source)
        self.assertIn("int(args.seed) + 20000", source)
        self.assertIn("args.seed + 30000", source)
        self.assertIn('"gamma": 0.997', source)
        self.assertIn('"symmetry_cfg"', source)
        self.assertIn(
            'summary.get("mean_arm_swing_match", 0.0)', source
        )
        self.assertIn('"model_best.pt"', source)
        launcher = (
            ROOT / "sim2sim" / "run_mujoco_native_train.sh"
        ).read_text()
        self.assertIn("TARGET_ITERATIONS=800", launcher)
        self.assertIn("TARGET_ITERATIONS=3000", launcher)
        self.assertIn("TARGET_ITERATIONS=4000", launcher)
        self.assertIn(
            'MAX_ITERATIONS_OVERRIDE="${N2_MAX_ITERATIONS:-}"',
            launcher,
        )
        self.assertIn(
            'RESUME_ACTION_NOISE_STD="${N2_RESUME_NOISE_STD:-0.12}"',
            launcher,
        )
        self.assertIn(
            'RESUME_SYMMETRY_LOSS_COEFF="${N2_RESUME_SYMMETRY_LOSS_COEFF:-0.50}"',
            launcher,
        )
        self.assertIn(
            'TARGET_ITERATIONS="${MAX_ITERATIONS_OVERRIDE:-6000}"',
            launcher,
        )
        self.assertIn(
            '--max_iterations="${TARGET_ITERATIONS}"', launcher
        )
        self.assertIn('TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"', launcher)
        self.assertIn('TRAIN_SEED="${N2_SEED:-42}"', launcher)
        self.assertIn(
            'INIT_CHECKPOINT="${N2_INIT_CHECKPOINT:-auto}"', launcher
        )
        self.assertIn(
            'NO_WARM_START="${N2_NO_WARM_START:-0}"', launcher
        )
        self.assertIn('WARM_START_ARGS=("--no_warm_start")', launcher)
        self.assertIn("RUN_VARIANT=\"_scratch\"", launcher)
        self.assertIn("LEARNING_RATE=2e-4", launcher)
        self.assertIn("ACTION_NOISE_STD=0.45", launcher)
        self.assertIn("FREEZE_ACTOR_ITERATIONS=0", launcher)
        self.assertIn("--selection_interval=50", launcher)
        self.assertIn("--selection_episodes=32", launcher)
        self.assertIn("--tournament_episodes=32", launcher)
        self.assertIn("SYMMETRY_LOSS_COEFF=0.75", launcher)
        self.assertIn(
            '--symmetry_loss_coeff="${SYMMETRY_LOSS_COEFF}"',
            launcher,
        )
        self.assertIn(
            "pilot|long|recover|retune|climb-long", launcher
        )
        self.assertIn("stop)", launcher)
        self.assertIn('kill -TERM "${TRAIN_PIDS[@]}"', launcher)
        self.assertIn(
            'RUN_NAME="mujoco_curriculum_v13_recover_s${TRAIN_SEED}"',
            launcher,
        )
        self.assertIn("LEARNING_RATE=1e-5", launcher)
        self.assertIn("ACTION_NOISE_STD=0.12", launcher)
        self.assertIn("FREEZE_ACTOR_ITERATIONS=25", launcher)
        self.assertIn("guide-check)", launcher)
        self.assertIn("--gait_guide_sweep", launcher)
        self.assertIn(
            'RUN_NAME="mujoco_curriculum_v13_climb_long_s${TRAIN_SEED}"',
            launcher,
        )
        self.assertIn("LEARNING_RATE=3e-6", launcher)
        self.assertIn("ACTION_NOISE_STD=0.08", launcher)
        self.assertIn('LEARNING_RATE_ARGS=("--fixed_learning_rate")', launcher)
        self.assertIn(
            '--resume_action_noise_std="${RESUME_ACTION_NOISE_STD}"',
            launcher,
        )
        self.assertIn(
            '--symmetry_loss_coeff="${RESUME_SYMMETRY_LOSS_COEFF}"',
            launcher,
        )
        self.assertIn(
            '"${WARM_START_ARGS[@]}"', launcher
        )
        self.assertNotIn("--init_checkpoint=auto_v2", launcher)
        self.assertIn("stream_stairs_mujoco.py", launcher)
        self.assertIn('checkpoint_path="${LATEST_BEST}"', launcher)
        self.assertIn("model_stage_best.pt", launcher)
        self.assertIn("model_progress_best.pt", launcher)
        self.assertIn("Viewing progress-best policy", launcher)
        self.assertIn('resume="${LATEST_MODEL}"', launcher)
        self.assertIn(
            '-path "*mujoco_curriculum_v[0-9]*_*_s${TRAIN_SEED}*"',
            launcher,
        )
        self.assertIn("! -path '*smoke*'", launcher)
        self.assertNotIn("humanoid/scripts/train.py", launcher)

        for script_name in (
            "train.py",
            "play.py",
            "eval_stairs.py",
            "stream_stairs.py",
        ):
            script_source = (
                ROOT / "humanoid" / "scripts" / script_name
            ).read_text()
            self.assertIn(
                "from humanoid.utils.helpers import",
                script_source,
            )
            self.assertIn(
                "parse_humanoid_args(", script_source
            )
            self.assertIn(
                "from humanoid.utils.task_registry import task_registry",
                script_source,
            )
            self.assertNotIn(
                "from humanoid.utils import task_registry", script_source
            )
            self.assertIn(
                "sys.path.insert(0, _REPOSITORY_ROOT)", script_source
            )

        ppo_source = (
            ROOT / "humanoid" / "algo" / "ppo" / "ppo.py"
        ).read_text()
        self.assertIn("symmetry_cfg: Optional[dict] = None", ppo_source)
        self.assertIn("mirror_observations", ppo_source)
        self.assertIn("mirror_actions", ppo_source)
        self.assertIn('"symmetry": mean_symmetry_loss', ppo_source)

        stream_source = (
            ROOT / "sim2sim" / "stream_stairs_mujoco.py"
        ).read_text()
        self.assertIn('os.environ.setdefault("MUJOCO_GL", "egl")', stream_source)
        self.assertIn("mujoco.Renderer", stream_source)
        self.assertIn("latest_native_checkpoint", stream_source)
        self.assertIn("latest saved checkpoint", stream_source)
        self.assertIn("step_callback=callback", stream_source)
        evaluator_source = (
            ROOT / "sim2sim" / "eval_stairs_mujoco.py"
        ).read_text()
        self.assertIn('"mean_arm_swing_match"', evaluator_source)
        evaluator_root = evaluator_source.index(
            "_REPOSITORY_ROOT ="
        )
        evaluator_humanoid = evaluator_source.index(
            "from humanoid import LEGGED_GYM_ROOT_DIR"
        )
        self.assertLess(evaluator_root, evaluator_humanoid)
        self.assertIn(
            "sys.path.insert(1, _REPOSITORY_ROOT)",
            evaluator_source,
        )
        sim2sim_source = (
            ROOT / "sim2sim" / "sim2sim.py"
        ).read_text()
        sim2sim_root = sim2sim_source.index("_REPOSITORY_ROOT =")
        sim2sim_humanoid = sim2sim_source.index(
            "from humanoid import LEGGED_GYM_ROOT_DIR"
        )
        self.assertLess(sim2sim_root, sim2sim_humanoid)

    def test_algorithm_utilities_do_not_eagerly_import_isaacgym(self):
        init_source = (
            ROOT / "humanoid" / "utils" / "__init__.py"
        ).read_text()
        self.assertIn("def __getattr__", init_source)
        self.assertNotIn("from .helpers import", init_source)
        utility_source = (
            ROOT / "humanoid" / "utils" / "utils.py"
        ).read_text()
        self.assertNotIn("\nimport git\n", utility_source)


if __name__ == "__main__":
    unittest.main()
