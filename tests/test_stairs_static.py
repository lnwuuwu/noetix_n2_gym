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
            train_cfg = stairs_module.N2StairsCfgPPO()
            walk_train_cfg = stairs_module.N2StairsWalkCfgPPO()
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
        self.assertIn("N2StairsEnv", registration)

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
            "humanoid/scripts/eval_stairs.py",
            "humanoid/scripts/play.py",
            "humanoid/scripts/stream_stairs.py",
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
        self.assertEqual(curriculum["successes_before_promotion"], 2)
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
            "sagittal_foot_phase",
            "single_support",
            "swing_knee",
            "arm_swing",
            "foot_riser_collision",
            "lower_leg_collision",
            "tread_advance",
            "alternating_tread",
            "repeated_lead",
            "same_tread_join",
            "completion",
            "natural_completion",
            "fall",
            "path_failure",
            "stall",
        }
        self.assertTrue(required_rewards.issubset(training["reward_scales"]))

        env_source = (
            ROOT / "sim2sim" / "mujoco_stairs_env.py"
        ).read_text()
        self.assertIn("self.num_privileged_obs = self.num_obs + 2", env_source)
        self.assertIn("def mirror_observations", env_source)
        self.assertIn("def mirror_actions", env_source)
        self.assertIn("scaled *= self.dt", env_source)
        self.assertIn("self.path_violation_steps", env_source)
        self.assertIn("def _advance_curriculum", env_source)
        self.assertIn("target_contact_reached", env_source)
        self.assertIn('"version": 3', env_source)
        self.assertIn("-16.0 * float(state[\"yaw\"] ** 2)", env_source)

    def test_native_mujoco_trainer_has_curriculum_and_robust_selection(self):
        source = (
            ROOT / "sim2sim" / "train_stairs_mujoco.py"
        ).read_text()
        self.assertIn("set_actor_trunk_trainable", source)
        self.assertIn("freeze_actor_iterations", source)
        self.assertIn("critic and Adam reset", source)
        self.assertIn("selection_score", source)
        self.assertIn("robust_checkpoint_tournament", source)
        self.assertIn("NATIVE_MUJOCO_CURRICULUM", source)
        self.assertIn("NATIVE_MUJOCO_TOURNAMENT", source)
        self.assertIn("NATIVE_MUJOCO_ROBUST_BEST", source)
        self.assertIn("args.seed + 20000", source)
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
        self.assertIn("TARGET_ITERATIONS=400", launcher)
        self.assertIn("TARGET_ITERATIONS=1600", launcher)
        self.assertIn("--max_iterations=1600", launcher)
        self.assertIn("--selection_interval=100", launcher)
        self.assertIn("--selection_episodes=16", launcher)
        self.assertIn("--tournament_episodes=32", launcher)
        self.assertIn("--symmetry_loss_coeff=0.50", launcher)
        self.assertIn("pilot|long", launcher)
        self.assertIn("--init_checkpoint=auto_v2", launcher)
        self.assertIn("stream_stairs_mujoco.py", launcher)
        self.assertIn('resume="${LATEST_MODEL}"', launcher)
        self.assertIn("-path '*mujoco_curriculum_v3_*'", launcher)
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
                "import humanoid.utils.helpers as humanoid_helpers",
                script_source,
            )
            self.assertIn(
                "humanoid_helpers.get_args(", script_source
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
