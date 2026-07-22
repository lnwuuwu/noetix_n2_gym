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
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_pure_stairs_module():
    path = ROOT / "humanoid" / "utils" / "stairs_terrain.py"
    spec = importlib.util.spec_from_file_location("n2_stairs_geometry_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            self.assertLessEqual(walk_cfg.env.top_speed_tolerance, 0.06)
            self.assertLessEqual(
                walk_cfg.env.success_max_mean_command_error, 0.06
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
                walk_cfg.env.max_sagittal_foot_separation,
                walk_cfg.env.success_max_sagittal_foot_separation,
            )
            self.assertLessEqual(
                walk_cfg.env.success_max_sagittal_foot_separation, 0.44
            )
            self.assertGreater(
                walk_cfg.env.swing_knee_max_target,
                walk_cfg.env.swing_knee_base_target,
            )
            self.assertGreater(walk_cfg.env.arm_swing_amplitude, 0.0)
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
            "stairs_heading_alignment",
            "stairs_leg_alignment",
            "stairs_feet_yaw",
            "stairs_alternating_tread",
            "stairs_repeated_lead",
            "stairs_same_tread_join",
            "stairs_skipped_tread",
            "stairs_overstride",
            "stairs_swing_knee_flexion",
            "stairs_arm_swing",
        ):
            self.assertIn(required_reward, walk_scales)

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
        self.assertIn("opposite_was_supported", stairs_source)
        self.assertIn("one_new_contact", stairs_source)
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
            "self.tread_advance_count + self.same_tread_join_count",
            stairs_source,
        )
        train_source = (
            ROOT / "humanoid" / "scripts" / "train.py"
        ).read_text()
        self.assertIn("--reset_optimizer", train_source)
        self.assertIn("load_optimizer=not args.reset_optimizer", train_source)
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
            "mean_forward_speed_m_s",
            "mean_command_error_m_s",
            "mean_phase_contact_match",
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


if __name__ == "__main__":
    unittest.main()
