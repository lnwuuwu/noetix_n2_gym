"""Parallel, headless MuJoCo environment for native N2 stair PPO."""

import math
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

from humanoid.algo import VecEnv
from humanoid.utils.stairs_terrain import smooth_swing_trajectory

from eval_stairs_mujoco import (
    ContactLayout,
    GaitTracker,
    _build_observation,
    _configure_solver,
    contact_synchronized_phase_offset,
    sample_contacts,
    terrain_height_at_x,
)
try:
    from gait_guidance import (
        apply_swing_action_residual,
        phase_swing_action_residual,
        support_guard_allows_residual,
    )
except ImportError:
    from sim2sim.gait_guidance import (
        apply_swing_action_residual,
        phase_swing_action_residual,
        support_guard_allows_residual,
    )
from sim2sim import load_mujoco_model, pd_control, resolve_joint_layout


class MujocoStairsVecEnv(VecEnv):
    """Independent data sharing a model resized only between rollout chunks."""

    LEGACY_PHYSICAL_STEP_HEIGHTS = (0.02, 0.04, 0.06, 0.08, 0.10)

    EVENT_REWARD_TERMS = {
        "tread_advance",
        "alternating_tread",
        "repeated_lead",
        "same_tread_join",
        "skipped_tread",
        "completion",
        "unnatural_completion",
        "gait_completion",
        "gait_failure",
        "natural_completion",
        "fall",
        "path_failure",
        "stall",
    }

    def __init__(
        self,
        config,
        num_envs=32,
        num_workers=8,
        seed=42,
    ):
        self.cfg = config
        self.training_cfg = config["mujoco_training"]
        self.curriculum_cfg = self.training_cfg["curriculum"]
        self.validation_cfg = config["validation"]
        self.stair_cfg = config["stairs"]
        if abs(float(self.stair_cfg["step_height"]) - 0.10) > 1.0e-9:
            raise ValueError(
                "Native stair training is fixed at the requested 0.10 m"
            )

        self.num_envs = int(num_envs)
        self.num_actions = int(config["num_actions"])
        self.num_obs = int(config["num_obs"])
        # The Actor remains deployable with the original 410-D observation.
        # The Critic additionally receives the current curriculum target,
        # avoiding a partially observable value function when two otherwise
        # identical initial states terminate at different stair goals.
        self.num_privileged_obs = self.num_obs + 3
        self.device = torch.device("cpu")
        self.control_decimation = int(config["control_decimation"])
        self.simulation_dt = float(config["simulation_dt"])
        self.dt = self.simulation_dt * self.control_decimation
        self.max_episode_length = int(
            round(
                float(self.training_cfg["episode_duration"])
                / self.dt
            )
        )
        self.frame_stack = int(config["frame_stack"])
        self.num_single_obs = int(config["num_single_obs"])
        self.rng = np.random.default_rng(int(seed))
        self.adaptation_stage = True
        self.curriculum_target_steps = np.asarray(
            self.curriculum_cfg["target_steps"], dtype=np.int64
        )
        self.curriculum_target_x = np.asarray(
            self.curriculum_cfg["target_x_m"], dtype=np.float64
        )
        if (
            self.curriculum_target_steps.ndim != 1
            or len(self.curriculum_target_steps) < 2
            or self.curriculum_target_x.shape
            != self.curriculum_target_steps.shape
        ):
            raise ValueError(
                "MuJoCo curriculum target_steps/target_x_m must be "
                "equal-length one-dimensional arrays"
            )
        if (
            self.curriculum_target_steps[0] != 0
            or self.curriculum_target_steps[-1]
            != int(self.stair_cfg["num_steps"])
            or np.any(np.diff(self.curriculum_target_steps) < 0)
            or np.any(np.diff(self.curriculum_target_x) <= 0.0)
        ):
            raise ValueError(
                "MuJoCo curriculum must progress monotonically from flat "
                "approach to the full staircase"
            )
        self.final_curriculum_level = (
            len(self.curriculum_target_steps) - 1
        )
        self.physical_step_heights = np.asarray(
            self.curriculum_cfg["physical_step_heights_m"],
            dtype=np.float64,
        )
        if (
            self.physical_step_heights.ndim != 1
            or len(self.physical_step_heights) < 2
            or np.any(self.physical_step_heights <= 0.0)
            or np.any(np.diff(self.physical_step_heights) <= 0.0)
            or not math.isclose(
                float(self.physical_step_heights[-1]),
                float(self.stair_cfg["step_height"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(
                "MuJoCo physical_step_heights_m must be positive, strictly "
                "increasing, and end at stairs.step_height"
            )
        self.physical_height_index = 0
        self.physical_promotion_streak = 0
        for key, values in self.curriculum_cfg[
            "physical_promotion"
        ].items():
            if key.endswith("_by_height") and len(values) != (
                len(self.physical_step_heights) - 1
            ):
                raise ValueError(
                    "{} must have one value per promotion".format(key)
                )

        xml_path = config["_resolved_xml_path"]
        urdf_path = config.get("_resolved_urdf_path")
        self.model = load_mujoco_model(
            xml_path,
            self.stair_cfg,
            config.get("mujoco_physics"),
            urdf_path=urdf_path,
        )
        _configure_solver(self.model, config)
        self.datas = [
            mujoco.MjData(self.model) for _ in range(self.num_envs)
        ]
        self.layout = ContactLayout(
            self.model, int(self.stair_cfg["num_steps"])
        )
        self._apply_physical_step_height(
            float(self.physical_step_heights[0]),
            rebuild_trackers=False,
        )

        joint_order = list(config["joint_order"])
        qpos, qvel, control = resolve_joint_layout(
            self.model, joint_order
        )
        self.qpos_indices = np.asarray(qpos, dtype=np.int64)
        self.qvel_indices = np.asarray(qvel, dtype=np.int64)
        self.control_indices = np.asarray(control, dtype=np.int64)
        self.left_knee_index = joint_order.index("L_leg_knee_joint")
        self.right_knee_index = joint_order.index("R_leg_knee_joint")
        self.sagittal_joint_indices = np.asarray(
            [
                [
                    joint_order.index("L_leg_hip_pitch_joint"),
                    self.left_knee_index,
                    joint_order.index("L_leg_ankle_joint"),
                ],
                [
                    joint_order.index("R_leg_hip_pitch_joint"),
                    self.right_knee_index,
                    joint_order.index("R_leg_ankle_joint"),
                ],
            ],
            dtype=np.int64,
        )
        self.left_shoulder_pitch_index = joint_order.index(
            "L_arm_shoulder_pitch_joint"
        )
        self.right_shoulder_pitch_index = joint_order.index(
            "R_arm_shoulder_pitch_joint"
        )
        self.left_hip_yaw_index = joint_order.index(
            "L_leg_hip_yaw_joint"
        )
        self.right_hip_yaw_index = joint_order.index(
            "R_leg_hip_yaw_joint"
        )
        self._build_mirror_layout(joint_order)
        gyro_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "angular-velocity",
        )
        if gyro_id < 0:
            raise ValueError("MuJoCo angular-velocity sensor is missing")
        self.gyro_address = int(self.model.sensor_adr[gyro_id])

        self.default_angles = np.asarray(
            config["default_angles"], dtype=np.float64
        )
        self.kps = np.asarray(config["kps"], dtype=np.float64)
        self.kds = np.asarray(config["kds"], dtype=np.float64)
        self.torque_limits = np.asarray(
            config["torque_limits"], dtype=np.float64
        )
        self.action_scale = float(config["action_scale"])
        self.clip_actions = float(config.get("clip_actions", 18.0))
        self.gait_guidance_cfg = self.training_cfg.get(
            "gait_guidance", {}
        )
        self.gait_assistance_scale = 0.0

        workers = max(1, min(int(num_workers), self.num_envs))
        self.num_workers = workers
        self.executor = (
            ThreadPoolExecutor(max_workers=workers)
            if workers > 1
            else None
        )
        self.worker_chunks = [
            chunk
            for chunk in np.array_split(
                np.arange(self.num_envs, dtype=np.int64), workers
            )
            if len(chunk)
        ]

        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.phase_offsets = np.zeros(self.num_envs, dtype=np.float64)
        self.actions = np.zeros(
            (self.num_envs, self.num_actions), dtype=np.float32
        )
        self.executed_actions = np.zeros_like(self.actions)
        self.last_actions = np.zeros_like(self.actions)
        self.last_last_actions = np.zeros_like(self.actions)
        self.last_torques = np.zeros_like(self.actions, dtype=np.float64)
        self.history = np.zeros(
            (
                self.num_envs,
                self.frame_stack,
                self.num_single_obs,
            ),
            dtype=np.float32,
        )
        self.trackers = [
            self._new_tracker() for _ in range(self.num_envs)
        ]
        self.swing_start_pos = np.zeros(
            (self.num_envs, 2, 3), dtype=np.float64
        )
        self.last_physical_swing = np.zeros(
            (self.num_envs, 2), dtype=bool
        )
        self.swing_elapsed_time = np.zeros(
            (self.num_envs, 2), dtype=np.float64
        )
        self.expected_swing_foot = np.full(
            self.num_envs, -1, dtype=np.int64
        )
        self.foot_surface_offset = np.full(
            (self.num_envs, 2),
            float(
                self.training_cfg["nominal_foot_surface_offset_m"]
            ),
            dtype=np.float64,
        )
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long
        )
        self.previous_x = np.zeros(self.num_envs, dtype=np.float64)
        self.progress_checkpoint = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.last_progress_step = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.completion_steps = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.path_violation_steps = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.mastery_levels = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.episode_levels = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.curriculum_success_streak = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.episode_reward = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.speed_sum = np.zeros(self.num_envs, dtype=np.float64)
        self.phase_match_sum = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.arm_match_sum = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.max_lateral = np.zeros(self.num_envs, dtype=np.float64)
        self.max_yaw = np.zeros(self.num_envs, dtype=np.float64)
        self.max_climb = np.zeros(self.num_envs, dtype=np.float64)
        self.previous_base_z = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.reward_sums = {
            name: np.zeros(self.num_envs, dtype=np.float64)
            for name in self.training_cfg["reward_scales"]
        }

        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float32
        )
        self.privileged_obs_buf = torch.zeros(
            self.num_envs,
            self.num_privileged_obs,
            dtype=torch.float32,
        )
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.extras = {}
        self.reset()

    def _new_tracker(self):
        return GaitTracker(
            self.dt, self.stair_cfg, self.validation_cfg
        )

    @property
    def physical_step_height(self):
        return float(
            self.physical_step_heights[self.physical_height_index]
        )

    @property
    def physical_curriculum_complete(self):
        return self.physical_height_index == (
            len(self.physical_step_heights) - 1
        )

    def _apply_physical_step_height(
        self, step_height, rebuild_trackers=True
    ):
        """Resize the shared static staircase between PPO rollout chunks."""
        step_height = float(step_height)
        num_steps = int(self.stair_cfg["num_steps"])
        for index in range(num_steps):
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "n2_stair_{:02d}".format(index + 1),
            )
            if geom_id < 0:
                raise ValueError(
                    "Generated MuJoCo stair geom is missing at index "
                    + str(index + 1)
                )
            total_height = (index + 1) * step_height
            self.model.geom_size[geom_id, 2] = 0.5 * total_height
            self.model.geom_pos[geom_id, 2] = 0.5 * total_height

        top_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "n2_stair_top",
        )
        if top_id < 0:
            raise ValueError("Generated MuJoCo stair top geom is missing")
        top_height = num_steps * step_height
        self.model.geom_size[top_id, 2] = 0.5 * top_height
        self.model.geom_pos[top_id, 2] = 0.5 * top_height
        self.stair_cfg["step_height"] = step_height
        mujoco.mj_setConst(self.model, self.datas[0])

        if rebuild_trackers and hasattr(self, "trackers"):
            self.trackers = [
                self._new_tracker() for _ in range(self.num_envs)
            ]

    def _physical_gate_value(self, gate, key):
        scheduled = gate.get(key + "_by_height")
        if scheduled is None:
            return float(gate[key])
        # Once every configured height has been promoted, the index points
        # one element past the schedule.  Keep using the strictest (last)
        # gate for checkpoint ranking and post-curriculum evaluation.
        schedule_index = min(
            self.physical_height_index,
            len(scheduled) - 1,
        )
        return float(scheduled[schedule_index])

    def _physical_gate_passed(self, summary, gate):
        expected_climb = (
            int(self.stair_cfg["num_steps"]) * self.physical_step_height
        )
        gait_quality_passed = (
            not bool(gate.get("require_gait_quality", True))
            or (
                float(summary["mean_alternating_tread_rate"])
                >= self._physical_gate_value(
                    gate, "min_alternating_tread_rate"
                )
                and float(summary["mean_same_tread_join_rate"])
                <= self._physical_gate_value(
                    gate, "max_same_tread_join_rate"
                )
            )
        )
        return (
            math.isclose(
                float(summary["step_height_m"]),
                self.physical_step_height,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and float(summary["completion_rate"])
            >= float(gate["min_completion_rate"])
            and float(summary["fall_rate"])
            <= float(gate["max_fall_rate"])
            and float(summary["path_failure_rate"])
            <= float(gate["max_path_failure_rate"])
            and float(summary["mean_climb_height_m"])
            >= self._physical_gate_value(
                gate, "min_climb_fraction"
            )
            * expected_climb
            and gait_quality_passed
            and abs(
                float(summary["mean_forward_speed_m_s"])
                - float(summary["command_speed_m_s"])
            )
            <= self._physical_gate_value(
                gate, "max_speed_error_m_s"
            )
            and float(summary["mean_max_yaw_deviation_rad"])
            <= self._physical_gate_value(
                gate, "max_yaw_deviation_rad"
            )
        )

    def update_physical_curriculum(self, summary):
        """Promote only after repeatable deterministic full-flight climbs."""
        if self.physical_curriculum_complete:
            self.physical_promotion_streak = 0
            return False
        gate = self.curriculum_cfg["physical_promotion"]
        if self._physical_gate_passed(summary, gate):
            self.physical_promotion_streak += 1
        else:
            self.physical_promotion_streak = 0
        required = int(gate["consecutive_evaluations"])
        if self.physical_promotion_streak < required:
            return False

        previous = self.physical_step_height
        self.physical_height_index += 1
        self.physical_promotion_streak = 0
        self.mastery_levels[:] = 0
        self.episode_levels[:] = 0
        self.curriculum_success_streak[:] = 0
        self._apply_physical_step_height(self.physical_step_height)
        self.reset()
        print(
            "NATIVE_MUJOCO_HEIGHT_PROMOTION {:.3f}m -> {:.3f}m".format(
                previous, self.physical_step_height
            ),
            flush=True,
        )
        return True

    def _build_mirror_layout(self, joint_order):
        """Build an exact left/right reflection for policy regularization."""
        sign_by_suffix = {
            "arm_shoulder_pitch_joint": 1.0,
            "arm_shoulder_roll_joint": -1.0,
            "arm_shoulder_yaw_joint": -1.0,
            "arm_elbow_joint": -1.0,
            "leg_hip_yaw_joint": -1.0,
            "leg_hip_roll_joint": -1.0,
            "leg_hip_pitch_joint": 1.0,
            "leg_knee_joint": 1.0,
            "leg_ankle_joint": 1.0,
        }
        joint_index = {name: index for index, name in enumerate(joint_order)}
        action_source = np.zeros(self.num_actions, dtype=np.int64)
        action_sign = np.ones(self.num_actions, dtype=np.float32)
        for index, name in enumerate(joint_order):
            if not (name.startswith("L_") or name.startswith("R_")):
                raise ValueError("Cannot mirror unpaired N2 joint " + name)
            side = name[0]
            suffix = name[2:]
            opposite = ("R_" if side == "L" else "L_") + suffix
            if suffix not in sign_by_suffix or opposite not in joint_index:
                raise ValueError("Cannot mirror N2 joint " + name)
            action_source[index] = joint_index[opposite]
            action_sign[index] = sign_by_suffix[suffix]
        self.mirror_action_source = action_source
        self.mirror_action_sign = action_sign

        source = np.arange(self.num_single_obs, dtype=np.int64)
        signs = np.ones(self.num_single_obs, dtype=np.float32)
        cursor = 0
        # command x/y/yaw
        signs[cursor:cursor + 3] = np.asarray([1.0, -1.0, -1.0])
        cursor += 3
        # A sagittal reflection swaps legs, hence advances gait phase by 0.5.
        signs[cursor:cursor + 2] = -1.0
        cursor += 2
        # body-frame linear velocity
        signs[cursor:cursor + 3] = np.asarray([1.0, -1.0, 1.0])
        cursor += 3
        # route-relative lateral position and yaw
        signs[cursor:cursor + 2] = -1.0
        cursor += 2
        # Angular velocity is an axial vector under reflection.
        signs[cursor:cursor + 3] = np.asarray([-1.0, 1.0, -1.0])
        cursor += 3
        # Projected gravity is a polar vector.
        signs[cursor:cursor + 3] = np.asarray([1.0, -1.0, 1.0])
        cursor += 3
        for _ in range(3):
            block = slice(cursor, cursor + self.num_actions)
            source[block] = cursor + action_source
            signs[block] = action_sign
            cursor += self.num_actions

        height_cfg = self.cfg["height_measurements"]
        points_x = list(height_cfg["points_x"])
        points_y = list(height_cfg["points_y"])
        height_count = len(points_x) * len(points_y)
        if cursor + height_count != self.num_single_obs:
            raise ValueError(
                "Mirror layout {} + {} != {}".format(
                    cursor, height_count, self.num_single_obs
                )
            )
        for x_index in range(len(points_x)):
            for y_index in range(len(points_y)):
                output = cursor + x_index * len(points_y) + y_index
                mirrored_y = len(points_y) - 1 - y_index
                source[output] = (
                    cursor + x_index * len(points_y) + mirrored_y
                )
        self.mirror_single_source = source
        self.mirror_single_sign = signs

    def mirror_actions(self, actions):
        source = torch.as_tensor(
            self.mirror_action_source,
            dtype=torch.long,
            device=actions.device,
        )
        signs = torch.as_tensor(
            self.mirror_action_sign,
            dtype=actions.dtype,
            device=actions.device,
        )
        return actions.index_select(-1, source) * signs

    def mirror_observations(self, observations):
        actor = observations[..., :self.num_obs]
        shape = actor.shape
        frames = actor.reshape(
            *shape[:-1], self.frame_stack, self.num_single_obs
        )
        source = torch.as_tensor(
            self.mirror_single_source,
            dtype=torch.long,
            device=observations.device,
        )
        signs = torch.as_tensor(
            self.mirror_single_sign,
            dtype=observations.dtype,
            device=observations.device,
        )
        mirrored_actor = (
            frames.index_select(-1, source) * signs
        ).reshape(shape)
        if observations.shape[-1] == self.num_obs:
            return mirrored_actor
        # Critic-only curriculum features are invariant under reflection.
        return torch.cat(
            [mirrored_actor, observations[..., self.num_obs:]], dim=-1
        )

    def set_adaptation_stage(self, active):
        """Use a wide route gate while only the Actor head is adapting."""
        self.adaptation_stage = bool(active)
        stage = "actor-head adaptation" if active else "full-policy"
        print("MuJoCo training stage: " + stage, flush=True)

    def set_gait_guidance(self, assistance_scale):
        """Set the optional training-only swing-action residual scale."""
        assistance_scale = float(assistance_scale)
        if not 0.0 <= assistance_scale <= 1.0:
            raise ValueError("gait assistance scale must be in [0, 1]")
        if not bool(self.gait_guidance_cfg.get("enabled", False)):
            assistance_scale = 0.0
        self.gait_assistance_scale = assistance_scale
        print(
            "NATIVE_MUJOCO_GAIT_GUIDANCE residual={:.3f}".format(
                self.gait_assistance_scale
            ),
            flush=True,
        )

    def _prepare_executed_actions(self):
        """Apply an optional support-protected swing residual to the plant.

        ``self.actions`` always remains the raw Actor output: it is stored in
        observations, receives smoothness penalties, and is what checkpoints
        export.  The residual never attenuates that output, is blocked unless
        the opposite foot has stable support, and is disabled in the climb-
        first configuration and all deterministic selection evaluations.
        """
        self.executed_actions[:] = self.actions
        if not bool(self.gait_guidance_cfg.get("enabled", False)):
            return

        minimum_steps = int(
            self.gait_guidance_cfg.get("min_target_steps", 2)
        )
        activation_x = (
            float(self.stair_cfg["start_x"])
            - float(
                self.gait_guidance_cfg.get(
                    "activation_distance_m", 0.30
                )
            )
        )
        for env_id in range(self.num_envs):
            _, target_steps, _ = self._target_for_env(env_id)
            if (
                target_steps < minimum_steps
                or float(self.datas[env_id].qpos[0]) < activation_x
            ):
                continue
            (
                residual,
                mask,
                swing_foot,
                _,
                phase_weight,
            ) = phase_swing_action_residual(
                self._phase_fraction(env_id),
                self.physical_step_height,
                self.action_scale,
                self.num_actions,
                self.sagittal_joint_indices,
                self.gait_guidance_cfg,
            )
            tracker = self.trackers[env_id]
            if not support_guard_allows_residual(
                swing_foot,
                tracker.stable_tread,
                tracker.pending_swing,
                tracker.accepted_tread,
            ):
                continue
            (
                self.executed_actions[env_id],
                _,
            ) = apply_swing_action_residual(
                self.actions[env_id],
                residual,
                mask,
                self.gait_assistance_scale,
                phase_weight,
            )
        np.clip(
            self.executed_actions,
            -self.clip_actions,
            self.clip_actions,
            out=self.executed_actions,
        )

    def _select_episode_level(self, env_id):
        level = int(self.mastery_levels[env_id])
        replay_probability = float(
            self.curriculum_cfg.get("replay_probability", 0.15)
        )
        if level > 0 and self.rng.random() < replay_probability:
            level -= 1
        return level

    def _target_for_env(self, env_id):
        level = int(self.episode_levels[env_id])
        return (
            level,
            int(self.curriculum_target_steps[level]),
            float(self.curriculum_target_x[level]),
        )

    def _update_privileged_observation(self, env_id):
        self.privileged_obs_buf[env_id, :self.num_obs] = (
            self.obs_buf[env_id]
        )
        level, _, target_x = self._target_for_env(env_id)
        self.privileged_obs_buf[env_id, self.num_obs] = (
            float(level) / max(1, self.final_curriculum_level)
        )
        remaining = target_x - float(self.datas[env_id].qpos[0])
        self.privileged_obs_buf[env_id, self.num_obs + 1] = float(
            np.clip(remaining / 2.60, -1.0, 1.0)
        )
        self.privileged_obs_buf[env_id, self.num_obs + 2] = float(
            self.physical_height_index
            / max(1, len(self.physical_step_heights) - 1)
        )

    def _gait_frequency(self, env_id):
        phase = self.cfg["gait_phase"]
        return float(phase.get("frequency", 1.25)) + float(
            phase.get("frequency_gain", 0.0)
        ) * max(
            float(self.commands[env_id, 0])
            - float(phase.get("reference_speed", 0.0)),
            0.0,
        )

    def _phase_fraction(self, env_id):
        elapsed = self.episode_steps[env_id] * self.dt
        return (
            self.phase_offsets[env_id]
            + elapsed * self._gait_frequency(env_id)
        ) % 1.0

    def _synchronize_phase_after_advance(self, env_id):
        """Make the next observable half-cycle swing the opposite foot."""
        if not bool(
            self.cfg["gait_phase"].get("contact_phase_reset", False)
        ):
            return
        advanced_foot = int(
            self.trackers[env_id].last_advanced_foot
        )
        if advanced_foot < 0:
            return
        # Right swing occupies phase (0.0, 0.5), and left swing occupies
        # (0.5, 1.0).  A touchdown therefore anchors right at 0.5 and left
        # at 0.0, followed by the opposite foot's scheduled half-cycle.
        elapsed = self.episode_steps[env_id] * self.dt
        self.phase_offsets[env_id] = (
            contact_synchronized_phase_offset(
                elapsed,
                self._gait_frequency(env_id),
                advanced_foot,
            )
        )

    @staticmethod
    def _rotation_matrix_wxyz(quaternion):
        w, x, y, z = quaternion
        return np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    def _state(self, env_id):
        data = self.datas[env_id]
        quaternion_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion_wxyz))
        if norm > 1.0e-9:
            quaternion_wxyz = quaternion_wxyz / norm
        rotation = self._rotation_matrix_wxyz(quaternion_wxyz)
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        local_velocity = rotation.T.dot(
            np.asarray(data.qvel[:3], dtype=np.float64)
        )
        gravity = rotation.T.dot(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        )
        angular_velocity = np.asarray(
            data.sensordata[
                self.gyro_address:self.gyro_address + 3
            ],
            dtype=np.float64,
        ).copy()
        q = np.asarray(
            data.qpos[self.qpos_indices], dtype=np.float64
        ).copy()
        dq = np.asarray(
            data.qvel[self.qvel_indices], dtype=np.float64
        ).copy()
        yaw = math.atan2(
            2.0
            * (
                quaternion_wxyz[0] * quaternion_wxyz[3]
                + quaternion_wxyz[1] * quaternion_wxyz[2]
            ),
            1.0
            - 2.0
            * (
                quaternion_wxyz[2] ** 2
                + quaternion_wxyz[3] ** 2
            ),
        )
        pitch = math.asin(
            float(
                np.clip(
                    2.0
                    * (
                        quaternion_wxyz[0] * quaternion_wxyz[2]
                        - quaternion_wxyz[3] * quaternion_wxyz[1]
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        return {
            "quat": quaternion_xyzw,
            "velocity": local_velocity,
            "omega": angular_velocity,
            "gravity": gravity,
            "q": q,
            "dq": dq,
            "yaw": yaw,
            "pitch": pitch,
        }

    def _observation(self, env_id, state):
        frequency = max(self._gait_frequency(env_id), 1.0e-6)
        elapsed = self.episode_steps[env_id] * self.dt
        configured_offset = float(
            self.cfg["gait_phase"].get("phase_offset", 0.0)
        )
        phase_adjusted_time = (
            elapsed
            + (
                self.phase_offsets[env_id] - configured_offset
            )
            / frequency
        )
        single = _build_observation(
            self.cfg,
            self.datas[env_id],
            state["quat"],
            state["velocity"],
            state["omega"],
            state["gravity"],
            state["q"],
            state["dq"],
            self.actions[env_id],
            self.commands[env_id],
            phase_adjusted_time,
        )[0]
        self.history[env_id, :-1] = self.history[env_id, 1:]
        self.history[env_id, -1] = single
        return self.history[env_id].reshape(-1)

    def _reset_one(self, env_id):
        data = self.datas[env_id]
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.model.qpos0
        data.qvel[:] = 0.0

        joint_noise = float(
            self.training_cfg["initial_joint_noise"]
        )
        data.qpos[self.qpos_indices] = (
            self.default_angles
            + self.rng.uniform(
                -joint_noise,
                joint_noise,
                size=self.num_actions,
            )
        )
        lateral_noise = float(
            self.training_cfg["initial_lateral_noise"]
        )
        yaw_noise = float(self.training_cfg["initial_yaw_noise"])
        data.qpos[0] += self.rng.uniform(-0.03, 0.03)
        data.qpos[1] += self.rng.uniform(
            -lateral_noise, lateral_noise
        )
        yaw = self.rng.uniform(-yaw_noise, yaw_noise)
        data.qpos[3:7] = np.asarray(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        )
        mujoco.mj_forward(self.model, data)

        command_range = self.training_cfg["command_speed_range"]
        self.commands[env_id, 0] = self.rng.uniform(
            float(command_range[0]), float(command_range[1])
        )
        self.commands[env_id, 1:] = 0.0
        self.phase_offsets[env_id] = (
            self.rng.uniform(0.0, 1.0)
            if bool(self.training_cfg["randomize_gait_phase"])
            else float(self.cfg["gait_phase"].get("phase_offset", 0.0))
        )
        self.actions[env_id] = 0.0
        self.executed_actions[env_id] = 0.0
        self.last_actions[env_id] = 0.0
        self.last_last_actions[env_id] = 0.0
        self.last_torques[env_id] = 0.0
        self.history[env_id] = 0.0
        self.trackers[env_id] = self._new_tracker()
        foot_positions = np.asarray(
            [
                data.site_xpos[site_id]
                for site_id in self.layout.foot_sites
            ],
            dtype=np.float64,
        )
        self.swing_start_pos[env_id] = foot_positions
        self.last_physical_swing[env_id] = False
        self.swing_elapsed_time[env_id] = 0.0
        self.expected_swing_foot[env_id] = -1
        self.foot_surface_offset[env_id] = float(
            self.training_cfg["nominal_foot_surface_offset_m"]
        )
        self.episode_steps[env_id] = 0
        self.episode_length_buf[env_id] = 0
        self.previous_x[env_id] = float(data.qpos[0])
        self.progress_checkpoint[env_id] = float(data.qpos[0])
        self.last_progress_step[env_id] = 0
        self.completion_steps[env_id] = 0
        self.path_violation_steps[env_id] = 0
        self.episode_levels[env_id] = self._select_episode_level(env_id)
        self.episode_reward[env_id] = 0.0
        self.speed_sum[env_id] = 0.0
        self.phase_match_sum[env_id] = 0.0
        self.arm_match_sum[env_id] = 0.0
        self.max_lateral[env_id] = abs(float(data.qpos[1]))
        self.max_yaw[env_id] = abs(yaw)
        self.max_climb[env_id] = 0.0
        self.previous_base_z[env_id] = float(data.qpos[2])
        for values in self.reward_sums.values():
            values[env_id] = 0.0
        state = self._state(env_id)
        self.obs_buf[env_id] = torch.from_numpy(
            self._observation(env_id, state).copy()
        )
        self._update_privileged_observation(env_id)

    def reset(self, env_ids=None):
        if env_ids is None:
            indices = range(self.num_envs)
        elif isinstance(env_ids, torch.Tensor):
            indices = env_ids.detach().cpu().numpy().reshape(-1)
        else:
            indices = env_ids
        for env_id in indices:
            self._reset_one(int(env_id))
        return self.obs_buf, self.privileged_obs_buf

    def _simulate_chunk(self, indices, targets):
        for raw_env_id in indices:
            env_id = int(raw_env_id)
            data = self.datas[env_id]
            target = targets[env_id]
            torque = self.last_torques[env_id]
            for _ in range(self.control_decimation):
                q = np.asarray(
                    data.qpos[self.qpos_indices], dtype=np.float64
                )
                dq = np.asarray(
                    data.qvel[self.qvel_indices], dtype=np.float64
                )
                torque = pd_control(
                    target,
                    q,
                    self.kps,
                    np.zeros_like(self.kds),
                    dq,
                    self.kds,
                )
                torque = np.clip(
                    torque, -self.torque_limits, self.torque_limits
                )
                data.ctrl[self.control_indices] = torque
                mujoco.mj_step(self.model, data)
            self.last_torques[env_id] = torque

    def _simulate(self, targets):
        if self.executor is None:
            self._simulate_chunk(self.worker_chunks[0], targets)
            return
        futures = [
            self.executor.submit(
                self._simulate_chunk, chunk, targets
            )
            for chunk in self.worker_chunks
        ]
        for future in futures:
            future.result()

    def _desired_contacts(self, env_id):
        sine = math.sin(2.0 * math.pi * self._phase_fraction(env_id))
        ratio = float(self.training_cfg["double_support_ratio"])
        threshold = math.sin(0.5 * math.pi * ratio)
        double_support = abs(sine) < threshold
        left_stance = sine >= 0.0
        return (
            np.asarray(
                [
                    left_stance or double_support,
                    (not left_stance) or double_support,
                ],
                dtype=bool,
            ),
            sine,
        )

    @staticmethod
    def _smoothstep01(value):
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _swing_reference_state(
        self,
        env_id,
        state,
        desired_contacts,
        raw_tread,
        riser,
        lower_leg,
    ):
        """Return dense next-tread targets for the scheduled opposite foot.

        Sparse touchdown events are too delayed to reliably turn an inherited
        step-to policy into stair-over-stair walking.  This reference supplies
        a bounded correction throughout the expected swing while retaining the
        contact tracker as the source of truth for actual gait success.
        """
        data = self.datas[env_id]
        tracker = self.trackers[env_id]
        foot_positions = np.asarray(
            [
                data.site_xpos[site_id]
                for site_id in self.layout.foot_sites
            ],
            dtype=np.float64,
        )

        for foot in range(2):
            tread = int(tracker.stable_tread[foot])
            if tread < 0:
                continue
            observed = (
                foot_positions[foot, 2]
                - tread * float(self.stair_cfg["step_height"])
            )
            observed = float(np.clip(observed, 0.02, 0.14))
            rate = float(
                self.training_cfg["foot_surface_offset_update_rate"]
            )
            self.foot_surface_offset[env_id, foot] = (
                (1.0 - rate) * self.foot_surface_offset[env_id, foot]
                + rate * observed
            )

        scheduled_swing = ~np.asarray(desired_contacts, dtype=bool)
        physical_swing = np.asarray(tracker.pending_swing, dtype=bool)
        physical_airborne = physical_swing & np.asarray(
            tracker.airborne_seen, dtype=bool
        )
        new_swing = physical_swing & ~self.last_physical_swing[env_id]
        self.swing_start_pos[env_id, new_swing] = foot_positions[new_swing]
        self.swing_elapsed_time[env_id, physical_swing] += self.dt
        self.swing_elapsed_time[env_id, ~physical_swing] = 0.0
        self.last_physical_swing[env_id] = physical_swing

        if np.any(scheduled_swing):
            scheduled_foot = int(np.argmax(scheduled_swing))
        else:
            scheduled_foot = -1
        # The target foot must be derivable from the Actor observation.
        # Contact-phase reset above aligns this observable clock with the
        # latest landing, so the scheduled foot is also the next foot that
        # should advance.  The former hidden last-advanced-foot target made
        # identical observations receive contradictory left/right rewards.
        expected_foot = scheduled_foot
        self.expected_swing_foot[env_id] = expected_foot

        inactive = {
            "score": 0.0,
            "error": 0.0,
            "next_tread_score": 0.0,
            "lateral_score": 0.0,
            "clearance_score": 0.0,
            "clearance_deficit": 0.0,
            "expected_liftoff": 0.0,
            "expected_delay": 0.0,
            "wrong_foot_swing": 0.0,
            "scheduled_active": 0.0,
            "active": 0.0,
            "progress": 0.0,
            "expected_foot": expected_foot,
        }
        if expected_foot < 0:
            return inactive

        # This clock is observable by the Actor and exists before liftoff.
        # The former reference used only tracker.pending_swing time, which
        # meant every lift/knee/trajectory term remained zero until the policy
        # had already discovered the correct swing by chance.
        phase = self._phase_fraction(env_id)
        scheduled_progress = float((2.0 * phase) % 1.0)
        nominal_swing_duration = (
            1.0 - float(self.training_cfg["double_support_ratio"])
        ) / max(2.0 * self._gait_frequency(env_id), 0.10)
        physical_progress = float(
            np.clip(
                self.swing_elapsed_time[env_id, expected_foot]
                / max(nominal_swing_duration, self.dt),
                0.0,
                1.0,
            )
        )
        inactive["progress"] = scheduled_progress
        near_stairs = (
            float(data.qpos[0])
            >= float(self.stair_cfg["start_x"])
            - float(
                self.training_cfg[
                    "gait_reward_activation_distance_m"
                ]
            )
        )
        behavior_active = (
            near_stairs
            and float(state["velocity"][0]) > 0.03
            and float(state["gravity"][2]) < -0.80
        )
        scheduled_expected = bool(scheduled_swing[expected_foot])
        scheduled_active = bool(behavior_active and scheduled_expected)
        inactive["scheduled_active"] = float(scheduled_active)
        actual_contacts = np.asarray(raw_tread, dtype=np.int64) >= 0
        opposite_foot = 1 - expected_foot

        # A smooth phase target gives a gradient while the foot is still
        # planted.  Raw contacts are intentional here: using the debounced
        # pending_swing/airborne flags delayed credit enough that the v8
        # expected-liftoff reward was almost always exactly zero.
        lift_profile = math.sin(math.pi * scheduled_progress) ** 2
        inactive["expected_liftoff"] = float(
            scheduled_active
            and not bool(actual_contacts[expected_foot])
            and bool(actual_contacts[opposite_foot])
        )
        inactive["expected_delay"] = float(
            scheduled_active and bool(actual_contacts[expected_foot])
        ) * lift_profile
        inactive["wrong_foot_swing"] = float(
            scheduled_active and not bool(actual_contacts[opposite_foot])
        )

        accepted_tread = max(
            0, int(tracker.accepted_tread[expected_foot])
        )
        support_surface_z = (
            accepted_tread * float(self.stair_cfg["step_height"])
            + self.foot_surface_offset[env_id, expected_foot]
        )
        actual_clearance = max(
            float(foot_positions[expected_foot, 2]) - support_surface_z,
            0.0,
        )
        peak_clearance = min(
            float(self.training_cfg["scheduled_clearance_max_m"]),
            float(self.training_cfg["scheduled_clearance_base_m"])
            + float(
                self.training_cfg["scheduled_clearance_height_gain"]
            )
            * float(self.stair_cfg["step_height"]),
        )
        target_clearance = peak_clearance * lift_profile
        clearance_normalizer = max(
            float(
                self.training_cfg["scheduled_clearance_normalizer_m"]
            ),
            1.0e-3,
        )
        normalized_clearance_error = float(
            np.clip(
                (actual_clearance - target_clearance)
                / clearance_normalizer,
                -2.0,
                2.0,
            )
        )
        normalized_clearance_deficit = float(
            np.clip(
                (target_clearance - actual_clearance)
                / clearance_normalizer,
                0.0,
                2.0,
            )
        )
        inactive["clearance_score"] = (
            math.exp(
                -float(
                    self.training_cfg[
                        "scheduled_clearance_sharpness"
                    ]
                )
                * normalized_clearance_error
                * normalized_clearance_error
            )
            * float(scheduled_active)
        )
        inactive["clearance_deficit"] = (
            normalized_clearance_deficit
            * normalized_clearance_deficit
            * float(scheduled_active)
        )

        expected_airborne = bool(physical_airborne[expected_foot])
        active = (
            scheduled_active
            and expected_airborne
            and bool(tracker.opposite_valid[expected_foot])
        )
        if not active:
            return inactive

        next_tread = int(
            np.clip(
                tracker.last_advanced_tread + 1,
                1,
                int(self.stair_cfg["num_steps"]),
            )
        )
        lateral_offset = float(
            self.training_cfg["foothold_lateral_offset_m"]
        )
        landing = np.asarray(
            [
                float(self.stair_cfg["start_x"])
                + (next_tread - 0.5)
                * float(self.stair_cfg["step_width"]),
                lateral_offset if expected_foot == 0 else -lateral_offset,
                next_tread * float(self.stair_cfg["step_height"])
                + self.foot_surface_offset[env_id, expected_foot],
            ],
            dtype=np.float64,
        )
        start = self.swing_start_pos[env_id, expected_foot]
        target = smooth_swing_trajectory(
            start[None, :],
            landing[None, :],
            np.asarray([physical_progress], dtype=np.float64),
            float(self.training_cfg["swing_trajectory_arc_base_m"])
            + float(
                self.training_cfg["swing_trajectory_arc_height_gain"]
            )
            * float(self.stair_cfg["step_height"]),
            float(self.training_cfg["swing_trajectory_forward_delay"]),
            float(self.training_cfg["swing_trajectory_lift_end"]),
            float(self.training_cfg["swing_trajectory_descent_start"]),
        )[0]
        normalizers = np.asarray(
            [
                self.training_cfg["swing_trajectory_x_normalizer_m"],
                self.training_cfg["swing_trajectory_y_normalizer_m"],
                self.training_cfg["swing_trajectory_z_normalizer_m"],
            ],
            dtype=np.float64,
        )
        error_clip = float(
            self.training_cfg["swing_trajectory_error_clip"]
        )
        normalized_error = np.clip(
            (foot_positions[expected_foot] - target) / normalizers,
            -error_clip,
            error_clip,
        )
        squared_error = float(np.mean(np.square(normalized_error)))
        collision_free = not (
            bool(riser[expected_foot]) or bool(lower_leg[expected_foot])
        )
        score = (
            math.exp(
                -float(self.training_cfg["swing_trajectory_sharpness"])
                * squared_error
            )
            if collision_free
            else 0.0
        )

        start_phase = float(
            self.training_cfg["next_tread_target_start_phase"]
        )
        full_phase = float(
            self.training_cfg["next_tread_target_full_phase"]
        )
        late_weight = self._smoothstep01(
            (physical_progress - start_phase)
            / max(full_phase - start_phase, 1.0e-3)
        )
        x_error = (
            foot_positions[expected_foot, 0] - landing[0]
        ) / max(float(self.stair_cfg["step_width"]), 1.0e-3)
        next_tread_score = (
            math.exp(-2.0 * float(np.clip(x_error, -2.0, 2.0) ** 2))
            * late_weight
        )
        y_error = (
            foot_positions[expected_foot, 1] - landing[1]
        ) / max(lateral_offset, 1.0e-3)
        lateral_score = math.exp(
            -2.0 * float(np.clip(y_error, -2.0, 2.0) ** 2)
        )
        return {
            "score": score,
            "error": squared_error,
            "next_tread_score": next_tread_score,
            "lateral_score": lateral_score,
            "clearance_score": inactive["clearance_score"],
            "clearance_deficit": inactive["clearance_deficit"],
            "expected_liftoff": inactive["expected_liftoff"],
            "expected_delay": inactive["expected_delay"],
            "wrong_foot_swing": inactive["wrong_foot_swing"],
            "scheduled_active": inactive["scheduled_active"],
            "active": 1.0,
            "progress": scheduled_progress,
            "expected_foot": expected_foot,
        }

    def _reward_one(
        self,
        env_id,
        state,
        raw_tread,
        riser,
        lower_leg,
        event_delta,
    ):
        scales = self.training_cfg["reward_scales"]
        data = self.datas[env_id]
        command_x = float(self.commands[env_id, 0])
        velocity_x = float(state["velocity"][0])
        progress_velocity = (
            float(data.qpos[0]) - self.previous_x[env_id]
        ) / self.dt
        vertical_progress_velocity = (
            float(data.qpos[2]) - self.previous_base_z[env_id]
        ) / self.dt
        # Forward motion only counts as useful stair progress while it follows
        # the route.  The former -4*yaw^2 gate still paid 53% of directed
        # progress at the 0.40 rad acceptance boundary, which encouraged the
        # characteristic fast diagonal exit seen in v2.
        route_gate = math.exp(
            -16.0 * float(state["yaw"] ** 2)
            - 6.0 * float(data.qpos[1] ** 2)
        )
        desired_contacts, phase_sine = self._desired_contacts(env_id)
        actual_contacts = raw_tread >= 0
        phase_match = 1.0 - float(
            np.mean(
                np.abs(
                    actual_contacts.astype(np.float64)
                    - desired_contacts.astype(np.float64)
                )
            )
        )
        single_support = float(np.sum(actual_contacts) == 1)

        foot_x = np.asarray(
            [
                data.site_xpos[site_id, 0]
                for site_id in self.layout.foot_sites
            ],
            dtype=np.float64,
        )
        # Right-minus-left follows -cos(phase): the scheduled swing foot starts
        # behind, crosses at mid-swing, and lands one tread ahead.
        phase = self._phase_fraction(env_id)
        sagittal_separation = float(foot_x[1] - foot_x[0])
        target_separation = (
            -float(
                self.training_cfg["target_sagittal_separation_m"]
            )
            * math.cos(2.0 * math.pi * phase)
        )
        sagittal_error = sagittal_separation - target_separation
        sagittal_match = math.exp(
            -12.0 * sagittal_error * sagittal_error
        )
        normalized_sagittal_error = float(
            np.clip(
                sagittal_error
                / max(
                    float(
                        self.training_cfg[
                            "target_sagittal_separation_m"
                        ]
                    ),
                    1.0e-3,
                ),
                -float(
                    self.training_cfg[
                        "sagittal_foot_phase_error_clip"
                    ]
                ),
                float(
                    self.training_cfg[
                        "sagittal_foot_phase_error_clip"
                    ]
                ),
            )
        )

        swing_reference = self._swing_reference_state(
            env_id,
            state,
            desired_contacts,
            raw_tread,
            riser,
            lower_leg,
        )
        expected_foot = int(swing_reference["expected_foot"])
        if expected_foot == 0:
            swing_knee_index = self.left_knee_index
        elif expected_foot == 1:
            swing_knee_index = self.right_knee_index
        else:
            swing_knee_index = (
                self.right_knee_index
                if phase_sine >= 0.0
                else self.left_knee_index
            )
        landing_knee = (
            float(self.training_cfg["swing_knee_landing_target_rad"])
            + float(
                self.training_cfg["swing_knee_landing_height_gain"]
            )
            * float(self.stair_cfg["step_height"])
        )
        peak_knee = min(
            float(self.training_cfg["swing_knee_max_target_rad"]),
            float(self.training_cfg["swing_knee_peak_target_rad"])
            + float(self.training_cfg["swing_knee_peak_height_gain"])
            * float(self.stair_cfg["step_height"]),
        )
        flexion_arc = (
            4.0
            * float(swing_reference["progress"])
            * (1.0 - float(swing_reference["progress"]))
        )
        target_knee = landing_knee + (
            peak_knee - landing_knee
        ) * flexion_arc
        knee_position = float(state["q"][swing_knee_index])
        knee_error = knee_position - target_knee
        knee_match = (
            math.exp(
                -float(
                    self.training_cfg["swing_knee_tracking_sharpness"]
                )
                * knee_error
                * knee_error
            )
            * float(swing_reference["scheduled_active"])
        )
        knee_deficit = (
            max(target_knee - knee_position, 0.0)
            / max(target_knee, 0.10)
        ) ** 2 * float(swing_reference["scheduled_active"])
        arm_amplitude = float(
            self.training_cfg["arm_swing_amplitude_rad"]
        )
        arm_targets = np.asarray(
            [-arm_amplitude * phase_sine, arm_amplitude * phase_sine],
            dtype=np.float64,
        )
        arm_positions = state["q"][
            [
                self.left_shoulder_pitch_index,
                self.right_shoulder_pitch_index,
            ]
        ]
        arm_match = math.exp(
            -float(self.training_cfg["arm_swing_sharpness"])
            * float(np.mean(np.square(arm_positions - arm_targets)))
        )

        terrain_height = float(
            terrain_height_at_x(
                np.asarray([float(data.qpos[0])]),
                start_x=float(self.stair_cfg["start_x"]),
                step_width=float(self.stair_cfg["step_width"]),
                step_height=float(self.stair_cfg["step_height"]),
                num_steps=int(self.stair_cfg["num_steps"]),
            )[0]
        )
        initial_height = float(
            self.validation_cfg["initial_base_height"]
        )
        height_error = (
            float(data.qpos[2]) - terrain_height - initial_height
        )
        near_stairs = (
            float(data.qpos[0])
            >= float(self.stair_cfg["start_x"])
            - float(
                self.training_cfg["posture_activation_distance_m"]
            )
        )
        posture_active = (
            near_stairs
            and velocity_x > 0.03
            and bool(np.any(actual_contacts))
            and state["gravity"][2] < -0.80
        )
        target_pitch = (
            float(
                self.training_cfg["forward_pitch_base_target_rad"]
            )
            + float(self.training_cfg["forward_pitch_height_gain"])
            * float(self.stair_cfg["step_height"])
        )
        forward_pitch_score = (
            math.exp(
                -float(self.training_cfg["forward_pitch_sharpness"])
                * (float(state["pitch"]) - target_pitch) ** 2
            )
            if posture_active
            else 0.0
        )
        support_x = (
            float(np.mean(foot_x[actual_contacts]))
            if np.any(actual_contacts)
            else float(data.qpos[0])
        )
        base_behind = max(
            support_x
            - float(data.qpos[0])
            - float(
                self.training_cfg[
                    "base_support_backward_allowance_m"
                ]
            ),
            0.0,
        )
        base_behind_penalty = (
            min(
                base_behind
                / max(
                    float(
                        self.training_cfg[
                            "base_support_backward_scale_m"
                        ]
                    ),
                    1.0e-3,
                ),
                2.0,
            )
            ** 2
            if posture_active
            else 0.0
        )
        foot_pitch_errors = []
        for foot_index, site_id in enumerate(self.layout.foot_sites):
            if not actual_contacts[foot_index]:
                continue
            rotation = np.asarray(
                data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3)
            foot_pitch_errors.append(
                math.atan2(
                    -float(rotation[2, 0]),
                    math.hypot(
                        float(rotation[2, 1]),
                        float(rotation[2, 2]),
                    ),
                )
            )
        foot_pitch_penalty = (
            float(
                np.mean(
                    np.square(
                        np.clip(
                            np.asarray(foot_pitch_errors)
                            / max(
                                float(
                                    self.training_cfg[
                                        "foot_pitch_normalizer_rad"
                                    ]
                                ),
                                1.0e-3,
                            ),
                            -2.0,
                            2.0,
                        )
                    )
                )
            )
            if foot_pitch_errors
            else 0.0
        )
        action_rate = float(
            np.mean(
                np.square(
                    self.actions[env_id] - self.last_actions[env_id]
                )
            )
        )
        action_smoothness = float(
            np.mean(
                np.square(
                    self.actions[env_id]
                    - 2.0 * self.last_actions[env_id]
                    + self.last_last_actions[env_id]
                )
            )
        )
        normalized_torque = (
            self.last_torques[env_id] / self.torque_limits
        )
        normalized_energy = (
            np.abs(
                self.last_torques[env_id] * state["dq"]
            )
            / np.maximum(self.torque_limits, 1.0)
        )
        tracker = self.trackers[env_id]
        gait_active = float(
            self.episode_steps[env_id] * self.dt
            >= float(self.training_cfg["gait_reward_grace_s"])
            and velocity_x > 0.03
            and float(state["gravity"][2]) < -0.80
        )
        overspeed = (
            max(
                velocity_x
                - command_x
                - float(
                    self.training_cfg["overspeed_tolerance_m_s"]
                ),
                0.0,
            )
            if state["gravity"][2] < -0.80
            else 0.0
        )
        terms = {
            "tracking_speed": math.exp(
                -((velocity_x - command_x) / 0.08) ** 2
            ),
            "overspeed": float(
                min(
                    (
                        overspeed
                        / max(
                            float(
                                self.training_cfg[
                                    "overspeed_normalizer_m_s"
                                ]
                            ),
                            1.0e-3,
                        )
                    )
                    ** 2,
                    2.0,
                )
            ),
            "forward_progress": float(
                np.clip(progress_velocity, -0.20, 0.35)
            ),
            "directed_progress": float(
                np.clip(progress_velocity, -0.20, 0.35)
                * route_gate
            ),
            "vertical_progress": float(
                np.clip(vertical_progress_velocity, -0.25, 0.25)
            ),
            "alive": 1.0,
            "upright_error": float(
                state["gravity"][0] ** 2
                + state["gravity"][1] ** 2
            ),
            "heading_alignment": math.exp(
                -20.0 * float(state["yaw"] ** 2)
            ),
            "heading_error": float(state["yaw"] ** 2),
            "lateral_error": float(data.qpos[1] ** 2),
            "lateral_velocity": float(state["velocity"][1] ** 2),
            "yaw_rate": float(state["omega"][2] ** 2),
            "base_height_error": float(height_error ** 2),
            "forward_pitch": forward_pitch_score,
            "base_behind_support": base_behind_penalty,
            "foot_pitch": foot_pitch_penalty,
            "phase_contact": phase_match * gait_active,
            "phase_contact_mismatch": (1.0 - phase_match) * gait_active,
            "sagittal_foot_phase": sagittal_match * gait_active,
            "sagittal_foot_phase_error": (
                normalized_sagittal_error
                * normalized_sagittal_error
                * gait_active
            ),
            "single_support": single_support * gait_active,
            "swing_knee": knee_match,
            "swing_knee_deficit": knee_deficit,
            "swing_clearance": float(
                swing_reference["clearance_score"]
            ),
            "swing_clearance_deficit": float(
                swing_reference["clearance_deficit"]
            ),
            "swing_trajectory": float(swing_reference["score"])
            * float(swing_reference["active"]),
            "swing_trajectory_error": float(
                swing_reference["error"]
            )
            * float(swing_reference["active"]),
            "next_tread_target": float(
                swing_reference["next_tread_score"]
            )
            * float(swing_reference["active"]),
            "foothold_lateral": float(
                swing_reference["lateral_score"]
            )
            * float(swing_reference["active"]),
            "expected_swing_liftoff": float(
                swing_reference["expected_liftoff"]
            ),
            "expected_swing_delay": float(
                swing_reference["expected_delay"]
            ),
            "wrong_foot_swing": float(
                swing_reference["wrong_foot_swing"]
            ),
            "arm_swing": arm_match * gait_active,
            "no_progress": float(progress_velocity < 0.03),
            "double_flight": float(np.all(raw_tread < 0)),
            "same_tread_support": float(
                tracker.stable_tread[0] == tracker.stable_tread[1]
                and 0 < tracker.stable_tread[0] < tracker.num_steps
            ),
            "foot_riser_collision": float(np.any(riser)),
            "lower_leg_collision": float(np.any(lower_leg)),
            "hip_yaw": float(
                state["q"][self.left_hip_yaw_index] ** 2
                + state["q"][self.right_hip_yaw_index] ** 2
            ),
            "action_rate": action_rate,
            "action_smoothness": action_smoothness,
            "torque": float(np.mean(np.square(normalized_torque))),
            "energy": float(np.mean(normalized_energy)),
            "angular_velocity_xy": float(
                np.sum(np.square(state["omega"][:2]))
            ),
            "vertical_velocity": float(state["velocity"][2] ** 2),
            "tread_advance": float(event_delta["advance"]),
            "alternating_tread": float(event_delta["alternating"]),
            "repeated_lead": float(event_delta["repeated"]),
            "same_tread_join": float(event_delta["joined"]),
            "skipped_tread": float(event_delta["skipped"]),
            "completion": 0.0,
            "unnatural_completion": 0.0,
            "gait_completion": 0.0,
            "gait_failure": 0.0,
            "natural_completion": 0.0,
            "fall": 0.0,
            "path_failure": 0.0,
            "stall": 0.0,
        }
        self.phase_match_sum[env_id] += phase_match
        self.arm_match_sum[env_id] += arm_match
        reward = 0.0
        for name, value in terms.items():
            scaled = float(scales[name]) * float(value)
            if name not in self.EVENT_REWARD_TERMS:
                scaled *= self.dt
            self.reward_sums[name][env_id] += scaled
            reward += scaled
        return reward, terms, terrain_height, progress_velocity

    def _path_limits(self, env_id):
        level_fraction = (
            float(self.episode_levels[env_id])
            / max(1, self.final_curriculum_level)
        )
        lateral = (
            (1.0 - level_fraction)
            * float(self.curriculum_cfg["initial_corridor_half_width_m"])
            + level_fraction
            * float(self.curriculum_cfg["final_corridor_half_width_m"])
        )
        yaw = (
            (1.0 - level_fraction)
            * float(self.curriculum_cfg["initial_corridor_yaw_limit_rad"])
            + level_fraction
            * float(self.curriculum_cfg["final_corridor_yaw_limit_rad"])
        )
        if self.adaptation_stage:
            lateral = max(
                lateral,
                float(
                    self.training_cfg[
                        "adaptation_corridor_half_width_m"
                    ]
                ),
            )
            yaw = max(
                yaw,
                float(
                    self.training_cfg[
                        "adaptation_corridor_yaw_limit_rad"
                    ]
                ),
            )
        return lateral, yaw

    def _curriculum_goal_state(self, env_id, state):
        level, target_steps, target_x = self._target_for_env(env_id)
        level_fraction = level / max(1, self.final_curriculum_level)
        lateral_tolerance = (
            (1.0 - level_fraction)
            * float(
                self.curriculum_cfg[
                    "initial_completion_lateral_tolerance_m"
                ]
            )
            + level_fraction
            * float(
                self.training_cfg[
                    "completion_lateral_tolerance_m"
                ]
            )
        )
        yaw_tolerance = (
            (1.0 - level_fraction)
            * float(
                self.curriculum_cfg[
                    "initial_completion_yaw_tolerance_rad"
                ]
            )
            + level_fraction
            * float(
                self.training_cfg[
                    "completion_yaw_tolerance_rad"
                ]
            )
        )
        target_height = (
            target_steps * float(self.stair_cfg["step_height"])
        )
        target_contact_reached = (
            target_steps == 0
            or int(np.max(self.trackers[env_id].accepted_tread))
            >= target_steps
        )
        at_goal = (
            float(self.datas[env_id].qpos[0]) >= target_x
            and float(self.datas[env_id].qpos[2])
            >= float(self.validation_cfg["initial_base_height"])
            + target_height
            - float(self.curriculum_cfg["height_tolerance_m"])
            and abs(float(self.datas[env_id].qpos[1]))
            <= lateral_tolerance
            and abs(float(state["yaw"])) <= yaw_tolerance
            and target_contact_reached
        )
        self.completion_steps[env_id] = (
            self.completion_steps[env_id] + 1 if at_goal else 0
        )
        dwell_s = (
            float(self.training_cfg["completion_dwell_s"])
            if level == self.final_curriculum_level
            else float(self.curriculum_cfg["intermediate_dwell_s"])
        )
        goal_reached = (
            self.completion_steps[env_id]
            >= max(1, int(round(dwell_s / self.dt)))
        )
        gait = self.trackers[env_id].summary()
        natural_success = (
            goal_reached
            and level == self.final_curriculum_level
            and self.trackers[env_id].alternating_count
            >= int(
                self.validation_cfg[
                    "success_min_alternating_tread_count"
                ]
            )
            and gait["alternating_tread_rate"]
            >= float(
                self.curriculum_cfg[
                    "natural_min_alternating_tread_rate"
                ]
            )
            and gait["same_tread_join_rate"]
            <= float(
                self.curriculum_cfg[
                    "natural_max_same_tread_join_rate"
                ]
            )
            and gait["double_flight_fraction"]
            <= float(
                self.validation_cfg[
                    "success_max_double_flight_fraction"
                ]
            )
        )
        return goal_reached, natural_success

    def _curriculum_gait_gate_active(self, ended_level):
        if not bool(
            self.curriculum_cfg.get(
                "require_gait_for_logical_promotion", True
            )
        ):
            return False
        target_steps = int(
            self.curriculum_target_steps[int(ended_level)]
        )
        return target_steps >= int(
            self.curriculum_cfg["gait_promotion_min_target_steps"]
        )

    def _curriculum_micro_gait_gate_active(self, ended_level):
        """Return whether this is the two-step gait-discovery stage."""
        target_steps = int(
            self.curriculum_target_steps[int(ended_level)]
        )
        return target_steps == int(
            self.curriculum_cfg["gait_promotion_min_target_steps"]
        )

    def _curriculum_gait_gate_passed(self, env_id, ended_level):
        if not self._curriculum_gait_gate_active(ended_level):
            return True
        gait = self.trackers[int(env_id)].summary()
        return (
            gait["alternating_tread_rate"]
            >= float(
                self.curriculum_cfg[
                    "gait_promotion_min_alternating_tread_rate"
                ]
            )
            and gait["same_tread_join_rate"]
            <= float(
                self.curriculum_cfg[
                    "gait_promotion_max_same_tread_join_rate"
                ]
            )
            and gait["skipped_tread_rate"]
            <= float(
                self.curriculum_cfg[
                    "gait_promotion_max_skipped_tread_rate"
                ]
            )
        )

    def _advance_curriculum(
        self, env_ids, curriculum_completed, ended_levels
    ):
        required = int(
            self.curriculum_cfg["successes_before_promotion"]
        )
        for env_id, ended_level in zip(env_ids, ended_levels):
            if (
                curriculum_completed[env_id]
                and ended_level == self.mastery_levels[env_id]
                and self._curriculum_gait_gate_passed(
                    env_id, ended_level
                )
            ):
                self.curriculum_success_streak[env_id] += 1
                if (
                    self.curriculum_success_streak[env_id] >= required
                    and self.mastery_levels[env_id]
                    < self.final_curriculum_level
                ):
                    self.mastery_levels[env_id] += 1
                    self.curriculum_success_streak[env_id] = 0
            elif ended_level == self.mastery_levels[env_id]:
                self.curriculum_success_streak[env_id] = 0

    def _episode_info(
        self,
        env_ids,
        final_completed,
        curriculum_completed,
        natural_success,
        fell,
        gait_failed,
        path_failed,
        stalled,
        timed_out,
        ended_levels,
        ended_mastery,
    ):
        if not env_ids:
            return {}
        summaries = [self.trackers[index].summary() for index in env_ids]
        lengths = np.maximum(self.episode_steps[env_ids], 1)
        info = {
            "mujoco_completion_rate": final_completed[env_ids],
            "mujoco_curriculum_completion_rate": (
                curriculum_completed[env_ids]
            ),
            "mujoco_curriculum_gait_gate_active_rate": np.asarray(
                [
                    self._curriculum_gait_gate_active(level)
                    for level in ended_levels
                ],
                dtype=np.float64,
            ),
            "mujoco_curriculum_gait_pass_rate": np.asarray(
                [
                    bool(curriculum_completed[index])
                    and self._curriculum_gait_gate_passed(index, level)
                    for index, level in zip(env_ids, ended_levels)
                ],
                dtype=np.float64,
            ),
            "mujoco_natural_success_rate": natural_success[env_ids],
            "mujoco_fall_rate": fell[env_ids],
            "mujoco_gait_failure_rate": gait_failed[env_ids],
            "mujoco_path_failure_rate": path_failed[env_ids],
            "mujoco_stall_rate": stalled[env_ids],
            "mujoco_timeout_rate": timed_out[env_ids],
            "mujoco_curriculum_level": np.asarray(
                ended_levels, dtype=np.float64
            ),
            "mujoco_mastery_level": np.asarray(
                ended_mastery, dtype=np.float64
            ),
            "mujoco_gait_assistance_scale": np.full(
                len(env_ids),
                self.gait_assistance_scale,
                dtype=np.float64,
            ),
            "mujoco_forward_distance": np.asarray(
                [
                    float(self.datas[index].qpos[0])
                    for index in env_ids
                ]
            ),
            "mujoco_climb_height": self.max_climb[env_ids],
            "mujoco_mean_speed": self.speed_sum[env_ids] / lengths,
            "mujoco_phase_match": self.phase_match_sum[env_ids] / lengths,
            "mujoco_arm_swing_match": (
                self.arm_match_sum[env_ids] / lengths
            ),
            "mujoco_max_lateral": self.max_lateral[env_ids],
            "mujoco_max_yaw": self.max_yaw[env_ids],
            "mujoco_alternating_rate": np.asarray(
                [summary["alternating_tread_rate"] for summary in summaries]
            ),
            "mujoco_join_rate": np.asarray(
                [summary["same_tread_join_rate"] for summary in summaries]
            ),
            "mujoco_riser_fraction": np.asarray(
                [
                    summary["foot_riser_collision_fraction"]
                    for summary in summaries
                ]
            ),
            "mujoco_total_reward": self.episode_reward[env_ids],
        }
        for name, values in self.reward_sums.items():
            info["mujoco_rew_" + name] = values[env_ids]
        return {
            name: torch.as_tensor(
                np.asarray(value, dtype=np.float32).reshape(-1)
            )
            for name, value in info.items()
        }

    def step(self, actions):
        clipped = torch.clamp(
            actions.detach().to("cpu"),
            -self.clip_actions,
            self.clip_actions,
        )
        self.actions[:] = clipped.numpy()
        self._prepare_executed_actions()
        targets = (
            self.default_angles[None, :]
            + self.action_scale * self.executed_actions
        )
        self._simulate(targets)
        self.episode_steps += 1
        self.episode_length_buf += 1

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        curriculum_completed = np.zeros(self.num_envs, dtype=bool)
        final_completed = np.zeros(self.num_envs, dtype=bool)
        natural_success = np.zeros(self.num_envs, dtype=bool)
        fell = np.zeros(self.num_envs, dtype=bool)
        gait_failed = np.zeros(self.num_envs, dtype=bool)
        path_failed = np.zeros(self.num_envs, dtype=bool)
        stalled = np.zeros(self.num_envs, dtype=bool)
        timed_out = np.zeros(self.num_envs, dtype=bool)
        numerical = np.zeros(self.num_envs, dtype=bool)

        for env_id in range(self.num_envs):
            data = self.datas[env_id]
            if not (
                np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
            ):
                numerical[env_id] = True
                fell[env_id] = True
                rewards[env_id] = -50.0
                continue

            state = self._state(env_id)
            raw_tread, riser, lower_leg = sample_contacts(
                self.model,
                data,
                self.layout,
                self.validation_cfg,
            )
            tracker = self.trackers[env_id]
            previous_counts = {
                "advance": tracker.advance_count,
                "alternating": tracker.alternating_count,
                "repeated": tracker.repeated_count,
                "joined": tracker.join_count,
                "skipped": tracker.skipped_count,
            }
            foot_site_z = np.asarray(
                [
                    data.site_xpos[site_id, 2]
                    for site_id in self.layout.foot_sites
                ],
                dtype=np.float64,
            )
            tracker.update(
                raw_tread, foot_site_z, riser, lower_leg
            )
            current_counts = {
                "advance": tracker.advance_count,
                "alternating": tracker.alternating_count,
                "repeated": tracker.repeated_count,
                "joined": tracker.join_count,
                "skipped": tracker.skipped_count,
            }
            event_delta = {
                name: int(current_counts[name]) - previous
                for name, previous in previous_counts.items()
            }
            # At the two-step discovery stage, a step-to join, repeated lead,
            # skipped tread, or simultaneous hop is an immediate failed
            # attempt.  Soft terminal scoring left all 32 v9 environments in
            # the step-to local optimum: they could collect completion reward
            # first and tolerate the delayed gait penalty.  The first proper
            # landing is classified as alternating, so advance > alternating
            # precisely catches every non-alternating advance event here.
            gait_failed[env_id] = (
                self._curriculum_micro_gait_gate_active(
                    self.episode_levels[env_id]
                )
                and (
                    event_delta["joined"] > 0
                    or event_delta["repeated"] > 0
                    or event_delta["skipped"] > 0
                    or event_delta["advance"]
                    > event_delta["alternating"]
                )
            )
            if event_delta["advance"] > 0:
                self._synchronize_phase_after_advance(env_id)
            reward, _, terrain_height, progress_velocity = (
                self._reward_one(
                    env_id,
                    state,
                    raw_tread,
                    riser,
                    lower_leg,
                    event_delta,
                )
            )

            elapsed = self.episode_steps[env_id] * self.dt
            base_clearance = float(data.qpos[2]) - terrain_height
            fell[env_id] = (
                base_clearance
                < float(self.validation_cfg["fall_base_height"])
                or state["gravity"][2]
                > float(self.validation_cfg["fall_upright_z"])
            )
            lateral_limit, yaw_limit = self._path_limits(env_id)
            outside_path = (
                elapsed
                >= float(self.training_cfg["progress_grace_s"])
                and (
                    abs(float(data.qpos[1])) > lateral_limit
                    or abs(float(state["yaw"])) > yaw_limit
                )
            )
            self.path_violation_steps[env_id] = (
                self.path_violation_steps[env_id] + 1
                if outside_path
                else 0
            )
            path_failed[env_id] = (
                self.path_violation_steps[env_id]
                >= max(
                    1,
                    int(
                        round(
                            float(
                                self.curriculum_cfg[
                                    "path_violation_dwell_s"
                                ]
                            )
                            / self.dt
                        )
                    ),
                )
            )

            current_x = float(data.qpos[0])
            if (
                current_x - self.progress_checkpoint[env_id]
                >= float(self.training_cfg["progress_epsilon_m"])
            ):
                self.progress_checkpoint[env_id] = current_x
                self.last_progress_step[env_id] = self.episode_steps[env_id]
            progress_timeout_steps = int(
                float(self.training_cfg["progress_timeout_s"]) / self.dt
            )
            grace_steps = int(
                float(self.training_cfg["progress_grace_s"]) / self.dt
            )
            stalled[env_id] = (
                self.episode_steps[env_id] >= grace_steps
                and self.episode_steps[env_id]
                - self.last_progress_step[env_id]
                >= progress_timeout_steps
            )

            (
                curriculum_completed[env_id],
                natural_success[env_id],
            ) = self._curriculum_goal_state(env_id, state)
            final_completed[env_id] = (
                curriculum_completed[env_id]
                and self.episode_levels[env_id]
                == self.final_curriculum_level
            )
            timed_out[env_id] = (
                self.episode_steps[env_id] >= self.max_episode_length
            )

            # Assign exactly one terminal reason. A physical fall wins over
            # all; at the two-step discovery stage a gait violation must win
            # over physical completion so step-to motion cannot collect its
            # former completion shortcut.
            if fell[env_id]:
                curriculum_completed[env_id] = False
                final_completed[env_id] = False
                natural_success[env_id] = False
                gait_failed[env_id] = False
                path_failed[env_id] = False
                stalled[env_id] = False
                timed_out[env_id] = False
            elif gait_failed[env_id]:
                curriculum_completed[env_id] = False
                final_completed[env_id] = False
                natural_success[env_id] = False
                path_failed[env_id] = False
                stalled[env_id] = False
                timed_out[env_id] = False
            elif curriculum_completed[env_id]:
                path_failed[env_id] = False
                stalled[env_id] = False
                timed_out[env_id] = False
            elif path_failed[env_id]:
                stalled[env_id] = False
                timed_out[env_id] = False
            elif stalled[env_id]:
                timed_out[env_id] = False

            scales = self.training_cfg["reward_scales"]
            gait_gate_active = self._curriculum_gait_gate_active(
                self.episode_levels[env_id]
            )
            gait_gate_passed = self._curriculum_gait_gate_passed(
                env_id, self.episode_levels[env_id]
            )
            unnatural_completion = (
                curriculum_completed[env_id]
                and gait_gate_active
                and not gait_gate_passed
            )
            gait_completion = (
                curriculum_completed[env_id]
                and gait_gate_active
                and gait_gate_passed
            )
            terminal_terms = {
                # Reaching the physical goal must always beat deliberate
                # falling.  The old qualified-only reward made an unnatural
                # completion (-35) worse than a fall (-25), so PPO learned to
                # stop or fall instead of converting its gait.
                "completion": float(curriculum_completed[env_id]),
                "unnatural_completion": float(unnatural_completion),
                "gait_completion": float(gait_completion),
                "gait_failure": float(gait_failed[env_id]),
                "natural_completion": float(natural_success[env_id]),
                "fall": float(fell[env_id]),
                "path_failure": float(path_failed[env_id]),
                "stall": float(stalled[env_id]),
            }
            for name, value in terminal_terms.items():
                scaled = float(scales[name]) * value
                reward += scaled
                self.reward_sums[name][env_id] += scaled

            rewards[env_id] = reward
            self.episode_reward[env_id] += reward
            self.speed_sum[env_id] += float(state["velocity"][0])
            self.max_lateral[env_id] = max(
                self.max_lateral[env_id], abs(float(data.qpos[1]))
            )
            self.max_yaw[env_id] = max(
                self.max_yaw[env_id], abs(float(state["yaw"]))
            )
            self.max_climb[env_id] = max(
                self.max_climb[env_id],
                float(data.qpos[2])
                - float(self.validation_cfg["initial_base_height"]),
            )
            self.previous_x[env_id] = current_x
            self.previous_base_z[env_id] = float(data.qpos[2])
            self.obs_buf[env_id] = torch.from_numpy(
                self._observation(env_id, state).copy()
            )
            self._update_privileged_observation(env_id)

        dones = (
            curriculum_completed
            | fell
            | gait_failed
            | path_failed
            | stalled
            | timed_out
            | numerical
        )
        pure_time_outs = (
            timed_out
            & ~curriculum_completed
            & ~fell
            & ~gait_failed
            & ~path_failed
            & ~stalled
            & ~numerical
        )
        done_ids = np.flatnonzero(dones).tolist()
        ended_levels = self.episode_levels[done_ids].copy()
        ended_mastery = self.mastery_levels[done_ids].copy()
        episode_info = self._episode_info(
            done_ids,
            final_completed,
            curriculum_completed,
            natural_success,
            fell,
            gait_failed,
            path_failed,
            stalled,
            pure_time_outs,
            ended_levels,
            ended_mastery,
        )
        self.rew_buf = torch.from_numpy(rewards)
        self.reset_buf = torch.from_numpy(dones)
        infos = {
            "time_outs": torch.from_numpy(
                pure_time_outs.astype(np.float32)
            )
        }
        if episode_info:
            infos["episode"] = episode_info

        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        if done_ids:
            self._advance_curriculum(
                done_ids, curriculum_completed, ended_levels
            )
            self.reset(torch.tensor(done_ids, dtype=torch.long))
        termination_ids = torch.tensor(done_ids, dtype=torch.long)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            infos,
            termination_ids,
            None,
        )

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def curriculum_summary(self):
        return {
            "mean_mastery_level": float(np.mean(self.mastery_levels)),
            "max_mastery_level": int(np.max(self.mastery_levels)),
            "final_level_fraction": float(
                np.mean(
                    self.mastery_levels >= self.final_curriculum_level
                )
            ),
            "level_histogram": np.bincount(
                self.mastery_levels,
                minlength=self.final_curriculum_level + 1,
            ).tolist(),
            "physical_height_index": int(self.physical_height_index),
            "physical_step_height_m": self.physical_step_height,
            "physical_promotion_streak": int(
                self.physical_promotion_streak
            ),
            "physical_curriculum_complete": bool(
                self.physical_curriculum_complete
            ),
        }

    def get_checkpoint_state(self):
        return {
            "version": 13,
            "mastery_levels": self.mastery_levels.copy(),
            "curriculum_success_streak": (
                self.curriculum_success_streak.copy()
            ),
            "physical_height_index": int(self.physical_height_index),
            "physical_step_height_m": self.physical_step_height,
            "physical_promotion_streak": int(
                self.physical_promotion_streak
            ),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_checkpoint_state(self, state):
        state_version = int(state.get("version", -1))
        if state_version not in (
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        ):
            raise ValueError("Unsupported MuJoCo curriculum state")
        mastery = np.asarray(
            state["mastery_levels"], dtype=np.int64
        )
        streak = np.asarray(
            state["curriculum_success_streak"], dtype=np.int64
        )
        if mastery.shape != (self.num_envs,) or streak.shape != (
            self.num_envs,
        ):
            raise ValueError(
                "MuJoCo curriculum checkpoint environment count differs"
            )
        saved_height = state.get("physical_step_height_m")
        if saved_height is None:
            if state_version >= 13:
                raise ValueError(
                    "MuJoCo checkpoint physical height is missing"
                )
            legacy_index = int(state["physical_height_index"])
            if not 0 <= legacy_index < len(
                self.LEGACY_PHYSICAL_STEP_HEIGHTS
            ):
                raise ValueError(
                    "Legacy MuJoCo checkpoint height index is invalid"
                )
            saved_height = self.LEGACY_PHYSICAL_STEP_HEIGHTS[
                legacy_index
            ]
        matches = np.flatnonzero(
            np.isclose(
                self.physical_step_heights,
                float(saved_height),
                rtol=0.0,
                atol=1.0e-9,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                "MuJoCo checkpoint height is absent from this curriculum"
            )
        self.physical_height_index = int(matches[0])
        self.physical_promotion_streak = max(
            0, int(state.get("physical_promotion_streak", 0))
        )
        self._apply_physical_step_height(self.physical_step_height)
        self.mastery_levels[:] = np.clip(
            mastery, 0, self.final_curriculum_level
        )
        self.curriculum_success_streak[:] = np.maximum(streak, 0)
        self.rng.bit_generator.state = state["rng_state"]
        # The simulator states intentionally reset on resume, but every
        # environment resumes from its saved curriculum mastery.
        self.reset()
        print(
            "Restored MuJoCo curriculum: " + str(self.curriculum_summary()),
            flush=True,
        )

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
