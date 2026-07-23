"""Parallel, headless MuJoCo environment for native N2 stair PPO."""

import math
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

from humanoid.algo import VecEnv

from eval_stairs_mujoco import (
    ContactLayout,
    GaitTracker,
    _build_observation,
    _configure_solver,
    sample_contacts,
    terrain_height_at_x,
)
from sim2sim import load_mujoco_model, pd_control, resolve_joint_layout


class MujocoStairsVecEnv(VecEnv):
    """Independent data sharing a model resized only between rollout chunks."""

    EVENT_REWARD_TERMS = {
        "tread_advance",
        "alternating_tread",
        "repeated_lead",
        "same_tread_join",
        "skipped_tread",
        "completion",
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

    def _physical_gate_passed(self, summary, gate):
        expected_climb = (
            int(self.stair_cfg["num_steps"]) * self.physical_step_height
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
            >= float(gate["min_climb_fraction"]) * expected_climb
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
            "NATIVE_MUJOCO_HEIGHT_PROMOTION {:.2f}m -> {:.2f}m".format(
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
        phase_adjusted_time = (
            elapsed + self.phase_offsets[env_id] / frequency
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
        self.last_actions[env_id] = 0.0
        self.last_last_actions[env_id] = 0.0
        self.last_torques[env_id] = 0.0
        self.history[env_id] = 0.0
        self.trackers[env_id] = self._new_tracker()
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
        sagittal_separation = float(foot_x[0] - foot_x[1])
        target_separation = (
            -float(
                self.training_cfg["target_sagittal_separation_m"]
            )
            * phase_sine
        )
        sagittal_match = math.exp(
            -22.0 * (sagittal_separation - target_separation) ** 2
        )

        swing_knee_index = (
            self.right_knee_index
            if phase_sine >= 0.0
            else self.left_knee_index
        )
        knee_error = (
            float(state["q"][swing_knee_index])
            - float(self.training_cfg["target_swing_knee_rad"])
        )
        knee_match = math.exp(-4.0 * knee_error * knee_error)
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
        terms = {
            "tracking_speed": math.exp(
                -((velocity_x - command_x) / 0.08) ** 2
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
            "phase_contact": phase_match,
            "sagittal_foot_phase": sagittal_match,
            "single_support": single_support,
            "swing_knee": knee_match,
            "arm_swing": arm_match,
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
            "mujoco_natural_success_rate": natural_success[env_ids],
            "mujoco_fall_rate": fell[env_ids],
            "mujoco_path_failure_rate": path_failed[env_ids],
            "mujoco_stall_rate": stalled[env_ids],
            "mujoco_timeout_rate": timed_out[env_ids],
            "mujoco_curriculum_level": np.asarray(
                ended_levels, dtype=np.float64
            ),
            "mujoco_mastery_level": np.asarray(
                ended_mastery, dtype=np.float64
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
        targets = (
            self.default_angles[None, :]
            + self.action_scale * self.actions
        )
        self._simulate(targets)
        self.episode_steps += 1
        self.episode_length_buf += 1

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        curriculum_completed = np.zeros(self.num_envs, dtype=bool)
        final_completed = np.zeros(self.num_envs, dtype=bool)
        natural_success = np.zeros(self.num_envs, dtype=bool)
        fell = np.zeros(self.num_envs, dtype=bool)
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

            # Assign exactly one terminal reason. A completed goal wins over
            # route/stall/timeout gates, while a physical fall wins over all.
            if fell[env_id]:
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
            terminal_terms = {
                "completion": float(curriculum_completed[env_id]),
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
            | path_failed
            | stalled
            | timed_out
            | numerical
        )
        pure_time_outs = (
            timed_out
            & ~curriculum_completed
            & ~fell
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
            "version": 4,
            "mastery_levels": self.mastery_levels.copy(),
            "curriculum_success_streak": (
                self.curriculum_success_streak.copy()
            ),
            "physical_height_index": int(self.physical_height_index),
            "physical_promotion_streak": int(
                self.physical_promotion_streak
            ),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_checkpoint_state(self, state):
        if int(state.get("version", -1)) != 4:
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
        physical_height_index = int(state["physical_height_index"])
        if not 0 <= physical_height_index < len(
            self.physical_step_heights
        ):
            raise ValueError(
                "MuJoCo checkpoint physical height index is invalid"
            )
        self.physical_height_index = physical_height_index
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
