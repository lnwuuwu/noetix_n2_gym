"""N2 environment specialized for stable, directional upstairs locomotion."""

from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_rotate_inverse,
    torch_rand_float,
)
import torch

from humanoid.envs.n2.n2_env import N2Env
from humanoid.utils.stairs_terrain import (
    classify_tread_transition,
    retained_swing_support_mask,
    same_tread_support_mask,
    select_height_indices,
    smooth_swing_trajectory,
    stable_tread_advance_mask,
    true_airborne_mask,
)
from humanoid.utils.terrain import N2StairsTerrain


class N2StairsEnv(N2Env):
    """N2 task with verified stair geometry, climb curriculum, and metrics."""

    _BASE_PROPRIO_OBS = 63

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        expected_order = list(self.cfg.asset.expected_dof_order)
        actual_order = list(self.dof_names)
        if actual_order != expected_order:
            raise RuntimeError(
                "Isaac Gym N2 DOF order differs from the policy/PD order.\n"
                "Expected: {}\nActual: {}".format(expected_order, actual_order)
            )
        self._build_mirror_layout(actual_order)
        # N2Env historically used numeric indices based on a different joint
        # ordering. Resolve every semantic group by name for this task.
        dof_index = {name: index for index, name in enumerate(actual_order)}
        self.left_yaw_roll = [
            dof_index["L_leg_hip_yaw_joint"],
            dof_index["L_leg_hip_roll_joint"],
        ]
        self.right_yaw_roll = [
            dof_index["R_leg_hip_yaw_joint"],
            dof_index["R_leg_hip_roll_joint"],
        ]
        self.leg_alignment_idxs = self.left_yaw_roll + self.right_yaw_roll
        self.ankle_dof_idxs = [
            dof_index["L_leg_ankle_joint"],
            dof_index["R_leg_ankle_joint"],
        ]
        self.knee_dof_idxs = [
            dof_index["L_leg_knee_joint"],
            dof_index["R_leg_knee_joint"],
        ]
        self.shoulder_pitch_dof_idxs = [
            dof_index["L_arm_shoulder_pitch_joint"],
            dof_index["R_arm_shoulder_pitch_joint"],
        ]
        self.up_joint_idxs = [
            dof_index[name] for name in actual_order if "_arm_" in name
        ]

    def _build_mirror_layout(self, joint_order):
        """Build the exact left/right reflection for the stacked Actor input."""
        sign_by_suffix = {
            "arm_shoulder_pitch_joint": 1.0,
            "arm_shoulder_roll_joint": -1.0,
            "arm_shoulder_yaw_joint": -1.0,
            # Both elbow axes are +Y in mirrored child frames.  URDF forward
            # kinematics therefore maps q_L -> q_R, not q_L -> -q_R.
            "arm_elbow_joint": 1.0,
            "leg_hip_yaw_joint": -1.0,
            "leg_hip_roll_joint": -1.0,
            "leg_hip_pitch_joint": 1.0,
            "leg_knee_joint": 1.0,
            "leg_ankle_joint": 1.0,
        }
        joint_index = {name: index for index, name in enumerate(joint_order)}
        action_source = [0] * self.num_actions
        action_sign = [1.0] * self.num_actions
        for index, name in enumerate(joint_order):
            if not (name.startswith("L_") or name.startswith("R_")):
                raise ValueError("Cannot mirror unpaired N2 joint " + name)
            suffix = name[2:]
            opposite = ("R_" if name.startswith("L_") else "L_") + suffix
            if suffix not in sign_by_suffix or opposite not in joint_index:
                raise ValueError("Cannot mirror N2 joint " + name)
            action_source[index] = joint_index[opposite]
            action_sign[index] = sign_by_suffix[suffix]
        self.mirror_action_source = torch.tensor(
            action_source, dtype=torch.long
        )
        self.mirror_action_sign = torch.tensor(
            action_sign, dtype=torch.float
        )

        source = list(range(self.cfg.env.num_single_obs))
        signs = [1.0] * self.cfg.env.num_single_obs
        cursor = 0
        # command x/y/yaw
        signs[cursor:cursor + 3] = [1.0, -1.0, -1.0]
        cursor += 3
        if self.include_gait_phase:
            # Swapping left/right advances the gait clock by half a cycle.
            signs[cursor:cursor + 2] = [-1.0, -1.0]
            cursor += 2
        if self.include_base_lin_vel:
            signs[cursor:cursor + 3] = [1.0, -1.0, 1.0]
            cursor += 3
        if self.include_navigation_state:
            signs[cursor:cursor + 2] = [-1.0, -1.0]
            cursor += 2
        # Angular velocity is axial; projected gravity is polar.
        signs[cursor:cursor + 3] = [-1.0, 1.0, -1.0]
        cursor += 3
        signs[cursor:cursor + 3] = [1.0, -1.0, 1.0]
        cursor += 3
        for _ in range(3):
            for output, input_index in enumerate(action_source):
                source[cursor + output] = cursor + input_index
                signs[cursor + output] = action_sign[output]
            cursor += self.num_actions

        points_x = list(self.cfg.terrain.actor_measured_points_x)
        points_y = list(self.cfg.terrain.actor_measured_points_y)
        for x_index in range(len(points_x)):
            for y_index, point_y in enumerate(points_y):
                candidates = [
                    index for index, value in enumerate(points_y)
                    if abs(value + point_y) <= 1.0e-6
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Actor height Y point has no unique mirror: {}".format(
                            point_y
                        )
                    )
                output = cursor + x_index * len(points_y) + y_index
                source[output] = (
                    cursor + x_index * len(points_y) + candidates[0]
                )
        height_count = len(points_x) * len(points_y)
        if cursor + height_count != self.cfg.env.num_single_obs:
            raise ValueError(
                "Mirror layout {} + {} != {}".format(
                    cursor, height_count, self.cfg.env.num_single_obs
                )
            )
        self.mirror_single_source = torch.tensor(source, dtype=torch.long)
        self.mirror_single_sign = torch.tensor(signs, dtype=torch.float)

    def mirror_actions(self, actions):
        source = self.mirror_action_source.to(device=actions.device)
        signs = self.mirror_action_sign.to(
            device=actions.device, dtype=actions.dtype
        )
        return actions.index_select(-1, source) * signs

    def mirror_observations(self, observations):
        expected_width = int(self.cfg.env.num_observations)
        if observations.shape[-1] != expected_width:
            raise ValueError(
                "N2 Actor mirror expects {} observations, received {}".format(
                    expected_width, observations.shape[-1]
                )
            )
        shape = observations.shape
        frames = observations.reshape(
            *shape[:-1],
            int(self.cfg.env.frame_stack),
            int(self.cfg.env.num_single_obs),
        )
        source = self.mirror_single_source.to(device=observations.device)
        signs = self.mirror_single_sign.to(
            device=observations.device, dtype=observations.dtype
        )
        return (
            frames.index_select(-1, source) * signs
        ).reshape(shape)

    def create_sim(self):
        """Create the directional stair generator instead of mixed HumanoidTerrain."""
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return super().create_sim()

        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.terrain = N2StairsTerrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "heightfield":
            self._create_heightfield()
        else:
            self._create_trimesh()
        self._create_envs()

    def _get_env_origins(self):
        super()._get_env_origins()
        fixed_level = int(getattr(self.cfg.terrain, "fixed_level", -1))
        level_mix = list(getattr(self.cfg.terrain, "level_mix", []))
        if level_mix:
            if any(
                level < 0 or level >= self.max_terrain_level
                for level in level_mix
            ):
                raise ValueError(
                    "mixed stair levels must all be inside [0, {}]".format(
                        self.max_terrain_level - 1
                    )
                )
            mixed_levels = torch.tensor(
                level_mix,
                dtype=self.terrain_levels.dtype,
                device=self.device,
            )
            indices = (
                torch.arange(self.num_envs, device=self.device)
                % len(mixed_levels)
            )
            self.terrain_levels[:] = mixed_levels[indices]
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        elif fixed_level >= 0:
            if fixed_level >= self.max_terrain_level:
                raise ValueError(
                    "fixed stair level {} is outside [0, {}]".format(
                        fixed_level, self.max_terrain_level - 1
                    )
                )
            self.terrain_levels[:] = fixed_level
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]

    def _init_buffers(self):
        super()._init_buffers()
        self.include_gait_phase = bool(
            getattr(self.cfg.env, "include_gait_phase", False)
        )
        self.enforce_walk_gait = bool(
            getattr(self.cfg.env, "enforce_walk_gait", False)
        )
        self.include_base_lin_vel = bool(
            getattr(self.cfg.env, "include_base_lin_vel", False)
        )
        self.include_navigation_state = bool(
            getattr(self.cfg.env, "include_navigation_state", False)
        )
        self.proprio_obs_size = self._BASE_PROPRIO_OBS
        self.proprio_obs_size += 2 if self.include_gait_phase else 0
        self.proprio_obs_size += 3 if self.include_base_lin_vel else 0
        self.proprio_obs_size += 2 if self.include_navigation_state else 0
        if len(self.feet_indices) != 2:
            raise RuntimeError(
                "n2_stairs expects exactly two ankle/foot bodies, found {}".format(
                    len(self.feet_indices)
                )
            )
        foot_body_names = [
            self.body_names[index] for index in self.feet_indices.tolist()
        ]
        if not (
            foot_body_names[0].startswith("L_")
            and foot_body_names[1].startswith("R_")
        ):
            raise RuntimeError(
                "n2_stairs expects [left, right] foot order, found {}".format(
                    foot_body_names
                )
            )
        lower_leg_names = ["L_leg_knee_link", "R_leg_knee_link"]
        missing_lower_legs = [
            name for name in lower_leg_names if name not in self.body_names
        ]
        if missing_lower_legs:
            raise RuntimeError(
                "n2_stairs cannot resolve lower-leg bodies: {}".format(
                    missing_lower_legs
                )
            )
        self.lower_leg_indices = torch.tensor(
            [self.body_names.index(name) for name in lower_leg_names],
            dtype=torch.long,
            device=self.device,
        )
        self.contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.stable_contacts = torch.zeros_like(self.contacts)
        self.last_stable_contacts = torch.zeros_like(self.contacts)
        self.stable_contact_candidate_time = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.stable_contact_loss_time = torch.zeros_like(
            self.stable_contact_candidate_time
        )
        self.current_foot_tread = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.long,
            device=self.device,
        )
        self.accepted_foot_tread = torch.zeros_like(self.current_foot_tread)
        self.same_tread_support = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.swing_active = torch.zeros_like(self.contacts)
        self.swing_true_airborne_seen = torch.zeros_like(self.contacts)
        self.swing_opposite_support_valid = torch.zeros_like(self.contacts)
        self.swing_start_valid = torch.zeros_like(self.contacts)
        self.swing_start_pos = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            3,
            dtype=torch.float,
            device=self.device,
        )
        self.swing_elapsed_time = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.swing_pending_time = torch.zeros_like(
            self.swing_elapsed_time
        )
        self.max_swing_duration = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.foot_surface_offset = torch.full(
            (self.num_envs, len(self.feet_indices)),
            float(getattr(self.cfg.env, "nominal_foot_surface_offset", 0.045)),
            dtype=torch.float,
            device=self.device,
        )

        actor_indices = select_height_indices(
            self.cfg.terrain.measured_points_x,
            self.cfg.terrain.measured_points_y,
            self.cfg.terrain.actor_measured_points_x,
            self.cfg.terrain.actor_measured_points_y,
        )
        self.actor_height_indices = torch.tensor(
            actor_indices, dtype=torch.long, device=self.device
        )
        expected_actor_heights = (
            self.cfg.env.num_single_obs - self.proprio_obs_size
        )
        if len(actor_indices) != expected_actor_heights:
            raise ValueError(
                "n2_stairs Actor height count {} does not match observation config {}".format(
                    len(actor_indices), expected_actor_heights
                )
            )

        self.stair_top_x = torch.tensor(
            self.terrain.stair_top_x, dtype=torch.float, device=self.device
        )
        self.stair_start_x = torch.tensor(
            self.terrain.stair_start_x, dtype=torch.float, device=self.device
        )
        self.stair_top_heights = torch.tensor(
            self.terrain.stair_top_heights, dtype=torch.float, device=self.device
        )
        self.stair_step_heights = torch.tensor(
            self.terrain.stair_step_heights, dtype=torch.float, device=self.device
        )

        self.best_forward_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.progress_checkpoint = torch.zeros_like(self.best_forward_progress)
        self.max_climb_height = torch.zeros_like(self.best_forward_progress)
        self.last_progress_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terrain_height_delta = torch.zeros_like(self.best_forward_progress)
        self.double_flight_time = torch.zeros_like(self.best_forward_progress)
        self.foot_contact_height_delta = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.max_foot_contact_height = torch.zeros_like(
            self.foot_contact_height_delta
        )
        # Strict walking tracks which foot most recently advanced to a new
        # tread. A phase clock alone cannot distinguish stair-over-stair gait
        # from a step-to pattern in which the same lead foot always advances.
        self.last_advanced_tread = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_advanced_foot = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # A trailing foot may join an advanced tread at most once. Without
        # this memory, every later stride on the flat top platform is
        # repeatedly mislabeled as another step-to join.
        self.last_joined_tread = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.alternating_tread_event = torch.zeros_like(
            self.best_forward_progress
        )
        self.repeated_lead_event = torch.zeros_like(
            self.best_forward_progress
        )
        self.same_tread_join_event = torch.zeros_like(
            self.best_forward_progress
        )
        self.skipped_tread_event = torch.zeros_like(
            self.best_forward_progress
        )
        self.tread_advance_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.alternating_tread_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.repeated_lead_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.same_tread_join_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.skipped_tread_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.gait_phase_offset = torch.zeros_like(self.best_forward_progress)
        self.episode_gait_frequency = torch.zeros_like(
            self.best_forward_progress
        )
        # Fresh policies use the tread-matched clock immediately. Loading a
        # legacy checkpoint without saved transition state enables this flag.
        self.gait_frequency_transition_active = False
        self.desired_contacts = torch.ones(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.phase_contact_match = torch.ones_like(self.best_forward_progress)
        self.top_stable_time = torch.zeros_like(self.best_forward_progress)
        self.path_failure_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.forward_speed_sum = torch.zeros_like(self.best_forward_progress)
        self.command_error_sum = torch.zeros_like(self.best_forward_progress)
        self.phase_match_sum = torch.zeros_like(self.best_forward_progress)
        self.sagittal_foot_phase_match_sum = torch.zeros_like(
            self.best_forward_progress
        )
        self.double_flight_step_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.max_lateral_deviation = torch.zeros_like(
            self.best_forward_progress
        )
        self.max_yaw_deviation = torch.zeros_like(self.best_forward_progress)
        self.max_sagittal_foot_separation = torch.zeros_like(
            self.best_forward_progress
        )
        self.swing_knee_flexion_sum = torch.zeros_like(
            self.best_forward_progress
        )
        self.swing_knee_sample_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.arm_swing_match_sum = torch.zeros_like(
            self.best_forward_progress
        )
        self.same_tread_support_step_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.lower_leg_collision_step_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.foot_riser_collision_step_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.swing_timeout_step_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.base_behind_support_sum = torch.zeros_like(
            self.best_forward_progress
        )
        self.foot_pitch_error_sum = torch.zeros_like(
            self.best_forward_progress
        )
        self.foot_pitch_sample_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.left_tread_advance_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.right_tread_advance_count = torch.zeros_like(
            self.best_forward_progress
        )
        # Measure actual forward displacement from physical lift-off to
        # confirmed touchdown for each foot. Tread counts alone cannot reveal
        # the visually obvious case where one leg consistently takes a longer
        # step than the other.
        self.last_swing_forward_displacement = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
        )
        self.swing_displacement_valid = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.swing_forward_displacement_sum = torch.zeros_like(
            self.last_swing_forward_displacement
        )
        self.swing_forward_displacement_count = torch.zeros_like(
            self.last_swing_forward_displacement
        )
        # Track each foot independently in world Y.  Pelvis-only corridor
        # metrics cannot identify an inward right-foot placement that later
        # pushes the whole robot to the left.
        self.foot_lateral_inward_error_sum = torch.zeros_like(
            self.last_swing_forward_displacement
        )
        self.foot_lateral_position_sum = torch.zeros_like(
            self.last_swing_forward_displacement
        )
        self.foot_lateral_sample_count = torch.zeros_like(
            self.last_swing_forward_displacement
        )
        self.top_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.top_position_reached_buf = torch.zeros_like(self.top_reached_buf)
        self.completion_buf = torch.zeros_like(self.top_reached_buf)
        self.curriculum_completion_buf = torch.zeros_like(
            self.top_reached_buf
        )
        self.completion_stable_time = torch.zeros_like(
            self.best_forward_progress
        )
        self.stall_buf = torch.zeros_like(self.top_reached_buf)
        self.fall_event_buf = torch.zeros_like(self.top_reached_buf)
        self.episode_started = torch.zeros_like(self.top_reached_buf)

        self.curriculum_success_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.curriculum_failure_streak = torch.zeros_like(
            self.curriculum_success_streak
        )

        # Persistent per-environment results are consumed by eval_stairs.py
        # after reset_idx() has already placed the robot in its next episode.
        self.last_episode_success = torch.zeros_like(self.best_forward_progress)
        self.last_episode_completion = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_curriculum_completion = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_top_reached = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_forward_progress = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_climb_height = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_survival_time = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_fall = torch.zeros_like(self.best_forward_progress)
        self.last_episode_stall = torch.zeros_like(self.best_forward_progress)
        self.last_episode_first_step = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_max_foot_height = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_forward_speed = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_command_error = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_phase_contact_match = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_sagittal_foot_phase_match = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_double_flight_fraction = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_max_lateral_deviation = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_max_yaw_deviation = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_path_failure = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_alternating_tread_count = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_alternating_tread_rate = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_repeated_lead_rate = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_same_tread_join_rate = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_skipped_tread_rate = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_max_sagittal_foot_separation = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_swing_knee_flexion = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_arm_swing_match = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_gait_frequency = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_same_tread_support_fraction = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_lower_leg_collision_fraction = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_foot_riser_collision_fraction = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_swing_timeout_fraction = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_base_behind_support = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_foot_pitch_error = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_max_swing_duration = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_left_tread_advances = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_right_tread_advances = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_final_lateral_position = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_left_swing_length = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_right_swing_length = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_left_foot_inward_error = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_right_foot_inward_error = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_left_foot_lateral_position = torch.zeros_like(
            self.best_forward_progress
        )
        self.last_episode_mean_right_foot_lateral_position = torch.zeros_like(
            self.best_forward_progress
        )

    def _get_noise_scale_vec(self, cfg):
        """Noise layout for commands, optional phase, proprioception, and terrain."""
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        self.add_noise = cfg.noise.add_noise
        scales = cfg.noise.noise_scales

        cursor = 0
        noise_vec[cursor:cursor + 3] = 0.0
        cursor += 3
        if bool(getattr(cfg.env, "include_gait_phase", False)):
            noise_vec[cursor:cursor + 2] = 0.0
            cursor += 2
        if bool(getattr(cfg.env, "include_base_lin_vel", False)):
            noise_vec[cursor:cursor + 3] = (
                scales.lin_vel * self.obs_scales.lin_vel
            )
            cursor += 3
        if bool(getattr(cfg.env, "include_navigation_state", False)):
            noise_vec[cursor:cursor + 2] = 0.0
            cursor += 2
        noise_vec[cursor:cursor + 3] = scales.ang_vel * self.obs_scales.ang_vel
        cursor += 3
        noise_vec[cursor:cursor + 3] = scales.gravity
        cursor += 3
        noise_vec[cursor:cursor + self.num_actions] = (
            scales.dof_pos * self.obs_scales.dof_pos
        )
        cursor += self.num_actions
        noise_vec[cursor:cursor + self.num_actions] = (
            scales.dof_vel * self.obs_scales.dof_vel
        )
        cursor += self.num_actions
        noise_vec[cursor:cursor + self.num_actions] = 0.0
        cursor += self.num_actions
        noise_vec[cursor:] = (
            scales.height_measurements * self.obs_scales.height_measurements
        )
        return noise_vec

    def _target_gait_frequency(self):
        """Return the final tread-matched frequency for each command."""
        base_frequency = float(getattr(self.cfg.env, "gait_frequency", 1.25))
        frequency_gain = float(
            getattr(self.cfg.env, "gait_frequency_gain", 0.0)
        )
        reference_speed = float(
            getattr(self.cfg.env, "gait_reference_speed", 0.0)
        )
        return base_frequency + frequency_gain * torch.clamp(
            self.commands[:, 0] - reference_speed, min=0.0
        )

    def _gait_frequency_transition_fraction(self):
        """Return global old-clock to tread-clock blend in ``[0, 1]``."""
        transition_steps = int(
            getattr(self.cfg.env, "gait_frequency_transition_steps", 0)
        )
        if not self.enforce_walk_gait or transition_steps <= 0:
            return 1.0
        if not self.gait_frequency_transition_active:
            return 1.0
        return min(max(self.common_step_counter / transition_steps, 0.0), 1.0)

    def _scheduled_gait_frequency(self):
        """Blend legacy timing into the deployable target without a phase jump."""
        target = self._target_gait_frequency()
        start_base = float(
            getattr(self.cfg.env, "gait_frequency_start", 1.25)
        )
        start_gain = float(
            getattr(self.cfg.env, "gait_frequency_start_gain", 0.0)
        )
        start_reference_speed = float(
            getattr(
                self.cfg.env,
                "gait_frequency_start_reference_speed",
                0.0,
            )
        )
        start = start_base + start_gain * torch.clamp(
            self.commands[:, 0] - start_reference_speed, min=0.0
        )
        blend = self._gait_frequency_transition_fraction()
        return start + blend * (target - start)

    def _get_gait_phase(self):
        """Return a per-environment deployable left/right gait clock in [0, 1)."""
        frequency = torch.where(
            self.episode_gait_frequency > 0.0,
            self.episode_gait_frequency,
            self._scheduled_gait_frequency(),
        )
        elapsed = self.episode_length_buf.float() * self.dt
        return torch.remainder(
            self.gait_phase_offset + elapsed * frequency, 1.0
        )

    def _update_desired_contacts(self):
        """Update the phase-scheduled left/right stance mask."""
        if not self.include_gait_phase:
            self.desired_contacts[:] = torch.logical_or(
                self.contacts, self.last_contacts
            )
            self.phase_contact_match[:] = 1.0
            return

        phase = self._get_gait_phase()
        phase_sine = torch.sin(2.0 * torch.pi * phase)
        ratio = float(self.cfg.env.double_support_ratio)
        transition_threshold = torch.sin(
            torch.tensor(
                0.5 * torch.pi * ratio,
                dtype=torch.float,
                device=self.device,
            )
        )
        double_support = torch.abs(phase_sine) < transition_threshold
        left_stance = phase_sine >= 0.0
        self.desired_contacts[:, 0] = left_stance | double_support
        self.desired_contacts[:, 1] = ~left_stance | double_support

        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        self.phase_contact_match[:] = 1.0 - torch.mean(
            torch.abs(
                contact_filt.float() - self.desired_contacts.float()
            ),
            dim=1,
        )

    def _reshape_critic_feature(self, name, value, expected_width):
        """Return one privileged-observation component as an ``(N, F)`` matrix."""
        raw_shape = tuple(value.shape)
        if value.shape[0] != self.num_envs:
            raise RuntimeError(
                "Critic feature '{}' has {} environments, expected {}".format(
                    name, value.shape[0], self.num_envs
                )
            )
        value = value.reshape(self.num_envs, -1)
        if value.shape[1] != expected_width:
            raise RuntimeError(
                "Critic feature '{}' has width {}, expected {} (raw shape {})".format(
                    name, value.shape[1], expected_width, raw_shape
                )
            )
        return value

    def compute_observations(self):
        proprio_parts = [self.commands[:, :3] * self.commands_scale]
        if self.include_gait_phase:
            phase = self._get_gait_phase().unsqueeze(1)
            proprio_parts.extend(
                (
                    torch.sin(2.0 * torch.pi * phase),
                    torch.cos(2.0 * torch.pi * phase),
                )
            )
        if self.include_base_lin_vel:
            proprio_parts.append(self.base_lin_vel * self.obs_scales.lin_vel)
        if self.include_navigation_state:
            lateral_position = self.root_states[:, 1] - self.env_origins[:, 1]
            yaw = self.base_euler_xyz[:, 2]
            yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            proprio_parts.append(
                torch.stack(
                    (
                        lateral_position
                        * float(self.cfg.env.lateral_position_obs_scale),
                        yaw_error * float(self.cfg.env.yaw_error_obs_scale),
                    ),
                    dim=1,
                )
            )
        proprio_parts.extend(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            )
        )
        proprio = torch.cat(tuple(proprio_parts), dim=-1)
        all_heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        critic_height_count = len(self.cfg.terrain.measured_points_x) * len(
            self.cfg.terrain.measured_points_y
        )
        all_heights = self._reshape_critic_feature(
            "terrain_heights", all_heights, critic_height_count
        )
        actor_heights = all_heights[:, self.actor_height_indices]
        obs_now = torch.cat((proprio, actor_heights), dim=-1)

        critic_features = (
            ("proprio", proprio, self.proprio_obs_size),
            ("base_lin_vel", self.base_lin_vel * self.obs_scales.lin_vel, 3),
            ("payload", self.payload * 0.5, 1),
            ("friction", self.friction_coeffs, 1),
            ("restitution", self.restitution_coeffs, 1),
            ("kp_factor", self.Kp_factors, self.num_actions),
            ("kd_factor", self.Kd_factors, self.num_actions),
            ("motor_strength", self.motor_strength, self.num_actions),
            ("foot_contacts", self.contacts, len(self.feet_indices)),
            ("terrain_heights", all_heights, critic_height_count),
        )
        self.privileged_obs_buf = torch.cat(
            tuple(
                self._reshape_critic_feature(name, value, width)
                for name, value, width in critic_features
            ),
            dim=1,
        )

        if obs_now.shape[1] != self.cfg.env.num_single_obs:
            raise RuntimeError(
                "Actor observation mismatch: runtime {} vs config {}".format(
                    obs_now.shape[1], self.cfg.env.num_single_obs
                )
            )
        if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                "Critic observation mismatch: runtime {} vs config {}".format(
                    self.privileged_obs_buf.shape[1],
                    self.cfg.env.num_privileged_obs,
                )
            )

        if self.add_noise:
            obs_now = (
                obs_now
                + torch.randn_like(obs_now)
                * self.noise_scale_vec
                * self.cfg.noise.noise_level
            )

        self.obs_history.append(obs_now)
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        lower = float(self.command_ranges["lin_vel_x"][0])
        if self.cfg.commands.curriculum:
            upper = self._command_upper_for_levels(self.terrain_levels[env_ids])
            uniform = torch.rand(len(env_ids), device=self.device)
            self.commands[env_ids, 0] = lower + (upper - lower) * uniform
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                lower,
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        self.commands[env_ids, 1:] = 0.0

    def _command_upper_for_levels(self, levels):
        upper = self.cfg.commands.initial_max_speed + (
            self.cfg.commands.speed_per_terrain_level * levels.float()
        )
        return torch.clamp(upper, max=float(self.cfg.commands.max_curriculum))

    def update_command_curriculum(self, env_ids):
        """Per-environment speed limits are applied in _resample_commands()."""
        del env_ids

    def _reset_dofs(self, env_ids):
        """Reset around the configured crouch, not around zero joint angles."""
        position_noise = float(self.cfg.init_state.dof_position_noise)
        velocity_noise = float(self.cfg.init_state.dof_velocity_noise)
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -position_noise,
            position_noise,
            (len(env_ids), self.num_dof),
            device=self.device,
        )
        self.dof_vel[env_ids] = torch_rand_float(
            -velocity_noise,
            velocity_noise,
            (len(env_ids), self.num_dof),
            device=self.device,
        )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _current_stair_targets(self, env_ids=None):
        if env_ids is None:
            levels = self.terrain_levels
            types = self.terrain_types
        else:
            levels = self.terrain_levels[env_ids]
            types = self.terrain_types[env_ids]
        return (
            self.stair_top_x[levels, types],
            self.stair_top_heights[levels, types],
            self.stair_step_heights[levels, types],
        )

    def _sample_terrain_height_xy(self, points_xy):
        """Sample the height field directly below world-frame XY points."""
        points = points_xy.clone()
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clamp(points[..., 0], 0, self.height_samples.shape[0] - 1)
        py = torch.clamp(points[..., 1], 0, self.height_samples.shape[1] - 1)
        return self.height_samples[px, py] * self.terrain.cfg.vertical_scale

    def _update_foot_step_progress(self):
        """Track stable landings and strict stair-over-stair alternation."""
        foot_surface_height_world = self._sample_terrain_height_xy(
            self.feet_pos[:, :, :2]
        )
        foot_surface_height = (
            foot_surface_height_world
            - self.env_origins[:, 2].unsqueeze(1)
        )
        foot_surface_height = torch.clamp(foot_surface_height, min=0.0)
        horizontal_speed = torch.norm(self.feet_vel[:, :, :2], dim=2)

        if self.foot_force_sensor_forces is not None:
            horizontal_force = torch.norm(
                self.foot_force_sensor_forces[:, :, :2], dim=2
            )
            vertical_force = torch.abs(self.foot_force_sensor_forces[:, :, 2])
            vertical_support = vertical_force >= (
                float(
                    getattr(
                        self.cfg.env,
                        "stable_contact_min_vertical_ratio",
                        1.0,
                    )
                )
                * horizontal_force
            )
        else:
            vertical_support = torch.ones_like(self.contacts)

        raw_stable_candidate = (
            self.contacts
            & vertical_support
            & (
                horizontal_speed
                < float(
                    getattr(
                        self.cfg.env,
                        "stable_contact_max_horizontal_speed",
                        0.20,
                    )
                )
            )
        )
        self.stable_contact_candidate_time[:] = torch.where(
            raw_stable_candidate,
            self.stable_contact_candidate_time + self.dt,
            torch.zeros_like(self.stable_contact_candidate_time),
        )
        self.stable_contact_loss_time[:] = torch.where(
            raw_stable_candidate,
            torch.zeros_like(self.stable_contact_loss_time),
            self.stable_contact_loss_time + self.dt,
        )
        previous_stable_support = self.last_stable_contacts.clone()
        confirmed_support = self.stable_contact_candidate_time >= float(
            getattr(self.cfg.env, "stable_contact_confirmation_s", self.dt)
        )
        retained_support = self.stable_contacts & (
            self.stable_contact_loss_time
            < float(getattr(self.cfg.env, "stable_contact_release_s", self.dt))
        )
        stable_support = confirmed_support | retained_support
        self.stable_contacts[:] = stable_support
        true_airborne_now = self._true_airborne_measurement()
        stable_measurement = stable_support & raw_stable_candidate
        attained_height = torch.where(
            stable_measurement,
            foot_surface_height,
            self.max_foot_contact_height,
        )
        raw_delta = torch.clamp(
            attained_height - self.max_foot_contact_height, min=0.0
        )

        # Preserve the original raw-contact event for the legacy stair tasks.
        new_contact = self.contacts & ~self.last_contacts
        one_new_contact = torch.sum(new_contact.int(), dim=1) == 1
        opposite_was_supported = torch.flip(self.last_contacts, dims=[1])
        raw_alternating_landing = (
            new_contact
            & opposite_was_supported
            & one_new_contact.unsqueeze(1)
            & stable_support
        )

        self.alternating_tread_event[:] = 0.0
        self.repeated_lead_event[:] = 0.0
        self.same_tread_join_event[:] = 0.0
        self.skipped_tread_event[:] = 0.0
        confirmed_landing = torch.zeros_like(self.contacts)

        if self.enforce_walk_gait:
            # Quantize the supporting surface into the configured stair row.
            # A true stair-over-stair sequence advances one tread at a time
            # and changes the advancing foot on every new tread. Merely
            # joining the lead foot on its tread is explicitly classified as
            # step-to gait, even if the open-loop contact phase looks valid.
            _, _, step_height = self._current_stair_targets()
            tread_index = torch.round(
                foot_surface_height
                / torch.clamp(step_height.unsqueeze(1), min=0.02)
            ).long()
            tread_index = torch.clamp(
                tread_index, min=0, max=int(self.cfg.terrain.num_steps)
            )

            # Reject ankle samples on a vertical boundary or far from the
            # inferred sole height.  This prevents a shin/riser contact from
            # advancing the physical stair target before the sole is actually
            # resting inside the tread.
            stair_start_x = self.stair_start_x[
                self.terrain_levels, self.terrain_types
            ].unsqueeze(1)
            step_width = float(self.cfg.terrain.step_width)
            tread_margin = float(self.cfg.env.stable_landing_tread_margin)
            tread_start_x = stair_start_x + torch.clamp(
                tread_index.float() - 1.0, min=0.0
            ) * step_width
            tread_end_x = stair_start_x + tread_index.float() * step_width
            approach_or_top = (
                (tread_index == 0)
                | (tread_index == int(self.cfg.terrain.num_steps))
            )
            inside_tread = approach_or_top | (
                (self.feet_pos[:, :, 0] >= tread_start_x + tread_margin)
                & (self.feet_pos[:, :, 0] <= tread_end_x - tread_margin)
            )
            nominal_ankle_height = (
                foot_surface_height_world + self.foot_surface_offset
            )
            height_tolerance = torch.minimum(
                torch.full_like(
                    step_height,
                    float(self.cfg.env.stable_landing_height_tolerance),
                ),
                float(self.cfg.env.stable_landing_height_tolerance_ratio)
                * step_height,
            ).unsqueeze(1)
            height_consistent = torch.abs(
                self.feet_pos[:, :, 2] - nominal_ankle_height
            ) <= height_tolerance
            landing_geometry_valid = inside_tread & height_consistent
            valid_stable_tread = confirmed_support & landing_geometry_valid
            confirmed_landing = self.swing_active & valid_stable_tread

            # Only a foot with a latched physical lift-off can produce a tread
            # transition. This blocks the old shortcut in which a planted foot
            # slid over a height-field boundary and collected an alternating
            # landing without ever swinging.
            tread_advance_candidates = stable_tread_advance_mask(
                confirmed_support,
                landing_geometry_valid,
                self.swing_active & self.swing_true_airborne_seen,
                tread_index,
                self.accepted_foot_tread,
            )
            candidate_count = torch.sum(
                tread_advance_candidates.int(), dim=1
            )
            one_candidate = candidate_count == 1
            simultaneous_candidates = candidate_count > 1
            physical_landing = (
                tread_advance_candidates & one_candidate.unsqueeze(1)
            )

            # Natural gait additionally requires the opposite foot to have
            # remained in real contact for the whole swing. A brief double
            # flight can still synchronize physical target state, but cannot
            # collect the alternating-tread reward.
            opposite_stable_now = torch.flip(stable_support, dims=[1])
            opposite_true_airborne = torch.flip(
                true_airborne_now, dims=[1]
            )
            stable_alternating_landing = (
                physical_landing
                & self.swing_opposite_support_valid
                & opposite_stable_now
                & ~opposite_true_airborne
            )

            # Actual stance tread may decrease after a slip or recovery step;
            # the separate accepted high-watermark prevents replaying an old
            # ascent event when that foot climbs back onto the same tread.
            self.current_foot_tread[:] = torch.where(
                valid_stable_tread,
                tread_index,
                self.current_foot_tread,
            )
            accepted_tread = torch.maximum(
                self.accepted_foot_tread, tread_index
            )
            self.accepted_foot_tread[:] = torch.where(
                tread_advance_candidates,
                accepted_tread,
                self.accepted_foot_tread,
            )
            self.same_tread_support[:] = same_tread_support_mask(
                stable_support & self.contacts,
                self.current_foot_tread,
                int(self.cfg.terrain.num_steps),
            )

            # Learn the ankle-link-to-sole height from stable stance samples.
            # This makes the touchdown endpoint land on the tread instead of
            # repeatedly adding the whole riser height to the airborne foot.
            observed_offset = torch.clamp(
                self.feet_pos[:, :, 2] - foot_surface_height_world,
                min=0.02,
                max=0.14,
            )
            offset_rate = float(self.cfg.env.foot_surface_offset_update_rate)
            filtered_offset = (
                (1.0 - offset_rate) * self.foot_surface_offset
                + offset_rate * observed_offset
            )
            # Bootstrap the ankle-to-sole offset on the flat approach without
            # first requiring the offset-dependent height gate to pass. Once
            # climbing starts, retain the full geometry check so a riser edge
            # cannot corrupt the calibration.
            offset_calibration_valid = (
                confirmed_support
                & inside_tread
                & ((tread_index == 0) | height_consistent)
            )
            self.foot_surface_offset[:] = torch.where(
                offset_calibration_valid,
                filtered_offset,
                self.foot_surface_offset,
            )

            # Reject boundary-sampled heights from first-step/max-height
            # diagnostics. Only a confirmed sole placement inside a tread may
            # raise the per-foot attained height.
            attained_height = torch.where(
                valid_stable_tread,
                foot_surface_height,
                self.max_foot_contact_height,
            )

            has_physical_landing = torch.any(physical_landing, dim=1)
            has_natural_landing = torch.any(
                stable_alternating_landing, dim=1
            )
            candidate_foot = torch.argmax(
                physical_landing.long(), dim=1
            )
            candidate_tread = torch.gather(
                tread_index, 1, candidate_foot.unsqueeze(1)
            ).squeeze(1)

            previous_tread = self.last_advanced_tread.clone()
            previous_foot = self.last_advanced_foot.clone()
            (
                advanced,
                alternating_advance,
                repeated_lead,
                same_tread_join,
                skipped_tread,
            ) = classify_tread_transition(
                has_physical_landing,
                candidate_tread,
                candidate_foot,
                previous_tread,
                previous_foot,
                self.last_joined_tread,
            )
            # Physical state must progress after a valid one-foot touchdown
            # even if a brief double-flight means it was not a rewarded
            # support-to-support transition.
            alternating_advance &= has_natural_landing

            self.alternating_tread_event[:] = alternating_advance.float()
            self.repeated_lead_event[:] = repeated_lead.float()
            self.same_tread_join_event[:] = same_tread_join.float()
            self.skipped_tread_event[:] = skipped_tread.float()
            self.tread_advance_count += advanced.float()
            self.alternating_tread_count += alternating_advance.float()
            self.repeated_lead_count += repeated_lead.float()
            self.same_tread_join_count += same_tread_join.float()
            self.skipped_tread_count += skipped_tread.float()
            self.left_tread_advance_count += (
                advanced & (candidate_foot == 0)
            ).float()
            self.right_tread_advance_count += (
                advanced & (candidate_foot == 1)
            ).float()
            self.last_joined_tread[same_tread_join] = candidate_tread[
                same_tread_join
            ]

            event_one_hot = torch.nn.functional.one_hot(
                candidate_foot, num_classes=len(self.feet_indices)
            ).float()
            self.foot_contact_height_delta[:] = (
                event_one_hot
                * alternating_advance.unsqueeze(1).float()
                * step_height.unsqueeze(1)
            )

            # Consume every advance, including a repeated lead or skipped
            # tread, so the invalid event cannot later be relabeled as valid.
            self.last_advanced_tread[advanced] = candidate_tread[advanced]
            self.last_advanced_foot[advanced] = candidate_foot[advanced]

            # A simultaneous two-foot ascent is never a natural step, but it
            # must still synchronize the global physical target. Otherwise the
            # per-foot high-watermarks consume the landing while
            # ``last_advanced_tread + 1`` remains permanently one riser behind.
            simultaneous_tread = torch.max(
                torch.where(
                    tread_advance_candidates,
                    tread_index,
                    torch.zeros_like(tread_index),
                ),
                dim=1,
            ).values
            synchronize_hop = simultaneous_candidates & (
                simultaneous_tread > self.last_advanced_tread
            )
            self.last_advanced_tread[synchronize_hop] = simultaneous_tread[
                synchronize_hop
            ]
            self.last_advanced_foot[synchronize_hop] = -1
            self.last_joined_tread[synchronize_hop] = -1
        else:
            # Preserve the original n2_stairs event definition and checkpoint
            # behavior. The stricter tread sequence belongs only to the
            # dedicated n2_stairs_walk task.
            self.foot_contact_height_delta[:] = (
                raw_delta * raw_alternating_landing.float()
            )
            confirmed_landing = self.swing_active & confirmed_support

        # Latch a pending lift on the first support-loss frame, independently
        # of stable-contact release hysteresis. A separate force-and-clearance
        # measurement must later prove true flight before this transaction can
        # advance a tread. Keep it alive through impact until a geometrically
        # valid, confirmed touchdown clears it.
        raw_airborne = ~self.contacts
        foot_was_supported = previous_stable_support & self.last_contacts
        opposite_stable_now = torch.flip(stable_support, dims=[1])
        opposite_true_airborne = torch.flip(true_airborne_now, dims=[1])
        new_swing = (
            raw_airborne
            & ~self.swing_active
            & foot_was_supported
        )
        pending_before_touchdown = self.swing_active | new_swing
        support_valid = retained_swing_support_mask(
            pending_before_touchdown,
            new_swing,
            self.swing_opposite_support_valid,
            opposite_stable_now,
            opposite_true_airborne,
        )
        true_airborne_started = (
            pending_before_touchdown
            & true_airborne_now
            & ~self.swing_true_airborne_seen
        )
        self.swing_start_pos[:] = torch.where(
            true_airborne_started.unsqueeze(2),
            self.feet_pos,
            self.swing_start_pos,
        )
        self.swing_start_valid[true_airborne_started] = True
        measured_swing_landing = confirmed_landing & self.swing_start_valid
        swing_forward_displacement = torch.clamp(
            self.feet_pos[:, :, 0] - self.swing_start_pos[:, :, 0],
            min=-0.10,
            max=0.60,
        )
        self.last_swing_forward_displacement[:] = torch.where(
            measured_swing_landing,
            swing_forward_displacement,
            self.last_swing_forward_displacement,
        )
        self.swing_displacement_valid |= measured_swing_landing
        self.swing_forward_displacement_sum += (
            swing_forward_displacement * measured_swing_landing.float()
        )
        self.swing_forward_displacement_count += (
            measured_swing_landing.float()
        )
        self.swing_start_valid[confirmed_landing] = False
        continued_pending_time = self.swing_pending_time + self.dt
        pending_duration = torch.where(
            new_swing,
            torch.zeros_like(continued_pending_time),
            continued_pending_time,
        )
        continued_airborne_time = torch.where(
            pending_before_touchdown & true_airborne_now,
            self.swing_elapsed_time + self.dt,
            self.swing_elapsed_time,
        )
        airborne_duration = torch.where(
            true_airborne_started,
            torch.zeros_like(continued_airborne_time),
            continued_airborne_time,
        )
        self.max_swing_duration[:] = torch.maximum(
            self.max_swing_duration,
            torch.max(
                torch.where(
                    pending_before_touchdown,
                    pending_duration,
                    torch.zeros_like(pending_duration),
                ),
                dim=1,
            ).values,
        )
        pending_after_touchdown = (
            pending_before_touchdown & ~confirmed_landing
        )
        true_airborne_seen = (
            self.swing_true_airborne_seen
            | (pending_before_touchdown & true_airborne_now)
        )
        self.swing_true_airborne_seen[:] = (
            true_airborne_seen & pending_after_touchdown
        )
        self.swing_pending_time[:] = torch.where(
            pending_after_touchdown,
            pending_duration,
            torch.zeros_like(self.swing_pending_time),
        )
        self.swing_elapsed_time[:] = torch.where(
            pending_after_touchdown,
            airborne_duration,
            torch.zeros_like(self.swing_elapsed_time),
        )
        self.swing_active[:] = pending_after_touchdown
        self.swing_opposite_support_valid[:] = torch.where(
            pending_after_touchdown,
            support_valid,
            torch.zeros_like(support_valid),
        )
        self.last_stable_contacts[:] = stable_support
        # Even an invalid simultaneous landing consumes that height event, so
        # hopping cannot land once and collect it later by lifting one foot.
        self.max_foot_contact_height[:] = torch.maximum(
            self.max_foot_contact_height, attained_height
        )

    def _swing_knee_state(self):
        """Return scheduled swing mask and left/right knee flexion."""
        knees = self.dof_pos[:, self.knee_dof_idxs]
        if self.enforce_walk_gait:
            swing = self._physical_airborne_mask()
        elif self.include_gait_phase:
            swing = ~self.desired_contacts & ~self.contacts
        else:
            swing = ~self.contacts
        return swing, knees

    def _physical_airborne_mask(self):
        """Return latched swing feet that are physically out of contact."""
        return self.swing_active & self._true_airborne_measurement()

    def _true_airborne_measurement(self):
        """Measure real foot clearance without trusting vertical force alone."""
        if self.foot_force_sensor_forces is not None:
            foot_force = self.foot_force_sensor_forces
        else:
            foot_force = self.contact_forces[:, self.feet_indices, :]
        force_norm = torch.norm(foot_force, dim=2)
        _, _, step_height = self._current_stair_targets()
        support_ankle_z = (
            self.env_origins[:, 2].unsqueeze(1)
            + self.current_foot_tread.float() * step_height.unsqueeze(1)
            + self.foot_surface_offset
        )
        ankle_clearance = self.feet_pos[:, :, 2] - support_ankle_z
        clearance_threshold = torch.minimum(
            torch.full_like(
                step_height,
                float(self.cfg.env.true_airborne_clearance),
            ),
            float(self.cfg.env.true_airborne_clearance_ratio) * step_height,
        ).unsqueeze(1)
        return true_airborne_mask(
            force_norm,
            ankle_clearance,
            float(self.cfg.env.true_airborne_force_threshold),
            clearance_threshold,
        )

    def _arm_swing_tracking_score(self):
        """Score small contralateral arm motion synchronized to leg phase."""
        if not self.include_gait_phase:
            return torch.ones_like(self.best_forward_progress)
        phase_sine = torch.sin(2.0 * torch.pi * self._get_gait_phase())
        amplitude = float(self.cfg.env.arm_swing_amplitude)
        # URDF kinematics show negative shoulder pitch moves either hand in
        # world +X. During left stance/right swing, the left arm therefore
        # moves forward while the right arm moves backward.
        target = torch.stack(
            (-amplitude * phase_sine, amplitude * phase_sine), dim=1
        )
        shoulder_pitch = self.dof_pos[:, self.shoulder_pitch_dof_idxs]
        sharpness = float(self.cfg.env.arm_swing_tracking_sharpness)
        return torch.exp(
            -sharpness * torch.mean(
                torch.square(shoulder_pitch - target), dim=1
            )
        )

    def _sagittal_foot_phase_state(self):
        """Return dense right-minus-left foot-order tracking quantities.

        Just after phase zero the right leg starts swing behind the left leg.
        It crosses the stance leg near phase 0.25 and lands ahead near phase
        0.5. The cosine reference mirrors this trajectory for left swing in
        the second half-cycle, directly distinguishing stair-over-stair motion
        from a step-to gait that repeatedly brings both feet together.
        """
        phase = self._get_gait_phase()
        amplitude = max(
            float(self.cfg.env.sagittal_foot_phase_amplitude), 1.0e-3
        )
        target_separation = -amplitude * torch.cos(2.0 * torch.pi * phase)
        actual_separation = (
            self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0]
        )
        error = actual_separation - target_separation
        sharpness = float(self.cfg.env.sagittal_foot_phase_sharpness)
        score = torch.exp(-sharpness * torch.square(error))
        error_clip = float(self.cfg.env.sagittal_foot_phase_error_clip)
        normalized_error = torch.clamp(
            error / amplitude, min=-error_clip, max=error_clip
        )
        return score, normalized_error

    def _sagittal_foot_phase_score(self):
        score, _ = self._sagittal_foot_phase_state()
        return score

    def _next_tread_swing_state(self):
        """Return the next foot and a cheat-resistant single-support mask."""
        scheduled_swing_mask = ~self.desired_contacts
        scheduled_foot = torch.argmax(
            scheduled_swing_mask.long(), dim=1
        )
        physical_airborne = self._physical_airborne_mask()
        physical_foot = torch.argmax(physical_airborne.long(), dim=1)
        has_physical_swing = torch.any(physical_airborne, dim=1)
        initial_expected_foot = torch.where(
            has_physical_swing, physical_foot, scheduled_foot
        )
        expected_foot = torch.where(
            self.last_advanced_foot < 0,
            initial_expected_foot,
            torch.clamp(1 - self.last_advanced_foot, min=0, max=1),
        )
        scheduled_swing = torch.gather(
            scheduled_swing_mask.long(),
            1,
            expected_foot.unsqueeze(1),
        ).squeeze(1).bool()
        expected_in_flight = torch.gather(
            self.swing_active.long(),
            1,
            expected_foot.unsqueeze(1),
        ).squeeze(1).bool()
        actually_airborne = torch.gather(
            self._physical_airborne_mask().long(),
            1,
            expected_foot.unsqueeze(1),
        ).squeeze(1).bool()
        opposite_supported = torch.gather(
            self.swing_opposite_support_valid.long(),
            1,
            expected_foot.unsqueeze(1),
        ).squeeze(1).bool()
        levels = self.terrain_levels
        types = self.terrain_types
        stair_start_x = self.stair_start_x[levels, types]
        near_stairs = self.root_states[:, 0] >= (
            stair_start_x
            - float(self.cfg.env.first_tread_target_activation_distance)
        )
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = (
            (self.last_advanced_tread < int(self.cfg.terrain.num_steps))
            & (scheduled_swing | expected_in_flight)
            & actually_airborne
            & opposite_supported
            & near_stairs
            & (self.episode_length_buf > grace_steps)
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        return expected_foot, active

    def _nominal_swing_duration(self):
        """Return physical airborne time implied by the deployed gait clock."""
        frequency = torch.where(
            self.episode_gait_frequency > 0.0,
            self.episode_gait_frequency,
            self._scheduled_gait_frequency(),
        )
        single_support_fraction = 1.0 - float(
            self.cfg.env.double_support_ratio
        )
        return single_support_fraction / torch.clamp(
            2.0 * frequency, min=0.10
        )

    def _swing_progress_state(self):
        """Return per-foot progress measured from the real lift-off event."""
        nominal = self._nominal_swing_duration().unsqueeze(1)
        return torch.clamp(
            self.swing_elapsed_time / torch.clamp(nominal, min=self.dt),
            min=0.0,
            max=1.0,
        )

    def _swing_trajectory_state(self):
        """Track a smooth lift-off-to-next-tread 3-D swing trajectory."""
        expected_foot, active = self._next_tread_swing_state()
        gather_xyz = expected_foot.view(-1, 1, 1).expand(-1, 1, 3)
        gather_scalar = expected_foot.unsqueeze(1)
        start = torch.gather(
            self.swing_start_pos, 1, gather_xyz
        ).squeeze(1)
        start_valid = torch.gather(
            self.swing_start_valid.long(), 1, gather_scalar
        ).squeeze(1).bool()
        progress = torch.gather(
            self._swing_progress_state(), 1, gather_scalar
        ).squeeze(1)

        levels = self.terrain_levels
        types = self.terrain_types
        stair_start_x = self.stair_start_x[levels, types]
        _, _, step_height = self._current_stair_targets()
        next_tread = torch.clamp(
            self.last_advanced_tread + 1,
            min=1,
            max=int(self.cfg.terrain.num_steps),
        )
        step_width = float(self.cfg.terrain.step_width)
        landing_x = stair_start_x + (
            next_tread.float() - 0.5
        ) * step_width
        lateral_offset = float(self.cfg.env.foothold_lateral_offset)
        landing_y = self.env_origins[:, 1] + torch.where(
            expected_foot == 0,
            torch.full_like(landing_x, lateral_offset),
            torch.full_like(landing_x, -lateral_offset),
        )
        ankle_offset = torch.gather(
            self.foot_surface_offset, 1, gather_scalar
        ).squeeze(1)
        landing_z = (
            self.env_origins[:, 2]
            + next_tread.float() * step_height
            + ankle_offset
        )
        landing = torch.stack((landing_x, landing_y, landing_z), dim=1)

        target = smooth_swing_trajectory(
            start,
            landing,
            progress,
            float(self.cfg.env.swing_trajectory_arc_base)
            + float(self.cfg.env.swing_trajectory_arc_height_gain)
            * step_height,
            float(self.cfg.env.swing_trajectory_forward_delay),
            float(self.cfg.env.swing_trajectory_lift_end),
            float(self.cfg.env.swing_trajectory_descent_start),
        )
        actual = torch.gather(
            self.feet_pos, 1, gather_xyz
        ).squeeze(1)
        normalizers = torch.tensor(
            [
                float(self.cfg.env.swing_trajectory_x_normalizer),
                float(self.cfg.env.swing_trajectory_y_normalizer),
                float(self.cfg.env.swing_trajectory_z_normalizer),
            ],
            dtype=torch.float,
            device=self.device,
        )
        error_clip = float(self.cfg.env.swing_trajectory_error_clip)
        normalized_error = torch.clamp(
            (actual - target) / normalizers,
            min=-error_clip,
            max=error_clip,
        )
        squared_error = torch.mean(torch.square(normalized_error), dim=1)
        score = torch.exp(
            -float(self.cfg.env.swing_trajectory_sharpness)
            * squared_error
        )
        target_active = active & start_valid

        # Do not pay positive tracking reward while the leg or foot is pushing
        # into a riser.  The bounded error and explicit collision terms remain.
        lower_leg_collision = torch.gather(
            self._lower_leg_collision_per_foot(), 1, gather_scalar
        ).squeeze(1)
        foot_riser_collision = torch.gather(
            self._foot_riser_collision_per_foot(), 1, gather_scalar
        ).squeeze(1)
        collision_free = (
            (lower_leg_collision <= 0.0)
            & (foot_riser_collision <= 0.0)
        )
        score *= collision_free.float()
        return score, squared_error, target_active.float()

    def _swing_timeout_state(self):
        """Return bounded severity for an unresolved swing transaction."""
        allowed = (
            self._nominal_swing_duration().unsqueeze(1)
            * float(self.cfg.env.swing_timeout_ratio)
            + float(self.cfg.env.swing_timeout_margin_s)
        )
        severity = torch.clamp(
            (self.swing_pending_time - allowed)
            / torch.clamp(allowed, min=self.dt),
            min=0.0,
            max=1.0,
        )
        severity *= self.swing_active.float()
        return torch.max(severity, dim=1).values

    def _lower_leg_collision_per_foot(self):
        """Return clipped left/right shin-contact severity."""
        force = torch.norm(
            self.contact_forces[:, self.lower_leg_indices, :], dim=2
        )
        return torch.clamp(
            (
                force - float(self.cfg.env.lower_leg_contact_threshold)
            )
            / max(float(self.cfg.env.lower_leg_contact_scale), 1.0e-3),
            min=0.0,
            max=1.0,
        )

    def _foot_riser_collision_per_foot(self):
        """Separate horizontal riser impacts from normal floor support."""
        if self.foot_force_sensor_forces is not None:
            foot_force = self.foot_force_sensor_forces
        else:
            foot_force = self.contact_forces[:, self.feet_indices, :]
        horizontal = torch.norm(foot_force[:, :, :2], dim=2)
        vertical = torch.abs(foot_force[:, :, 2])
        required_horizontal = torch.maximum(
            torch.full_like(
                horizontal,
                float(self.cfg.env.foot_riser_horizontal_threshold),
            ),
            float(self.cfg.env.foot_riser_force_ratio) * vertical,
        )
        severity = torch.clamp(
            (horizontal - required_horizontal)
            / max(float(self.cfg.env.foot_riser_contact_scale), 1.0e-3),
            min=0.0,
            max=1.0,
        )
        return severity * (~self.stable_contacts).float()

    def _base_behind_support_state(self):
        """Measure how far the pelvis trails the stable support polygon."""
        support_weight = self.stable_contacts.float()
        support_count = torch.sum(support_weight, dim=1)
        support_x = torch.sum(
            self.feet_pos[:, :, 0] * support_weight, dim=1
        ) / torch.clamp(support_count, min=1.0)
        excess = torch.clamp(
            support_x
            - self.root_states[:, 0]
            - float(self.cfg.env.base_support_backward_allowance),
            min=0.0,
        )
        stair_start_x = self.stair_start_x[
            self.terrain_levels, self.terrain_types
        ]
        near_stairs = self.root_states[:, 0] >= (
            stair_start_x
            - float(self.cfg.env.first_tread_target_activation_distance)
        )
        active = (
            (support_count > 0)
            & near_stairs
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        raw_excess = excess * active.float()
        normalized = torch.clamp(
            raw_excess
            / max(float(self.cfg.env.base_support_backward_scale), 1.0e-3),
            max=float(self.cfg.env.base_support_backward_error_clip),
        )
        return raw_excess, torch.square(normalized)

    def _foot_pitch_state(self):
        """Keep stance and late-swing soles approximately level."""
        feet_quat = self.feet_quat.reshape(-1, 4)
        _, foot_pitch, _ = get_euler_xyz(feet_quat)
        foot_pitch = torch.atan2(
            torch.sin(foot_pitch), torch.cos(foot_pitch)
        ).reshape(self.num_envs, len(self.feet_indices))
        progress = self._swing_progress_state()
        mask = self.stable_contacts | (
            self._physical_airborne_mask()
            & (progress >= float(self.cfg.env.late_swing_progress))
        )
        normalized = torch.clamp(
            torch.abs(foot_pitch)
            / max(float(self.cfg.env.foot_pitch_normalizer), 1.0e-3),
            max=float(self.cfg.env.foot_pitch_error_clip),
        )
        sample_count = torch.sum(mask.float(), dim=1)
        mean_abs = torch.sum(
            torch.abs(foot_pitch) * mask.float(), dim=1
        ) / torch.clamp(sample_count, min=1.0)
        penalty = torch.sum(
            torch.square(normalized) * mask.float(), dim=1
        ) / torch.clamp(sample_count, min=1.0)
        return penalty, mean_abs, sample_count

    def _next_tread_foot_target_state(self):
        """Target the center of the next riser with the opposite swing foot.

        This target activates only after the first stair has been reached, so
        it cannot make the robot reach across the flat approach platform. A
        step-to landing leaves the target one full tread ahead and therefore
        keeps a dense correction signal until that foot actually advances.
        """
        levels = self.terrain_levels
        types = self.terrain_types
        stair_start_x = self.stair_start_x[levels, types]
        next_tread = torch.clamp(
            self.last_advanced_tread + 1,
            min=1,
            max=int(self.cfg.terrain.num_steps),
        )
        step_width = max(float(self.cfg.terrain.step_width), 1.0e-3)
        target_x = stair_start_x + (
            next_tread.float() - 0.5
        ) * step_width

        expected_foot, active = self._next_tread_swing_state()
        expected_foot_x = torch.gather(
            self.feet_pos[:, :, 0], 1, expected_foot.unsqueeze(1)
        ).squeeze(1)
        normalized_error = (expected_foot_x - target_x) / step_width
        error_clip = float(self.cfg.env.next_tread_target_error_clip)
        normalized_error = torch.clamp(
            normalized_error, min=-error_clip, max=error_clip
        )
        sharpness = float(self.cfg.env.next_tread_target_sharpness)
        score = torch.exp(-sharpness * torch.square(normalized_error))

        phase = self._get_gait_phase()
        right_swing_progress = 2.0 * phase
        left_swing_progress = 2.0 * (phase - 0.5)
        swing_progress = torch.where(
            expected_foot == 1,
            right_swing_progress,
            left_swing_progress,
        )
        swing_progress = torch.clamp(swing_progress, min=0.0, max=1.0)
        start_phase = float(self.cfg.env.next_tread_target_start_phase)
        full_phase = float(self.cfg.env.next_tread_target_full_phase)
        phase_span = max(full_phase - start_phase, 1.0e-3)
        landing_weight = torch.clamp(
            (swing_progress - start_phase) / phase_span,
            min=0.0,
            max=1.0,
        )
        # Smoothstep avoids an abrupt reward change halfway through swing.
        landing_weight = torch.square(landing_weight) * (
            3.0 - 2.0 * landing_weight
        )
        return score, normalized_error, landing_weight * active.float()

    def _next_tread_lateral_target_state(self):
        """Keep the scheduled swing foot at its natural side of the tread."""
        expected_foot, active = self._next_tread_swing_state()
        expected_foot_y = torch.gather(
            self.feet_pos[:, :, 1], 1, expected_foot.unsqueeze(1)
        ).squeeze(1)
        lateral_offset = max(
            float(self.cfg.env.foothold_lateral_offset), 1.0e-3
        )
        target_sign = torch.where(
            expected_foot == 0,
            torch.ones_like(expected_foot_y),
            -torch.ones_like(expected_foot_y),
        )
        target_y = self.env_origins[:, 1] + target_sign * lateral_offset
        normalized_error = (expected_foot_y - target_y) / lateral_offset
        error_clip = float(self.cfg.env.foothold_lateral_error_clip)
        normalized_error = torch.clamp(
            normalized_error, min=-error_clip, max=error_clip
        )
        sharpness = float(self.cfg.env.foothold_lateral_sharpness)
        score = torch.exp(-sharpness * torch.square(normalized_error))

        return score, normalized_error, active.float()

    def _foot_crossover_state(self):
        """Return per-foot inward error and a symmetric stair activity mask.

        World ``+Y`` is the robot's left side (verified by the URDF hip
        anchors).  The left foot therefore needs positive centerline
        clearance and the right foot negative clearance.  Converting both to
        a signed outward clearance makes this calculation exactly symmetric.
        """
        relative_y = (
            self.feet_pos[:, :, 1] - self.env_origins[:, 1].unsqueeze(1)
        )
        outward_clearance = torch.stack(
            (relative_y[:, 0], -relative_y[:, 1]), dim=1
        )
        minimum_half_width = float(
            self.cfg.env.foothold_min_half_width
        )
        inward_error = torch.clamp(
            minimum_half_width - outward_clearance, min=0.0
        )
        normalizer = max(
            float(self.cfg.env.foothold_crossover_normalizer), 1.0e-3
        )
        normalized_error = torch.clamp(
            inward_error / normalizer, min=0.0, max=2.0
        )

        levels = self.terrain_levels
        types = self.terrain_types
        stair_start_x = self.stair_start_x[levels, types]
        near_stairs = self.root_states[:, 0] >= (
            stair_start_x
            - float(self.cfg.env.first_tread_target_activation_distance)
        )
        active = (
            near_stairs
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        return relative_y, inward_error, normalized_error, active

    def _foot_lane_error_state(self):
        """Return dense left/right foot error from the two stair lanes.

        Crossover alone cannot see a foot that is too far outside its lane.
        Penalising both feet against ``(+offset, -offset)`` also detects the
        common-mode lane shift observed in the 9100 checkpoint while remaining
        exactly invariant under left/right reflection.
        """
        relative_y = (
            self.feet_pos[:, :, 1] - self.env_origins[:, 1].unsqueeze(1)
        )
        offset = max(float(self.cfg.env.foothold_lateral_offset), 1.0e-3)
        targets = torch.tensor(
            (offset, -offset),
            dtype=relative_y.dtype,
            device=relative_y.device,
        ).unsqueeze(0)
        normalized_error = torch.clamp(
            (relative_y - targets) / offset,
            min=-float(self.cfg.env.foothold_lateral_error_clip),
            max=float(self.cfg.env.foothold_lateral_error_clip),
        )
        levels = self.terrain_levels
        types = self.terrain_types
        stair_start_x = self.stair_start_x[levels, types]
        active = (
            (
                self.root_states[:, 0]
                >= (
                    stair_start_x
                    - float(
                        self.cfg.env.first_tread_target_activation_distance
                    )
                )
            )
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        return relative_y, normalized_error, active

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_desired_contacts()

        forward_progress = self.root_states[:, 0] - self.env_origins[:, 0]
        new_best_progress = torch.maximum(
            self.best_forward_progress, forward_progress
        )
        advanced = new_best_progress > (
            self.progress_checkpoint + self.cfg.env.progress_epsilon
        )
        self.best_forward_progress[:] = new_best_progress
        self.progress_checkpoint[advanced] = new_best_progress[advanced]
        self.last_progress_step[advanced] = self.episode_length_buf[advanced]
        current_climb_height = self.terrain_h - self.env_origins[:, 2]
        # Reward only newly attained stair height. Moving back and forth across
        # one riser cannot repeatedly collect vertical-progress reward.
        self.terrain_height_delta[:] = torch.clamp(
            current_climb_height - self.max_climb_height, min=0.0
        )
        self.max_climb_height[:] = torch.maximum(
            self.max_climb_height, current_climb_height
        )
        self._update_foot_step_progress()

        if self.enforce_walk_gait:
            (
                foot_lateral_position,
                foot_lateral_inward_error,
                _,
                lateral_placement_active,
            ) = self._foot_crossover_state()
            lateral_samples = (
                self.stable_contacts
                & lateral_placement_active.unsqueeze(1)
            )
            self.foot_lateral_inward_error_sum += (
                foot_lateral_inward_error * lateral_samples.float()
            )
            self.foot_lateral_position_sum += (
                foot_lateral_position * lateral_samples.float()
            )
            self.foot_lateral_sample_count += lateral_samples.float()

        lateral_position = self.root_states[:, 1] - self.env_origins[:, 1]
        yaw = self.base_euler_xyz[:, 2]
        yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        forward_speed = self.root_states[:, 7]
        self.forward_speed_sum += forward_speed
        self.command_error_sum += torch.abs(
            forward_speed - self.commands[:, 0]
        )
        self.phase_match_sum += self.phase_contact_match
        if self.enforce_walk_gait:
            self.sagittal_foot_phase_match_sum += (
                self._sagittal_foot_phase_score()
            )
        self.double_flight_step_count += (~torch.any(self.contacts, dim=1)).float()
        self.max_lateral_deviation[:] = torch.maximum(
            self.max_lateral_deviation, torch.abs(lateral_position)
        )
        self.max_yaw_deviation[:] = torch.maximum(
            self.max_yaw_deviation, torch.abs(yaw_error)
        )
        sagittal_foot_separation = torch.abs(
            self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]
        )
        self.max_sagittal_foot_separation[:] = torch.maximum(
            self.max_sagittal_foot_separation,
            sagittal_foot_separation,
        )
        swing, knees = self._swing_knee_state()
        self.swing_knee_flexion_sum += torch.sum(
            knees * swing.float(), dim=1
        )
        self.swing_knee_sample_count += torch.sum(swing.float(), dim=1)
        self.arm_swing_match_sum += self._arm_swing_tracking_score()
        if self.enforce_walk_gait:
            lower_leg_collision = torch.max(
                self._lower_leg_collision_per_foot(), dim=1
            ).values
            foot_riser_collision = torch.max(
                self._foot_riser_collision_per_foot(), dim=1
            ).values
            swing_timeout = self._swing_timeout_state()
            base_behind_support, _ = self._base_behind_support_state()
            _, mean_foot_pitch_error, foot_pitch_samples = (
                self._foot_pitch_state()
            )
            self.same_tread_support_step_count += (
                self.same_tread_support.float()
            )
            self.lower_leg_collision_step_count += (
                lower_leg_collision > 0.0
            ).float()
            self.foot_riser_collision_step_count += (
                foot_riser_collision > 0.0
            ).float()
            self.swing_timeout_step_count += (
                swing_timeout > 0.0
            ).float()
            self.base_behind_support_sum += base_behind_support
            self.foot_pitch_error_sum += (
                mean_foot_pitch_error * foot_pitch_samples
            )
            self.foot_pitch_sample_count += foot_pitch_samples

        if self.enforce_walk_gait:
            corridor_grace_steps = int(
                self.cfg.env.corridor_grace_s / self.dt
            )
            outside_corridor = torch.abs(lateral_position) > float(
                self.cfg.env.corridor_half_width
            )
            excessive_yaw = torch.abs(yaw_error) > float(
                self.cfg.env.corridor_yaw_limit
            )
            self.path_failure_buf[:] = (
                (outside_corridor | excessive_yaw)
                & (self.episode_length_buf > corridor_grace_steps)
            )
        else:
            self.path_failure_buf[:] = False

        top_x, top_height, _ = self._current_stair_targets()
        reached_position_and_height = torch.logical_and(
            self.root_states[:, 0] >= top_x,
            self.max_climb_height >= (
                top_height - self.cfg.terrain.success_height_tolerance
            ),
        )
        upright = -self.projected_gravity[:, 2] > 0.85
        stable_support = torch.any(
            self.contacts
            & (torch.norm(self.feet_vel[:, :, :2], dim=2) < 0.20),
            dim=1,
        )
        self.top_position_reached_buf[:] = reached_position_and_height
        physical_stable_top = (
            self.top_position_reached_buf & upright & stable_support
        )
        stable_top = physical_stable_top.clone()
        if self.enforce_walk_gait:
            self.completion_stable_time += self.dt
            self.completion_stable_time *= physical_stable_top.float()
            self.completion_buf[:] = physical_stable_top & (
                self.completion_stable_time
                >= float(self.cfg.env.completion_dwell_s)
            )
            centered = torch.abs(lateral_position) <= float(
                self.cfg.env.success_lateral_tolerance
            )
            facing_forward = torch.abs(yaw_error) <= float(
                self.cfg.env.success_yaw_tolerance
            )
            path_consistent = (
                self.max_lateral_deviation
                <= float(self.cfg.env.success_max_lateral_deviation)
            ) & (
                self.max_yaw_deviation
                <= float(self.cfg.env.success_max_yaw_deviation)
            )
            elapsed_steps = torch.clamp(
                self.episode_length_buf.float(), min=1.0
            )
            mean_forward_speed = self.forward_speed_sum / elapsed_steps
            command_consistent = (
                torch.abs(mean_forward_speed - self.commands[:, 0])
                <= float(self.cfg.env.success_max_mean_speed_bias)
            )
            gait_consistent = (
                self.phase_match_sum / elapsed_steps
                >= float(self.cfg.env.success_min_phase_contact_match)
            ) & (
                self.double_flight_step_count / elapsed_steps
                <= float(self.cfg.env.success_max_double_flight_fraction)
            )
            # Step-to joins are sequence events but not advances. Include
            # them in the denominator so every reported rate remains in
            # [0, 1] and a join-heavy gait cannot produce percentages above
            # 100%. Repeated-lead and skipped-tread events are already a
            # subset of tread_advance_count.
            tread_sequence_denominator = torch.clamp(
                self.tread_advance_count + self.same_tread_join_count,
                min=1.0,
            )
            alternating_tread_rate = (
                self.alternating_tread_count / tread_sequence_denominator
            )
            same_tread_join_rate = (
                self.same_tread_join_count / tread_sequence_denominator
            )
            skipped_tread_rate = (
                self.skipped_tread_count / tread_sequence_denominator
            )
            natural_step_sequence = (
                self.alternating_tread_count
                >= float(self.cfg.env.success_min_alternating_tread_count)
            ) & (
                alternating_tread_rate
                >= float(self.cfg.env.success_min_alternating_tread_rate)
            ) & (
                same_tread_join_rate
                <= float(self.cfg.env.success_max_same_tread_join_rate)
            ) & (
                skipped_tread_rate
                <= float(self.cfg.env.success_max_skipped_tread_rate)
            ) & (
                self.max_sagittal_foot_separation
                <= float(
                    self.cfg.env.success_max_sagittal_foot_separation
                )
            )
            stable_top &= (
                centered
                & facing_forward
                & command_consistent
                & path_consistent
                & gait_consistent
                & natural_step_sequence
            )
            self.top_stable_time += self.dt
            self.top_stable_time *= stable_top.float()
            self.top_reached_buf[:] = stable_top & (
                self.top_stable_time >= float(self.cfg.env.top_dwell_s)
            )
            # Strict success is also a physical completion, even though its
            # shorter dwell may fire before the non-strict completion timer.
            self.completion_buf |= self.top_reached_buf
        else:
            self.top_stable_time[:] = 0.0
            self.completion_stable_time[:] = 0.0
            self.completion_buf[:] = stable_top
            self.top_reached_buf[:] = stable_top

        self.curriculum_completion_buf[:] = (
            self._terrain_curriculum_success_mask(None)
        )

        grace_steps = int(self.cfg.env.progress_grace_s / self.dt)
        stall_steps = int(self.cfg.env.stall_timeout_s / self.dt)
        self.stall_buf[:] = torch.logical_and(
            self.episode_length_buf > grace_steps,
            (self.episode_length_buf - self.last_progress_step) > stall_steps,
        )
        self.stall_buf &= ~self.completion_buf

    def check_termination(self):
        super().check_termination()
        physical_failure = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 5.0,
            dim=1,
        )
        if hasattr(self, "fallen_buf"):
            physical_failure |= self.fallen_buf
        self.fall_event_buf[:] = physical_failure
        # Keep the raw position/height flag for evaluation: reaching the top
        # and falling there should count as top_reached but not as success.
        self.top_reached_buf &= ~physical_failure
        self.completion_buf &= ~physical_failure
        self.curriculum_completion_buf &= ~physical_failure
        self.stall_buf &= ~physical_failure
        self.path_failure_buf &= ~physical_failure
        # A stable physical completion is a true terminal state and must not
        # receive timeout bootstrapping in PPO. A physical failure on the final
        # time step is likewise a failure, not a benign timeout.
        self.time_out_buf &= ~self.completion_buf
        self.time_out_buf &= ~physical_failure
        self.reset_buf |= self.stall_buf
        self.reset_buf |= self.path_failure_buf
        self.reset_buf |= self.completion_buf

    def _terrain_curriculum_success_mask(self, env_ids):
        """Return the training promotion gate without weakening evaluation."""
        index = slice(None) if env_ids is None else env_ids
        if not self.enforce_walk_gait:
            return self.top_reached_buf[index]

        episode_steps = torch.clamp(
            self.episode_length_buf[index].float(), min=1.0
        )
        sequence_denominator = torch.clamp(
            self.tread_advance_count[index]
            + self.same_tread_join_count[index],
            min=1.0,
        )
        alternating_rate = (
            self.alternating_tread_count[index] / sequence_denominator
        )
        same_tread_join_rate = (
            self.same_tread_join_count[index] / sequence_denominator
        )
        skipped_tread_rate = (
            self.skipped_tread_count[index] / sequence_denominator
        )
        phase_match = self.phase_match_sum[index] / episode_steps
        double_flight_fraction = (
            self.double_flight_step_count[index] / episode_steps
        )

        return (
            self.completion_buf[index]
            & ~self.fall_event_buf[index]
            & ~self.path_failure_buf[index]
            & (
                self.alternating_tread_count[index]
                >= float(
                    self.cfg.env.curriculum_min_alternating_tread_count
                )
            )
            & (
                alternating_rate
                >= float(self.cfg.env.curriculum_min_alternating_tread_rate)
            )
            & (
                same_tread_join_rate
                <= float(self.cfg.env.curriculum_max_same_tread_join_rate)
            )
            & (
                skipped_tread_rate
                <= float(self.cfg.env.curriculum_max_skipped_tread_rate)
            )
            & (
                phase_match
                >= float(self.cfg.env.curriculum_min_phase_contact_match)
            )
            & (
                double_flight_fraction
                <= float(
                    self.cfg.env.curriculum_max_double_flight_fraction
                )
            )
        )

    def _update_terrain_curriculum(self, env_ids):
        valid = self.episode_started[env_ids]
        success = self.curriculum_completion_buf[env_ids] & valid

        failure = (
            self.fall_event_buf[env_ids]
            | self.stall_buf[env_ids]
            | self.path_failure_buf[env_ids]
            | self.time_out_buf[env_ids]
        ) & ~success & valid

        success_streak = torch.where(
            success,
            self.curriculum_success_streak[env_ids] + 1,
            torch.zeros_like(self.curriculum_success_streak[env_ids]),
        )
        failure_streak = torch.where(
            failure,
            self.curriculum_failure_streak[env_ids] + 1,
            torch.zeros_like(self.curriculum_failure_streak[env_ids]),
        )
        # The initial reset is not an episode and must not clear a preloaded
        # curriculum state. All real outcomes require consecutive episodes.
        self.curriculum_success_streak[env_ids] = torch.where(
            valid, success_streak, self.curriculum_success_streak[env_ids]
        )
        self.curriculum_failure_streak[env_ids] = torch.where(
            valid, failure_streak, self.curriculum_failure_streak[env_ids]
        )

        move_up = self.curriculum_success_streak[env_ids] >= int(
            self.cfg.terrain.curriculum_successes
        )
        move_down = self.curriculum_failure_streak[env_ids] >= int(
            self.cfg.terrain.curriculum_failures
        )
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.clamp(
            self.terrain_levels[env_ids], 0, self.max_terrain_level - 1
        )
        self.curriculum_success_streak[env_ids] *= ~move_up
        self.curriculum_failure_streak[env_ids] *= ~move_down
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        valid = self.episode_started[env_ids].clone()
        success = (self.top_reached_buf[env_ids] & valid).float()
        completion = (self.completion_buf[env_ids] & valid).float()
        curriculum_pass = (
            self.curriculum_completion_buf[env_ids] & valid
        ).float()
        top_reached = (
            self.top_position_reached_buf[env_ids] & valid
        ).float()
        fall = (self.fall_event_buf[env_ids] & valid).float()
        stall = (self.stall_buf[env_ids] & valid).float()
        forward = self.best_forward_progress[env_ids].clone() * valid.float()
        climb = self.max_climb_height[env_ids].clone() * valid.float()
        survival = self.episode_length_buf[env_ids].float() * self.dt * valid.float()
        episode_command = self.commands[env_ids, 0].clone() * valid.float()
        episode_gait_frequency = (
            self.episode_gait_frequency[env_ids].clone() * valid.float()
        )
        _, _, episode_step_height = self._current_stair_targets(env_ids)
        max_foot_height = torch.max(
            self.max_foot_contact_height[env_ids], dim=1
        ).values
        first_step = (
            max_foot_height
            >= episode_step_height - self.cfg.terrain.success_height_tolerance
        ).float() * valid.float()
        episode_steps = torch.clamp(
            self.episode_length_buf[env_ids].float(), min=1.0
        )
        mean_forward_speed = (
            self.forward_speed_sum[env_ids] / episode_steps
        ) * valid.float()
        mean_command_error = (
            self.command_error_sum[env_ids] / episode_steps
        ) * valid.float()
        phase_contact_match = (
            self.phase_match_sum[env_ids] / episode_steps
        ) * valid.float()
        sagittal_foot_phase_match = (
            self.sagittal_foot_phase_match_sum[env_ids] / episode_steps
        ) * valid.float()
        double_flight_fraction = (
            self.double_flight_step_count[env_ids] / episode_steps
        ) * valid.float()
        max_lateral_deviation = (
            self.max_lateral_deviation[env_ids] * valid.float()
        )
        max_yaw_deviation = self.max_yaw_deviation[env_ids] * valid.float()
        path_failure = (
            self.path_failure_buf[env_ids] & valid
        ).float()
        tread_sequence_denominator = torch.clamp(
            self.tread_advance_count[env_ids]
            + self.same_tread_join_count[env_ids],
            min=1.0,
        )
        alternating_tread_count = (
            self.alternating_tread_count[env_ids] * valid.float()
        )
        alternating_tread_rate = (
            self.alternating_tread_count[env_ids]
            / tread_sequence_denominator
        ) * valid.float()
        repeated_lead_rate = (
            self.repeated_lead_count[env_ids]
            / tread_sequence_denominator
        ) * valid.float()
        same_tread_join_rate = (
            self.same_tread_join_count[env_ids]
            / tread_sequence_denominator
        ) * valid.float()
        skipped_tread_rate = (
            self.skipped_tread_count[env_ids]
            / tread_sequence_denominator
        ) * valid.float()
        max_sagittal_foot_separation = (
            self.max_sagittal_foot_separation[env_ids] * valid.float()
        )
        mean_swing_knee_flexion = (
            self.swing_knee_flexion_sum[env_ids]
            / torch.clamp(
                self.swing_knee_sample_count[env_ids], min=1.0
            )
        ) * valid.float()
        mean_arm_swing_match = (
            self.arm_swing_match_sum[env_ids] / episode_steps
        ) * valid.float()
        same_tread_support_fraction = (
            self.same_tread_support_step_count[env_ids] / episode_steps
        ) * valid.float()
        lower_leg_collision_fraction = (
            self.lower_leg_collision_step_count[env_ids] / episode_steps
        ) * valid.float()
        foot_riser_collision_fraction = (
            self.foot_riser_collision_step_count[env_ids] / episode_steps
        ) * valid.float()
        swing_timeout_fraction = (
            self.swing_timeout_step_count[env_ids] / episode_steps
        ) * valid.float()
        mean_base_behind_support = (
            self.base_behind_support_sum[env_ids] / episode_steps
        ) * valid.float()
        mean_foot_pitch_error = (
            self.foot_pitch_error_sum[env_ids]
            / torch.clamp(
                self.foot_pitch_sample_count[env_ids], min=1.0
            )
        ) * valid.float()
        max_swing_duration = (
            self.max_swing_duration[env_ids] * valid.float()
        )
        left_tread_advances = (
            self.left_tread_advance_count[env_ids] * valid.float()
        )
        right_tread_advances = (
            self.right_tread_advance_count[env_ids] * valid.float()
        )
        final_lateral_position = (
            self.root_states[env_ids, 1]
            - self.env_origins[env_ids, 1]
        ) * valid.float()
        mean_swing_lengths = (
            self.swing_forward_displacement_sum[env_ids]
            / torch.clamp(
                self.swing_forward_displacement_count[env_ids], min=1.0
            )
        ) * valid.float().unsqueeze(1)
        mean_left_swing_length = mean_swing_lengths[:, 0]
        mean_right_swing_length = mean_swing_lengths[:, 1]
        mean_foot_inward_error = (
            self.foot_lateral_inward_error_sum[env_ids]
            / torch.clamp(
                self.foot_lateral_sample_count[env_ids], min=1.0
            )
        ) * valid.float().unsqueeze(1)
        mean_foot_lateral_position = (
            self.foot_lateral_position_sum[env_ids]
            / torch.clamp(
                self.foot_lateral_sample_count[env_ids], min=1.0
            )
        ) * valid.float().unsqueeze(1)
        mean_left_foot_inward_error = mean_foot_inward_error[:, 0]
        mean_right_foot_inward_error = mean_foot_inward_error[:, 1]
        mean_left_foot_lateral_position = mean_foot_lateral_position[:, 0]
        mean_right_foot_lateral_position = mean_foot_lateral_position[:, 1]

        self.last_episode_success[env_ids] = success
        self.last_episode_completion[env_ids] = completion
        self.last_episode_curriculum_completion[env_ids] = curriculum_pass
        self.last_episode_top_reached[env_ids] = top_reached
        self.last_episode_forward_progress[env_ids] = forward
        self.last_episode_climb_height[env_ids] = climb
        self.last_episode_survival_time[env_ids] = survival
        self.last_episode_fall[env_ids] = fall
        self.last_episode_stall[env_ids] = stall
        self.last_episode_first_step[env_ids] = first_step
        self.last_episode_max_foot_height[env_ids] = (
            max_foot_height * valid.float()
        )
        self.last_episode_mean_forward_speed[env_ids] = mean_forward_speed
        self.last_episode_mean_command_error[env_ids] = mean_command_error
        self.last_episode_phase_contact_match[env_ids] = phase_contact_match
        self.last_episode_sagittal_foot_phase_match[env_ids] = (
            sagittal_foot_phase_match
        )
        self.last_episode_double_flight_fraction[env_ids] = (
            double_flight_fraction
        )
        self.last_episode_max_lateral_deviation[env_ids] = (
            max_lateral_deviation
        )
        self.last_episode_max_yaw_deviation[env_ids] = max_yaw_deviation
        self.last_episode_path_failure[env_ids] = path_failure
        self.last_episode_alternating_tread_count[env_ids] = (
            alternating_tread_count
        )
        self.last_episode_alternating_tread_rate[env_ids] = (
            alternating_tread_rate
        )
        self.last_episode_repeated_lead_rate[env_ids] = repeated_lead_rate
        self.last_episode_same_tread_join_rate[env_ids] = (
            same_tread_join_rate
        )
        self.last_episode_skipped_tread_rate[env_ids] = skipped_tread_rate
        self.last_episode_max_sagittal_foot_separation[env_ids] = (
            max_sagittal_foot_separation
        )
        self.last_episode_mean_swing_knee_flexion[env_ids] = (
            mean_swing_knee_flexion
        )
        self.last_episode_mean_arm_swing_match[env_ids] = (
            mean_arm_swing_match
        )
        self.last_episode_gait_frequency[env_ids] = episode_gait_frequency
        self.last_episode_same_tread_support_fraction[env_ids] = (
            same_tread_support_fraction
        )
        self.last_episode_lower_leg_collision_fraction[env_ids] = (
            lower_leg_collision_fraction
        )
        self.last_episode_foot_riser_collision_fraction[env_ids] = (
            foot_riser_collision_fraction
        )
        self.last_episode_swing_timeout_fraction[env_ids] = (
            swing_timeout_fraction
        )
        self.last_episode_mean_base_behind_support[env_ids] = (
            mean_base_behind_support
        )
        self.last_episode_mean_foot_pitch_error[env_ids] = (
            mean_foot_pitch_error
        )
        self.last_episode_max_swing_duration[env_ids] = max_swing_duration
        self.last_episode_left_tread_advances[env_ids] = left_tread_advances
        self.last_episode_right_tread_advances[env_ids] = right_tread_advances
        self.last_episode_final_lateral_position[env_ids] = (
            final_lateral_position
        )
        self.last_episode_mean_left_swing_length[env_ids] = (
            mean_left_swing_length
        )
        self.last_episode_mean_right_swing_length[env_ids] = (
            mean_right_swing_length
        )
        self.last_episode_mean_left_foot_inward_error[env_ids] = (
            mean_left_foot_inward_error
        )
        self.last_episode_mean_right_foot_inward_error[env_ids] = (
            mean_right_foot_inward_error
        )
        self.last_episode_mean_left_foot_lateral_position[env_ids] = (
            mean_left_foot_lateral_position
        )
        self.last_episode_mean_right_foot_lateral_position[env_ids] = (
            mean_right_foot_lateral_position
        )

        super().reset_idx(env_ids)

        # Base reset updates root state and orientation, but the base velocity
        # buffers otherwise still describe the just-finished episode. The
        # strict Actor observes velocity, so refresh it before the first frame.
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )

        if self.include_gait_phase and bool(
            getattr(self.cfg.env, "randomize_gait_phase", False)
        ):
            self.gait_phase_offset[env_ids] = torch.rand(
                len(env_ids), device=self.device
            )
        else:
            self.gait_phase_offset[env_ids] = 0.0
        if self.include_gait_phase:
            scheduled_frequency = self._scheduled_gait_frequency()
            self.episode_gait_frequency[env_ids] = scheduled_frequency[env_ids]
        else:
            self.episode_gait_frequency[env_ids] = 0.0

        metric_mask = valid
        metric_weight = metric_mask.float()
        metric_count = torch.clamp(metric_weight.sum(), min=1.0)

        def masked_mean(values):
            return torch.sum(values * metric_weight) / metric_count

        self.extras["episode"].update(
            {
                "stairs_success_rate": masked_mean(success),
                "stairs_completion_rate": masked_mean(completion),
                "stairs_curriculum_pass_rate": masked_mean(curriculum_pass),
                "stairs_top_rate": masked_mean(top_reached),
                "stairs_fall_rate": masked_mean(fall),
                "stairs_stall_rate": masked_mean(stall),
                "stairs_first_step_rate": masked_mean(first_step),
                "stairs_forward_distance": masked_mean(forward),
                "stairs_climb_height": masked_mean(climb),
                "stairs_max_foot_height": masked_mean(max_foot_height),
                "stairs_mean_forward_speed": masked_mean(mean_forward_speed),
                "stairs_mean_command_error": masked_mean(mean_command_error),
                "stairs_phase_contact_match": masked_mean(phase_contact_match),
                "stairs_sagittal_foot_phase_match": masked_mean(
                    sagittal_foot_phase_match
                ),
                "stairs_double_flight_fraction": masked_mean(
                    double_flight_fraction
                ),
                "stairs_max_lateral_deviation": masked_mean(
                    max_lateral_deviation
                ),
                "stairs_max_yaw_deviation": masked_mean(max_yaw_deviation),
                "stairs_path_failure_rate": masked_mean(path_failure),
                "stairs_alternating_tread_count": masked_mean(
                    alternating_tread_count
                ),
                "stairs_alternating_tread_rate": masked_mean(
                    alternating_tread_rate
                ),
                "stairs_repeated_lead_rate": masked_mean(
                    repeated_lead_rate
                ),
                "stairs_same_tread_join_rate": masked_mean(
                    same_tread_join_rate
                ),
                "stairs_skipped_tread_rate": masked_mean(
                    skipped_tread_rate
                ),
                "stairs_max_sagittal_foot_separation": masked_mean(
                    max_sagittal_foot_separation
                ),
                "stairs_mean_swing_knee_flexion": masked_mean(
                    mean_swing_knee_flexion
                ),
                "stairs_mean_arm_swing_match": masked_mean(
                    mean_arm_swing_match
                ),
                "stairs_gait_frequency_hz": masked_mean(
                    episode_gait_frequency
                ),
                "stairs_gait_frequency_transition": (
                    self._gait_frequency_transition_fraction()
                ),
                "stairs_same_tread_support_fraction": masked_mean(
                    same_tread_support_fraction
                ),
                "stairs_lower_leg_collision_fraction": masked_mean(
                    lower_leg_collision_fraction
                ),
                "stairs_foot_riser_collision_fraction": masked_mean(
                    foot_riser_collision_fraction
                ),
                "stairs_swing_timeout_fraction": masked_mean(
                    swing_timeout_fraction
                ),
                "stairs_mean_base_behind_support": masked_mean(
                    mean_base_behind_support
                ),
                "stairs_mean_foot_pitch_error": masked_mean(
                    mean_foot_pitch_error
                ),
                "stairs_max_swing_duration": masked_mean(
                    max_swing_duration
                ),
                "stairs_left_tread_advances": masked_mean(
                    left_tread_advances
                ),
                "stairs_right_tread_advances": masked_mean(
                    right_tread_advances
                ),
                "stairs_final_lateral_position": masked_mean(
                    final_lateral_position
                ),
                "stairs_mean_left_swing_length": masked_mean(
                    mean_left_swing_length
                ),
                "stairs_mean_right_swing_length": masked_mean(
                    mean_right_swing_length
                ),
                "stairs_mean_left_foot_inward_error": masked_mean(
                    mean_left_foot_inward_error
                ),
                "stairs_mean_right_foot_inward_error": masked_mean(
                    mean_right_foot_inward_error
                ),
                "stairs_mean_left_foot_lateral_position": masked_mean(
                    mean_left_foot_lateral_position
                ),
                "stairs_mean_right_foot_lateral_position": masked_mean(
                    mean_right_foot_lateral_position
                ),
                "stairs_survival_time": masked_mean(survival),
                "stairs_command_x": masked_mean(episode_command),
            }
        )
        command_lower = float(self.command_ranges["lin_vel_x"][0])
        command_upper = float(self.command_ranges["lin_vel_x"][1])
        if abs(command_upper - command_lower) < 1.0e-9:
            logged_command_upper = command_upper
        else:
            logged_command_upper = self._command_upper_for_levels(
                self.terrain_levels[env_ids]
            ).mean()
        self.extras["episode"]["max_command_x"] = logged_command_upper
        self.extras["episode"]["min_command_x"] = command_lower

        self.best_forward_progress[env_ids] = 0.0
        self.progress_checkpoint[env_ids] = 0.0
        self.max_climb_height[env_ids] = 0.0
        self.last_progress_step[env_ids] = 0
        self.terrain_height_delta[env_ids] = 0.0
        self.double_flight_time[env_ids] = 0.0
        self.foot_contact_height_delta[env_ids] = 0.0
        self.max_foot_contact_height[env_ids] = 0.0
        self.last_advanced_tread[env_ids] = 0
        self.last_advanced_foot[env_ids] = -1
        self.last_joined_tread[env_ids] = -1
        self.alternating_tread_event[env_ids] = 0.0
        self.repeated_lead_event[env_ids] = 0.0
        self.same_tread_join_event[env_ids] = 0.0
        self.skipped_tread_event[env_ids] = 0.0
        self.tread_advance_count[env_ids] = 0.0
        self.alternating_tread_count[env_ids] = 0.0
        self.repeated_lead_count[env_ids] = 0.0
        self.same_tread_join_count[env_ids] = 0.0
        self.skipped_tread_count[env_ids] = 0.0
        self.top_stable_time[env_ids] = 0.0
        self.completion_stable_time[env_ids] = 0.0
        self.path_failure_buf[env_ids] = False
        self.forward_speed_sum[env_ids] = 0.0
        self.command_error_sum[env_ids] = 0.0
        self.phase_match_sum[env_ids] = 0.0
        self.sagittal_foot_phase_match_sum[env_ids] = 0.0
        self.double_flight_step_count[env_ids] = 0.0
        self.max_lateral_deviation[env_ids] = 0.0
        self.max_yaw_deviation[env_ids] = 0.0
        self.max_sagittal_foot_separation[env_ids] = 0.0
        self.swing_knee_flexion_sum[env_ids] = 0.0
        self.swing_knee_sample_count[env_ids] = 0.0
        self.arm_swing_match_sum[env_ids] = 0.0
        self.same_tread_support_step_count[env_ids] = 0.0
        self.lower_leg_collision_step_count[env_ids] = 0.0
        self.foot_riser_collision_step_count[env_ids] = 0.0
        self.swing_timeout_step_count[env_ids] = 0.0
        self.base_behind_support_sum[env_ids] = 0.0
        self.foot_pitch_error_sum[env_ids] = 0.0
        self.foot_pitch_sample_count[env_ids] = 0.0
        self.left_tread_advance_count[env_ids] = 0.0
        self.right_tread_advance_count[env_ids] = 0.0
        self.last_swing_forward_displacement[env_ids] = 0.0
        self.swing_displacement_valid[env_ids] = False
        self.swing_forward_displacement_sum[env_ids] = 0.0
        self.swing_forward_displacement_count[env_ids] = 0.0
        self.foot_lateral_inward_error_sum[env_ids] = 0.0
        self.foot_lateral_position_sum[env_ids] = 0.0
        self.foot_lateral_sample_count[env_ids] = 0.0
        self.top_reached_buf[env_ids] = False
        self.top_position_reached_buf[env_ids] = False
        self.completion_buf[env_ids] = False
        self.curriculum_completion_buf[env_ids] = False
        self.stall_buf[env_ids] = False
        self.fall_event_buf[env_ids] = False
        self.contacts[env_ids] = False
        self.last_contacts[env_ids] = False
        self.stable_contacts[env_ids] = False
        self.last_stable_contacts[env_ids] = False
        self.stable_contact_candidate_time[env_ids] = 0.0
        self.stable_contact_loss_time[env_ids] = 0.0
        self.current_foot_tread[env_ids] = 0
        self.accepted_foot_tread[env_ids] = 0
        self.same_tread_support[env_ids] = False
        self.swing_active[env_ids] = False
        self.swing_true_airborne_seen[env_ids] = False
        self.swing_opposite_support_valid[env_ids] = False
        self.swing_start_valid[env_ids] = False
        self.swing_start_pos[env_ids] = 0.0
        self.swing_elapsed_time[env_ids] = 0.0
        self.swing_pending_time[env_ids] = 0.0
        self.max_swing_duration[env_ids] = 0.0
        self.foot_surface_offset[env_ids] = float(
            getattr(self.cfg.env, "nominal_foot_surface_offset", 0.045)
        )
        for history_frame in self.obs_history:
            history_frame[env_ids] = 0.0
        self.episode_started[env_ids] = True

    def get_checkpoint_state(self):
        """Return the curriculum state needed for a faithful training resume."""
        transition_steps = int(
            getattr(self.cfg.env, "gait_frequency_transition_steps", 0)
        )
        transition_step = transition_steps
        if self.gait_frequency_transition_active:
            transition_step = min(self.common_step_counter, transition_steps)
        return {
            "version": 2,
            "task_name": getattr(self.cfg.env, "task_name", None),
            "terrain_levels": self.terrain_levels.detach().cpu(),
            "success_streak": self.curriculum_success_streak.detach().cpu(),
            "failure_streak": self.curriculum_failure_streak.detach().cpu(),
            "gait_frequency_transition_step": transition_step,
        }

    def load_checkpoint_state(self, state):
        """Restore curriculum state while allowing a different environment count."""
        if not state:
            return
        source_task = state.get("task_name")
        current_task = getattr(self.cfg.env, "task_name", None)
        if source_task is not None and current_task is not None:
            if source_task != current_task:
                print(
                    "Skipping curriculum state from task '{}' while loading '{}'".format(
                        source_task, current_task
                    )
                )
                return

        transition_steps = int(
            getattr(self.cfg.env, "gait_frequency_transition_steps", 0)
        )
        saved_transition_step = state.get("gait_frequency_transition_step")
        if transition_steps > 0 and self.enforce_walk_gait:
            # Version-1 strict checkpoints used a fixed 1.25 Hz clock and do
            # not contain this key. Start their compatibility transition at
            # zero. New checkpoints preserve progress, while fresh training
            # never calls this loader and uses the final clock immediately.
            restored_step = 0
            if saved_transition_step is not None:
                restored_step = max(
                    0, min(int(saved_transition_step), transition_steps)
                )
            self.gait_frequency_transition_active = (
                restored_step < transition_steps
            )
            self.common_step_counter = restored_step
            self.episode_gait_frequency[:] = self._scheduled_gait_frequency()
            print(
                "Gait frequency transition: step {}/{} (active={})".format(
                    restored_step,
                    transition_steps,
                    self.gait_frequency_transition_active,
                )
            )

        if not self.cfg.terrain.curriculum:
            return
        if int(getattr(self.cfg.terrain, "fixed_level", -1)) >= 0:
            return

        def expand_to_envs(key, target):
            source = state.get(key)
            if source is None or source.numel() == 0:
                return
            source = source.flatten().to(device=self.device, dtype=target.dtype)
            indices = torch.arange(self.num_envs, device=self.device) % len(source)
            target[:] = source[indices]

        expand_to_envs("terrain_levels", self.terrain_levels)
        self.terrain_levels[:] = torch.clamp(
            self.terrain_levels, 0, self.max_terrain_level - 1
        )
        expand_to_envs("success_streak", self.curriculum_success_streak)
        expand_to_envs("failure_streak", self.curriculum_failure_streak)
        self.env_origins[:] = self.terrain_origins[
            self.terrain_levels, self.terrain_types
        ]

        # OnPolicyRunner has already performed its construction reset. Mark
        # that short episode invalid, then reset again at the restored origins.
        self.episode_started[:] = False
        self.reset()

    # ------------------------------ rewards ------------------------------

    def _reward_stairs_forward_progress(self):
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        supported = torch.any(self.contacts, dim=1).float()
        world_forward_velocity = torch.clamp(self.root_states[:, 7], min=0.0)
        if self.enforce_walk_gait:
            rewarded_speed = torch.clamp(
                1.25 * self.commands[:, 0], min=0.10
            )
            world_forward_velocity = torch.minimum(
                world_forward_velocity, rewarded_speed
            )
        else:
            world_forward_velocity = torch.clamp(
                world_forward_velocity, max=0.8
            )
        lateral_gate = torch.exp(-4.0 * torch.abs(self.root_states[:, 8]))
        heading_gate = torch.ones_like(world_forward_velocity)
        if self.enforce_walk_gait:
            yaw = self.base_euler_xyz[:, 2]
            yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            heading_gate = torch.exp(-6.0 * torch.square(yaw_error))
        return (
            world_forward_velocity
            * upright
            * supported
            * lateral_gate
            * heading_gate
        )

    def _reward_tracking_lin_vel(self):
        """Track +X without rewarding a stationary robot at the stair foot."""
        world_velocity = self.root_states[:, 7:9]
        error = torch.sum(
            torch.square(self.commands[:, :2] - world_velocity), dim=1
        )
        tracking_sharpness = 20.0 if self.enforce_walk_gait else 5.0
        score = torch.exp(-tracking_sharpness * error)
        target = torch.clamp(self.commands[:, 0], min=0.05)
        progress_gate = torch.clamp(
            world_velocity[:, 0] / (0.5 * target), min=0.0, max=1.0
        )
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        supported = torch.any(self.contacts, dim=1).float()
        heading_gate = torch.ones_like(score)
        if self.enforce_walk_gait:
            yaw = self.base_euler_xyz[:, 2]
            yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            heading_gate = torch.exp(-6.0 * torch.square(yaw_error))
        return score * progress_gate * upright * supported * heading_gate

    def _reward_stairs_overspeed(self):
        allowed_speed = 1.15 * self.commands[:, 0] + 0.02
        excess_forward = torch.clamp(
            self.root_states[:, 7] - allowed_speed, min=0.0
        )
        lateral_speed = self.root_states[:, 8]
        return torch.square(excess_forward) + torch.square(lateral_speed)

    def _reward_stairs_command_speed_error(self):
        forward_error = self.root_states[:, 7] - self.commands[:, 0]
        return torch.square(forward_error)

    def _reward_stairs_vertical_progress(self):
        _, _, step_height = self._current_stair_targets()
        normalized_rise = torch.clamp(
            self.terrain_height_delta / torch.clamp(step_height, min=0.02),
            0.0,
            1.0,
        )
        moving_forward = (self.root_states[:, 7] > 0.03).float()
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        supported = torch.any(self.contacts, dim=1).float()
        # Reward preparation multiplies every scale by dt. This is a discrete
        # height-transition event, so cancel dt to make the configured scale
        # the actual reward per newly attained riser.
        return normalized_rise * moving_forward * upright * supported / self.dt

    def _reward_stairs_foot_step_progress(self):
        """Reward one-shot, alternating landings on a newly higher tread."""
        _, _, step_height = self._current_stair_targets()
        normalized_rise = torch.clamp(
            self.foot_contact_height_delta
            / torch.clamp(step_height.unsqueeze(1), min=0.02),
            0.0,
            1.0,
        )
        upright = (-self.projected_gravity[:, 2] > 0.80).float()
        # This is a discrete landing event, so cancel reward preparation's dt.
        return torch.sum(normalized_rise, dim=1) * upright / self.dt

    def _reward_stairs_alternating_tread(self):
        """Reward the opposite foot advancing exactly one new tread."""
        upright = -self.projected_gravity[:, 2] > 0.80
        return self.alternating_tread_event * upright.float() / self.dt

    def _reward_stairs_repeated_lead(self):
        """Penalize one foot repeatedly leading onto successive treads."""
        return self.repeated_lead_event / self.dt

    def _reward_stairs_same_tread_join(self):
        """Penalize the trailing foot joining the lead foot step-to style."""
        return self.same_tread_join_event / self.dt

    def _reward_stairs_skipped_tread(self):
        """Penalize reaching over a tread instead of climbing sequentially."""
        return self.skipped_tread_event / self.dt

    def _reward_stairs_overstride(self):
        """Penalize excessive fore-aft leg splits and full-leg reaching."""
        base_x = self.root_states[:, 0].unsqueeze(1)
        foot_offset = torch.abs(self.feet_pos[:, :, 0] - base_x)
        foot_separation = torch.abs(
            self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]
        )
        offset_excess = torch.clamp(
            foot_offset - float(self.cfg.env.max_sagittal_foot_offset),
            min=0.0,
        )
        separation_excess = torch.clamp(
            foot_separation
            - float(self.cfg.env.max_sagittal_foot_separation),
            min=0.0,
        )
        normalizer = max(float(self.cfg.env.overstride_soft_margin), 1.0e-3)
        return torch.mean(
            torch.square(offset_excess / normalizer), dim=1
        ) + torch.square(separation_excess / normalizer)

    def _swing_knee_target(self):
        """Flex near mid-swing and extend again before touchdown."""
        _, _, step_height = self._current_stair_targets()
        landing_target = (
            float(self.cfg.env.swing_knee_landing_target)
            + float(self.cfg.env.swing_knee_landing_height_gain) * step_height
        )
        peak_target = torch.clamp(
            float(self.cfg.env.swing_knee_peak_target)
            + float(self.cfg.env.swing_knee_peak_height_gain) * step_height,
            max=float(self.cfg.env.swing_knee_max_target),
        )
        swing_progress = self._swing_progress_state()
        flexion_arc = 4.0 * swing_progress * (1.0 - swing_progress)
        return landing_target.unsqueeze(1) + (
            peak_target - landing_target
        ).unsqueeze(1) * flexion_arc

    def _active_swing_knee_average(self, values, swing):
        """Average a per-knee quantity only over scheduled airborne legs."""
        swing_count = torch.sum(swing.float(), dim=1)
        average = torch.sum(values * swing.float(), dim=1) / torch.clamp(
            swing_count, min=1.0
        )
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        active = (swing_count > 0) & moving & upright
        return average * active.float()

    def _reward_stairs_swing_knee_flexion(self):
        """Guide the airborne leg to bend rather than reach out straight."""
        swing, knees = self._swing_knee_state()
        target = self._swing_knee_target()
        sharpness = float(self.cfg.env.swing_knee_tracking_sharpness)
        score = torch.exp(-sharpness * torch.square(knees - target))
        return self._active_swing_knee_average(score, swing)

    def _reward_stairs_swing_knee_deficit(self):
        """Penalize an under-flexed swing knee with a non-vanishing gradient."""
        swing, knees = self._swing_knee_state()
        target = self._swing_knee_target()
        normalized_deficit = torch.clamp(
            target - knees, min=0.0
        ) / torch.clamp(target, min=0.10)
        return self._active_swing_knee_average(
            torch.square(normalized_deficit), swing
        )

    def _reward_stairs_arm_swing(self):
        """Encourage modest contralateral arms instead of a frozen torso."""
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = self.episode_length_buf > grace_steps
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        return self._arm_swing_tracking_score() * (
            active & moving & upright
        ).float()

    def _reward_stairs_swing_trajectory(self):
        """Reward the expected foot following its continuous 3-D reference."""
        score, _, active = self._swing_trajectory_state()
        return score * active

    def _reward_stairs_swing_trajectory_error(self):
        """Provide a bounded gradient when the swing foot misses its arc."""
        _, squared_error, active = self._swing_trajectory_state()
        return squared_error * active

    def _reward_stairs_swing_timeout(self):
        """Penalize leaving either leg suspended beyond its nominal swing."""
        return self._swing_timeout_state()

    def _reward_stairs_same_tread_support(self):
        """Continuously penalize step-to stance on an intermediate tread."""
        return self.same_tread_support.float()

    def _reward_stairs_lower_leg_collision(self):
        """Penalize bounded shin contact with a stair riser."""
        return torch.max(
            self._lower_leg_collision_per_foot(), dim=1
        ).values

    def _reward_stairs_foot_riser_collision(self):
        """Penalize a swing foot pushing horizontally into a riser."""
        return torch.max(
            self._foot_riser_collision_per_foot(), dim=1
        ).values

    def _reward_stairs_forward_pitch(self):
        """Keep the body slightly forward over the stance leg while climbing."""
        _, _, step_height = self._current_stair_targets()
        target_pitch = (
            float(self.cfg.env.forward_pitch_base_target)
            + float(self.cfg.env.forward_pitch_height_gain) * step_height
        )
        pitch = self.base_euler_xyz[:, 1]
        score = torch.exp(
            -float(self.cfg.env.forward_pitch_sharpness)
            * torch.square(pitch - target_pitch)
        )
        stair_start_x = self.stair_start_x[
            self.terrain_levels, self.terrain_types
        ]
        active = (
            (
                self.root_states[:, 0]
                >= stair_start_x
                - float(
                    self.cfg.env.first_tread_target_activation_distance
                )
            )
            & torch.any(self.stable_contacts, dim=1)
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        return score * active.float()

    def _reward_stairs_base_behind_support(self):
        """Penalize a pelvis that lags behind its stance foot."""
        _, normalized_penalty = self._base_behind_support_state()
        return normalized_penalty

    def _reward_stairs_foot_pitch(self):
        """Penalize toe-up stance and late-swing landing posture."""
        penalty, _, _ = self._foot_pitch_state()
        return penalty

    def _reward_stairs_success(self):
        # One-shot terminal event; see the dt note above.
        return self.top_reached_buf.float() / self.dt

    def _reward_stairs_completion(self):
        """Reward a stable physical climb, independent of style gates."""
        return self.completion_buf.float() / self.dt

    def _reward_stairs_curriculum_completion(self):
        """Reward completion with the intermediate curriculum gait gates."""
        return self.curriculum_completion_buf.float() / self.dt

    def _reward_termination(self):
        # In this completion task, an unfinished time limit is also a failure
        # even though PPO still bootstraps its critic value at that time limit.
        failure = self.reset_buf.bool() & ~self.completion_buf
        return failure.float() / self.dt

    def _reward_stairs_stride_symmetry(self):
        """Penalize unequal measured left/right forward swing lengths."""
        both_measured = torch.all(self.swing_displacement_valid, dim=1)
        stride_difference = (
            self.last_swing_forward_displacement[:, 0]
            - self.last_swing_forward_displacement[:, 1]
        )
        normalized_error = torch.clamp(
            stride_difference / float(self.cfg.terrain.step_width),
            min=-1.0,
            max=1.0,
        )
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        return (
            torch.square(normalized_error)
            * (both_measured & moving & upright).float()
        )

    def _reward_stairs_lateral_drift(self):
        lateral_position = self.root_states[:, 1] - self.env_origins[:, 1]
        lateral_velocity = self.root_states[:, 8]
        yaw_rate = self.base_ang_vel[:, 2]
        yaw = self.base_euler_xyz[:, 2]
        yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return (
            lateral_position.square()
            + lateral_velocity.square()
            + 0.2 * yaw_rate.square()
            + 0.5 * yaw_error.square()
        )

    def _reward_stairs_heading_alignment(self):
        lateral_position = self.root_states[:, 1] - self.env_origins[:, 1]
        yaw = self.base_euler_xyz[:, 2]
        yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        aligned = torch.exp(
            -8.0 * torch.square(yaw_error)
            -4.0 * torch.square(lateral_position)
        )
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        target_speed = torch.clamp(self.commands[:, 0], min=0.05)
        progress_gate = torch.clamp(
            self.root_states[:, 7] / (0.5 * target_speed),
            min=0.0,
            max=1.0,
        )
        return aligned * upright * progress_gate

    def _reward_stairs_leg_alignment(self):
        joint_error = self.dof_pos[:, self.leg_alignment_idxs] - (
            self.default_dof_pos[:, self.leg_alignment_idxs]
        )
        return torch.sum(torch.square(joint_error), dim=1)

    def _reward_stairs_feet_yaw(self):
        feet_quat = self.feet_quat.reshape(-1, 4)
        _, _, foot_yaw = get_euler_xyz(feet_quat)
        foot_yaw = torch.atan2(torch.sin(foot_yaw), torch.cos(foot_yaw))
        foot_yaw = foot_yaw.reshape(self.num_envs, len(self.feet_indices))
        return torch.mean(torch.square(foot_yaw), dim=1)

    def _reward_stairs_no_progress(self):
        grace_steps = int(self.cfg.env.progress_grace_s / self.dt)
        active = self.episode_length_buf > grace_steps
        slow = self.root_states[:, 7] < 0.03
        return (active & slow & ~self.top_reached_buf).float()

    def _reward_stairs_double_flight(self):
        both_airborne = ~torch.any(self.contacts, dim=1)
        self.double_flight_time += self.dt
        self.double_flight_time *= both_airborne.float()
        allowed = float(self.cfg.env.max_double_flight_s)
        severity = torch.clamp(
            (self.double_flight_time - allowed) / max(allowed, self.dt),
            min=0.0,
            max=1.0,
        )
        grace_steps = int(self.cfg.env.flight_grace_s / self.dt)
        return severity * (self.episode_length_buf > grace_steps).float()

    def _reward_stairs_phase_contact(self):
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = self.episode_length_buf > grace_steps
        upright = -self.projected_gravity[:, 2] > 0.80
        return self.phase_contact_match * (active & upright).float()

    def _reward_stairs_phase_contact_mismatch(self):
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = self.episode_length_buf > grace_steps
        return (1.0 - self.phase_contact_match) * active.float()

    def _reward_stairs_sagittal_foot_phase(self):
        """Reward the scheduled swing foot crossing ahead of the stance foot."""
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = self.episode_length_buf > grace_steps
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        score, _ = self._sagittal_foot_phase_state()
        return score * (active & moving & upright).float()

    def _reward_stairs_sagittal_foot_phase_error(self):
        """Supply a dense gradient when the feet have the wrong phase order."""
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = self.episode_length_buf > grace_steps
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        _, normalized_error = self._sagittal_foot_phase_state()
        return torch.square(normalized_error) * (
            active & moving & upright
        ).float()

    def _reward_stairs_next_tread_target(self):
        """Reward the opposite swing foot approaching the next tread center."""
        score, _, target_weight = self._next_tread_foot_target_state()
        return score * target_weight

    def _reward_stairs_next_tread_target_error(self):
        """Penalize remaining fore-aft distance to the next tread center."""
        _, normalized_error, target_weight = (
            self._next_tread_foot_target_state()
        )
        return torch.square(normalized_error) * target_weight

    def _reward_stairs_foothold_lateral(self):
        """Reward left/right swing-foot placement around the centerline."""
        score, _, active = self._next_tread_lateral_target_state()
        return score * active

    def _reward_stairs_foothold_lateral_error(self):
        """Penalize crossing or widening the scheduled swing foothold."""
        _, normalized_error, active = self._next_tread_lateral_target_state()
        return torch.square(normalized_error) * active

    def _reward_stairs_foot_crossover(self):
        """Penalize either foot moving inward across its minimum half-width."""
        _, _, normalized_error, active = self._foot_crossover_state()
        return (
            torch.mean(torch.square(normalized_error), dim=1)
            * active.float()
        )

    def _reward_stairs_foot_lane_error(self):
        """Keep both feet on their own centered lateral stair lanes."""
        _, normalized_error, active = self._foot_lane_error_state()
        return (
            torch.mean(torch.square(normalized_error), dim=1)
            * active.float()
        )

    def _single_support_stability_state(self):
        """Return support masks and a dense body/action shake cost."""
        support = self.stable_contacts & self.contacts
        single_support = torch.sum(support.int(), dim=1) == 1
        roll_tilt = self.projected_gravity[:, 1]
        roll_rate = self.base_ang_vel[:, 0]
        lateral_velocity = self.base_lin_vel[:, 1]
        action_rate = torch.mean(
            torch.square(self.actions - self.last_actions), dim=1
        )
        action_accel = torch.mean(
            torch.square(
                self.actions
                + self.last_last_actions
                - 2.0 * self.last_actions
            ),
            dim=1,
        )
        cost = (
            torch.square(roll_tilt)
            + float(self.cfg.env.single_support_roll_rate_scale)
            * torch.square(roll_rate)
            + float(self.cfg.env.single_support_lateral_velocity_scale)
            * torch.square(lateral_velocity)
            + float(self.cfg.env.single_support_action_rate_scale)
            * action_rate
            + float(self.cfg.env.single_support_action_accel_scale)
            * action_accel
        )
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.80
        active = single_support & moving & upright
        return support, cost, active

    def _reward_stairs_single_support_stability(self):
        """Suppress roll/lateral/action shaking on either support foot."""
        _, cost, active = self._single_support_stability_state()
        return cost * active.float()

    def _reward_stairs_right_support_stability(self):
        """Extra damping for the measured right-support/left-swing shake."""
        support, cost, active = self._single_support_stability_state()
        right_support_only = support[:, 1] & ~support[:, 0]
        return cost * (active & right_support_only).float()

    def _reward_stairs_single_support(self):
        """Prefer a moving single-support gait over dual-foot hopping."""
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        single_support = torch.sum(contact_filt.int(), dim=1) == 1
        moving_forward = self.root_states[:, 7] > 0.02
        upright = -self.projected_gravity[:, 2] > 0.80
        return (single_support & moving_forward & upright).float()

    def _reward_stairs_swing_clearance(self):
        _, _, step_height = self._current_stair_targets()
        target = (
            step_height.unsqueeze(1)
            + self.cfg.rewards.swing_clearance_margin
        )
        foot_ground_height = self._sample_terrain_height_xy(
            self.feet_pos[:, :, :2]
        )
        clearance = self.feet_pos[:, :, 2] - foot_ground_height
        if self.include_gait_phase:
            swing = ~self.desired_contacts & ~self.contacts
        else:
            swing = ~self.contacts
        score = torch.exp(-40.0 * torch.square(clearance - target)) * swing.float()
        swing_count = torch.sum(swing.float(), dim=1)
        score = torch.sum(score, dim=1) / torch.clamp(swing_count, min=1.0)
        moving = self.root_states[:, 7] > 0.03
        has_support = torch.any(self.contacts, dim=1)
        return (
            score
            * moving.float()
            * has_support.float()
            * (swing_count > 0).float()
        )

    def _reward_stairs_stable_contact(self):
        horizontal_speed = torch.norm(self.feet_vel[:, :, :2], dim=2)
        stable = self.contacts & (horizontal_speed < 0.12)
        moving = self.root_states[:, 7] > 0.03
        upright = -self.projected_gravity[:, 2] > 0.8
        return torch.mean(stable.float(), dim=1) * moving.float() * upright.float()

    def _reward_stumble(self):
        if self.foot_force_sensor_forces is None:
            return super()._reward_stumble()
        horizontal = torch.norm(self.foot_force_sensor_forces[:, :, :2], dim=2)
        vertical = torch.abs(self.foot_force_sensor_forces[:, :, 2])
        return torch.any(horizontal > 5.0 * vertical, dim=1)

    def _reward_feet_contact_forces(self):
        if self.foot_force_sensor_forces is None:
            return super()._reward_feet_contact_forces()
        force_norm = torch.norm(self.foot_force_sensor_forces, dim=2)
        return torch.sum(
            torch.clamp(force_norm - self.cfg.rewards.max_contact_force, min=0.0),
            dim=1,
        )

    def _reward_feet_air_time(self):
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        self.feet_air_time += self.dt
        first_contact = self.contacts & ~self.last_contacts
        one_landing = torch.sum(first_contact.int(), dim=1) == 1
        opposite_was_supported = torch.flip(self.last_contacts, dims=[1])
        alternating_landing = (
            first_contact
            & opposite_was_supported
            & one_landing.unsqueeze(1)
        )
        reward = torch.sum(
            torch.clamp(self.feet_air_time - 0.10, min=0.0, max=0.25)
            * alternating_landing.float(),
            dim=1,
        )
        self.feet_air_time *= ~contact_filt
        moving = (self.root_states[:, 7] > 0.02).float()
        upright = (-self.projected_gravity[:, 2] > 0.80).float()
        # Landing bonus is an event, so cancel reward preparation's dt.
        return reward * moving * upright / self.dt
