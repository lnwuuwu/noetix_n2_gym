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
    select_height_indices,
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
        if fixed_level >= 0:
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
        self.contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
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
        self.top_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.top_position_reached_buf = torch.zeros_like(self.top_reached_buf)
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
        foot_surface_height = self._sample_terrain_height_xy(
            self.feet_pos[:, :, :2]
        ) - self.env_origins[:, 2].unsqueeze(1)
        foot_surface_height = torch.clamp(foot_surface_height, min=0.0)
        horizontal_speed = torch.norm(self.feet_vel[:, :, :2], dim=2)

        if self.foot_force_sensor_forces is not None:
            horizontal_force = torch.norm(
                self.foot_force_sensor_forces[:, :, :2], dim=2
            )
            vertical_force = torch.abs(self.foot_force_sensor_forces[:, :, 2])
            vertical_support = vertical_force > horizontal_force
        else:
            vertical_support = torch.ones_like(self.contacts)

        stable_support = (
            self.contacts
            & vertical_support
            & (horizontal_speed < 0.20)
        )
        attained_height = torch.where(
            stable_support,
            foot_surface_height,
            self.max_foot_contact_height,
        )
        raw_delta = torch.clamp(
            attained_height - self.max_foot_contact_height, min=0.0
        )

        # A valid step landing has exactly one newly contacting foot and the
        # opposite foot supported on the preceding frame. Synchronous hopping
        # and impacts into a vertical riser therefore receive no event reward.
        new_contact = self.contacts & ~self.last_contacts
        one_new_contact = torch.sum(new_contact.int(), dim=1) == 1
        opposite_was_supported = torch.flip(self.last_contacts, dims=[1])
        alternating_landing = (
            new_contact
            & opposite_was_supported
            & one_new_contact.unsqueeze(1)
            & stable_support
        )
        self.alternating_tread_event[:] = 0.0
        self.repeated_lead_event[:] = 0.0
        self.same_tread_join_event[:] = 0.0
        self.skipped_tread_event[:] = 0.0

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
            has_valid_landing = torch.any(alternating_landing, dim=1)
            candidate_foot = torch.argmax(
                alternating_landing.long(), dim=1
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
                has_valid_landing,
                candidate_tread,
                candidate_foot,
                previous_tread,
                previous_foot,
                self.last_joined_tread,
            )

            self.alternating_tread_event[:] = alternating_advance.float()
            self.repeated_lead_event[:] = repeated_lead.float()
            self.same_tread_join_event[:] = same_tread_join.float()
            self.skipped_tread_event[:] = skipped_tread.float()
            self.tread_advance_count += advanced.float()
            self.alternating_tread_count += alternating_advance.float()
            self.repeated_lead_count += repeated_lead.float()
            self.same_tread_join_count += same_tread_join.float()
            self.skipped_tread_count += skipped_tread.float()
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
        else:
            # Preserve the original n2_stairs event definition and checkpoint
            # behavior. The stricter tread sequence belongs only to the
            # dedicated n2_stairs_walk task.
            self.foot_contact_height_delta[:] = (
                raw_delta * alternating_landing.float()
            )
        # Even an invalid simultaneous landing consumes that height event, so
        # hopping cannot land once and collect it later by lifting one foot.
        self.max_foot_contact_height[:] = torch.maximum(
            self.max_foot_contact_height, attained_height
        )

    def _swing_knee_state(self):
        """Return scheduled swing mask and left/right knee flexion."""
        knees = self.dof_pos[:, self.knee_dof_idxs]
        if self.include_gait_phase:
            swing = ~self.desired_contacts & ~self.contacts
        else:
            swing = ~self.contacts
        return swing, knees

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

        expected_foot = torch.clamp(
            1 - self.last_advanced_foot, min=0, max=1
        )
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

        expected_is_swing = torch.gather(
            (~self.desired_contacts).long(),
            1,
            expected_foot.unsqueeze(1),
        ).squeeze(1).bool()
        grace_steps = int(
            getattr(self.cfg.env, "gait_reward_grace_s", 0.0) / self.dt
        )
        active = (
            (self.last_advanced_tread >= 1)
            & (self.last_advanced_tread < int(self.cfg.terrain.num_steps))
            & expected_is_swing
            & (self.episode_length_buf > grace_steps)
            & (self.root_states[:, 7] > 0.03)
            & (-self.projected_gravity[:, 2] > 0.80)
        )
        return score, normalized_error, active

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
        stable_top = self.top_position_reached_buf & upright & stable_support
        if self.enforce_walk_gait:
            centered = torch.abs(lateral_position) <= float(
                self.cfg.env.success_lateral_tolerance
            )
            facing_forward = torch.abs(yaw_error) <= float(
                self.cfg.env.success_yaw_tolerance
            )
            command_matched = torch.abs(
                forward_speed - self.commands[:, 0]
            ) <= float(self.cfg.env.top_speed_tolerance)
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
            command_consistent = (
                self.command_error_sum / elapsed_steps
                <= float(self.cfg.env.success_max_mean_command_error)
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
                & command_matched
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
        else:
            self.top_stable_time[:] = 0.0
            self.top_reached_buf[:] = stable_top

        grace_steps = int(self.cfg.env.progress_grace_s / self.dt)
        stall_steps = int(self.cfg.env.stall_timeout_s / self.dt)
        self.stall_buf[:] = torch.logical_and(
            self.episode_length_buf > grace_steps,
            (self.episode_length_buf - self.last_progress_step) > stall_steps,
        )
        self.stall_buf &= ~self.top_reached_buf

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
        self.stall_buf &= ~physical_failure
        self.path_failure_buf &= ~physical_failure
        # Reaching the top is a true terminal state and must not receive
        # timeout bootstrapping in PPO. A physical failure on the final time
        # step is likewise a failure, not a benign timeout.
        self.time_out_buf &= ~self.top_reached_buf
        self.time_out_buf &= ~physical_failure
        self.reset_buf |= self.stall_buf
        self.reset_buf |= self.path_failure_buf
        self.reset_buf |= self.top_reached_buf

    def _update_terrain_curriculum(self, env_ids):
        valid = self.episode_started[env_ids]
        success = self.top_reached_buf[env_ids] & valid

        failure = (
            self.fall_event_buf[env_ids]
            | self.stall_buf[env_ids]
            | self.path_failure_buf[env_ids]
            | (self.time_out_buf[env_ids] & ~success)
        ) & valid

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

        self.last_episode_success[env_ids] = success
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
        self.top_reached_buf[env_ids] = False
        self.top_position_reached_buf[env_ids] = False
        self.stall_buf[env_ids] = False
        self.fall_event_buf[env_ids] = False
        self.contacts[env_ids] = False
        self.last_contacts[env_ids] = False
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
        """Return the per-environment knee target for the current riser."""
        _, _, step_height = self._current_stair_targets()
        return torch.clamp(
            float(self.cfg.env.swing_knee_base_target)
            + float(self.cfg.env.swing_knee_height_gain) * step_height,
            max=float(self.cfg.env.swing_knee_max_target),
        ).unsqueeze(1)

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

    def _reward_stairs_success(self):
        # One-shot terminal event; see the dt note above.
        return self.top_reached_buf.float() / self.dt

    def _reward_termination(self):
        # In this completion task, an unfinished time limit is also a failure
        # even though PPO still bootstraps its critic value at that time limit.
        failure = self.reset_buf.bool() & ~self.top_reached_buf
        return failure.float() / self.dt

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
        return aligned * upright

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
        score, _, active = self._next_tread_foot_target_state()
        return score * active.float()

    def _reward_stairs_next_tread_target_error(self):
        """Penalize remaining fore-aft distance to the next tread center."""
        _, normalized_error, active = self._next_tread_foot_target_state()
        return torch.square(normalized_error) * active.float()

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
