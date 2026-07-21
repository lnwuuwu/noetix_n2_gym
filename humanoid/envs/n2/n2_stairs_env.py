"""N2 environment specialized for stable, directional upstairs locomotion."""

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
import torch

from humanoid.envs.n2.n2_env import N2Env
from humanoid.utils.stairs_terrain import select_height_indices
from humanoid.utils.terrain import N2StairsTerrain


class N2StairsEnv(N2Env):
    """N2 task with verified stair geometry, climb curriculum, and metrics."""

    _PROPRIO_OBS = 63

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
        self.ankle_dof_idxs = [
            dof_index["L_leg_ankle_joint"],
            dof_index["R_leg_ankle_joint"],
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
        expected_actor_heights = self.cfg.env.num_single_obs - self._PROPRIO_OBS
        if len(actor_indices) != expected_actor_heights:
            raise ValueError(
                "n2_stairs Actor height count {} does not match observation config {}".format(
                    len(actor_indices), expected_actor_heights
                )
            )

        self.stair_top_x = torch.tensor(
            self.terrain.stair_top_x, dtype=torch.float, device=self.device
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

    def _get_noise_scale_vec(self, cfg):
        """Noise layout for 63 proprioceptive + compact terrain observations."""
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        self.add_noise = cfg.noise.add_noise
        scales = cfg.noise.noise_scales

        noise_vec[0:3] = 0.0
        noise_vec[3:6] = scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[6:9] = scales.gravity
        noise_vec[9:27] = scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[27:45] = scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[45:63] = 0.0
        noise_vec[63:] = (
            scales.height_measurements * self.obs_scales.height_measurements
        )
        return noise_vec

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
        proprio = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
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
            ("proprio", proprio, self._PROPRIO_OBS),
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

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

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
        self.top_reached_buf[:] = (
            self.top_position_reached_buf & upright & stable_support
        )

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
        # Reaching the top is a true terminal state and must not receive
        # timeout bootstrapping in PPO. A physical failure on the final time
        # step is likewise a failure, not a benign timeout.
        self.time_out_buf &= ~self.top_reached_buf
        self.time_out_buf &= ~physical_failure
        self.reset_buf |= self.stall_buf
        self.reset_buf |= self.top_reached_buf

    def _update_terrain_curriculum(self, env_ids):
        valid = self.episode_started[env_ids]
        success = self.top_reached_buf[env_ids] & valid

        failure = (
            self.fall_event_buf[env_ids]
            | self.stall_buf[env_ids]
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

        self.last_episode_success[env_ids] = success
        self.last_episode_top_reached[env_ids] = top_reached
        self.last_episode_forward_progress[env_ids] = forward
        self.last_episode_climb_height[env_ids] = climb
        self.last_episode_survival_time[env_ids] = survival
        self.last_episode_fall[env_ids] = fall
        self.last_episode_stall[env_ids] = stall

        super().reset_idx(env_ids)

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
                "stairs_forward_distance": masked_mean(forward),
                "stairs_climb_height": masked_mean(climb),
                "stairs_survival_time": masked_mean(survival),
                "stairs_command_x": masked_mean(episode_command),
            }
        )
        self.extras["episode"]["max_command_x"] = self._command_upper_for_levels(
            self.terrain_levels[env_ids]
        ).mean()
        self.extras["episode"]["min_command_x"] = float(
            self.command_ranges["lin_vel_x"][0]
        )

        self.best_forward_progress[env_ids] = 0.0
        self.progress_checkpoint[env_ids] = 0.0
        self.max_climb_height[env_ids] = 0.0
        self.last_progress_step[env_ids] = 0
        self.terrain_height_delta[env_ids] = 0.0
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
        return {
            "version": 1,
            "task_name": getattr(self.cfg.env, "task_name", None),
            "terrain_levels": self.terrain_levels.detach().cpu(),
            "success_streak": self.curriculum_success_streak.detach().cpu(),
            "failure_streak": self.curriculum_failure_streak.detach().cpu(),
        }

    def load_checkpoint_state(self, state):
        """Restore curriculum state while allowing a different environment count."""
        if not state or not self.cfg.terrain.curriculum:
            return
        if int(getattr(self.cfg.terrain, "fixed_level", -1)) >= 0:
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
        world_forward_velocity = torch.clamp(self.root_states[:, 7], 0.0, 0.8)
        lateral_gate = torch.exp(-4.0 * torch.abs(self.root_states[:, 8]))
        return world_forward_velocity * upright * supported * lateral_gate

    def _reward_tracking_lin_vel(self):
        """Track +X without rewarding a stationary robot at the stair foot."""
        world_velocity = self.root_states[:, 7:9]
        error = torch.sum(
            torch.square(self.commands[:, :2] - world_velocity), dim=1
        )
        score = torch.exp(-5.0 * error)
        target = torch.clamp(self.commands[:, 0], min=0.05)
        progress_gate = torch.clamp(
            world_velocity[:, 0] / (0.5 * target), min=0.0, max=1.0
        )
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)
        supported = torch.any(self.contacts, dim=1).float()
        return score * progress_gate * upright * supported

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

    def _reward_stairs_no_progress(self):
        grace_steps = int(self.cfg.env.progress_grace_s / self.dt)
        active = self.episode_length_buf > grace_steps
        slow = self.root_states[:, 7] < 0.03
        return (active & slow & ~self.top_reached_buf).float()

    def _reward_stairs_double_flight(self):
        grace_steps = int(self.cfg.env.progress_grace_s / self.dt)
        both_airborne = ~torch.any(self.contacts, dim=1)
        return (both_airborne & (self.episode_length_buf > grace_steps)).float()

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
        first_contact = contact_filt & (self.feet_air_time > 0.0)
        reward = torch.sum(
            torch.clamp(self.feet_air_time - 0.15, min=0.0, max=0.35)
            * first_contact.float(),
            dim=1,
        )
        self.feet_air_time *= ~contact_filt
        return reward * (self.root_states[:, 7] > 0.03).float()
