import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float
from isaacgym import gymapi, gymutil
from humanoid.utils.math import wrap_to_pi
from humanoid.utils.terrain import ParkourCourseTerrain, ParkourTerrain
from humanoid.envs.n2.n2_perceptive_env import N2PerceptiveEnv


class N2ParkourEnv(N2PerceptiveEnv):
    """Extreme Parkour(arXiv:2309.14341)架构的复刻。
    """

    # ---------------- terrain / goals ----------------
    def _create_terrain_impl(self):
        """Use either the training grid or the separate continuous course."""
        terrain_cls = (ParkourCourseTerrain
                       if getattr(self.cfg.terrain, 'course_mode', False)
                       else ParkourTerrain)
        self.terrain = terrain_cls(self.cfg.terrain, self.num_envs)

    def create_sim(self):
        # 复刻基类 create_sim，但把地形类换掉。只在 trimesh/heightfield 下有意义。
        if getattr(self.cfg.asset, 'require_self_collisions', False):
            mode = getattr(self.cfg.asset, 'self_collision_mode', None)
            if self.cfg.asset.self_collisions != 0:
                raise ValueError(
                    "该任务要求自碰撞，但 asset.self_collisions=%r（必须为 0）"
                    % self.cfg.asset.self_collisions)
            if mode not in (None, 'cross_leg'):
                raise ValueError(
                    "不支持的自碰撞模式: %r" % mode)
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self._create_terrain_impl()
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _check_goal_reach_consistency(self):
        """goal_reach_dist 必须小于最小 goal 间距，否则指针连跳、课程虚高。
        """
        # 最小间距是【所有启用地形类型】里最小的那个，不能只看台阶——多地形混训后
        # 台阶不一定还是最紧的那一种，写死 parkour_x_range 会漏掉新类型。
        gap = self.terrain.min_goal_spacing()
        reach = self.cfg.rewards.goal_reach_dist
        if gap > 0 and reach >= gap:
            raise ValueError(
                "goal_reach_dist=%.3f 必须 < 各地形类型的最小 goal 间距 %.3f，"
                "否则相邻 goal 落在到达半径内、指针连跳导致课程虚高" % (reach, gap))

    def _reward_stumble(self):
        """摆动脚撞到竖直面的惩罚：连续量 + 只算摆动相。

        基类版本(legged_robot.py:1159)是二值判据 any(|Fxy| > 5|Fz|)，两个毛病：
          1) 轻蹭和狠踢同价，梯度不带强度信息；
          2) 它是【比值】判据，摆动脚只要分担了一点体重比值就不成立，会漏掉。
        实测(model_999，上楼梯 lvl5)：触发率 2.61%，累计只占正奖励栈的 0.82%，
        所以"加了 stumble 还是踢台阶"不是权重不够，是这一项根本没有分辨力。

        阈值全部取自实测接触力分布(自重 327N)：
          支撑相 Fz 中位数 207.6N、p90 381N；摆动相 Fz p90 只有 100.7N
            -> stance_force=80N 能干净地把"还没承重的脚"分出来。
          摆动相 Fxy p90=16.8N(正常噪声)、p99=265N(真撞上)
            -> min_force=20N 滤掉噪声，ref_force=200N 做归一化。
        归一化到单脚 [0,1]、两脚合计 [0,2]，与原来的二值量纲相当，所以 scale 不用重调；
        同时天然封顶，不会像 world_progress 那次被单步尖峰把 PPO 的价值函数打飞。
        """
        c = self.cfg.rewards
        f = self.contact_forces[:, self.feet_indices, :]
        fxy = torch.norm(f[:, :, :2], dim=2)
        fz = torch.abs(f[:, :, 2])
        swing = fz < c.stumble_stance_force              # 脚还没真正承重
        hit = ((fxy - c.stumble_min_force).clip(min=0.) / c.stumble_ref_force).clip(max=1.)
        return torch.sum(hit * swing.float(), dim=1)

    def _check_spawn_clearance(self, max_drop=0.15, samples=64):
        """出生点抖动之后必须仍落在实地上。

        goals[0] 由地形函数决定、抖动幅度由 terrain 配置决定，两者分处不同文件，改一边
        很容易忘掉另一边——踏石就踩过：goals[0] 放在平台边缘(离末端 0.1m)，配 ±0.3m 抖动
        后 6.7% 的出生点直接落到坑上、最深掉 0.52m，而机器人还是按平台高度摆的，等于每
        次 reset 都从半空掉进沟里。这里在构造时把这个跨段耦合钉死。
        """
        jit = getattr(self.cfg.terrain, 'parkour_spawn_jitter', 0.0)
        if jit <= 0 or not hasattr(self.terrain, 'height_field_raw'):
            return
        hs = self.cfg.terrain.horizontal_scale
        vs = self.cfg.terrain.vertical_scale
        border = int(self.cfg.terrain.border_size / hs)
        hf = self.terrain.height_field_raw
        g0 = self.terrain.goals[:, :, 0, :]                      # (rows, cols, 3)
        # 角点最危险，直接取抖动方框的边界网格
        off = np.linspace(-jit, jit, int(np.sqrt(samples)))
        dx, dy = np.meshgrid(off, off, indexing='ij')
        px = np.clip(((g0[..., None, None, 0] + dx) / hs).astype(int) + border,
                     0, hf.shape[0] - 1)
        py = np.clip(((g0[..., None, None, 1] + dy) / hs).astype(int) + border,
                     0, hf.shape[1] - 1)
        drop = hf[px, py] * vs - g0[..., None, None, 2]
        worst = float(drop.min())
        if worst < -max_drop:
            i, j = np.unravel_index(np.argmin(drop.min(axis=(2, 3))), drop.shape[:2])
            raise ValueError(
                "出生点抖动 ±%.2fm 会把机器人扔到比 goals[0] 低 %.2fm 的地方"
                "(最差在 row %d / col %d)。把该地形的 goals[0] 往平台内挪，"
                "或调小 terrain.parkour_spawn_jitter。" % (jit, -worst, i, j))

    def _init_goal_buffers(self):
        self._check_goal_reach_consistency()
        self._check_spawn_clearance()
        # (num_rows, num_cols, num_goals, 3) 的世界坐标路点表
        self.terrain_goals = torch.tensor(self.terrain.goals, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.num_goals = self.terrain_goals.shape[2]
        # 见 reset_idx：goals[0] 是出生点，第一个真正的目标是 goals[1]
        self.cur_goal_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.reached_goals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_delta_yaw = torch.zeros(self.num_envs, device=self.device)
        if getattr(self.cfg.terrain, 'course_mode', False):
            self.course_goal_stage = torch.tensor(
                self.terrain.goal_stage_indices, dtype=torch.long,
                device=self.device, requires_grad=False)
            self.course_stage_goal_starts = torch.tensor(
                self.terrain.course_stage_goal_starts, dtype=torch.long,
                device=self.device, requires_grad=False)
            self.course_stage_spawn_goals = torch.tensor(
                self.terrain.course_stage_spawn_goals, dtype=torch.float,
                device=self.device, requires_grad=False)
            self.course_stage_idx = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device)
            self.course_finished = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            speeds = getattr(
                self.cfg.commands, 'course_command_speeds',
                [self.cfg.commands.ranges.lin_vel_x[1]]
                * len(self.terrain.course_stage_names))
            if len(speeds) != len(self.terrain.course_stage_names):
                raise ValueError(
                    "course_command_speeds 必须与 course stage 数量相同")
            if any(float(speed) <= 0 for speed in speeds):
                raise ValueError("course_command_speeds 必须全部为正数")
            self.course_command_speeds = torch.tensor(
                speeds, dtype=torch.float, device=self.device,
                requires_grad=False)
            self._apply_course_commands()

    def _env_goals(self):
        """当前每个环境所在地形块的 goal 表 -> (N, num_goals, 3)。"""
        return self.terrain_goals[self.terrain_levels, self.terrain_types]

    def _cur_goal(self):
        g = self._env_goals()
        return g[torch.arange(self.num_envs, device=self.device), self.cur_goal_idx]

    def _apply_course_commands(self, env_ids=None):
        """Apply the MuJoCo-matched speed for each robot's current stage."""
        if not getattr(self.cfg.terrain, 'course_mode', False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        stage = self.course_stage_idx[env_ids]
        speed = self.course_command_speeds[stage]
        speed = torch.where(
            self.course_finished[env_ids], torch.zeros_like(speed), speed)
        self.commands[env_ids, 0] = speed
        self.commands[env_ids, 1] = 0.
        self.commands[env_ids, 2] = 0.
        if hasattr(self, 'standing_cmd'):
            self.standing_cmd[env_ids] = (
                torch.norm(self.commands[env_ids, :3], dim=1)
                <= self.min_cmd_vel)

    def _update_goals(self):
        """推进 goal 指针：走到当前 goal 的 reach 半径内就切下一个。"""
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        cur = self._cur_goal()
        self.target_pos_rel = cur[:, :2] - self.root_states[:, :2]
        dist = torch.norm(self.target_pos_rel, dim=1)
        reach = self.cfg.rewards.goal_reach_dist


        passed = (self.target_pos_rel[:, 0] < 0) & \
                 (self.target_pos_rel[:, 1].abs() < self.cfg.rewards.goal_pass_lateral_tol)
        reached_or_passed = (dist < reach) | passed
        advance = reached_or_passed & (self.cur_goal_idx < self.num_goals - 1)
        self.cur_goal_idx[advance] += 1
        self.reached_goals[advance] += 1
        if getattr(self.cfg.terrain, 'course_mode', False):
            # The pointer cannot advance beyond the final goal. Record that
            # arrival once, then stand without closing the simulation/viewer.
            newly_finished = (
                reached_or_passed
                & (self.cur_goal_idx == self.num_goals - 1)
                & ~advance
                & ~self.course_finished)
            self.course_finished |= newly_finished
            self.reached_goals[newly_finished] += 1
        # 指针推进后重新取，保证 reward/obs 用的是同一步的目标
        cur = self._cur_goal()
        self.target_pos_rel = cur[:, :2] - self.root_states[:, :2]
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        target_yaw = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
        self.goal_delta_yaw = wrap_to_pi(target_yaw - yaw)
        if getattr(self.cfg.terrain, 'course_mode', False):
            self.course_stage_idx = self.course_goal_stage[self.cur_goal_idx]
            self._apply_course_commands()
            # The final goal is behind once crossed; hide that stale 180-degree
            # request while holding a zero-speed terminal command.
            self.target_pos_rel[self.course_finished] = 0.
            self.goal_delta_yaw[self.course_finished] = 0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_goals()
        if getattr(self.cfg.terrain, 'course_mode', False):
            # Once all five stages are clear, keep the terminal scene alive
            # instead of letting the ordinary episode timeout restart it.
            self.episode_length_buf[self.course_finished] = 0

    # ---------------- commands: 只发前进速度 ----------------
    def _resample_commands(self, env_ids):
        """EP 的指令空间只有 vx。这里在基类采样之后把 vy/wz 清零、vx 取正。
        """
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        if getattr(self.cfg.terrain, 'course_mode', False):
            if hasattr(self, 'course_stage_idx'):
                self._apply_course_commands(env_ids)
            else:
                self.commands[env_ids, 0] = float(
                    self.cfg.commands.course_command_speeds[0])
                self.commands[env_ids, 1] = 0.
                self.commands[env_ids, 2] = 0.
            return
        lo = self.cfg.commands.parkour_min_vx
        # hi 兜底：play.py 的 CONTROL_ROBOT 分支会把 lin_vel_x 范围压成 [0,0]，
        # 那样 torch_rand_float(0.3, 0.0) 会反向取到 [0, 0.3]，指令悄悄失真。
        hi = max(self.command_ranges["lin_vel_x"][1], lo)
        self.commands[env_ids, 0] = torch_rand_float(lo, hi, (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0.
        self.commands[env_ids, 2] = 0.

    # ---------------- spawn / reset ----------------
    def _reset_root_states(self, env_ids):
        """出生在 goals[0]（跑道起点）而不是块中心——EP 的机器人也是从通道口出发。"""
        super()._reset_root_states(env_ids)
        if not self.custom_origins or len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        if getattr(self.cfg.terrain, 'course_mode', False):
            # A fall retries only the current stage, matching the MuJoCo
            # campaign. Earlier cleared stages remain cleared.
            g0 = self.course_stage_spawn_goals[
                self.course_stage_idx[env_ids]]
        else:
            g0 = self.terrain_goals[
                self.terrain_levels[env_ids], self.terrain_types[env_ids], 0]
        jit = self.cfg.terrain.parkour_spawn_jitter
        self.root_states[env_ids, 0] = g0[:, 0] + torch_rand_float(
            -jit, jit, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 1] = g0[:, 1] + torch_rand_float(
            -jit, jit, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 2] = g0[:, 2] + self.base_init_state[2] + 0.05
        ids32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def reset_idx(self, env_ids):
        retry_stage = None
        if (getattr(self.cfg.terrain, 'course_mode', False)
                and hasattr(self, 'course_stage_idx')):
            retry_stage = self.course_stage_idx[env_ids].clone()
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        # 从 1 开始：goals[0] 就是出生点本身，从 0 开始会在出生瞬间白送一次"到达"，
        # reached_goals 恒定虚高 1。第一个真正的目标是 goals[1]（第一级台阶）。
        self.cur_goal_idx[env_ids] = 1
        self.reached_goals[env_ids] = 0
        if getattr(self.cfg.terrain, 'course_mode', False):
            if retry_stage is None:
                retry_stage = torch.zeros_like(env_ids, dtype=torch.long)
            self.course_finished[env_ids] = False
            self.course_stage_idx[env_ids] = retry_stage
            self.cur_goal_idx[env_ids] = self.course_stage_goal_starts[
                retry_stage]
            goals_per_stage = self.course_stage_goal_starts[1] - \
                self.course_stage_goal_starts[0]
            self.reached_goals[env_ids] = retry_stage * goals_per_stage
            self._apply_course_commands(env_ids)

    # ---------------- curriculum: 按走到第几个 goal 升降级 ----------------
    def _update_terrain_curriculum(self, env_ids):
        """EP 的课程是"走完这条通道就升级"。这里直接用 goal 计数，比投影距离更贴切：
        它天然只认"沿通道穿越"，横向乱走不会推进 goal 指针。"""
        if not self.init_done or len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        reached = self.reached_goals[env_ids]
        move_up = reached >= self.cfg.terrain.parkour_goals_to_level_up
        move_down = (reached <= self.cfg.terrain.parkour_goals_to_level_down) & ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                         self.terrain_types[env_ids]]

    # ---------------- observations: 加入到 goal 的 delta yaw ----------------
    def compute_observations(self):
        if not hasattr(self, 'goal_delta_yaw'):
            self._init_goal_buffers()
            self._update_goals()
        # EP 观测的是"到 goal 的相对朝向"。用 cos/sin 而不是裸角度，避免 ±pi 处的跳变。
        goal_obs = torch.stack((torch.cos(self.goal_delta_yaw),
                                torch.sin(self.goal_delta_yaw)), dim=1)

        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            goal_obs,
        ), dim=-1)

        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.) * self.obs_scales.height_measurements
        obs_buf = torch.cat((obs_buf, heights), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            goal_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.payload * 0.5,
            self.friction_coeffs,
            self.restitution_coeffs,
            self.Kp_factors,
            self.Kd_factors,
            self.motor_strength,
            self.contacts,
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            obs_buf_all = torch.stack([self.obs_history[i]
                                       for i in range(self.obs_history.maxlen)], dim=1)
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        else:
            self.obs_buf = obs_now

    # ---------------- rewards: EP 的两项核心 ----------------
    def _reward_tracking_goal_vel(self):
        """min(<d_hat, v_world>, vx_cmd) / vx_cmd，逐字对应 EP 的 _reward_tracking_goal_vel。

        """
        if not hasattr(self, 'target_pos_rel'):
            return torch.zeros(self.num_envs, device=self.device)
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        d_hat = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        cmd = self.commands[:, 0]
        rew = torch.minimum(torch.sum(d_hat * cur_vel, dim=-1), cmd) / (cmd + 1e-5)
        return torch.clamp(rew, min=-1.0, max=1.0)

    def _reward_overspeed(self):
        """Bounded penalty for moving faster than the commanded planar speed.

        ``tracking_goal_vel`` intentionally saturates once projected velocity
        reaches ``vx_cmd`` (matching Extreme Parkour), so it cannot distinguish
        0.5 m/s from a dangerous 2.0 m/s descent.  This complementary term uses
        total world-frame XY speed, leaves a small tolerance for gait
        oscillation, and is squared then capped so a fall cannot create an
        unbounded PPO target.
        """
        planar_speed = torch.norm(self.root_states[:, 7:9], dim=1)
        commanded_speed = torch.abs(self.commands[:, 0])
        excess = torch.clamp(
            planar_speed - commanded_speed
            - self.cfg.rewards.overspeed_tolerance,
            min=0.0)
        normalized = excess / self.cfg.rewards.overspeed_reference
        return torch.clamp(
            torch.square(normalized),
            max=self.cfg.rewards.overspeed_max_penalty)

    def _reward_feet_separation(self):
        """Bounded penalty for narrow or crossed left/right ankle positions.

        The signed distance is measured along the robot's yaw-frame left
        axis, so normal turning and terrain height differences are harmless.
        Physical cross-leg collision prevents penetration; this term teaches
        the policy to avoid reaching that contact in the first place.
        """
        if self.feet_pos.shape[1] != 2:
            raise RuntimeError(
                "feet_separation 奖励要求恰好两只脚，当前为 %d"
                % self.feet_pos.shape[1])
        if not hasattr(self, '_feet_lr_slots'):
            foot_body_names = [
                self.body_names[index]
                for index in self.feet_indices.detach().cpu().tolist()]
            left = [
                slot for slot, name in enumerate(foot_body_names)
                if name.startswith("L_leg_")]
            right = [
                slot for slot, name in enumerate(foot_body_names)
                if name.startswith("R_leg_")]
            if len(left) != 1 or len(right) != 1:
                raise RuntimeError(
                    "无法唯一识别左右脚: %r" % foot_body_names)
            self._feet_lr_slots = (left[0], right[0])
        left_slot, right_slot = self._feet_lr_slots
        forward = quat_apply(self.base_quat, self.forward_vec)[:, :2]
        forward = forward / torch.norm(
            forward, dim=1, keepdim=True).clamp(min=1e-6)
        left_axis = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
        left_minus_right = (
            self.feet_pos[:, left_slot, :2]
            - self.feet_pos[:, right_slot, :2])
        signed_separation = torch.sum(
            left_minus_right * left_axis, dim=1)
        minimum = self.cfg.rewards.feet_separation_min
        shortfall = torch.clamp(
            minimum - signed_separation, min=0.0) / minimum
        return torch.clamp(
            torch.square(shortfall),
            max=self.cfg.rewards.feet_separation_max_penalty)

    def _reward_tracking_yaw(self):
        """exp(-|Δψ|)，Δψ 是到当前 goal 的方位角误差（EP 的 _reward_tracking_yaw）。"""
        if not hasattr(self, 'goal_delta_yaw'):
            return torch.zeros(self.num_envs, device=self.device)
        return torch.exp(-torch.abs(self.goal_delta_yaw))

    def _reward_goal_reached(self):
        """踩到一个新 goal 的稀疏奖励。EP 没有这一项（
        这里保留一个很小的权重作为课程信号，可在 config 里置 0 完全对齐 EP。"""
        if not hasattr(self, 'cur_goal_idx'):
            return torch.zeros(self.num_envs, device=self.device)
        return (self.cur_goal_idx > 0).float() * 0.  # 由 config scale 控制，默认关闭

    # ---------------- debug 可视化：把 goal 画出来 ----------------
    def _draw_debug_vis(self):
        """在基类的高度图散点之外，额外画出本块地形的 goal 路点。
        """
        super()._draw_debug_vis()
        if self.viewer is None or not hasattr(self, 'cur_goal_idx'):
            return
        goals = self._env_goals()          # (N, num_goals, 3)
        pending = gymutil.WireframeSphereGeometry(0.08, 6, 6, None, color=(0, 1, 0))
        current = gymutil.WireframeSphereGeometry(0.16, 8, 8, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            gi = int(self.cur_goal_idx[i])
            g = goals[i].cpu().numpy()
            for k in range(g.shape[0]):
                geom = current if k == gi else pending
                pose = gymapi.Transform(gymapi.Vec3(g[k, 0], g[k, 1], g[k, 2] + 0.05), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

    # ---------------- 步态：堵住"单腿拖行"这个局部最优 ----------------
    def _reward_feet_air_time(self):
        """每只脚独立计腾空时长后相加，不再对两脚取 min。
        """
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        if not hasattr(self, 'last_air_time'):
            self.last_air_time = torch.zeros_like(self.feet_air_time)

        touchdown = contact_filt & (self.feet_air_time > 0)
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.last_air_time = torch.where(touchdown, self.feet_air_time, self.last_air_time)
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt

        stale = self.feet_contact_time > self.cfg.rewards.feet_air_time_stale
        self.last_air_time = torch.where(stale, torch.zeros_like(self.last_air_time),
                                         self.last_air_time)

        single_stance = torch.sum(contact_filt.int(), dim=1) == 1
        per_foot = torch.clamp(self.last_air_time, max=0.5)
        rew = 0.5 * torch.sum(per_foot, dim=1) * single_stance
        return rew
