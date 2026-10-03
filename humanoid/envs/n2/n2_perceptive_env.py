import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi
from humanoid.envs.n2.n2_10dof_env import N2_10dof_Env

class N2PerceptiveEnv(N2_10dof_Env):
    # ---- 世界系参考朝向 (反绕路) ----
    # 参考 yaw 必须【积分】指令偏航率，不能在 resample 时冻结：commands[:,2] 可以
    # 在一段 5~15s 的指令内要求最多 ±1 rad/s，冻结的目标会按设计漂走 ±15 rad，
    # 于是这两项被偏航指令本身饱和、不含任何绕路信息(实测完美服从的机器人得分与
    # 训练策略相同)，反而在策略观测不到的维度上注入方差。
    # 积分之后：服从全部三个指令通道即得满分，只有【未被指令的】偏航偏移(绕路转弯)
    # 或位置上的侧移/后退才扣分。它罚的是偏航误差的积分——正是速率型 tracking_ang_vel
    # 罚不到的东西(转身 2s 只付一次速率误差，之后 10s 照拿满额 tracking_lin_vel)。
    def _resample_commands(self, env_ids):
        """Seed the reference yaw to the robot's actual yaw whenever a new
        command is drawn, so each command segment starts with zero heading
        error and any drift accumulated under the previous command is
        forgiven.

        Reads the quaternion from root_states rather than self.base_quat on
        purpose: this is called from reset_idx AFTER _reset_root_states but
        BEFORE the "fix reset gravity bug" block that refreshes base_quat
        (legged_robot.py:219), so base_quat is still the pre-reset orientation
        at this point while root_states is already the new spawn."""
        if not hasattr(self, 'yaw_ref'):
            self._init_world_progress_buffers()
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        if getattr(self.cfg.commands, 'stairs_forward_only', False):
            on_stairs = self._stairs_env_mask()[env_ids]
            if torch.any(on_stairs):
                ids = env_ids[on_stairs]
                # 站立指令保持站立（不把它强行变成前进），与基类语义一致
                moving = torch.norm(self.commands[ids, :3], dim=1) > self.min_cmd_vel
                ids = ids[moving]
                if len(ids) > 0:
                    lo = self.cfg.commands.stairs_min_vx
                    hi = max(self.command_ranges["lin_vel_x"][1], lo)
                    self.commands[ids, 0] = torch_rand_float(
                        lo, hi, (len(ids), 1), device=self.device).squeeze(1)
                    self.commands[ids, 1] = 0.
                    self.commands[ids, 2] = 0.

        forward = quat_apply(self.root_states[env_ids, 3:7], self.forward_vec[env_ids])
        self.yaw_ref[env_ids] = torch.atan2(forward[:, 1], forward[:, 0])

    def _stairs_env_mask(self):
        """每个环境是否位于直行楼梯列（terrain_proportions 的 index 7/8）。"""
        if not hasattr(self, 'terrain_types'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, 'stairs_tile'):
            self.stairs_tile = self._stairs_tile_mask()
        return self.stairs_tile[self.terrain_types]

    def _init_world_progress_buffers(self):
        self.yaw_ref = torch.zeros(self.num_envs, device=self.device)
        self.world_heading_err = torch.zeros(self.num_envs, device=self.device)
        self.commands_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self.commands_world_speed = torch.zeros(self.num_envs, device=self.device)
        self.world_progress_accum = torch.zeros(self.num_envs, device=self.device)

    def _post_physics_step_callback(self):
        # super() resamples any expired commands (seeding yaw_ref for those
        # envs) and refreshes standing_cmd; the world reference must be built
        # on top of that, and before check_termination/compute_reward, which
        # post_physics_step calls straight after this (legged_robot.py:113-117).
        super()._post_physics_step_callback()
        self._update_world_reference()

    def _update_world_reference(self):
        """Integrate the commanded yaw rate into yaw_ref, leak-clamp it to the
        robot's actual yaw, and rebuild the world-frame command direction."""
        if not hasattr(self, 'yaw_ref'):
            self._init_world_progress_buffers()

        self.yaw_ref += self.commands[:, 2] * self.dt

        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        # 泄漏钳制：限制参考 yaw 领先机器人的幅度。
        max_err = self.cfg.rewards.world_heading_max_err
        self.world_heading_err = torch.clamp(wrap_to_pi(self.yaw_ref - yaw), -max_err, max_err)
        self.yaw_ref = yaw + self.world_heading_err

        # 用【参考】yaw 而非实际 yaw 旋转指令——两者之差就是全部的反绕路信号。
        c, s = torch.cos(self.yaw_ref), torch.sin(self.yaw_ref)
        vx, vy = self.commands[:, 0], self.commands[:, 1]
        world_vel_cmd = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=1)
        self.commands_world_speed = torch.norm(world_vel_cmd, dim=1)
        self.commands_world_dir = world_vel_cmd / self.commands_world_speed.clamp(min=1e-6).unsqueeze(1)

        # 课程用的方向性进展。逐步累加(而非记录 episode 起点位移)
        prog = torch.sum(self.root_states[:, 7:9] * self.commands_world_dir, dim=1)

        on_stairs = self._stairs_env_mask()
        prog = torch.where(on_stairs, self.root_states[:, 7], prog)

        self.world_progress_accum += prog * self.dt

    def reset_idx(self, env_ids):
        # super() runs _update_terrain_curriculum first, which consumes
        # world_progress_accum for the episode that just ended, so only zero
        # it afterwards.
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, 'world_progress_accum'):
            self.world_progress_accum[env_ids] = 0.

    def _stairs_tile_mask(self):

        props = self.cfg.terrain.terrain_proportions
        cum = [float(np.sum(props[:i + 1])) for i in range(len(props))]
        n = self.cfg.terrain.num_cols
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for j in range(n):
            choice = j / n + 0.001
            idx = next((k for k, c in enumerate(cum) if choice < c), -1)
            mask[j] = idx in (7, 8)
        return mask

    def _reset_root_states(self, env_ids):

        super()._reset_root_states(env_ids)
        if not self.custom_origins or len(env_ids) == 0:
            return
        if not hasattr(self, 'stairs_tile'):
            self.stairs_tile = self._stairs_tile_mask()
        on_stairs = self.stairs_tile[self.terrain_types[env_ids]]
        if not torch.any(on_stairs):
            return
        ids = env_ids[on_stairs]
        plat = getattr(self.cfg.terrain, 'stairs_platform_size', 1.5)
        base_x = self.env_origins[ids, 0] - self.terrain.env_length / 2. + plat / 2.
        jit = min(plat * 0.4, 0.5)
        self.root_states[ids, 0] = base_x + torch_rand_float(
            -jit, jit, (len(ids), 1), device=self.device).squeeze(1)
        # 底部平台高度=0，覆盖掉基类按块中心高度设的 root_z
        self.root_states[ids, 2] = self.base_init_state[2] + 0.05
        ids32 = ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def _update_terrain_curriculum(self, env_ids):

        if not self.init_done:
            return
        if not hasattr(self, 'world_progress_accum'):
            super()._update_terrain_curriculum(env_ids)
            return
        progress = self.world_progress_accum[env_ids]

        # up-distance: base uses env_length/2 (2m). With center spawn on stairs that
        # means climbing the whole upper half-tile to level up -- too steep for early
        # stairs, which pins the stairs columns at level 0. Configurable, default 1.5m.
        up_dist = getattr(self.cfg.terrain, 'curriculum_up_distance', self.terrain.env_length / 2)
        move_up = progress > up_dist
        move_down = (progress < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def compute_observations(self):

        # ---- 单帧本体感知 (39) ----
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,        # 缩放后的命令
            self.base_ang_vel * self.obs_scales.ang_vel,       # 基座角速度
            self.projected_gravity,                            # 投影重力
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
            self.dof_vel * self.obs_scales.dof_vel,            # 关节速度
            self.actions,                                      # 当前动作
        ), dim=-1)

        # ---- 地形高度图    num_single_obs = 39 + 96 = 135 ----
        # 参考基准用 base_height_target(0.698m) 而不是硬编码 0.5:平地正常站立时
        # root_z≈base_height_target,这样 heights 才能在 0 附近居中,而不是带一个
        # ~+1.0(经 height_measurements=5 放大后)的固定偏置。
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.) * self.obs_scales.height_measurements
        obs_buf = torch.cat((obs_buf, heights), dim=-1)        # (N, num_single_obs)

        # ---- 特权观测 (critic) ----
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
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

        # ---- 噪声 (height 段的 noise_scale=0,见 _get_noise_scale_vec / height_measurements=0) ----
        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            obs_buf_all = torch.stack([self.obs_history[i]
                                       for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        else:
            self.obs_buf = obs_now

    # ---------------- foothold penalty ----------------
    def _init_foot_sample_points(self):
        """Grid of (x,y,0) sample points across the foot sole, in the foot frame.
        Built once. Shape: (n_samples, 3)."""
        cfg = self.cfg.rewards
        xs = torch.linspace(-cfg.foot_length / 2, cfg.foot_length / 2,
                            cfg.foot_n_x, device=self.device)
        ys = torch.linspace(-cfg.foot_width / 2, cfg.foot_width / 2,
                            cfg.foot_n_y, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        pts = torch.zeros(gx.numel(), 3, device=self.device)
        pts[:, 0] = gx.flatten()
        pts[:, 1] = gy.flatten()
        self.foot_sample_points = pts  # (S, 3)
        self.num_foot_samples = pts.shape[0]

    def _terrain_height_at(self, points_xy):
        """Terrain height lookup at arbitrary world XY.
        points_xy: (K, 2)  ->  heights: (K,).  Mirrors base _get_heights()."""
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points_xy.shape[0], device=self.device)
        pts = points_xy + self.terrain.cfg.border_size
        pts = (pts / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(pts[:, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(pts[:, 1], 0, self.height_samples.shape[1] - 2)
        # min over the neighbouring cells = conservative (matches _get_heights)
        h = torch.min(torch.min(self.height_samples[px, py],
                                self.height_samples[px + 1, py]),
                      self.height_samples[px, py + 1])
        return h * self.terrain.cfg.vertical_scale

    def _reward_foothold(self):
        # lazy one-time init (feet_pos etc. exist after _init_foot)
        if not hasattr(self, 'foot_sample_points'):
            self._init_foot_sample_points()

        E, F, S = self.num_envs, self.feet_num, self.num_foot_samples

        # rotate sole samples by each foot's yaw, translate to world
        quat = self.feet_quat.reshape(E * F, 4).unsqueeze(1).expand(-1, S, -1)  # (E*F, S, 4)
        pts = self.foot_sample_points.unsqueeze(0).expand(E * F, -1, -1)  # (E*F, S, 3)
        world = quat_apply_yaw(quat.reshape(-1, 4), pts.reshape(-1, 3))  # (E*F*S, 3)
        world = world.reshape(E, F, S, 3)
        world = world + self.feet_pos.unsqueeze(2)  # + foot xyz

        # terrain height under every sample
        terr = self._terrain_height_at(world[..., :2].reshape(-1, 2)).reshape(E, F, S)

        # 参考面 = 该脚自身采样点里最高的地形 = 它实际踩着的石块 / 横梁表面。
        # 用【相对高度】而非 feet_pos.z:feet_pos.z 是 ankle 关节原点,远高于脚底(~脚厚),
        # 平地站立时 foot_z - terr 恒 > ε,会把每个触地脚都判为悬空 → 逼出单腿跳。
        ref = terr.max(dim=-1, keepdim=True).values  # (E, F, 1) 支撑面
        # d_ij : 采样点地形比支撑面低多少 = 该点悬在支撑面之外的深度(m)
        d = (ref - terr).clamp(min=0.)  # (E, F, S)

        # raw = 每只触地脚 (1 - exp(-k*d̄)) 的均值 ∈ [0,1]，必须【有界且平滑】
        # (同类工作 Limx Oli 的 feet_stair_flat=exp(-4*r_D) 也是指数形式)。
        # 原实现是离散计数 sum(1{d > 4cm})：中位数恒为 0(一半以上的步没有梯度)，
        # 上界 12 使单步最坏惩罚 -3.5，而整个正项栈才 ~1.5 —— 正是尖峰致发散的通道。
        # d̄ 取每只脚采样点的均值而非最大值，避免单个边缘点主导整只脚的评分。
        k = self.cfg.rewards.foothold_flat_k
        per_foot = 1.0 - torch.exp(-k * d.mean(dim=-1))  # (E, F) in [0, 1]
        Ci = self.contacts.float()  # (E, F)
        n_contact = Ci.sum(dim=-1).clamp(min=1.0)
        # 腾空(无触地脚)时为 0：既不奖励也不惩罚，避免把"飞行相"变成可刷的状态。
        return (Ci * per_foot).sum(dim=-1) / n_contact  # (E,) in [0, 1]

    # ---- 世界系 progress / heading (反绕路)，设计理由见类顶部注释 ----
    def _reward_world_progress(self):
        """Actual world-frame velocity projected on the commanded world
        direction, Extreme-Parkour-style (arXiv:2309.14341's
        r_tracking = min(<v, d_hat>, v_cmd)). A robot obeying all three command
        channels keeps its heading aligned with yaw_ref and scores the full
        commanded speed; one that turns away to skirt an obstacle, or sidesteps
        or retreats, loses the projection."""
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        world_vel = self.root_states[:, 7:9]  # world-frame xy velocity (unrotated)
        proj = torch.sum(world_vel * self.commands_world_dir, dim=1)
        # 必须除以指令速度(同 EP 的 _reward_tracking_goal_vel)：返回原始投影会让
        # 该项速度相关——服从 0.3m/s 指令得 0.3、服从 0.8m/s 得 0.8，结构性地惩罚
        # 慢速，而慢正是小心爬楼需要的。归一化后任意指令速度下满分都是 1.0。
        # 分母下限防 wz 主导的指令(min_cmd_vel 只约束三维模长，|v_xy| 可接近 0)。
        # 对称 [-1,1] 钳制是尖峰防护：proj 下无界，一次摔倒/推挤反向打飞 world_vel
        # 就曾把 noise_std 从 1.0 推到 21.0。
        denom = self.commands_world_speed.clamp(min=self.cfg.rewards.world_progress_min_speed)
        rew = torch.clamp(proj / denom, min=-1.0, max=1.0)
        rew[self.standing_cmd] = 0.
        return rew

    def _reward_world_heading(self):
        """Penalise the *accumulated* yaw error against the commanded yaw rate.
        Target is yaw_ref, NOT the direction of the linear velocity command:
        that earlier choice demanded a mean 90-degree (median 90-degree)
        instantaneous body turn over the real command distribution -- 46% of
        commands asked for >90 degrees and any vx<0 command asked for ~180 --
        which fought tracking_lin_vel and world_progress simultaneously.
        Extreme Parkour has no such conflict only because its commands are
        always goal-directed/forward-facing."""
        if not hasattr(self, 'world_heading_err'):
            return torch.zeros(self.num_envs, device=self.device)
        rew = torch.exp(-torch.square(self.world_heading_err) * 2.0)
        rew[self.standing_cmd] = 0.
        return rew

    # ---------------- bounded foot-impact penalty ----------------
    def _reward_feet_contact_forces(self):
        """有界版的接触力惩罚，替代基类的 sum(clip(|F| - max_contact_force, 0, inf))。
        """
        excess = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                  - self.cfg.rewards.max_contact_force).clip(min=0.)
        return excess.clip(max=self.cfg.rewards.feet_contact_force_max_excess).sum(dim=1)

    # ---------------- anti-freeze (break the "stand still on stairs" optimum) ----------------
    def _reward_anti_freeze(self):
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        fwd_speed = torch.sum(self.root_states[:, 7:9] * self.commands_world_dir, dim=1)
        rew = torch.clamp(fwd_speed / self.cfg.rewards.anti_freeze_speed, min=0.0, max=1.0)
        rew[self.standing_cmd] = 0.
        return rew

    # ---------------- bounded stand-still penalty ----------------
    def _reward_stand_still(self):
        """De-weighted, capped version of N2_10dof_Env._reward_stand_still.


        Overridden here rather than in N2_10dof_Env so the blind n2_10dof/n2
        tasks are untouched."""
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        sc = self.standing_cmd
        rew[sc] = (torch.sum(torch.abs(self.dof_pos[sc] - self.default_dof_pos), dim=1)
                   + self.cfg.rewards.stand_still_vel_weight
                   * torch.sum(torch.square(self.dof_vel[sc]), dim=1))
        return torch.clamp(rew, max=self.cfg.rewards.stand_still_max)
