"""AMP On-Policy Runner: 在标准 PPO 训练循环基础上集成 AMP。

核心改动:
1. 在 super().__init__ 之后, 用 AMPPPO 包装已创建的 PPO alg
2. 加载参考动作数据
3. 在 rollout 中提取 AMP 观测, 注入到 transition 中
"""

import time
import os
import collections
import torch
import numpy as np

from humanoid.algo.ppo.on_policy_runner import OnPolicyRunner
from humanoid.amp_utils.motion_loader import MotionLoaderNing
from humanoid.algo.amp.amp_ppo import AMPPPO
from humanoid.algo.amp.discriminator import AMPDiscriminator


def extract_amp_obs(env):
    """从环境状态中提取 55 维 AMP 观测向量。

    特征组成:
      joint_pose (18) + toe_pos_local (12) + lin_vel (3)
      + ang_vel (3) + joint_vel (18) + base_height (1) = 55
    """
    joint_pose = env.dof_pos  # [N, 18]

    # 脚部位置 (体坐标系)
    if hasattr(env, "feet_pos") and env.feet_pos is not None:
        feet_pos = env.feet_pos  # 通常 [N, 2, 3]
        feet_flat = feet_pos.reshape(env.num_envs, -1)  # [N, 6 or 12]
        if feet_flat.shape[1] < 12:
            # 不足 12 维时用 ankle joint 位置补齐
            pad = torch.zeros(
                env.num_envs, 12 - feet_flat.shape[1], device=env.device
            )
            toe_pos_local = torch.cat([feet_flat, pad], dim=1)
        else:
            toe_pos_local = feet_flat[:, :12]
    else:
        toe_pos_local = torch.zeros(env.num_envs, 12, device=env.device)

    lin_vel = env.base_lin_vel  # [N, 3]
    ang_vel = env.base_ang_vel  # [N, 3]
    joint_vel = env.dof_vel  # [N, 18]

    # 基座相对地面高度
    if hasattr(env, "measured_heights") and env.measured_heights is not None:
        # 取机器人正下方的地形高度
        terrain_h = env.measured_heights.mean(dim=1)
    else:
        terrain_h = torch.zeros(env.num_envs, device=env.device)
    base_height = (env.root_states[:, 2] - terrain_h).unsqueeze(1)  # [N, 1]

    return torch.cat(
        [joint_pose, toe_pos_local, lin_vel, ang_vel, joint_vel, base_height],
        dim=1,
    )  # [N, 55]


class AMPOnPolicyRunner(OnPolicyRunner):
    """在标准 OnPolicyRunner 基础上增加 AMP 支持。

    使用方式:
      runner = AMPOnPolicyRunner(env, train_cfg, log_dir, device)
      runner.learn(num_learning_iterations)
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # 先让父类完成所有初始化 (创建 policy, PPO alg, storage, etc.)
        super().__init__(env, train_cfg, log_dir, device)

        # --- AMP 配置 ---
        amp_cfg = train_cfg.get("amp", {})
        style_reward_weight = float(amp_cfg.get("style_reward_weight", 0.5))
        disc_learning_rate = float(amp_cfg.get("disc_learning_rate", 1e-4))
        amp_replay_buffer_size = int(amp_cfg.get("replay_buffer_size", 100000))
        amp_batch_size = int(amp_cfg.get("batch_size", 512))
        motion_files = amp_cfg.get("motion_files", [])
        disc_hidden_dims = amp_cfg.get("disc_hidden_dims", [1024, 512])

        # --- 创建判别器 ---
        # 输入 = (s_t || s_{t+1}) 拼接, 每个 55 维
        amp_obs_dim = 55
        disc_input_dim = amp_obs_dim * 2  # 110
        self.discriminator = AMPDiscriminator(
            input_dim=disc_input_dim,
            hidden_dims=disc_hidden_dims,
        ).to(self.device)

        # --- 用 AMPPPO 替换 PPO ---
        old_alg = self.alg
        self.alg = AMPPPO(
            policy=old_alg.policy,
            discriminator=self.discriminator,
            num_learning_epochs=old_alg.num_learning_epochs,
            num_mini_batches=old_alg.num_mini_batches,
            clip_param=old_alg.clip_param,
            gamma=old_alg.gamma,
            lam=old_alg.lam,
            value_loss_coef=old_alg.value_loss_coef,
            entropy_coef=old_alg.entropy_coef,
            learning_rate=old_alg.learning_rate,
            max_grad_norm=old_alg.max_grad_norm,
            use_clipped_value_loss=old_alg.use_clipped_value_loss,
            schedule=old_alg.schedule,
            desired_kl=old_alg.desired_kl,
            device=self.device,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_batch_size=amp_batch_size,
            style_reward_weight=style_reward_weight,
            disc_learning_rate=disc_learning_rate,
        )

        # 复制对称性和参考策略设置
        if old_alg.symmetry is not None:
            self.alg.set_symmetry_config(old_alg.symmetry)
        if old_alg.actor_reference is not None:
            self.alg.actor_reference = old_alg.actor_reference
            self.alg.actor_reference_loss_coeff = old_alg.actor_reference_loss_coeff
            self.alg.actor_reference_symmetry_env = old_alg.actor_reference_symmetry_env
            self.alg.actor_reference_mirror_blend = old_alg.actor_reference_mirror_blend

        # 重新初始化存储 (使用 AMPRolloutStorage)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        num_obs = obs.shape[1]
        num_privileged_obs = privileged_obs.shape[1]

        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # --- 加载参考动作 ---
        if isinstance(motion_files, str):
            motion_files = [motion_files] if motion_files else []

        if motion_files and all(os.path.exists(f) for f in motion_files):
            print(f"[AMP] Loading reference motions from: {motion_files}")
            # IsaacGym 的 dt 通常为 0.005s * decimation
            sim_dt = getattr(env, "dt", 0.02)
            self.motion_loader = MotionLoaderNing(
                device=self.device,
                time_between_frames=sim_dt,
                reference_observation_horizon=2,
                num_preload_transitions=min(500000, amp_replay_buffer_size * 5),
                motion_files=motion_files,
            )
            self.alg.motion_loader = self.motion_loader
            print(
                f"[AMP] Loaded {self.motion_loader.num_motions} motions, "
                f"obs_dim={self.motion_loader.observation_dim}"
            )
        else:
            self.motion_loader = None
            print("[AMP] WARNING: No valid motion files. Discriminator will not train.")

    # ------------------------------------------------------------------
    # 覆盖 learn: 在 rollout 中提取并注入 AMP 观测
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 获取初始观测
        obs = self.env.get_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations().to(self.device)
        obs = self.obs_normalizer(obs)

        self.alg.policy.train()

        ep_infos = []
        rewbuffer = collections.deque(maxlen=100)
        lenbuffer = collections.deque(maxlen=100)
        style_rewbuffer = collections.deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # === Rollout 阶段 ===
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 1. 提取当前 AMP 观测
                    amp_obs_before = extract_amp_obs(self.env)

                    # 2. 采样动作
                    actions = self.alg.act(obs, privileged_obs)

                    # 3. 执行环境步
                    obs, privileged_obs, rewards, dones, infos, _, _ = (
                        self.env.step(actions.to(self.env.device))
                    )
                    obs = obs.to(self.device)
                    privileged_obs = privileged_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # 4. 提取步后 AMP 观测
                    amp_obs_after = extract_amp_obs(self.env)

                    # 5. 注入 AMP 观测到 transition (process_env_step 会用到)
                    self.alg.transition.amp_obs = amp_obs_before
                    self.alg.transition.amp_obs_next = amp_obs_after

                    # 6. 归一化并处理
                    obs = self.obs_normalizer(obs)
                    self.alg.process_env_step(rewards, dones, infos)

                    # 7. 记录
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 计算 GAE 回报
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # === 更新阶段 ===
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1

            # 日志
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )

            ep_infos.clear()
