"""PPO 扩展: 集成 AMP (Adversarial Motion Priors) 判别器和风格奖励。

在标准 PPO 训练循环中:
1. process_env_step 时计算 style_reward 并与 task_reward 组合
2. update 时用 WGAN-GP 训练判别器 (real=参考动作, fake=策略rollout)
"""

import torch
import torch.optim as optim
from humanoid.algo.ppo.ppo import PPO
from humanoid.algo.amp.amp_storage import AMPRolloutStorage


class AMPPPO(PPO):
    """在 PPO 基础上增加 AMP 判别器训练和风格奖励注入。"""

    def __init__(
        self,
        policy,
        discriminator,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        # --- AMP 专用参数 ---
        amp_replay_buffer_size=100000,
        amp_batch_size=512,
        style_reward_weight=0.5,
        disc_learning_rate=1e-4,
    ):
        super().__init__(
            policy,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
        )

        # 判别器
        self.discriminator = discriminator.to(device)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=disc_learning_rate
        )

        # AMP 超参
        self.style_reward_weight = style_reward_weight
        self.amp_batch_size = amp_batch_size

        # AMP 回放缓冲区 (存储策略产生的 (s_t||s_{t+1}) 对)
        self.amp_replay_buffer_size = amp_replay_buffer_size
        self.amp_replay_buffer = None  # 懒初始化
        self.amp_replay_ptr = 0
        self.amp_replay_full = False

        # 由 runner 在初始化后设置
        self.motion_loader = None

    # ------------------------------------------------------------------
    # 覆盖 init_storage: 使用 AMPRolloutStorage
    # ------------------------------------------------------------------
    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        self.storage = AMPRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # 覆盖 process_env_step: 注入 AMP 风格奖励
    # ------------------------------------------------------------------
    def process_env_step(self, rewards, dones, infos):
        # amp_obs / amp_obs_next 由 runner 在调用前设置到 self.transition 上
        s_t = self.transition.amp_obs
        s_t1 = self.transition.amp_obs_next

        # 计算风格奖励 (不需要梯度)
        with torch.no_grad():
            style_reward = self.discriminator.compute_style_reward(s_t, s_t1)

        # 组合: total_reward = task_reward + w * style_reward
        combined_rewards = rewards + self.style_reward_weight * style_reward

        # 更新 AMP 回放缓冲区
        self._update_replay_buffer(s_t, s_t1)

        # 调用父类 process_env_step (使用组合奖励)
        super().process_env_step(combined_rewards, dones, infos)

    def _update_replay_buffer(self, s_t, s_t1):
        """环形写入 AMP 回放缓冲区。"""
        amp_pair = torch.cat([s_t, s_t1], dim=-1)
        num_envs = amp_pair.shape[0]
        pair_dim = amp_pair.shape[1]

        # 懒初始化
        if self.amp_replay_buffer is None:
            self.amp_replay_buffer = torch.zeros(
                self.amp_replay_buffer_size, pair_dim, device=self.device
            )

        end_idx = self.amp_replay_ptr + num_envs
        if end_idx <= self.amp_replay_buffer_size:
            self.amp_replay_buffer[self.amp_replay_ptr : end_idx] = amp_pair
        else:
            overflow = end_idx - self.amp_replay_buffer_size
            self.amp_replay_buffer[self.amp_replay_ptr :] = amp_pair[: num_envs - overflow]
            self.amp_replay_buffer[:overflow] = amp_pair[num_envs - overflow :]
            self.amp_replay_full = True

        self.amp_replay_ptr = end_idx % self.amp_replay_buffer_size
        if self.amp_replay_ptr == 0 and num_envs > 0:
            self.amp_replay_full = True

    # ------------------------------------------------------------------
    # 覆盖 update: 在 PPO 更新后训练判别器
    # ------------------------------------------------------------------
    def update(self):
        # 先执行标准 PPO 更新
        loss_dict = super().update()

        # 然后训练判别器
        disc_loss_sum = 0.0
        grad_pen_sum = 0.0
        num_disc_updates = 0

        max_idx = (
            self.amp_replay_buffer_size
            if self.amp_replay_full
            else self.amp_replay_ptr
        )
        if max_idx < 2 or self.motion_loader is None:
            # 还没有足够的回放数据, 跳过判别器更新
            loss_dict["disc_loss"] = 0.0
            loss_dict["grad_pen"] = 0.0
            return loss_dict

        for _ in range(self.num_learning_epochs):
            # 采样真实数据 (参考动作)
            # MotionLoaderNing.feed_forward_generator yields:
            #   (states [B, horizon=2, 55], labels [B, num_motions])
            real_gen = self.motion_loader.feed_forward_generator(
                1, self.amp_batch_size
            )
            real_states, _labels = next(real_gen)
            # 拼接 horizon 维: [B, 2, 55] -> [B, 110]
            real_data = real_states.reshape(real_states.shape[0], -1).to(self.device)

            # 采样假数据 (策略回放, 已是 [B, 110])
            sample_size = min(self.amp_batch_size, max_idx)
            idx = torch.randint(0, max_idx, (sample_size,), device=self.device)
            fake_data = self.amp_replay_buffer[idx]

            # 确保维度匹配
            min_dim = min(real_data.shape[1], fake_data.shape[1])
            real_data = real_data[:, :min_dim]
            fake_data = fake_data[:, :min_dim]

            # WGAN 判别器前向
            d_real = self.discriminator(real_data)
            d_fake = self.discriminator(fake_data)

            disc_loss = d_fake.mean() - d_real.mean()
            grad_pen = self.discriminator.compute_grad_pen(real_data, fake_data)
            total_disc_loss = disc_loss + grad_pen

            self.disc_optimizer.zero_grad()
            total_disc_loss.backward()
            self.disc_optimizer.step()

            disc_loss_sum += disc_loss.item()
            grad_pen_sum += grad_pen.item()
            num_disc_updates += 1

        if num_disc_updates > 0:
            loss_dict["disc_loss"] = disc_loss_sum / num_disc_updates
            loss_dict["grad_pen"] = grad_pen_sum / num_disc_updates
        else:
            loss_dict["disc_loss"] = 0.0
            loss_dict["grad_pen"] = 0.0

        return loss_dict
