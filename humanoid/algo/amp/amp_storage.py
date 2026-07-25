"""AMP 扩展的回放缓冲区，额外存储 AMP 观测对 (s_t, s_{t+1})"""

import torch
from humanoid.algo.ppo.rollout_storage import RolloutStorage


class AMPRolloutStorage(RolloutStorage):
    """在标准 RolloutStorage 基础上增加 AMP 观测存储。"""

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
        amp_obs_dim=55,
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            device=device,
        )
        self.amp_obs_dim = amp_obs_dim
        self.amp_obs = torch.zeros(
            num_transitions_per_env, num_envs, amp_obs_dim, device=self.device
        )
        self.amp_obs_next = torch.zeros(
            num_transitions_per_env, num_envs, amp_obs_dim, device=self.device
        )

    def add_transitions(self, transition):
        """存储标准转换数据 + AMP 观测对。"""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! Call clear() before adding."
            )

        # --- 标准字段 (与父类 add_transitions 保持一致) ---
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.privileged_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(
                transition.actions_log_prob.view(-1, 1)
            )
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # --- AMP 扩展字段 ---
        if hasattr(transition, "amp_obs") and transition.amp_obs is not None:
            self.amp_obs[self.step].copy_(transition.amp_obs)
        if hasattr(transition, "amp_obs_next") and transition.amp_obs_next is not None:
            self.amp_obs_next[self.step].copy_(transition.amp_obs_next)

        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """与父类相同的 mini-batch 生成器，额外输出 AMP 观测对。"""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        amp_obs_flat = self.amp_obs.flatten(0, 1)
        amp_obs_next_flat = self.amp_obs_next.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),  # placeholder for hidden states
                    None,     # placeholder for masks
                )
