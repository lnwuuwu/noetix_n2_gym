# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from typing import Optional

from humanoid.algo.ppo.actor_critic import ActorCritic
from humanoid.algo.ppo.rollout_storage import RolloutStorage
from humanoid.utils.utils import string_to_callable


class PPO:
    """近端策略优化算法 (Proximal Policy Optimization, https://arxiv.org/abs/1707.06347)"""

    policy: ActorCritic
    """演员-评论家模块"""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,           # 学习周期数
        num_mini_batches=1,              # 小批次数量
        clip_param=0.2,                  # 裁剪参数
        gamma=0.998,                     # 折扣因子
        lam=0.95,                        # GAE参数
        value_loss_coef=1.0,             # 价值损失系数
        entropy_coef=0.0,                # 熵系数
        learning_rate=1e-3,              # 学习率
        max_grad_norm=1.0,               # 最大梯度范数
        use_clipped_value_loss=True,     # 是否使用裁剪的价值损失
        schedule="fixed",                # 学习率调度策略
        desired_kl=0.01,                 # 期望的KL散度
        device="cpu",                    # 计算设备
        normalize_advantage_per_mini_batch=False,  # 是否按小批次归一化优势函数
        symmetry_cfg: Optional[dict] = None,
        multi_gpu_cfg: Optional[dict] = None,      # 多GPU配置
    ):
        """
        初始化PPO算法
        
        Args:
            policy: 策略网络(演员-评论家)
            num_learning_epochs: 学习周期数
            num_mini_batches: 小批次数量
            clip_param: PPO裁剪参数
            gamma: 折扣因子，用于计算回报
            lam: GAE(广义优势估计)参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数，用于鼓励探索
            learning_rate: 学习率
            max_grad_norm: 梯度裁剪的最大范数
            use_clipped_value_loss: 是否使用裁剪的价值损失函数
            schedule: 学习率调度策略("fixed"或"adaptive")
            desired_kl: 期望的KL散度，用于自适应学习率调整
            device: 计算设备("cpu"或"cuda")
            normalize_advantage_per_mini_batch: 是否按小批次归一化优势函数
            multi_gpu_cfg: 多GPU配置字典
        """
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # 多GPU参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]  # 全局rank
            self.gpu_world_size = multi_gpu_cfg["world_size"]    # 世界大小
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO组件
        self.policy = policy  # 策略网络
        self.policy.to(self.device)  # 将策略网络移动到指定设备
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # 创建回合存储
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()  # 转换存储

        # PPO参数
        self.clip_param = clip_param                    # PPO裁剪参数
        self.num_learning_epochs = num_learning_epochs  # 学习周期数
        self.num_mini_batches = num_mini_batches        # 小批次数量
        self.value_loss_coef = value_loss_coef          # 价值损失系数
        self.entropy_coef = entropy_coef                # 熵系数
        self.gamma = gamma                              # 折扣因子
        self.lam = lam                                  # GAE参数
        self.max_grad_norm = max_grad_norm              # 最大梯度范数
        self.use_clipped_value_loss = use_clipped_value_loss  # 是否使用裁剪价值损失
        self.desired_kl = desired_kl                    # 期望KL散度
        self.schedule = schedule                        # 学习率调度策略
        self.learning_rate = learning_rate              # 学习率
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch  # 优势函数归一化标志
        self.symmetry = None
        self.set_symmetry_config(symmetry_cfg)
        self.actor_reference = None
        self.actor_reference_loss_coeff = 0.0
        self.actor_reference_symmetry_env = None

    def set_symmetry_config(self, symmetry_cfg):
        """Configure an optional exact actor-reflection consistency loss."""
        if symmetry_cfg is not None:
            symmetry_env = symmetry_cfg.get("_env")
            if symmetry_env is None:
                raise ValueError("symmetry_cfg requires _env")
            if not (
                hasattr(symmetry_env, "mirror_observations")
                and hasattr(symmetry_env, "mirror_actions")
            ):
                raise ValueError(
                    "Symmetry environment must mirror observations/actions"
                )
            actor_coefficient = float(
                symmetry_cfg.get("actor_loss_coeff", 0.0)
            )
            critic_coefficient = float(
                symmetry_cfg.get("critic_loss_coeff", 0.0)
            )
            if actor_coefficient < 0.0 or critic_coefficient < 0.0:
                raise ValueError("Symmetry loss coefficients cannot be negative")
        self.symmetry = symmetry_cfg

    def set_actor_reference(self, loss_coefficient, symmetry_env=None):
        """Freeze the loaded Actor as an optional reflection-symmetric teacher.

        A raw checkpoint teacher preserves every left/right bias in that
        checkpoint.  When ``symmetry_env`` is supplied, the target instead
        averages the teacher action with the mirrored teacher action:

        ``0.5 * (T(o) + M_a(T(M_o(o))))``.

        The resulting target is exactly reflection equivariant while still
        retaining the checkpoint's state-dependent climbing behaviour.
        """
        loss_coefficient = float(loss_coefficient)
        if loss_coefficient < 0.0:
            raise ValueError(
                "Actor reference loss coefficient cannot be negative"
            )
        if symmetry_env is not None and not (
            hasattr(symmetry_env, "mirror_observations")
            and hasattr(symmetry_env, "mirror_actions")
        ):
            raise ValueError(
                "Symmetric Actor reference requires observation/action mirrors"
            )
        self.actor_reference_loss_coeff = loss_coefficient
        self.actor_reference_symmetry_env = symmetry_env
        if loss_coefficient == 0.0:
            self.actor_reference = None
            self.actor_reference_symmetry_env = None
            return
        self.actor_reference = copy.deepcopy(self.policy.actor).to(self.device)
        self.actor_reference.eval()
        self.actor_reference.requires_grad_(False)

    def _actor_reference_target(self, observations):
        """Return the frozen raw or left/right-averaged teacher target."""
        if self.actor_reference is None:
            return None
        with torch.no_grad():
            direct = self.actor_reference(observations)
            if self.actor_reference_symmetry_env is None:
                return direct
            symmetry_env = self.actor_reference_symmetry_env
            mirrored_observations = symmetry_env.mirror_observations(
                observations
            )
            mirrored = self.actor_reference(mirrored_observations)
            reflected_back = symmetry_env.mirror_actions(mirrored)
            return 0.5 * (direct + reflected_back)

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        """
        初始化存储
        
        Args:
            training_type: 训练类型
            num_envs: 环境数量
            num_transitions_per_env: 每个环境的转换数
            actor_obs_shape: 演员观察形状
            critic_obs_shape: 评论家观察形状
            actions_shape: 动作形状
        """
        # 创建回合存储
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        """
        根据观察选择动作
        
        Args:
            obs: 观察值(用于演员网络)
            critic_obs: 评论家观察值(用于评论家网络)
            
        Returns:
            选择的动作
        """
        # 如果策略是循环的，获取隐藏状态
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # 计算动作和价值
        self.transition.actions = self.policy.act(obs).detach()  # 选择动作
        self.transition.values = self.policy.evaluate(critic_obs).detach()  # 评估状态价值
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()  # 动作对数概率
        self.transition.action_mean = self.policy.action_mean.detach()  # 动作均值
        self.transition.action_sigma = self.policy.action_std.detach()   # 动作标准差
        
        # 需要在env.step()之前记录obs和critic_obs
        self.transition.observations = obs  # 观察值
        self.transition.privileged_observations = critic_obs  # 特权观察值
        return self.transition.actions  # 返回动作

    def process_env_step(self, rewards, dones, infos):
        """
        处理环境步骤结果
        
        Args:
            rewards: 奖励
            dones: 完成标志
            infos: 附加信息
        """
        # 记录奖励和完成标志
        # 注意：这里进行克隆，因为后续会根据超时进行奖励引导
        self.transition.rewards = rewards.clone()  # 奖励
        self.transition.dones = dones  # 完成标志

        # 超时引导
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()  # 清除转换
        self.policy.reset(dones)  # 重置策略

    def compute_returns(self, last_critic_obs):
        """
        计算回报
        
        Args:
            last_critic_obs: 最后一步的评论家观察值
        """
        # 计算最后一步的价值
        last_values = self.policy.evaluate(last_critic_obs).detach()
        # 计算回报
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        """
        更新策略和价值网络
        
        Returns:
            包含损失值的字典
        """
        mean_value_loss = 0      # 平均价值损失
        mean_surrogate_loss = 0  # 平均替代损失
        mean_entropy = 0         # 平均熵
        mean_symmetry_loss = 0
        mean_actor_reference_loss = 0

        # 小批次生成器
        if self.policy.is_recurrent:
            # 循环网络使用循环小批次生成器
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # 非循环网络使用普通小批次生成器
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历批次
        for (
            obs_batch,                    # 观察批次
            critic_obs_batch,             # 评论家观察批次
            actions_batch,                # 动作批次
            target_values_batch,          # 目标价值批次
            advantages_batch,             # 优势函数批次
            returns_batch,                # 回报批次
            old_actions_log_prob_batch,   # 旧动作对数概率批次
            old_mu_batch,                 # 旧均值批次
            old_sigma_batch,              # 旧标准差批次
            hid_states_batch,             # 隐藏状态批次
            masks_batch,                  # 掩码批次
        ) in generator:
            
            # 确保所有输入张量都在正确的设备上
            obs_batch = obs_batch.to(self.device)
            critic_obs_batch = critic_obs_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)
            target_values_batch = target_values_batch.to(self.device)
            advantages_batch = advantages_batch.to(self.device)
            returns_batch = returns_batch.to(self.device)
            old_actions_log_prob_batch = old_actions_log_prob_batch.to(self.device)
            old_mu_batch = old_mu_batch.to(self.device)
            old_sigma_batch = old_sigma_batch.to(self.device)
            if hid_states_batch[0] is not None:
                hid_states_batch = (hid_states_batch[0].to(self.device), hid_states_batch[1].to(self.device))
            if masks_batch is not None:
                masks_batch = masks_batch.to(self.device)

            # 每个样本的增强数量
            # 从1开始，如果使用对称增强则增加
            num_aug = 1
            # 原始批次大小
            original_batch_size = obs_batch.shape[0]

            # 检查是否应该按小批次归一化优势函数
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 执行对称增强
            # if self.symmetry and self.symmetry["use_data_augmentation"]:
            #     # 返回形状: [batch_size * num_aug, ...]
            #     obs_batch, actions_batch = self.data_augmentation_func(
            #         obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
            #     )
            #     critic_obs_batch, _ = self.data_augmentation_func(
            #         obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
            #     )
            #     # 计算每个样本的增强数量
            #     num_aug = int(obs_batch.shape[0] / original_batch_size)
            #     # 重复批次的其余部分
            #     # -- 演员
            #     old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            #     # -- 评论家
            #     target_values_batch = target_values_batch.repeat(num_aug, 1)
            #     advantages_batch = advantages_batch.repeat(num_aug, 1)
            #     returns_batch = returns_batch.repeat(num_aug, 1)

            # 为当前批次的转换重新计算动作对数概率和熵
            # 注意：需要这样做因为我们用新参数更新了策略
            # -- 演员
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- 评论家
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- 熵
            # 我们只保留第一次增强(原始)的熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            symmetry_loss = torch.zeros((), device=self.device)
            if self.symmetry is not None:
                symmetry_env = self.symmetry["_env"]
                mirrored_obs = symmetry_env.mirror_observations(
                    obs_batch
                )
                mirrored_mean = self.policy.actor(mirrored_obs)
                expected_mirrored_mean = symmetry_env.mirror_actions(
                    self.policy.action_mean
                )
                symmetry_loss = float(
                    self.symmetry.get("actor_loss_coeff", 0.0)
                ) * F.mse_loss(
                    mirrored_mean, expected_mirrored_mean
                )
                critic_coefficient = float(
                    self.symmetry.get("critic_loss_coeff", 0.0)
                )
                if critic_coefficient:
                    mirrored_critic_obs = (
                        symmetry_env.mirror_observations(
                            critic_obs_batch
                        )
                    )
                    mirrored_value = self.policy.critic(
                        mirrored_critic_obs
                    )
                    symmetry_loss = symmetry_loss + (
                        critic_coefficient
                        * F.mse_loss(mirrored_value, value_batch)
                    )
            actor_reference_loss = torch.zeros((), device=self.device)
            if self.actor_reference is not None:
                reference_mean = self._actor_reference_target(obs_batch)
                actor_reference_loss = (
                    self.actor_reference_loss_coeff
                    * F.mse_loss(self.policy.action_mean, reference_mean)
                )

            # KL散度计算
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 在所有GPU间归约KL散度
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 更新学习率
                    # 仅在主进程上执行此调整
                    # TODO: 是否需要？如果KL散度在所有GPU上相同，
                    #       那么学习率应该在所有GPU上相同。
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 更新所有GPU的学习率
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 更新所有参数组的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 替代损失计算
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 价值函数损失计算
            if self.use_clipped_value_loss:
                # 使用裁剪的价值损失
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                # 使用普通的价值损失
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # PPO objective plus exact N2 sagittal-reflection consistency.
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + symmetry_loss
                + actor_reference_loss
            )

            # 计算梯度
            # -- 对于PPO
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播

            # 从所有GPU收集梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用梯度
            # -- 对于PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # 梯度裁剪
            self.optimizer.step()  # 优化器步骤

            # 存储损失值
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_symmetry_loss += symmetry_loss.item()
            mean_actor_reference_loss += actor_reference_loss.item()

        # -- 对于PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches  # 更新次数
        mean_value_loss /= num_updates      # 平均价值损失
        mean_surrogate_loss /= num_updates  # 平均替代损失
        mean_entropy /= num_updates         # 平均熵
        mean_symmetry_loss /= num_updates
        mean_actor_reference_loss /= num_updates
       
        # -- 清除存储
        self.storage.clear()

        # 构造损失字典
        loss_dict = {
            "value_function": mean_value_loss,    # 价值函数损失
            "surrogate": mean_surrogate_loss,     # 替代损失
            "entropy": mean_entropy,              # 熵
            "symmetry": mean_symmetry_loss,
            "actor_reference": mean_actor_reference_loss,
        }

        return loss_dict

    """
    辅助函数
    """

    def broadcast_parameters(self):
        """将模型参数广播到所有GPU"""
        # 获取当前GPU上的模型参数
        model_params = [self.policy.state_dict()]
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 从源GPU加载所有GPU上的模型参数
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """从所有GPU收集梯度并求平均
        
        此函数在反向传播后调用，用于同步所有GPU间的梯度
        """
        # 创建张量来存储梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # 在所有GPU间平均梯度
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 获取所有参数
        all_params = self.policy.parameters()

        # 使用归约后的梯度更新所有参数的梯度
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移量
                offset += numel
