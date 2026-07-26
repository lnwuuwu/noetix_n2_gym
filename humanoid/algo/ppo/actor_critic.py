# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid.utils.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    # 标记该网络是否为循环网络（RNN/LSTM等），这里设置为False表示使用前馈网络
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,           # 演员网络的观测维度（输入状态的维度）
        num_critic_obs,          # 评论家网络的观测维度（输入状态的维度）
        num_actions,             # 动作空间维度（输出动作的维度）
        actor_hidden_dims=[256, 256, 256],   # 演员网络隐藏层维度列表
        critic_hidden_dims=[256, 256, 256],  # 评论家网络隐藏层维度列表
        activation="elu",        # 激活函数类型
        init_noise_std=1.0,      # 初始化动作噪声标准差
        noise_std_type: str = "scalar",  # 噪声标准差类型："scalar"表示标量，"log"表示对数形式
        **kwargs,
    ):
        """
        ActorCritic类初始化函数
        实现了PPO算法中的演员-评论家网络架构，包含策略网络（演员）和价值网络（评论家）
        """
        # 处理未预期的关键字参数
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # 解析激活函数
        activation = resolve_nn_activation(activation)

        # 设置演员和评论家网络的输入维度
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        
        # 构建演员网络（策略网络）- 用于根据观测状态选择动作
        actor_layers = []
        # 添加第一层全连接层
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # 添加激活函数
        actor_layers.append(activation)
        # 构建后续隐藏层
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                # 最后一层输出动作维度
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                # 中间层连接
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        # 将所有层组合成一个序列模型
        self.actor = nn.Sequential(*actor_layers)

        # 构建评论家网络（价值网络）- 用于评估状态的价值
        critic_layers = []
        # 添加第一层全连接层
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        # 添加激活函数
        critic_layers.append(activation)
        # 构建后续隐藏层
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                # 最后一层输出价值（标量）
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                # 中间层连接
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        # 将所有层组合成一个序列模型
        self.critic = nn.Sequential(*critic_layers)

        # 打印网络结构信息
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # 动作噪声设置 - 用于在训练中增加探索性
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # 标量形式的噪声标准差，所有动作维度共享同一个标准差参数
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            # 对数形式的噪声标准差，每个动作维度有独立的标准差参数
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（在update_distribution方法中填充）- 用于存储当前策略下的动作分布
        self.distribution = None
        # 禁用参数验证以提高速度
        Normal.set_default_validate_args(False)

    @staticmethod
    # 当前未使用
    def init_weights(sequential, scales):
        """
        初始化网络权重（正交初始化）
        :param sequential: 网络层序列
        :param scales: 每层的增益系数
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        重置网络状态（对于循环网络有用）
        在前馈网络中为空操作
        """
        pass

    def forward(self):
        """
        前向传播函数
        这里未实现，因为演员和评论家分别使用不同的前向传播方法
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        获取动作分布的均值
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        获取动作分布的标准差
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        计算动作分布的熵（衡量策略的随机性）
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        根据观测更新动作分布
        :param observations: 当前观测状态
        """
        # 将观测数据移动到与网络相同的设备上
        observations = observations.to(self.actor[0].weight.device)
        # Keep the policy mean behind one method so constrained/residual
        # policies can reuse the standard PPO distribution implementation.
        mean = self.actor_mean(observations)
        # 计算动作标准差
        if self.noise_std_type == "scalar":
            # 标量形式的标准差
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            # 对数形式的标准差
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # 创建正态分布（高斯分布）
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """
        根据当前策略选择动作（用于训练阶段，包含噪声）
        :param observations: 当前观测状态
        :return: 从策略分布中采样的动作
        """
        # 更新动作分布
        self.update_distribution(observations)
        # 从分布中采样动作
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """
        计算给定动作的对数概率
        :param actions: 动作
        :return: 动作的对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        根据当前策略选择动作（用于推理阶段，确定性动作）
        :param observations: 当前观测状态
        :return: 确定性动作（均值）
        """
        # 直接使用演员网络计算动作均值
        actions_mean = self.actor_mean(observations)
        return actions_mean

    def actor_mean(self, observations):
        """Return the deterministic Actor mean for training and inference."""
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """
        评估状态的价值（评论家网络）
        :param critic_observations: 评论家网络的观测状态
        :return: 状态价值
        """
        # 使用评论家网络计算状态价值
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """加载演员-评论家模型的参数

        Args:
            state_dict (dict): 模型的状态字典
            strict (bool): 是否严格匹配状态字典的键与模型的键

        Returns:
            bool: 是否恢复了之前的训练。此标志由`OnPolicyRunner`的`load()`函数使用，
                  以确定如何加载其他参数（例如蒸馏相关参数）
        """

        super().load_state_dict(state_dict, strict=strict)
        return True


class ResidualActorCritic(ActorCritic):
    """Frozen base Actor plus a small, bounded leg-only correction.

    The base Actor keeps the exact observation width and weights of the
    approved checkpoint.  A second MLP receives the base observation plus
    deployable stair-target features and may alter only explicitly selected
    leg joints.  Its output layer is initialized to zero, so converting a
    normal checkpoint into this policy is behavior preserving at step zero.
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        residual_base_obs_dim=410,
        residual_hidden_dims=(128, 64),
        residual_action_indices=(4, 5, 6, 7, 8, 13, 14, 15, 16, 17),
        residual_action_scales=(
            0.08,
            0.12,
            0.16,
            0.20,
            0.12,
            0.08,
            0.12,
            0.16,
            0.20,
            0.12,
        ),
        residual_frozen_action_std=1.0e-4,
        **kwargs,
    ):
        self.residual_observation_dim = int(num_actor_obs)
        self.residual_base_obs_dim = int(residual_base_obs_dim)
        if self.residual_base_obs_dim <= 0:
            raise ValueError("residual_base_obs_dim must be positive")
        if self.residual_base_obs_dim > self.residual_observation_dim:
            raise ValueError(
                "Residual base observation width {} exceeds total width {}"
                .format(
                    self.residual_base_obs_dim,
                    self.residual_observation_dim,
                )
            )

        hidden_dims = [int(width) for width in residual_hidden_dims]
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("residual_hidden_dims must contain positive widths")
        action_indices = [int(index) for index in residual_action_indices]
        action_scales = [float(scale) for scale in residual_action_scales]
        if len(action_indices) != len(action_scales):
            raise ValueError(
                "Residual action indices/scales must have equal lengths"
            )
        if len(set(action_indices)) != len(action_indices):
            raise ValueError("Residual action indices must be unique")
        if any(index < 0 or index >= num_actions for index in action_indices):
            raise ValueError("Residual action index is outside the action space")
        if any(scale <= 0.0 for scale in action_scales):
            raise ValueError("Residual action scales must be positive")
        if residual_frozen_action_std <= 0.0:
            raise ValueError("residual_frozen_action_std must be positive")

        # Construct the checkpoint-compatible base Actor with its original
        # 410-wide input.  The Critic keeps its unchanged privileged input.
        super().__init__(
            self.residual_base_obs_dim,
            num_critic_obs,
            num_actions,
            **kwargs,
        )

        activation_name = kwargs.get("activation", "elu")
        activation = resolve_nn_activation(activation_name)
        residual_layers = []
        input_width = self.residual_observation_dim
        for output_width in hidden_dims:
            residual_layers.extend(
                (nn.Linear(input_width, output_width), activation)
            )
            input_width = output_width
        residual_layers.append(nn.Linear(input_width, len(action_indices)))
        self.residual_actor = nn.Sequential(*residual_layers)
        final_layer = self.residual_actor[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        self.register_buffer(
            "residual_action_indices",
            torch.tensor(action_indices, dtype=torch.long),
        )
        self.register_buffer(
            "residual_action_scales",
            torch.tensor(action_scales, dtype=torch.float),
        )
        action_mask = torch.zeros(num_actions, dtype=torch.bool)
        action_mask[self.residual_action_indices] = True
        self.register_buffer("residual_action_mask", action_mask)
        self.register_buffer(
            "residual_frozen_action_std",
            torch.tensor(float(residual_frozen_action_std)),
        )

        # The approved base policy is an immutable skill prior.  PPO trains
        # only the residual branch, Critic, and (unless separately frozen)
        # noise parameter.
        self.actor.requires_grad_(False)
        self._last_residual_action = None
        print(
            "Residual Actor MLP: {} (base_obs={} total_obs={} actions={})"
            .format(
                self.residual_actor,
                self.residual_base_obs_dim,
                self.residual_observation_dim,
                action_indices,
            )
        )

    def residual_correction(self, observations):
        """Return a full-width, bounded residual action tensor."""
        if observations.shape[-1] != self.residual_observation_dim:
            raise ValueError(
                "Residual Actor expects {} observations, received {}".format(
                    self.residual_observation_dim, observations.shape[-1]
                )
            )
        compact = torch.tanh(self.residual_actor(observations))
        compact = compact * self.residual_action_scales.to(
            device=compact.device, dtype=compact.dtype
        )
        residual = torch.zeros(
            *compact.shape[:-1],
            self.residual_action_mask.numel(),
            dtype=compact.dtype,
            device=compact.device,
        )
        residual[..., self.residual_action_indices] = compact
        return residual

    def actor_mean(self, observations):
        base_observations = observations[..., : self.residual_base_obs_dim]
        base_actions = self.actor(base_observations)
        residual = self.residual_correction(observations)
        self._last_residual_action = residual
        return base_actions + residual

    def update_distribution(self, observations):
        observations = observations.to(self.actor[0].weight.device)
        mean = self.actor_mean(observations)
        if self.noise_std_type == "scalar":
            learned_std = torch.clamp(self.std, min=1.0e-6)
        elif self.noise_std_type == "log":
            learned_std = torch.exp(self.log_std)
        else:
            raise ValueError(
                "Unknown standard deviation type: {}".format(
                    self.noise_std_type
                )
            )
        frozen_std = self.residual_frozen_action_std.to(
            device=mean.device, dtype=mean.dtype
        )
        effective_std = torch.where(
            self.residual_action_mask.to(device=mean.device),
            learned_std.to(device=mean.device, dtype=mean.dtype),
            frozen_std,
        )
        self.distribution = Normal(mean, effective_std.expand_as(mean))

    def normalized_residual_correction(self, observations):
        """Return compact residual usage normalized by each safety bound."""
        compact = torch.tanh(self.residual_actor(observations))
        return compact

    def checkpoint_metadata(self):
        return {
            "class_name": self.__class__.__name__,
            "residual_base_obs_dim": self.residual_base_obs_dim,
            "residual_observation_dim": self.residual_observation_dim,
            "residual_hidden_dims": [
                module.out_features
                for module in self.residual_actor
                if isinstance(module, nn.Linear)
            ][:-1],
            "residual_action_indices": self.residual_action_indices.tolist(),
            "residual_action_scales": self.residual_action_scales.tolist(),
            "residual_frozen_action_std": float(
                self.residual_frozen_action_std.item()
            ),
        }

    def load_state_dict(self, state_dict, strict=True):
        """Load either a base checkpoint or a residual checkpoint safely."""
        incompatible = nn.Module.load_state_dict(
            self, state_dict, strict=False
        )
        residual_prefixes = (
            "residual_actor.",
            "residual_action_indices",
            "residual_action_scales",
            "residual_action_mask",
            "residual_frozen_action_std",
        )
        loaded_residual = any(
            key.startswith("residual_actor.") for key in state_dict
        )
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if loaded_residual or not key.startswith(residual_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Residual checkpoint mismatch: missing={} unexpected={}"
                .format(invalid_missing, incompatible.unexpected_keys)
            )
        expected_mask = torch.zeros_like(self.residual_action_mask)
        expected_mask[self.residual_action_indices] = True
        if not torch.equal(expected_mask, self.residual_action_mask):
            raise RuntimeError(
                "Residual checkpoint action mask does not match its indices"
            )
        if not loaded_residual:
            # Loading a normal approved checkpoint deliberately keeps the
            # zero-initialized correction.  Assert that invariant instead of
            # silently accepting a non-zero branch.
            with torch.no_grad():
                final_layer = self.residual_actor[-1]
                final_layer.weight.zero_()
                final_layer.bias.zero_()
            print(
                "Initialized zero residual policy from base Actor checkpoint"
            )
        return True
