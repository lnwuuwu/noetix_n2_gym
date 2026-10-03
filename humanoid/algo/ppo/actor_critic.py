# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid.utils.utils import resolve_nn_activation


class ScanEncoderActor(nn.Module):
    """把高度图先压成低维 latent，再和本体感知一起送进主干。

    动机：本仓库把 96 维高度图 x frame_stack 帧原样摊平灌进 actor 第一层，
    parkour 任务下观测是 1370 维、其中 960 维是高度图，第一层 1370x256 占 actor
    参数的 91%。除了慢，更本质的问题是高度图和本体感知在同一层被平等对待，而两者
    的维度与统计特性差异极大。

    Extreme Parkour 的做法是 scan_encoder [128,64,32] 把 132 维 scandots 压到 32 维
    再与本体感知拼接（legged_gym 上游同样是 [512,256,128] 主干）。这里照此实现。

    观测布局是【逐帧交错】的：obs = concat_t([proprio_t, scan_t])，所以不能按前后
    切成两段，必须 reshape 成 (B, T, S) 再沿最后一维分。

    整个前向写在这一个 Module 里，因为 export_policy_as_jit 只拷贝
    actor_critic.actor 并对它 torch.jit.script——编码器若放在外面，导出的策略会
    缺掉它，sim2sim 拿到的网络与训练时不一致。
    """

    def __init__(self, frame_stack: int, n_proprio: int, n_scan: int,
                 scan_encoder_dims, trunk_dims, num_actions, activation):
        super().__init__()
        self.frame_stack = int(frame_stack)
        self.n_proprio = int(n_proprio)
        self.n_scan = int(n_scan)
        self.single_obs = self.n_proprio + self.n_scan
        # ONNX 导出要知道真实输入维度；此时第一个 Linear 是编码器的，其 in_features
        # 只有 scan 部分，会算错，所以显式暴露。
        self.expected_input_dim = self.single_obs * self.frame_stack

        enc = []
        d = self.n_scan * self.frame_stack
        for h in scan_encoder_dims:
            enc.append(nn.Linear(d, h)); enc.append(activation); d = h
        self.scan_encoder = nn.Sequential(*enc)
        self.latent_dim = d

        trunk_in = self.n_proprio * self.frame_stack + self.latent_dim
        layers = [nn.Linear(trunk_in, trunk_dims[0]), activation]
        for i in range(len(trunk_dims)):
            if i == len(trunk_dims) - 1:
                layers.append(nn.Linear(trunk_dims[i], num_actions))
            else:
                layers.append(nn.Linear(trunk_dims[i], trunk_dims[i + 1])); layers.append(activation)
        self.trunk = nn.Sequential(*layers)

    def forward(self, obs):
        b = obs.shape[0]
        x = obs.view(b, self.frame_stack, self.single_obs)
        proprio = x[:, :, :self.n_proprio].reshape(b, self.frame_stack * self.n_proprio)
        scan = x[:, :, self.n_proprio:].reshape(b, self.frame_stack * self.n_scan)
        return self.trunk(torch.cat([proprio, self.scan_encoder(scan)], dim=-1))


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
        # 高度图编码器（可选）。给了 scan_encoder_dims 就启用，否则完全保持原行为，
        # 所以未配置该项的任务（n2_10dof / n2_perceptive 等）不受影响。
        scan_encoder_dims=None,
        frame_stack=None,
        n_proprio=None,
        n_scan=None,
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
        if scan_encoder_dims and n_scan and frame_stack and n_proprio:
            # 启用高度图编码器：actor 变成 ScanEncoderActor（内部含编码器+主干），
            # 这样 export_policy_as_jit 拷 self.actor 时编码器一并被导出。
            self.actor = ScanEncoderActor(frame_stack, n_proprio, n_scan,
                                          scan_encoder_dims, actor_hidden_dims,
                                          num_actions, activation)
            assert frame_stack * (n_proprio + n_scan) == num_actor_obs, (
                "frame_stack*(n_proprio+n_scan)=%d 与 num_actor_obs=%d 不符"
                % (frame_stack * (n_proprio + n_scan), num_actor_obs))
        else:
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
        # 不用 self.actor[0]：启用 scan encoder 时 actor 是 ScanEncoderActor 而非
        # nn.Sequential，不能下标访问。取任意一个参数的 device 即可。
        observations = observations.to(next(self.actor.parameters()).device)
        # 计算动作均值
        mean = self.actor(observations)
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
        actions_mean = self.actor(observations)
        return actions_mean

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