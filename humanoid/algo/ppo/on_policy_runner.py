# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import json
from collections import deque
from typing import Optional

import humanoid
from humanoid.algo.ppo.ppo import PPO
from humanoid.algo.ppo.actor_critic import ActorCritic 
from humanoid.algo.ppo.normalizer import EmpiricalNormalization
from humanoid.algo import VecEnv
from humanoid.utils.utils import store_code_state


class OnPolicyRunner:
    """On-policy训练和评估的运行器类"""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: Optional[str] = None,
        device="cpu",
    ):
        """
        初始化OnPolicyRunner
        
        Args:
            env: 环境实例
            train_cfg: 训练配置字典
            log_dir: 日志目录路径
            device: 计算设备("cpu"或"cuda")
        """
        # 存储训练配置
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  # 算法配置
        self.policy_cfg = train_cfg["policy"]  # 策略配置
        self.device = device  # 计算设备
        self.env = env  # 环境实例

        # 检查是否启用多GPU训练
        self._configure_multi_gpu()

        # 根据算法确定训练类型
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"  # 强化学习
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"  # 蒸馏学习
        else:
            raise ValueError(f"算法 {self.alg_cfg['class_name']} 未找到对应的训练类型")

        # 获取观测空间维度
        obs = self.env.get_observations()  # 获取观测值
        privileged_obs = self.env.get_privileged_observations()  # 获取特权观测值
        num_obs = obs.shape[1]  # 观测维度

        # 确定特权观测类型
        if self.training_type == "rl":
            self.privileged_obs_type = "critic"  # 用于critic网络
        # if self.training_type == "distillation":
        #     if "teacher" in extras["observations"]:
        #         self.privileged_obs_type = "teacher"  # 策略蒸馏
        #     else:
        #         self.privileged_obs_type = None

        # 确定特权观测维度
        if self.privileged_obs_type is not None:
            num_privileged_obs = privileged_obs.shape[1]  # 特权观测维度
        else:
            num_privileged_obs = num_obs  # 与普通观测相同

        # 评估策略类
        policy_class = eval(self.policy_cfg.pop("class_name"))  # 获取策略类
        policy: ActorCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)  # 创建策略实例并移动到指定设备

        # 初始化算法
        alg_class = eval(self.alg_cfg.pop("class_name"))  # 获取算法类
        self.alg: PPO = alg_class(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)  # 创建算法实例

        # 存储训练配置参数
        self.num_steps_per_env = self.cfg["runner"]["num_steps_per_env"]  # 每个环境的步数
        self.save_interval = self.cfg["runner"]["save_interval"]  # 保存间隔
        self.empirical_normalization = self.cfg["runner"]["empirical_normalization"]  # 是否使用经验归一化
        
        # 初始化观测归一化器
        if self.empirical_normalization:
            # 使用经验归一化
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            # 不使用归一化，使用恒等映射
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # 无归一化
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # 无归一化

        # 初始化存储和模型
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,  # 环境数量
            self.num_steps_per_env,  # 每个环境的步数
            [num_obs],  # 观测维度
            [num_privileged_obs],  # 特权观测维度
            [self.env.num_actions],  # 动作维度
        )

        # 决定是否禁用日志记录
        # 只有rank为0的进程(主进程)才记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        
        # 日志配置
        self.log_dir = log_dir  # 日志目录
        self.writer = None  # 日志写入器
        self.tot_timesteps = 0  # 总时间步数
        self.tot_time = 0  # 总时间
        self.current_learning_iteration = 0  # 当前学习迭代次数
        self.git_status_repos = [humanoid.__file__]  # Git状态仓库

        # 重置环境
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """
        执行学习过程
        
        Args:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否以随机episode长度初始化
        """
        # 初始化日志写入器
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 启动Tensorboard或Neptune & Tensorboard摘要写入器，默认为Tensorboard
            self.logger_type = self.cfg.get("logger", "tensorboard")  # 获取日志类型
            self.logger_type = self.logger_type.lower()  # 转换为小写

            # 根据日志类型初始化对应的写入器
            # if self.logger_type == "neptune":
            #     from humanoid.utils.neptune_utils import NeptuneSummaryWriter
            #     self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            #     self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            # elif self.logger_type == "wandb":
            #     from humanoid.utils.wandb_utils import WandbSummaryWriter
            #     self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            #     self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)  # 创建Tensorboard写入器
            else:
                raise ValueError("未找到日志类型。请选择 'neptune', 'wandb' 或 'tensorboard'")

            # 保存训练配置到JSON文件
            with open(os.path.join(self.log_dir, 'train_cfg.json'), 'w') as f:
                print(self.log_dir)
                if "symmetry_cfg" in self.alg_cfg:
                    env = self.alg_cfg['symmetry_cfg'].pop("_env")
                    f.write(json.dumps(self.cfg, sort_keys=False, indent=4, separators=(',', ': ')))
                    self.alg_cfg['symmetry_cfg']["_env"] = env
                else:
                    f.write(json.dumps(self.cfg, sort_keys=False, indent=4, separators=(',', ': ')))

        # # 检查教师模型是否已加载
        # if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
        #     raise ValueError("教师模型参数未加载。请加载教师模型进行蒸馏")

        # 随机化初始episode长度(用于探索)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始学习
        obs = self.env.get_observations()  # 获取观测值
        privileged_obs = self.env.get_privileged_observations()  # 获取特权观测值
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)  # 移动到指定设备
        self.train_mode()  # 切换到训练模式(例如启用dropout)

        # 记录变量
        ep_infos = []  # episode信息
        rewbuffer = deque(maxlen=100)  # 奖励缓冲区
        lenbuffer = deque(maxlen=100)  # 长度缓冲区
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前奖励和
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前episode长度

        # 确保所有参数同步
        if self.is_distributed:
            print(f"同步rank {self.gpu_global_rank} 的参数...")
            self.alg.broadcast_parameters()
            # TODO: 是否需要同步经验归一化器?
            #   目前不需要，因为它们应该会"渐近地"收敛到相同值

        # 开始训练
        start_iter = self.current_learning_iteration  # 起始迭代次数
        tot_iter = start_iter + num_learning_iterations  # 总迭代次数
        for it in range(start_iter, tot_iter):
            start = time.time()  # 记录开始时间
            # 回合.rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 采样动作
                    actions = self.alg.act(obs, privileged_obs)
                    # 执行环境步骤
                    obs, privileged_obs, rewards, dones, infos, \
                        _, _ = self.env.step(actions.to(self.env.device))
                    # obs, privileged_obs, rewards, dones, infos, \
                    #     _, _, _ = self.env.step(actions.to(self.env.device))
                    # 移动到设备
                    obs, privileged_obs, rewards, dones = (
                        obs.to(self.device),
                        privileged_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # 执行归一化
                    obs = self.obs_normalizer(obs)

                    # 处理步骤结果
                    self.alg.process_env_step(rewards, dones, infos)

                    # 记录信息
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        # 更新episode长度
                        cur_episode_length += 1
                        # 清理已完成episode的数据
                        # -- 通用处理
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()  # 记录结束时间
                collection_time = stop - start  # 收集时间
                start = stop

                # 计算回报
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # 更新策略
            loss_dict = self.alg.update()

            stop = time.time()  # 记录结束时间
            learn_time = stop - start  # 学习时间
            # Store the number of completed iterations. This avoids repeating
            # the last PPO update after resume and makes --max_iterations an
            # exact total target.
            self.current_learning_iteration = it + 1
            
            # 记录日志信息
            if self.log_dir is not None and not self.disable_logs:
                # 记录信息
                self.log(locals())
                # 保存模型
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )

            # 清理episode信息
            ep_infos.clear()
            
            # 保存代码状态
            if it == start_iter and not self.disable_logs:
                # 获取所有差异文件
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可能，将它们存储到wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """
        记录训练日志
        
        Args:
            locs: 本地变量字典
            width: 日志输出宽度
            pad: 填充字符数
        """
        # 计算收集大小
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新总时间步数和时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode信息
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维张量信息
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 记录到日志和终端
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'平均episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()  # 平均动作标准差
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))  # 每秒帧数
        collection_fps = int(collection_size / max(locs["collection_time"], 1.0e-9))

        # -- 损失值
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- 策略
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- 性能
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_fps", collection_fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- 训练
        if len(locs["rewbuffer"]) > 0:
            # 记录奖励和episode长度
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb不支持非整数x轴记录
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- 损失值
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- 奖励
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode信息
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, 学习 {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timestep:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration tome:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """
        保存模型
        
        Args:
            path: 保存路径
            infos: 附加信息
        """
        # -- 保存模型
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),  # 模型状态字典
            "optimizer_state_dict": self.alg.optimizer.state_dict(),  # 优化器状态字典
            "iter": self.current_learning_iteration,  # 当前迭代次数
            "infos": infos,  # 附加信息
        }
        if hasattr(self.env, "get_checkpoint_state"):
            saved_dict["env_state"] = self.env.get_checkpoint_state()
        # -- 如果使用了观测归一化，则保存归一化器
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()  # 观测归一化器状态
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()  # 特权观测归一化器状态

        # 保存模型
        torch.save(saved_dict, path)

        # 上传模型到外部日志服务
        if (
            getattr(self, "logger_type", "tensorboard") in ["neptune", "wandb"]
            and not self.disable_logs
            and self.writer is not None
        ):
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """
        加载模型
        
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器
            
        Returns:
            加载的信息
        """
        # PyTorch 1.13 (the Isaac Gym Preview 4 target) does not accept the
        # newer ``weights_only`` keyword.
        try:
            loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        except TypeError:
            loaded_dict = torch.load(path, map_location=self.device)
        # -- 加载模型
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- 如果使用了观测归一化，则加载归一化器
        if self.empirical_normalization:
            if resumed_training:
                # 如果恢复了之前的训练，则为actor/student加载归一化器
                # 为critic/teacher加载归一化器
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果没有恢复训练但加载了模型，这必须是跟随RL训练的蒸馏训练
                # 因此为教师模型加载actor归一化器，不加载学生的归一化器
                # 因为观测空间可能与之前的RL训练不同
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- 如果使用了优化器则加载
        if load_optimizer and resumed_training:
            # -- 算法优化器
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- 加载当前学习迭代次数
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        if "env_state" in loaded_dict and hasattr(
            self.env, "load_checkpoint_state"
        ):
            self.env.load_checkpoint_state(loaded_dict["env_state"])
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """
        获取推理策略
        
        Args:
            device: 设备
            
        Returns:
            推理策略函数
        """
        self.eval_mode()  # 切换到评估模式(例如禁用dropout)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["runner"]["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """切换到训练模式"""
        # -- PPO
        self.alg.policy.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """切换到评估模式"""
        # -- PPO
        self.alg.policy.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """
        添加Git仓库到日志记录
        
        Args:
            repo_file_path: 仓库文件路径
        """
        self.git_status_repos.append(repo_file_path)

    """
    辅助函数
    """

    def _configure_multi_gpu(self):
        """配置多GPU训练"""
        # 检查是否启用分布式训练
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))  # GPU世界大小
        self.is_distributed = self.gpu_world_size > 1  # 是否分布式训练

        # 如果不是分布式训练，将本地和全局rank设置为0并返回
        if not self.is_distributed:
            self.gpu_local_rank = 0  # 本地rank
            self.gpu_global_rank = 0  # 全局rank
            self.multi_gpu_cfg = None  # 多GPU配置
            return

        # 获取rank和世界大小
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地rank
        self.gpu_global_rank = int(os.getenv("RANK", "0"))  # 全局rank

        # 创建配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的rank
            "local_rank": self.gpu_local_rank,  # 当前进程的rank
            "world_size": self.gpu_world_size,  # 总进程数
        }

        # 检查用户是否为本地rank指定了设备
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"设备 '{self.device}' 与本地rank '{self.gpu_local_rank}' 的预期设备不匹配")
        # 验证多GPU配置
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"本地rank '{self.gpu_local_rank}' 大于或等于世界大小 '{self.gpu_world_size}'")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"全局rank '{self.gpu_global_rank}' 大于或等于世界大小 '{self.gpu_world_size}'")

        # 初始化torch分布式
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # 将设备设置为本地rank
        torch.cuda.set_device(self.gpu_local_rank)
