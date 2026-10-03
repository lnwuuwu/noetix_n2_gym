# 吴淑林的个人贡献

## 个人职责

本人负责 **强化学习训练、参数调整、日志分析、候选 checkpoint 量化筛选**。个人职责以本人确认的实际工作为准；技术参数、模型和实验结果依据归档配置、源码及项目材料说明。本页只介绍本人承担的工作。

### 强化学习训练

在 Isaac Gym 中开展 N2 人形机器人多地形 PPO 训练，使用并行环境采样、保存训练日志并生成候选策略。项目归档配置采用 2048 个并行环境与每轮 24 步 rollout。

对应实现：[训练入口](../humanoid/scripts/train.py)、[PPO 训练流程](../humanoid/algo/ppo/on_policy_runner.py)、[任务配置](../humanoid/envs/n2/n2_parkour_config.py)。

### 参数调整

结合训练与回放表现调整训练、奖励及地形课程相关参数，对比不同配置下的运动稳定性、任务完成度和失败情况，保留与候选策略相对应的配置。

对应材料：[Parkour 配置](../humanoid/envs/n2/n2_parkour_config.py)、[checkpoint 3800 的归档配置](../sim2sim/configs/train_cfg_checkpoint_3800.json)。

### 日志分析

分析训练日志、奖励变化与回放表现，定位影响稳定性和地形通过的失败模式，为后续调参及候选策略筛选提供依据。

对应实现：[日志查看入口](../humanoid/scripts/tbpeek.py)、[训练过程日志记录](../humanoid/algo/ppo/on_policy_runner.py)。

### 候选 checkpoint 量化筛选

在一致评测条件下比较候选策略，综合完整课程是否完成、地形段完成度、失败次数、重置次数、进度和耗时等指标筛选稳定策略，而不是只看训练 reward 或单段录屏。

对应实现：[训练策略评测](../humanoid/scripts/evaluate_parkour_checkpoints.py)、[MuJoCo 批量回放](../sim2sim/evaluate_parkour_batch.py)、[量化排序规则](../sim2sim/parkour_evaluation.py)。

发布包含 [policy_best_stable.pt](../sim2sim/policy/policy_best_stable.pt)、对应配置与[历史固定跑道回放记录](results/mujoco_final_replay.json)。报告归档模型对应 checkpoint 3800；结果 JSON 中原有的 null checkpoint 字段保持不变，没有补造原始记录。

## 结果与来源

历史固定跑道的一次确定性回放完成 5 类地形、45/45 路点，0 次重置和失败，仿真时间为 119.902 s。这是项目已有记录，不是本次发布新测量，也不代表多随机种子统计成功率或实机结果，详见 [结果说明](RESULTS.md)。

源码来自既有答辩交付版本；保留原 GitHub 默认分支历史、开源许可证和原作者信息。视频为已有 Isaac Gym 与 MuJoCo 原始录屏。个人贡献、上游代码与项目实验结果分别说明。
