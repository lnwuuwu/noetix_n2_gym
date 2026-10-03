# 个人贡献与团队分工

## 依据

分工来自已有《N2人形机器人多地形运动控制——答辩讲解与问答手册》第 10 节，以及之前的全项目复习报告；技术链路与参数来自《N2设计报告（合并版）》和归档源码。

报告记载吴淑林负责总体技术路线、任务与方法设计、各模块协调和答辩材料整合。

## 可展示的工作

- 技术路线：梳理盲走、感知运动与 Parkour 三条路线，围绕失败模式组织方案比较。
- 任务与方法设计：参与确定双腿 10DOF、时序本体 / 高程输入、路点任务目标和多地形课程的总体方案。
- 接口协调：围绕观测顺序、历史帧、高程展平、关节映射、策略 / PD 频率、动作尺度及碰撞语义，协调训练与部署模块。
- 材料整合：把训练、筛选、失败模式与 Sim2Sim 结果组织为可核对的设计报告和演示。

上述为报告记载的总体设计和协作职责。不能据此宣称训练代码、参数调试或 MuJoCo 部署均由本人独立完成。

## 团队职责

| 成员 | 报告记载职责 |
|---|---|
| 吴淑林 | 总体技术路线、任务与方法设计、模块协调、材料整合 |
| 杜军 | 训练、调参、日志分析和 checkpoint 量化筛选 |
| 赵汝坤 | 模型回放评测、实验数据整理、结果分析 |
| 武天豪 | MuJoCo 验证、演示检查、部署文件整理 |

训练配置、观测 / 动作接口、模型筛选与 Sim2Sim 契约由团队共同核对；这份个人作品集保留团队与开源来源。

## 代码对应关系

| 设计议题 | 团队实现 |
|---|---|
| 时序高程输入与扫描压缩 | humanoid/envs/n2/n2_parkour_env.py、humanoid/algo/ppo/actor_critic.py |
| 任务目标、奖励与课程 | humanoid/envs/n2/n2_parkour_config.py、humanoid/utils/terrain.py |
| PPO 训练 / 采样 | humanoid/algo/ppo、humanoid/scripts/train.py |
| JIT 导出 | humanoid/scripts/play.py |
| 部署契约与回放 | sim2sim/sim2sim_parkour.py、sim2sim/configs/n2_parkour_slow_stable.yaml |
| 确定性跑道与筛选 | sim2sim/parkour_campaign.py、sim2sim/evaluate_parkour_batch.py |

表中路径用于解释方案如何落实，并非个人代码作者认定。

## 本次发布来源

源码使用已有答辩交付目录中的最终版本，补齐原 GitHub 仓库中的项目环境、测试、归档配置和部署模型；保留 GitHub 默认分支的历史。视频来自团队归档的 Isaac Gym 和 MuJoCo 两段录屏，未经剪辑。完整报告中的历史指标与本次发布检查分别记录。
