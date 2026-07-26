# FastStair-N2 训练与验收

这套入口把 FastStair 论文中最适合当前 N2 问题的第一阶段落到了
Isaac Gym：GPU 并行 DCM 落脚搜索、地形可行性约束、连续摆脚轨迹奖励，
以及独立留出集验收。

它不会覆盖现有 `n2_stairs_walk` 策略。新任务名是
`n2_faststair`，训练输出写入 `logs/n2_faststair`，最终只有通过
holdout 的模型才会写入
`logs/faststair_launcher/selected_s*/model_best.pt`。

## 为什么不再从随机策略训练

旧 PPO Actor 每帧输入 82 维，5 帧共 410 维。FastStair Actor 使用
70 维本体/导航信息和 9 x 5 地形图，每帧 115 维，5 帧共 575 维；
Critic 还额外接收 DCM 规划信息。启动器现在执行经过约束的
Actor-only bootstrap：

1. 每帧共同的 70 维输入和全部后续 Actor 层逐值复制；
2. 新 9 x 5 地形图明确包含旧 4 x 3 地形点的全部物理坐标，逐点精确映射；
3. 新增的 33 个地形输入权重置零；
4. Critic 和优化器从零开始。

这样初始 FastStair Actor 精确保留旧输入子空间上的动作，同时保留
学习更宽地形图的能力。旧 PPO 还继续作为不会覆盖的部署回退模型和
相同 holdout seed 上的性能基准。

正式训练前，启动器会先只生成 `model_0.pt`，不采样、不做 PPO
更新；随后分别以 `n2_faststair` 和 `n2_stairs_walk` 在同一 seed
上做确定性评估。只有日志出现
`FASTSTAIR_BOOTSTRAP_GATE passed=True` 才会启动 Stage 1。如果迁移、
地形图、相位时钟或物理配置破坏了旧策略，训练会在零次 PPO rollout
时停止。

## 当前实现范围

论文完整流程还包含低速/高速专家与 LoRA 融合。本仓库当前先实现
安全基座预训练，因为 N2 的目标速度仅为 0.10–0.22 m/s，而且在安全
基座通过验收前训练专家只会增加失败分支和 GPU 消耗。

启动器内部使用三个地形阶段，而不是三个速度专家：

- Stage 1：固定 2 cm，发现可靠抬脚和落脚。
- Stage 2：均衡训练 2/4/6 cm。
- Stage 3：加入 8/10 cm，并提高目标楼梯占比。

三个阶段默认都保持旧策略已经验证的 0.18 m/s 命令，只逐步增加楼梯
高度，避免在迁移的同时改变相位时钟输入。探索噪声固定为 0.05，而
不是会让首批采样直接跌倒的 0.12。每阶段都评估中间检查点，但只有
至少一个模型通过该阶段的完成率、首步率、跌倒、路径和规划有效率
门槛才会晋级。没有合格模型时立即停止，不再把“最不差的失败模型”
传给下一阶段消耗 GPU。

## 服务器操作

先让当前普通 PPO 训练正常结束，并保存它的已批准模型。下面示例把
它显式指定为基准；请把路径替换为本次训练实际生成的路径：

```bash
cd "/home/lcx/桌面/wsl/ultralytics-8.3.221/noetix_n2_gym_fork"
conda activate n2stairs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64"

PPO_BEST="$PWD/logs/isaac_launcher/stability_selected_s45/model_best.pt"
test -f "$PPO_BEST"
```

先运行迁移验收。它生成 `model_0.pt`，在相同 seed 下与旧模型逐级
对照，但不会执行任何 PPO 更新：

```bash
N2_SEED=46 \
N2_DEVICE=cuda:0 \
N2_FASTSTAIR_BASELINE_CHECKPOINT="$PPO_BEST" \
bash humanoid/scripts/run_faststair_n2.sh smoke
```

只有输出 `FASTSTAIR_SMOKE passed=True` 后才启动正式训练：

```bash
N2_SEED=46 \
N2_DEVICE=cuda:0 \
N2_FASTSTAIR_BASELINE_CHECKPOINT="$PPO_BEST" \
bash humanoid/scripts/run_faststair_n2.sh train
```

训练在后台运行，不打开服务器 GUI：

```bash
N2_SEED=46 bash humanoid/scripts/run_faststair_n2.sh status
N2_SEED=46 bash humanoid/scripts/run_faststair_n2.sh log
```

默认训练量为 400 + 600 + 800 = 1800 PPO iterations，1024 个并行
环境，每 100 iterations 保存并筛选一次。FastStair 的大地图在
4096 环境下会耗尽 PhysX `foundLostAggregatePairs` 缓冲并漏掉接触，
因此启动器默认并硬性保护在 1024 环境以内；任务配置同时把 GPU
contact-pair 容量提高到 `2**24`、内部 buffer multiplier 提高到 8。
不要再用 4096 环境运行。显存仍不足时可降到 512：

```bash
N2_SEED=46 \
N2_FASTSTAIR_NUM_ENVS=512 \
N2_FASTSTAIR_BASELINE_CHECKPOINT="$PPO_BEST" \
bash humanoid/scripts/run_faststair_n2.sh train
```

## 如何判断结果

长训开始前必须先看到：

```text
FASTSTAIR_BOOTSTRAP_GATE passed=True
FASTSTAIR_STAGE_TRAIN stage=1 ...
```

若出现 `FASTSTAIR_TRAINING_ABORT reason=bootstrap_preflight_failed`，
表示迁移模型没有复现旧策略，启动器尚未执行任何 PPO rollout。

只有日志出现以下内容，模型才被批准：

```text
FASTSTAIR_HOLDOUT_APPROVED=True
N2_FASTSTAIR_BEST=.../model_best.pt
```

若出现：

```text
FASTSTAIR_STAGE_STOP stage=... reason=promotion_gate_failed
```

说明当前阶段没有任何合格检查点，后续阶段不会启动。若三阶段均晋级
但最终独立留出集未批准，则会出现：

```text
FASTSTAIR_HOLDOUT_APPROVED=False
N2_FASTSTAIR_SCREEN_BEST=.../model_screen_best.pt
```

说明训练轨迹里有可供诊断的最佳候选，但它没有通过独立 holdout，
不能替代旧 PPO。启动器会保留以前已经批准的 FastStair 模型；若从未
有 FastStair 模型通过，旧 PPO 仍是部署模型。

验收同时检查：

- 10 cm、8 cm、6 cm 完成率；
- 跌倒率、路径失败率和横向偏移；
- 动作变化/加速度，防止用剧烈抖动换完成率；
- DCM 规划有效率；
- 实际落脚到规划目标的误差；
- 足底距踏板边缘的安全余量；
- 相对旧 PPO 的完成率、跌倒、路径、横移和动作平滑度回退。

## 无 GUI 可视化

只有已通过 holdout 的模型可以由 `view` 启动：

```bash
N2_SEED=46 \
N2_STREAM_PORT=18080 \
bash humanoid/scripts/run_faststair_n2.sh view
```

本地建立 SSH 隧道时可选择任意未占用的本地端口，例如：

```bash
ssh -N -L 18082:127.0.0.1:18080 lad427_4090
```

然后在本地浏览器打开 `http://127.0.0.1:18082`。

## 关键文件

- `humanoid/utils/faststair_planner.py`：纯 Torch GPU 并行 DCM 搜索。
- `humanoid/envs/n2/n2_stairs_env.py`：规划、轨迹奖励与 touchdown 统计。
- `humanoid/envs/n2/n2_stairs_config.py`：`n2_faststair` 任务配置。
- `humanoid/scripts/select_faststair_checkpoint.py`：绝对与相对验收门。
- `humanoid/scripts/run_faststair_n2.sh`：训练、筛选、holdout 与可视化入口。
