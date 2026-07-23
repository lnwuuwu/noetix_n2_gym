# Noetix N2 上楼梯训练与 AutoDL 部署

本文档对应独立任务 `n2_stairs`、`n2_stairs_robust` 和严格步态任务
`n2_stairs_walk`。原始 `n2`、`n2_10dof`、`n2_mimic` 仍然保留。
当前开发机没有 Isaac Gym/GPU，因此本文所说的“已验证”仅指静态检查；
真实 PhysX 仿真、PPO 收敛和 4090 显存占用必须在 AutoDL 执行本文的冒烟测试确认。

## 1. 审查结论

- `LeggedRobot.create_sim()` 对复杂地形实际实例化的是 `HumanoidTerrain`，不是基类
  `Terrain`。
- 当前原 `n2` 配置实际是 `terrain.curriculum=True`。`HumanoidTerrain` 把原始
  `[0.10, 0, 0.05, 0.05, 0.05, 0.45, 0.20]` 转为累积阈值
  `[0.10, 0.10, 0.15, 0.20, 0.25, 0.70, 0.90]`，尾部 10% 是平地。
  课程模式又用 `j / 10 + 0.001` 离散选 10 列，因此真正生成的是 1 列粗糙、
  1 列随机起伏、1 列下坡、4 列上楼梯（40%）、2 列下楼梯（20%）、1 列平地；
  障碍物和上坡实际没有列，不能把名义百分比直接当生成结果。
- 原楼梯高度是 `difficulty * 0.20`。10 行课程使用 `i / num_rows`，所以实际为
  0、2、4、...、18 cm，并非注释所暗示的最高 10 cm。
- 原课程只用机器人相对环境原点的平面 XY 距离升降级，没有验证脚下地形高度、
  台阶完成数或是否到达顶平台；最后一级还会随机回退。
- 原地形的环境原点位于金字塔楼梯中央，而 reset 又随机 XY 各 ±1 m。这不能保证
  从低平台出生，也不能保证面向上升方向。
- 原 `n2` Actor 每帧 63 维，只含命令、角速度、重力、关节位置/速度和动作；96 个
  地形高度仅提供给 Critic。规则楼梯可凭接触反应学习，但对从零训练和高度泛化不利。
- 原 `play.py` 强制 1 个环境、最新 checkpoint 和 1.0 m/s，还会重写所有任务的地形。
  它确实把比例改成全上楼，但同时关闭课程，使每块测试楼梯独立随机到约 0–20 cm
  单级高度并叠加粗糙度，不是固定、可复现的楼梯等级。
- 策略控制周期为 `0.002 * 10 = 0.02 s`，即 50 Hz；动作仍是 18 个关节的
  默认角度加 `0.25 * action`，PD 和 URDF 力矩限制继续生效。
- 原 PPO rollout 构造时把设备字符串误传成 `rnd_state_shape`，缓冲区实际落在 CPU；
  4090 训练会发生大量 CPU/GPU 拷贝。该调用已改为显式 `device=self.device`。
- 原摩擦和恢复系数随机化把 `(N, 1)` 张量压成 `(N,)`，与 N2 Critic 的二维特征
  拼接不一致；现已保持 `(N, 1)`。
- 原 resume 不加载优化器，也不保存 terrain curriculum；现在训练恢复会加载模型、
  优化器、已完成迭代数、每个环境的楼梯等级及连续成败计数，并把
  `--max_iterations` 解释为总目标迭代数。改变并行环境数时，课程状态会循环映射到
  新环境数；固定级别训练和评估不会被 checkpoint 的课程状态覆盖。
- N2 URDF 的 XML 非固定关节声明顺序是“左臂、右臂、左腿、右腿”，但原 N2 数字
  奖励索引、原 Sim2Sim 参数数组和 MJCF 树遍历三者都采用“左臂、左腿、右臂、
  右腿”；后者因此是当前最可信的 Isaac 策略顺序。专用任务按关节名建立奖励索引，
  并在 Isaac Gym 启动时把真实 DOF 顺序与该顺序逐项断言。Sim2Sim YAML 同样采用
  该策略顺序，再按名字解析 MJCF 的 qpos、qvel 和 actuator。开发机没有 Isaac Gym，
  所以最终确认必须由服务器冒烟测试完成；若导入器返回不同顺序，会打印
  expected/actual 并停止，不会静默错位训练。

## 2. 专用任务设计

### 严格交替步态任务

`n2_stairs` 保留为 375 维 Actor 的几何爬楼 baseline，以兼容已有 checkpoint。
仅凭到顶率不能证明策略在走路：策略可能无视 0.18 m/s 命令高速双脚蹦跳，或斜向
到顶后仍被统计为成功。遇到这种情况应从零训练 `n2_stairs_walk`，不要把旧 checkpoint
恢复到新任务（网络输入分别为 375 和 410，框架也会因维度不同拒绝加载）。

严格任务在原有地形前视基础上增加：

- 可部署的左右步态相位 `sin/cos`，以及按相位定义的单支撑/摆动接触目标；
- 相位频率按 `command_speed / (2 × 0.30 m踏面)` 变化，即 0.12 m/s 时约
  0.20 Hz、0.45 m/s 时约 0.75 Hz，使每半周期对应跨一阶，不再用过快相位迫使
  机器人在同一踏面内反复换脚。为了兼容已按 1.25 Hz 训练的严格任务 checkpoint，
  续训开始时仍使用旧频率，并在 800 个 PPO iteration（每次 24 个策略步）内平滑
  过渡到踏面匹配频率；退火进度随 checkpoint 保存和恢复；
- 机身坐标系线速度、相对楼梯中心线横向误差和相对 `+X` 航向误差；
- 超速、双脚腾空、相位接触不匹配、髋 yaw/roll 偏离和足部外八惩罚；
- 横偏超过 0.30 m 或偏航超过 0.40 rad（1 s 宽限后）直接作为路径失败；
- 训练终点分为三层：在顶平台直立并稳定支撑 0.60 s 即为物理完成并结束回合；
  同时满足中等步态门槛才用于课程晋级；满足全部严格门槛才计为最终自然步态成功。
  物理完成不再继续走到超时后被错误施加失败惩罚；三层分别获得 4、8、12 的一次性奖励；
- 到顶后需连续 0.30 s 满足支撑、直立、速度、横偏 0.12 m 和偏航 0.15 rad
  条件，且整段最大横偏不超过 0.20 m、最大偏航不超过 0.30 rad、相位接触匹配率
  至少 70%、双脚腾空占比不超过 8%，才记为成功。70% 仍高于永久双脚着地在该
  相位定义下约 58% 的理论上限，但允许台阶高度导致的自然提前/延后触地。
- 严格成功要求整段平均速度与命令的偏差不超过 0.05 m/s；不再用单个时刻或
  `mean(abs(v-command))` 拒绝自然步态固有的步内速度波动。每个策略步仍使用更尖锐的
  速度误差核，并惩罚超过 `1.15 * command + 0.02` 的超速。
- 环境按台阶高度量化两只脚的落点，并记住“上一只跨到新台阶的脚”。只有另一只脚
  依次跨上下一阶才获得交替奖励；同一只脚连续领步、后脚并到同一阶和跨过台阶分别
  受罚。6 级楼梯至少检测到 4 次有效交替，交替率至少 75%，并阶率和跳阶率均不得
  超过 20%，才允许记为成功。
- 整回合双脚前后最大分离不得超过 0.44 m；超过 0.40 m 已开始连续惩罚。摆动腿膝盖
  随台阶高度跟踪约 0.53–0.73 rad 的屈曲目标，双肩以 0.22 rad 小幅反相摆动，减少
  直腿远伸和上半身完全僵硬。N2 没有腰部自由度，因此不能人为加入躯干关节动作。
- 相位驱动的落脚参考同时约束下一踏面中心 X、左右脚约 ±0.09 m 的自然站宽 Y，
  并继续用随台阶高度变化的摆脚净空约束 Z；这比仅增强 X 方向奖励更能抑制交叉腿、
  外扩落脚和累积横向漂移。

这些改动没有改变 `n2_stairs_walk` 的 410/153 维 Actor/Critic 输入或 18 维动作，
所以该任务已有 checkpoint 可以继续微调；375 维的 `n2_stairs` checkpoint 仍不能
直接加载到 `n2_stairs_walk`。奖励或相位定义发生明显变化时，续训命令可加
`--reset_optimizer`：保留策略、归一化统计、迭代号和地形课程，但不继承旧 Adam
动量。

评估把 `top_reached_rate`（位置/高度）、`completion_rate`（稳定物理完成）、
`curriculum_completion_rate`（可晋级步态）和 `success_rate`（严格自然步态）分开输出，
并额外输出平均实际速度、命令误差、相位接触匹配率、双脚腾空占比、整回合最大
横偏/偏航、交替跨阶次数/比例、重复领步率、并阶率、跳阶率、最大前后脚距、摆动膝
平均屈曲和摆臂匹配度。因此，新的验收不能只看 `success_rate`。

### 地形与课程

`N2StairsTerrain` 的每一列都只生成沿世界/局部 `+X` 上升的楼梯，不会混入下楼梯、
沟壑、踏石或斜坡。每块地形为 5.0 m × 3.0 m：

| level | 单级高度 | 6 级总爬升 |
|---:|---:|---:|
| 0 | 0.02 m | 0.12 m |
| 1 | 0.04 m | 0.24 m |
| 2 | 0.06 m | 0.36 m |
| 3 | 0.08 m | 0.48 m |
| 4 | 0.10 m | 0.60 m |

踏面宽 0.30 m，共 6 级。出生点在低平台中央，距第一立面约 0.60 m；XY reset
扰动仅为 ±8 cm/±5 cm，朝向固定为 +X。三角网格的陡坡修正阈值为 0.30，低于
2 cm 立面在 5 cm 水平网格上的 0.4 斜率，因此最低难度也不会被转换成短坡。

训练从 level 0 开始。严格步态任务一次满足中等课程门槛即升级；连续两次失败才降级。
课程通过要求：

- 根部 X 已越过顶平台成功线；
- 脚下地形最大高度达到该楼梯顶高（允许 1.2 cm 采样误差）；
- 躯干保持直立、至少一只脚稳定支撑，且同一步没有 base 碰撞或高度跌倒。

连续两次跌倒、停滞、越界或未完成 timeout 才降一级；单纯物理完成但步态未达到课程
门槛是中性结果，不会被当成失败。中间出现其他结果会清零连续计数。最高一级
保持在 level 4，不随机回退。机器人超过 2 s 宽限期后，如果 4 s 内未增加至少
6 cm 的累计最佳前进距离，会提前按停滞失败终止；watchdog 使用独立的 6 cm
进度里程碑，不会错误地要求单个 20 ms 策略步移动 6 cm。
专用任务不随机偏置初始 episode 长度，避免进度 watchdog 把刚重置的环境误判为长时停滞。

### 命令

- 初始 X 速度范围：0.12–0.25 m/s；
- Y 速度和 yaw 速度固定为 0；
- 不采样站立命令；整回合保持同一条命令；
- 每个环境按自身 terrain level 解锁速度，X 上限从 0.25 逐级提高到 0.45 m/s；
  这样低台阶环境不会被已经升级的环境提前推到高速，也避免每次 reset 做 GPU→CPU
  同步来计算全局命令范围。

可视化和评估默认使用 0.25 m/s，且会按所测 level 拒绝训练课程未覆盖的速度：
level 0–4 的上限依次为 0.25、0.30、0.35、0.40、0.45 m/s；一次评估多个 level
时采用其中最低的上限。

### 观测

- Actor：每帧 `63 + 12 = 75` 维，堆叠 5 帧，总输入 375 维；
- 12 个新增量是 4 个前向距离 × 3 个横向位置的地形高度；
- Critic：63 维 proprio、62 维速度/域参数/接触特权量、21 个完整高度点，共 146 维；
- 动作：18 维。

`n2_stairs_walk` 每帧为 `82 = 63 + 相位2 + 机身线速度3 + 路径状态2 + 地形12`
维，堆叠 5 帧后 Actor 输入 410 维；Critic 为 153 维。路径状态在 Isaac Gym 中以
环境中心线和世界 `+X` 为参考；MuJoCo 配置提供完全一致的输入。真实机器人部署时，
除地形高度外还需要状态估计器提供机身速度、相对楼梯中心线位置及航向。

这会提升仿真楼梯学习能力，但也意味着真实机器人部署必须提供与训练定义一致的前向
地形高度（深度相机、激光或可靠局部高程图）。`sim2sim/configs/n2_stairs.yaml`
用解析楼梯几何生成同样的 12 个输入；它不是实际传感器替代方案。

### 奖励和防取巧

稠密奖励 scale 会被框架统一乘以策略 `dt=0.02`。新台阶高度、到顶和失败属于离散
事件，函数内部抵消该 `dt`，因此配置中的 1.0/25/-10 分别是每个新 riser、成功和
失败的实际量级，不会被无意缩小 50 倍。关键项均自动写入 TensorBoard 的
`Episode/rew_*`：

- 世界 +X 前进速度：需保持直立、至少一脚支撑，并用横向速度门控；
- 地形高度进度：只奖励脚下地形“新达到的最高高度”，不直接奖励 base Z 速度，
  反复跨同一立面不能重复刷分；
- X 速度跟踪：静止时门控为 0，避免停在楼梯前领取跟踪奖励；
- 到顶一次性奖励和失败终止惩罚；
- 直立、相对地形的机身高度、横向位置/速度、偏航漂移和停滞；
- 摆动脚相对当前台阶高度的对称间隙目标、稳定落脚、小权重 air-time；
- 相邻新台阶必须由左右脚轮流跨上；同脚重复领步、后脚并阶、跳阶和过大前后跨距；
- 摆动腿屈膝目标以及与腿相反的小幅肩部摆动；
- 双脚悬空、Z 速度、脚滑、立面撞脚、非足部碰撞和过大接触力；
- torque、DOF acceleration、energy、动作一阶/二阶变化和关节限位。

`only_positive_rewards=False`，因此惩罚不会被总奖励裁零。`feet_air_time` 权重很小，
摆动间隙是对称误差，且两者都要求实际向前运动；这抑制原地高抬腿和无限抬脚。
专用任务还在两个脚踝上启用了仅包含求解器约束力、不包含重力的世界坐标力
传感器，用于支撑/摆动、稳定落脚、立面撞脚和过大接触力判定。原始任务默认关闭
该选项，仍使用原有 net contact force 路径。

### 域随机化

`n2_stairs` 是从零训练的 baseline：摩擦 0.70–1.10、恢复 0–0.05、PD 和电机
强度 ±5%、动作延迟混合 0–0.15；不加质量/质心变化、外力或推搡。摩擦桶在创建
环境时分配，不在每回合通过 CPU Gym API 重写，以保证大量并行环境吞吐。

`n2_stairs_robust` 保持同一网络结构，用于已有可用 checkpoint 的第二阶段：摩擦
0.50–1.40、PD/电机 ±15%、质心 ±2 cm、质量 ±2 kg、动作延迟 0–0.35，并加入
较弱随机外力。它从所有 5 个高度随机初始化，不建议直接从零训练。
跨任务的 baseline→robust 迁移只加载网络和优化器，不覆盖 robust 的初始
地形分布；同一 robust 任务的中断恢复仍会恢复它自己的课程状态。

## 3. AutoDL 环境安装

以下假设仓库放在 `/root/autodl-tmp/noetix_n2_gym`，Isaac Gym Preview 4 解压到
`/root/autodl-tmp/isaacgym`。若目录不同，只改这两个明确路径。

```bash
cd /root/autodl-tmp
git clone https://github.com/JunDu-cyber/noetix_n2_gym.git
cd noetix_n2_gym
# 本地修改经用户审查、commit 并 push 后：
git fetch origin codex/n2-stairs
git switch --track origin/codex/n2-stairs

conda create -n n2stairs python=3.8.18 -y
conda activate n2stairs
python -m pip install --upgrade pip==23.3.2 setuptools==68.2.2 wheel
```

PyTorch 1.13.1 官方 Linux 包使用 CUDA 11.7 构建；NVIDIA 驱动显示 CUDA 11.8 或
更高仍可运行该二进制。不要使用不存在的 `torch==1.13.1+cu118` 包。若 AutoDL
镜像已经提供可用的 PyTorch 1.13.1，则先用下一节命令核对，无需重装。

```bash
python -m pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

cd /root/autodl-tmp/isaacgym/python
python -m pip install -e .

cd /root/autodl-tmp/noetix_n2_gym
python -m pip install -r requirements-autodl.txt
python -m pip install -e . --no-deps
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

## 4. 环境检查

```bash
nvidia-smi
nvcc --version || true

python - <<'PY'
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("Isaac Gym import: OK")
assert torch.__version__.startswith("1.13.1")
assert torch.cuda.is_available()
PY

python - <<'PY'
import isaacgym
import humanoid.envs  # noqa: F401 - register every task
from humanoid.utils import task_registry
required = {"n2", "n2_10dof", "n2_mimic", "n2_stairs", "n2_stairs_robust", "n2_stairs_walk"}
print("registered tasks:", sorted(task_registry.task_classes))
assert required.issubset(task_registry.task_classes)
PY

python -m unittest -v tests.test_stairs_static
python -m compileall -q humanoid sim2sim tests
git diff --check
```

Isaac Gym 应先于 `torch` 导入。第一次运行会编译 `gymtorch` 扩展；若曾在不同 CUDA
环境编译失败，应只删除对应用户缓存中的那个 `torch_extensions` 构建目录，再重试。

## 5. 训练命令

所有命令都显式指定同一张 GPU 给仿真和 PPO。

### 5.1 64 环境冒烟测试（从零，10 次迭代）

```bash
cd /root/autodl-tmp/noetix_n2_gym
conda activate n2stairs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python humanoid/scripts/train.py \
  --task=n2_stairs \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=64 \
  --max_iterations=10 \
  --run_name=smoke_64 \
  --seed=42
```

验收：环境创建时没有 DOF/观测维度/足端力传感器数量断言；迭代能推进；TensorBoard 中出现
`rew_stairs_*`、`stairs_completion_rate`、`stairs_curriculum_pass_rate`、
`stairs_success_rate`、terrain level、value/surrogate/entropy、
learning rate、`Perf/collection_fps` 和 `Perf/total_fps`；目录中有数字编号的
`model_*.pt`。

启动日志会打印 Isaac Gym 的 DOF names。若顺序断言失败，不要删除断言继续训练；保留
完整 expected/actual 输出，按 actual 同步 `expected_dof_order` 和两个 18-DoF Sim2Sim
YAML 的 `joint_order` 及对应参数数组后重新冒烟。这是当前开发机无法替代的服务器检查。

### 5.2 中等规模验证（1024 环境，750 次迭代）

```bash
python humanoid/scripts/train.py \
  --task=n2_stairs \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=1024 \
  --max_iterations=750 \
  --run_name=medium_1024 \
  --seed=42
```

先确认 1024 稳定，再分别用 2048 和 4096 做 20–50 iteration 的容量测试。4096
是否适合取决于 PhysX contact buffer、驱动、同时运行的进程和显存碎片，不能仅凭代码
保证。发生 OOM 时先退到 2048，不要同时降低 PhysX 接触容量掩盖接触溢出。

### 5.3 RTX 4090 正式后台训练

```bash
cd /root/autodl-tmp/noetix_n2_gym
conda activate n2stairs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p server_logs
RUN_TAG=stairs_4096_s42
TRAIN_STDOUT="server_logs/${RUN_TAG}.log"
TRAIN_PID_FILE="server_logs/${RUN_TAG}.pid"

nohup python -u humanoid/scripts/train.py \
  --task=n2_stairs \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=4096 \
  --max_iterations=8000 \
  --run_name="${RUN_TAG}" \
  --seed=42 \
  >"${TRAIN_STDOUT}" 2>&1 &
TRAIN_PROCESS_ID=$!
echo "${TRAIN_PROCESS_ID}" | tee "${TRAIN_PID_FILE}"
```

若 4096 OOM，把 `--num_envs` 改为 2048；PPO mini-batch 会随环境数自动缩放。

## 6. 查看进度、停止和恢复

```bash
tail -f server_logs/stairs_4096_s42.log
watch -n 2 nvidia-smi
tensorboard --logdir logs/n2_stairs --bind_all --port 6006
```

浏览器通过 AutoDL 端口映射访问 6006。找出最新训练目录：

```bash
find logs/n2_stairs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
  | sort -nr | head
find logs/n2_stairs -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head
```

优雅停止会触发一次最终 checkpoint 保存：

```bash
kill -TERM "$(cat server_logs/stairs_4096_s42.pid)"
tail -n 50 server_logs/stairs_4096_s42.log
```

从指定 run 的最新 checkpoint 恢复到总目标 8000 iteration：

```bash
cd /root/autodl-tmp/noetix_n2_gym
conda activate n2stairs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python humanoid/scripts/train.py \
  --task=n2_stairs \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=4096 \
  --max_iterations=8000 \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<run目录> \
  --checkpoint=-1 \
  --run_name=stairs_4096_resume \
  --seed=42
```

checkpoint 编号表示已经完成的 PPO iteration 数。恢复会加载优化器和楼梯课程状态，并只运行
`8000 - checkpoint迭代号` 的剩余部分。若要在 8000
之后继续训练，应把总目标改成更大的值，例如 `--max_iterations=10000`。

## 7. 可视化和批量评估

### 7.1 推荐：浏览器实时画面（容器/VNC 环境）

Isaac Gym Preview 4 的交互 Viewer 使用 Vulkan。在普通 TurboVNC/Xvnc
桌面中，桌面和 OpenGL 测试可能正常，但 Viewer 仍可能是全黑窗口。项目提供的
`stream_stairs.py` 不创建 Viewer，而是让 Isaac Gym GPU 离屏相机直接生成画面并通过
MJPEG 实时发送，因此无需在容器中配置 GPU-backed Xorg。

先在服务器安装轻量 JPEG 编码依赖（已经安装时会直接提示 satisfied）：

```bash
conda activate n2
python -m pip install Pillow
```

服务器终端运行（进程必须保持运行）：

```bash
cd /root/autodl-tmp/noetix_n2_gym
conda activate n2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64"
python humanoid/scripts/stream_stairs.py \
  --task=n2_stairs \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<run目录> \
  --checkpoint=-1 \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --terrain_level=0 \
  --command_speed=0.18 \
  --stream_port=8080 \
  --seed=42
```

然后在本地电脑另开一个终端建立隧道：

```bash
ssh -N -L 8080:127.0.0.1:8080 \
  -p 21508 root@connect.cqa1.seetacloud.com
```

SSH 终端登录后一直没有新输出是正常的，表示隧道正在工作。保持它不关闭，在本地浏览器
打开 `http://127.0.0.1:8080/` 即可实时观看。服务器端按 `Ctrl-C` 停止仿真。
HTTP 服务只绑定服务器 `127.0.0.1`，不会直接暴露公网端口。需要降低带宽时可增加
`--camera_width=640 --camera_height=360 --jpeg_quality=65`。

### 7.2 交互 Viewer（仅 GPU 图形桌面可用时）

10 cm 台阶、0.25 m/s、单机器人可视化：

```bash
cd /root/autodl-tmp/noetix_n2_gym
conda activate n2stairs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python humanoid/scripts/play.py \
  --task=n2_stairs \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<run目录> \
  --checkpoint=-1 \
  --num_envs=1 \
  --terrain_level=4 \
  --command_speed=0.25 \
  --seed=42
```

### 7.3 批量指标

5 个高度、128 环境、每个环境 3 回合的 headless 统计：

```bash
python humanoid/scripts/eval_stairs.py \
  --task=n2_stairs \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<run目录> \
  --checkpoint=-1 \
  --num_envs=128 \
  --headless \
  --terrain_levels=0,1,2,3,4 \
  --command_speed=0.25 \
  --episodes_per_env=3 \
  --output=reports/n2_stairs_eval.csv \
  --seed=42
```

程序同时保存 CSV 和 JSON，包含三层完成率、严格成功率、平均前进距离、平均爬升、平均存活时间、
跌倒率、到顶率、停滞率、第一阶踏上率和最大稳定落脚高度；JSON 还记录解析后的
checkpoint 绝对路径。“到顶”只要求位置和真实地形高度达标；“物理完成”还要求
直立和稳定足部支撑；“成功”还必须通过全部路线、速度和自然交替步态门槛。也可用
`--step_heights=0.02,0.06,0.10` 选择高度。

如果旧 checkpoint 已经形成双脚同步蹦跳或斜向漂移，只把它当作几何到顶对照。
单次落脚事件和双脚腾空惩罚仍不足以保证自然交替步态；正式训练应改用独立的
`n2_stairs_walk` 从零开始，使相位、速度和路径状态都进入 Actor 观测与成功判定。

导出 JIT/ONNX 并进行 MuJoCo 楼梯检查：

```bash
python humanoid/scripts/play.py \
  --task=n2_stairs_walk \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs_walk/<run目录> \
  --checkpoint=-1 \
  --terrain_level=2 \
  --command_speed=0.25 \
  --export_policy

python sim2sim/sim2sim.py --config_file=n2_stairs_walk.yaml
```

也可以跳过 Isaac Gym 环境和 JIT 导出，直接读取原始 checkpoint。评估器会只重建
Actor，并自动选择早期 375 维或当前 410 维观测：

```bash
python sim2sim/eval_stairs_mujoco.py \
  --checkpoint_path=/绝对路径/model_9000.pt \
  --physics_preset=isaac_aligned \
  --step_height=0.10 \
  --episodes=4 \
  --output=/root/autodl-tmp/n2_eval/raw_checkpoint.csv
```

若训练目录中保留了本项目的阶段 checkpoint，下面的一条命令会在完全相同的 MuJoCo
条件下比较 `5000/8000/8600/9000` 四个代表版本，并写出排序。缺失的版本会明确跳过：

```bash
python sim2sim/compare_isaac_checkpoints_mujoco.py
```

## 8. 低台阶预训练、迁移和 robust 微调

默认课程已经是首选方案：所有环境从 2 cm 开始，以每个环境的真实到顶结果逐级提升，
无需人工切 checkpoint。

如果自动课程始终无法离开 level 0，可先固定低台阶。以下 `max_iterations` 均为总目标：

```bash
# 阶段 A：只训练 2 cm
python humanoid/scripts/train.py --task=n2_stairs --headless \
  --sim_device=cuda:0 --rl_device=cuda:0 --num_envs=2048 \
  --fixed_terrain_level=0 --max_iterations=2000 --run_name=fixed_02m

# 阶段 B：加载阶段 A，固定 4 cm，继续到总计 3500 iteration
python humanoid/scripts/train.py --task=n2_stairs --headless \
  --sim_device=cuda:0 --rl_device=cuda:0 --num_envs=2048 \
  --fixed_terrain_level=1 --max_iterations=3500 --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<阶段A目录> \
  --checkpoint=-1 --run_name=fixed_04m
```

后续可按 level 2、3、4 重复，但每级都应先用 `eval_stairs.py` 检查较低高度是否退化。
网络观测和动作维度不变，因此 checkpoint 可直接迁移。

基础策略稳定后再做 robust：

```bash
python humanoid/scripts/train.py \
  --task=n2_stairs_robust \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=2048 \
  --max_iterations=10000 \
  --resume \
  --load_run=/root/autodl-tmp/noetix_n2_gym/logs/n2_stairs/<baseline目录> \
  --checkpoint=-1 \
  --run_name=robust_from_baseline
```

## 9. 结果不理想时的排查顺序

1. 先看 `stairs_completion_rate`、`stairs_curriculum_pass_rate`、严格
   `stairs_success_rate`、`terrain_level` 和不同高度的独立评估，不要只看总奖励。
   严格步态还必须同时检查实际速度/命令误差、相位接触匹配、双脚腾空占比、最大横偏、
   最大偏航和路径失败率；成功率高但这些指标差，仍属于策略取巧。
2. 再看 `stairs_forward_distance`、`stairs_climb_height`、fall/stall：
   - 距离低且 stall 高：检查是否走到第一立面、命令跟踪和前视高度输入；
   - 距离高但爬升低：可能绕行/侧滑，优先看 lateral drift、stumble 和碰撞；
   - 爬升高但 fall 高：优先看 orientation、base height、稳定落脚和接触力。
3. 看逐项奖励。若 clearance/air-time 上升但前进不升，先降低这两项，不能继续放大。
4. 看 base Z 速度、double-flight、feet contact force；跳跃明显时提高相应惩罚，保持
   `stairs_vertical_progress` 只使用地形高度变化。
5. 再检查 PD/力矩。先确认饱和比例和 ankle 接触振荡，再小幅调整踝阻尼/刚度；不要
   一开始扩大 `action_scale` 或取消 URDF 力矩限制。
6. 最后看 PPO 的 value loss、surrogate、entropy、learning rate。value loss 爆炸时先
   检查奖励尖峰/终止标记，再调学习率或 value loss；entropy 很快归零才考虑提高熵。
7. 同时看 `Perf/collection_fps`、`Perf/total_fps` 和 `nvidia-smi`。吞吐突然下降可能是接触对不足、CPU 属性
   更新、显存换页或另一个进程占用，而不是策略问题。

## 10. 上传范围

如果打包上传，上传整个仓库但排除 `.git`、`logs/`、录屏和本地缓存。若通过 Git，
应提交下面两个命令合并显示的所有源码、配置、测试、依赖和文档；不要提交 checkpoint：

```bash
git diff --name-only
git ls-files --others --exclude-standard
```

建议用户审查后执行：

```bash
git status --short
git diff --check
git diff --stat
git add .gitignore humanoid sim2sim tests docs requirements-autodl.txt setup.py README.md
git commit -m "Add dedicated N2 stair-climbing curriculum and evaluation"
# 仅在确认远端和权限后由用户执行：
git push -u origin codex/n2-stairs
```
