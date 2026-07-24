# N2 楼梯攀爬训练方案 — 实现介绍与已知问题记录

> 最后更新：2026-07-24
>
> 本文档包含两部分：
> 1. **训练方案实现介绍** — 完整描述三阶段训练流水线的实现细节
> 2. **已知问题记录** — 代码审查中发现的 bug、风险和潜在改进点

---

## 目录

- [一、项目文件结构](#一项目文件结构)
- [二、训练方案总体架构](#二训练方案总体架构)
- [三、阶段一：基线楼梯课程训练 (N2StairsCfg)](#三阶段一基线楼梯课程训练-n2stairscfg)
- [四、阶段二：自然步态训练 (N2StairsWalkCfg)](#四阶段二自然步态训练-n2stairswalkcfg)
- [五、阶段三：稳定性精调 (Stability Fine-tuning)](#五阶段三稳定性精调-stability-fine-tuning)
- [六、锦标赛检查点选择](#六锦标赛检查点选择)
- [七、辅助系统](#七辅助系统)
- [八、已知问题记录](#八已知问题记录)
- [九、改进建议摘要](#九改进建议摘要)

---

## 一、项目文件结构

```
noetix_n2_gym/
├── humanoid/
│   ├── envs/n2/
│   │   ├── n2_config.py              # 基础 N2 18-DOF 配置
│   │   ├── n2_env.py                 # 基础环境类 (继承 LeggedRobot)
│   │   ├── n2_stairs_config.py       # 楼梯任务三套配置
│   │   └── n2_stairs_env.py          # 楼梯环境 (3509行, 核心)
│   ├── scripts/
│   │   ├── train.py                  # 训练入口
│   │   ├── eval_stairs.py            # 确定性评估脚本
│   │   ├── isaac_stability_tournament.py  # 锦标赛选择算法
│   │   └── run_isaac_stability_curriculum.sh  # 精调流水线调度
│   └── utils/
│       ├── stairs_terrain.py         # 楼梯地形生成 & 摆动轨迹
│       └── terrain.py                # N2StairsTerrain 类
├── tests/
│   └── test_stairs_static.py         # 2830行 CPU 静态测试
├── resources/robots/N2/urdf/N2.urdf  # 机器人模型
└── sim2sim/                          # MuJoCo sim2sim 部署
```

---

## 二、训练方案总体架构

### 三阶段递进训练

```
阶段 1: N2StairsCfg          阶段 2: N2StairsWalkCfg        阶段 3: Stability Fine-tuning
┌──────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────────┐
│ 目标: 学会爬楼梯      │    │ 目标: 学自然交替步态      │    │ 目标: 消除左右不对称/抖动  │
│ 课程: 2→10cm 5级      │───>│ 相位时钟 + 严格步序检测   │───>│ 冻结底层 + 教师锚定        │
│ 奖励: 进度为主        │    │ 奖励: 36个多目标          │    │ 锦标赛 + holdout 验证      │
│ 训练: 8000 iter       │    │ 训练: 从阶段1 checkpoint  │    │ 训练: 400 iter 微调        │
│ 随机化: 温和          │    │ 随机化: 最小化            │    │ 随机化: 继承阶段2         │
└──────────────────────┘    └──────────────────────────┘    └────────────────────────────┘
```

### 技术栈

| 组件 | 技术选型 |
|------|----------|
| 物理模拟 | Isaac Gym (GPU 并行) |
| RL 算法 | PPO (Proximal Policy Optimization) |
| 网络结构 | Actor-Critic MLP `[512, 256, 128]` + ELU |
| 观测设计 | 帧堆叠 (frame_stack=5) + 地形高度场扫描 |
| 课程学习 | 基于连续成功/失败的自动难度推进 |
| 部署验证 | MuJoCo sim2sim 转换 |

---

## 三、阶段一：基线楼梯课程训练 (N2StairsCfg)

> 配置文件: `n2_stairs_config.py` 第 6–262 行

### 3.1 楼梯地形设计

由 `stairs_terrain.py → build_directional_stairs()` 生成确定性 +X 方向楼梯：

```
           ┌─────┐
       ┌───┘     │ ← 平台顶部 (top platform)
   ┌───┘         │ ← 6级台阶, 每级 0.30m 深
───┘             │ ← 起始平台 (start_platform_length=1.15m)
                 │
  spawn_x=0.55m  │
```

- **5个课程级别**: step_heights = [0.02, 0.04, 0.06, 0.08, 0.10] m
- **台阶参数**: step_width=0.30m, num_steps=6
- **地形网格**: 5行 × 8列, terrain_length=5.0m, terrain_width=3.0m
- **摩擦系数**: static=0.8, dynamic=0.8, restitution=0.0

### 3.2 观测空间

**Actor 观测** (`num_single_obs=75`, 5帧堆叠 → `num_observations=375`):

```
obs_now = [
    commands(3),              # 目标速度 [vx, vy, ωyaw]
    base_ang_vel(3),          # 机体角速度
    projected_gravity(3),     # 重力在机体坐标系的投影
    dof_pos - default(18),    # 关节位置偏差
    dof_vel(18),              # 关节速度
    last_actions(18),         # 上一步动作
    actor_heights(12),        # 前方 4×3 地形高度扫描
]
# → 75维 × 5帧 = 375维
```

**Critic 特权观测** (`num_privileged_obs=146`):

```
privileged = [
    proprio(63),              # 完整本体感受
    base_lin_vel(3),          # 真实线速度 (Actor 看不到)
    payload(1),               # 负载质量
    friction(1),              # 地面摩擦系数
    restitution(1),           # 弹性系数
    kp_factor(18),            # PD 增益随机化系数
    kd_factor(18),            # PD 阻尼随机化系数
    motor_strength(18),       # 电机力矩随机化系数
    foot_contacts(2),         # 左右脚接触
    terrain_heights(21),      # 完整 7×3 地形高度场
]
```

### 3.3 课程推进机制

实现在 `n2_stairs_env.py` 的 `_update_terrain_curriculum()` (第 2345-2389 行):

```python
# 每个环境独立维护成功/失败连胜计数
if curriculum_success_streak >= curriculum_successes(=2):
    terrain_level += 1  # 晋级到更高台阶
    
if curriculum_failure_streak >= curriculum_failures(=2):
    terrain_level -= 1  # 降级到更低台阶

# terrain_level 被夹紧在 [0, 4] 之间
```

成功判定条件（基线模式）：到达楼梯顶部 (`top_reached_buf`)。

### 3.4 终止条件

实现在 `check_termination()` (第 2254-2281 行):

| 终止条件 | 触发逻辑 |
|----------|----------|
| **跌倒** (`fall_event_buf`) | base 碰撞力 > 5N 或 `fallen_buf` |
| **停滞** (`stall_buf`) | 超过 grace 期后 6s 无前进进度 |
| **路径偏离** (`path_failure_buf`) | 横向偏移 > 0.30m 或偏航 > 0.40 rad |
| **完成** (`completion_buf`) | 稳定到达楼梯顶部 |
| **超时** (`time_out_buf`) | episode 达到 30s 上限 |

关键设计：物理跌倒会清除所有成功标志，避免"到顶后摔倒"被误判为成功。完成状态被标记为真正终止（非 timeout），防止 PPO 对完成状态做 value bootstrapping。

### 3.5 PPO 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 5.0e-4 | 自适应 KL 调度 |
| `entropy_coef` | 0.01 | 探索系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `num_learning_epochs` | 5 | 每轮 PPO 更新次数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `num_steps_per_env` | 24 | 每环境每轮采集步数 |
| `num_envs` | 4096 | 并行环境数 |
| `max_iterations` | 8000 | 最大训练迭代 |
| `init_noise_std` | 0.8 | 初始动作噪声 |

### 3.6 域随机化（基线阶段，温和）

| 随机化项 | 范围 | 说明 |
|----------|------|------|
| 动作延迟 | [0.0, 0.15]s | 模拟通信延迟 |
| PD 增益 | ×[0.95, 1.05] | 执行器不确定性 |
| 电机力矩 | ×[0.95, 1.05] | 力矩输出波动 |
| 摩擦系数 | [0.70, 1.10] | 地面材质变化 |
| 弹性系数 | [0.0, 0.05] | 碰撞弹性 |
| 质心偏移 | 关闭 | — |
| 基座附加质量 | 关闭 | — |
| 外力推动 | 关闭 | — |

---

## 四、阶段二：自然步态训练 (N2StairsWalkCfg)

> 配置文件: `n2_stairs_config.py` 第 296–623 行

### 4.1 与阶段一的关键差异

| 维度 | 阶段一 (N2StairsCfg) | 阶段二 (N2StairsWalkCfg) |
|------|---------------------|--------------------------|
| 目标 | 能爬上去就行 | 必须像人类一样交替步态 |
| 观测 | 75维/帧 | 82维/帧 (加入相位+速度+导航) |
| 步态时钟 | 无 | 有 (可部署的开环相位时钟) |
| 步序检测 | 简单高度进度 | 严格踏板交替分类 |
| 域随机化 | 温和 | 最小化 (先学步态再加噪声) |
| 奖励数量 | ~20个 | 36个 |
| 速度控制 | 追踪命令 | 严格限速 + 超速惩罚 |

### 4.2 步态相位时钟

实现在 `_get_gait_phase()` / `_scheduled_gait_frequency()` (第 784-794 行):

```python
# 可部署相位时钟：仅依赖 elapsed_time 和 command_speed
phase = (gait_phase_offset + elapsed_time * frequency) % 1.0

# 频率匹配踏板宽度
# 0.30m 踏板 → 半周期 = 0.30m / command_speed
# gait_frequency = 0.20 Hz + 1.667 * max(cmd_x - 0.12, 0)
```

相位映射到左/右支撑期：
- `phase ∈ [0.0, 0.5)` → 左脚支撑期 (右脚摆动)
- `phase ∈ [0.5, 1.0)` → 右脚支撑期 (左脚摆动)
- `double_support_ratio=0.28` → 每次切换有28%的双支撑缓冲

### 4.3 严格踏板交替检测

实现在 `_update_foot_step_progress()` (第 1016-1300+ 行):

```
稳定接触判定 (3步确认):
  1. 原始接触 AND 垂直力 >= 1.0 × 水平力
  2. 水平脚速 < 0.18 m/s
  3. 确认持续时间 >= 0.04s (confirmation_s)
  4. 释放滞后 <= 0.06s (release_s)

踏板量化:
  tread_index = round(foot_surface_height / step_height)

步序分类 (classify_tread_transition):
  ┌─ alternating_advance: 对侧脚前进到 last_tread+1 ✅
  ├─ repeated_lead:       同侧脚连续前进两次 ❌
  ├─ same_tread_join:     后脚加入前脚的踏板 (step-to) ❌
  └─ skipped_tread:       跳过一级以上台阶 ❌
```

### 4.4 C2 平滑摆动轨迹

实现在 `stairs_terrain.py → smooth_swing_trajectory()`:

三阶段摆动参考轨迹，使用五阶 smootherstep 确保 C2 连续性（位置、速度、加速度端点均为零）：

```
阶段 1 (0% → 35%):  抬腿 — Z 上升到弧顶
阶段 2 (35% → 72%): 越过立面 — XY 前进，Z 保持高位
阶段 3 (72% → 100%): 下放 — Z 下降到目标踏板

弧高 = 0.04 + 0.50 × step_height
(10cm 台阶 → 9cm 弧高，留约 4cm 脚底间隙)
```

C2 连续性的意义：PD 控制器跟踪不连续的参考轨迹会产生力矩脉冲和接触抖动。

### 4.5 左/右对称镜像

实现在 `_build_mirror_layout()` (第 67-154 行):

```python
# Actor 左右镜像映射
# 关节交换: L_leg_hip_yaw ↔ R_leg_hip_yaw (符号取反)
# 观测交换: 对应的关节位置/速度/动作通道
# 高度场: Y 轴对称点交换

# 用于:
# 1. symmetry_loss: ||π(obs) - mirror(π(mirror(obs)))||²
# 2. actor_reference_mirror_blend: 参考教师的镜像平均
```

### 4.6 奖励系统（36个函数）

#### 进度与攀爬奖励

| 函数 | 权重 | 机制 |
|------|------|------|
| `tracking_lin_vel` | +3.5 | `exp(-σ·‖v_cmd - v_actual‖²)`, 门控:前进进度+直立+朝向对齐 |
| `stairs_forward_progress` | +0.50 | 前进速度 × 直立 × 单脚支撑 × `exp(-4|v_y|)` × `exp(-6·yaw²)` |
| `stairs_foot_step_progress` | +2.0 | 一次性事件：脚落在新的更高踏板上, `/dt` 抵消框架缩放 |
| `stairs_command_speed_error` | -16.0 | `‖cmd_x - actual_x‖²`, 限速 |
| `stairs_overspeed` | -30.0 | 超过最大允许速度的惩罚 |

#### 步态交替奖励

| 函数 | 权重 | 机制 |
|------|------|------|
| `stairs_alternating_tread` | +10.0 | 一次性事件：对侧脚交替前进到下一级踏板 |
| `stairs_repeated_lead` | -8.0 | 一次性惩罚：同一只脚连续前进两次 |
| `stairs_same_tread_join` | -10.0 | 一次性惩罚：后脚加入前脚的踏板 (step-to 步态) |
| `stairs_skipped_tread` | -5.0 | 一次性惩罚：跳过台阶 |
| `stairs_phase_contact` | +2.0 | 实际接触匹配相位时钟调度 |
| `stairs_phase_contact_mismatch` | -3.0 | 实际接触与相位时钟不匹配 |
| `stairs_double_flight` | -8.0 | 双脚同时离地时间超限 |
| `stairs_single_support` | +0.40 | 单脚支撑 + 前进 + 直立 |

#### 摆动轨迹与膝盖引导

| 函数 | 权重 | 机制 |
|------|------|------|
| `stairs_swing_trajectory` | +5.0 | `exp(-σ·‖actual_xyz - ref_xyz‖²)`, 碰撞时归零 |
| `stairs_swing_trajectory_error` | -6.0 | 归一化3D位置误差的二次惩罚, 有界裁剪 |
| `stairs_swing_knee_flexion` | +2.0 | `exp(-σ·(knee - target)²)`, 摆动中期弯曲+落地前伸展 |
| `stairs_swing_knee_deficit` | -6.0 | 膝盖欠弯曲的非对称二次惩罚 (提供远距离梯度) |
| `stairs_swing_timeout` | -3.0 | 摆动腿超时未落地 |

#### 姿态与对齐

| 函数 | 权重 | 机制 |
|------|------|------|
| `stairs_lateral_drift` | -12.0 | 横向偏移² + 横向速度² + 偏航率² + 偏航误差² |
| `stairs_heading_alignment` | +3.0 | `exp(-σ·(yaw² + lateral²))` |
| `stairs_forward_pitch` | +1.25 | 鼓励身体微前倾 `target = 0.04 + 0.35·step_height` |
| `stairs_arm_swing` | +0.50 | 对侧手臂摆动与步态相位同步 |
| `stairs_stride_symmetry` | 0.0→-10.0 | 左右摆动长度差异 (精调阶段才启用) |
| `stairs_overstride` | -8.0 | 前后脚距离超过 max_sagittal_foot_separation |
| `stairs_feet_yaw` | -2.5 | 脚偏航偏离机体朝向 |

#### 碰撞与安全

| 函数 | 权重 | 机制 |
|------|------|------|
| `stairs_lower_leg_collision` | -5.0 | 小腿接触力超阈值 (12N), 归一化裁剪 |
| `stairs_foot_riser_collision` | -4.0 | 脚对立面的水平力超过垂直力比例 |
| `stairs_base_behind_support` | -4.0 | 骨盆落后于支撑脚 X 位置 |
| `stairs_foot_pitch` | -2.0 | 脚底非水平着地 |
| `termination` | -10.0 | 跌倒终止的一次性惩罚 |
| `stairs_success/completion/curriculum` | +12/+8/+4 | 分层完成奖励 |

#### 动作平滑性

| 函数 | 权重 | 机制 |
|------|------|------|
| `action_rate` | -0.10 | `‖a_t - a_{t-1}‖²` |
| `action_smoothness` | -0.05 | `‖a_t - 2·a_{t-1} + a_{t-2}‖²` (二阶差分) |
| `dof_acc` | -2.5e-7 | 关节加速度 |

### 4.7 成功判定（严格步态模式）

`completion_buf` 需要同时满足:

| 条件 | 阈值 |
|------|------|
| 到达楼梯顶部高度 | root_x ≥ top_x, height ≥ top_height - tolerance |
| 稳定停留时间 | ≥ 0.60s (completion_dwell_s) |
| 直立姿态 | -gravity_z > 0.85 |
| 脚部稳定 | foot_speed < 0.20 m/s |
| 横向偏移 | < 0.20m |
| 偏航偏差 | < 0.30 rad |
| 步态相位匹配 | ≥ 0.70 |
| 双脚离地比例 | < 0.08 |
| 交替踏板计数 | ≥ 4 |
| 交替率 | ≥ 0.75 |
| 前后脚分离 | < 0.40m |

---

## 五、阶段三：稳定性精调 (Stability Fine-tuning)

> 脚本文件: `run_isaac_stability_curriculum.sh`

### 5.1 精调策略

从阶段二的已验证 checkpoint (如 `model_9050.pt`) 出发，仅微调 Actor 网络后2层：

```
Actor MLP:
  Linear(375, 512)  ← 冻结
  Linear(512, 256)  ← 可训练 (layer 2)
  Linear(256, 128)  ← 可训练 (layer 1)
  Linear(128, 18)   ← 可训练 (output)
```

### 5.2 精调超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LEARNING_RATE` | 1.0e-6 | 极低学习率保护攀爬技能 |
| `ACTION_NOISE` | 0.05 | 动作噪声标准差 |
| `REFERENCE_COEFF` | 0.20 | 教师锚定损失系数 |
| `SYMMETRY_COEFF` | 0.006 | 左/右对称损失系数 |
| `ACTOR_LAYERS` | 2 | 可训练层数 |
| `TRAIN_ITERATIONS` | 400 | 训练迭代次数 |
| `CHECKPOINT_INTERVAL` | 20 | 检查点保存间隔 |
| `TERRAIN_MIX` | 2,3,4,4,4,4,4,4 | 62.5% 环境在10cm台阶 |
| `TRAIN_ENVS` | 256 | 并行环境数 (比训练少) |

### 5.3 精调过程中的额外损失

```python
total_loss = policy_loss_scale × PPO_surrogate_loss
            + reference_coeff × ||π(obs) - π_teacher(obs)||²
            + symmetry_coeff × ||π(obs) - mirror(π(mirror(obs)))||²
```

- **Reference Loss**: 锚定到加载时的 checkpoint，防止策略偏移过远
- **Symmetry Loss**: 强制策略对左右镜像输入产生镜像输出
- **Mirror Blend**: 可选将教师自身做镜像平均，消除教师的左右偏差

### 5.4 奖励覆盖

精调阶段通过 `--reward_scale_overrides` 动态修改部分奖励权重：

```bash
REWARD_OVERRIDES="
  action_rate=-0.16,           # 加强动作平滑 (原 -0.10)
  action_smoothness=-0.12,     # 加强二阶平滑 (原 -0.05)
  stairs_lateral_drift=-18,    # 加强横向约束 (原 -12)
  stairs_stride_symmetry=-10,  # 启用步幅对称 (原 0)
  stairs_foot_crossover=-10,   # 启用脚交叉惩罚 (原 0)
  stairs_foot_lane_error=-6,   # 启用脚车道误差 (原 0)
  stairs_single_support_stability=-5,  # 启用单支撑稳定性
  stairs_right_support_stability=-4,   # 右脚支撑期特别约束
"
```

### 5.5 运行模式

```bash
# 快速验证 (2次迭代)
bash run_isaac_stability_curriculum.sh smoke

# 短期试探 (80次迭代)
bash run_isaac_stability_curriculum.sh pilot

# 完整训练 (400次迭代)
bash run_isaac_stability_curriculum.sh long

# 诊断: 零训练镜像策略测试
bash run_isaac_stability_curriculum.sh diagnose

# 矫正: 预检门控的全Actor蒸馏
bash run_isaac_stability_curriculum.sh correct
```

---

## 六、锦标赛检查点选择

> 文件: `isaac_stability_tournament.py`

### 6.1 选择流程

```
1. 评估 baseline (加载时的原始 checkpoint)
2. 训练 400 次迭代, 每 20 次保存 checkpoint
3. 对每个 checkpoint 做确定性评估 (128 envs × 5 levels)
4. 锦标赛比较: 所有候选 vs baseline
5. 选出最佳候选
6. 独立 holdout 验证 (256 envs, 不同种子)
7. holdout 也通过 → 接受; 否则 → 保留 baseline
```

### 6.2 评估指标（每级别加权）

```python
LEVEL_WEIGHTS = {
    0: 0.125,  # 2cm 台阶
    1: 0.125,  # 4cm
    2: 0.125,  # 6cm
    3: 0.125,  # 8cm
    4: 0.500,  # 10cm ← 重点权重
}
```

### 6.3 安全门控（Hard Gates）

必须全部通过，否则直接淘汰：

| 门控 | 条件 | 容差 |
|------|------|------|
| 10cm 完成率 | candidate ≥ baseline - tolerance | 3.0% |
| 10cm 跌倒率 | candidate ≤ baseline + tolerance | 2.5% |
| 10cm 路径失败 | candidate ≤ baseline + tolerance | 2.5% |
| 加权完成率 | candidate ≥ baseline - tolerance | 3.0% |
| 加权跌倒率 | candidate ≤ baseline + tolerance | 2.5% |
| 各级别完成/跌倒 | 每级单独检查 | 10% |

### 6.4 风格评分（Soft Gates + 综合得分）

```python
style_cost = (
    action_motion              # 动作抖动
    + 4.0 × stride_imbalance   # 左右步长差
    + 2.0 × signed_lateral     # 横向偏移
    + max_lateral              # 最大横向偏移
    + 0.25 × yaw               # 偏航偏差
    + 0.5 × double_flight      # 双脚离地
    + 8.0 × foot_inward        # 脚内收
    + 8.0 × foot_lane_center   # 脚车道中心偏移
    + 2.0 × foot_lane_width    # 脚车道宽度偏差
    + 0.5 × phase_imbalance    # 左/右摆动相位动作不平衡
    + 0.5 × left_swing_body    # 右脚单支撑期身体晃动
    + 0.5 × phase_body_imb     # 身体晃动不平衡
    + actor_symmetry_error     # Actor 镜像误差
)

selection_score = style_gain + 0.20 × performance_delta
```

### 6.5 选择层级

1. **完全合格**: 通过所有 hard + soft 门控 → 选最高 `selection_score`
2. **安全回退**: 通过 hard 门控 + 最低 style/score 阈值 → 选最高得分
3. **保留 baseline**: 没有候选通过 → `improved=False`

---

## 七、辅助系统

### 7.1 地形高度场扫描

Actor 前方 4×3 = 12 个采样点:

```
actor_measured_points_x = [0.25, 0.45, 0.65, 0.85]  (前方)
actor_measured_points_y = [-0.24, 0.0, 0.24]          (左中右)
```

Critic 完整 7×3 = 21 个采样点（含后方）:

```
measured_points_x = [-0.20, 0.0, 0.25, 0.45, 0.65, 0.85, 1.05]
measured_points_y = [-0.24, 0.0, 0.24]
```

### 7.2 脚底-踝关节偏移动态校准

```python
# 在平地稳定着地时，用指数滑动平均估算踝关节到脚底的真实高度差
foot_surface_offset = lerp(foot_surface_offset, measured_offset, rate=0.10)
# 初始值: nominal_foot_surface_offset = 0.045m
# 防止立面边缘接触的测量污染落地终点
```

### 7.3 检查点状态持久化

`get_checkpoint_state()` 保存:
- 各环境的课程级别 (`terrain_levels`)
- 成功/失败连胜计数
- 步态频率过渡步数

`load_checkpoint_state()` 恢复:
- 重新加载所有课程状态
- 将过渡状态广播到所有环境

### 7.4 静态测试覆盖

`test_stairs_static.py` (2830行) 在无 GPU/Isaac Gym 的情况下验证:

- 楼梯几何正确性 (5级高度 × 6级台阶)
- 高度场索引展平/子集选择
- 踏板分类器逻辑 (交替 vs step-to 步态)
- 摆动轨迹端点连续性/间隙
- 配置观测维度一致性
- 所有非零奖励权重均有对应 `_reward_*` 函数
- MuJoCo sim2sim 一致性 (URDF/MJCF 惯量对齐)
- 源代码兼容性 (无 isaacgym 导入)

---

## 八、已知问题记录

### 🔴 P0 — 代码缺陷

#### Issue #1: `_reward_stairs_swing_clearance` 摆动掩码不一致

**位置**: `n2_stairs_env.py` 第 3449-3452 行

**问题**: `_swing_knee_state()` 会检查 `enforce_walk_gait` 条件并使用 `_physical_airborne_mask()`，但 `_reward_stairs_swing_clearance` 只检查 `include_gait_phase`，不检查 `enforce_walk_gait`。

**当前代码**:
```python
# _reward_stairs_swing_clearance (第 3449 行)
if self.include_gait_phase:
    swing = ~self.desired_contacts & ~self.contacts
else:
    swing = ~self.contacts
```

**应该是**:
```python
if self.enforce_walk_gait:
    swing = self._physical_airborne_mask()
elif self.include_gait_phase:
    swing = ~self.desired_contacts & ~self.contacts
else:
    swing = ~self.contacts
```

**影响**: 当 `enforce_walk_gait=True` 时，swing_clearance 使用不同于 swing_knee 的摆动检测逻辑，可能导致奖励信号在摆动/支撑边界不一致。

---

#### Issue #2: 观测中地形高度偏移硬编码为 -0.5

**位置**: `n2_stairs_env.py` 第 883 行

**问题**: 高度测量使用 `root_states[:, 2] - 0.5 - measured_heights`，但 N2 的名义骨盆高度为 `base_height_target = 0.698m`。

**当前代码**:
```python
all_heights = torch.clip(
    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
    -1.0, 1.0,
) * self.obs_scales.height_measurements
```

**影响**: Actor 看到的地形相对高度有约 0.2m 的系统性偏差。策略可能已经适应了这个偏差（因为训练和部署都用同样的偏移），但如果修改了 `base_height_target` 或者换了不同身高的机器人，这个硬编码就会出问题。

**修复方向**: 将 `-0.5` 替换为 `-self.cfg.rewards.base_height_target` 或从配置中读取。注意：修改后现有 checkpoint 的观测分布会改变，需要重新训练。

---

#### Issue #3: 摆动轨迹正奖励的不连续悬崖

**位置**: `n2_stairs_env.py` 第 1725-1729 行

**问题**: 碰撞检测使用硬切断 `score *= collision_free.float()`，从满分瞬间跳变到零。

**当前代码**:
```python
collision_free = (
    (lower_leg_collision <= 0.0) & (foot_riser_collision <= 0.0)
)
score *= collision_free.float()  # 硬 0/1 门控
```

**影响**: 在碰撞边界附近策略梯度不稳定。`swing_trajectory_error` 的二次惩罚仍然提供平滑梯度，所以这并非致命问题，但可能减慢碰撞边界附近的学习收敛速度。

**修复方向**: 使用 sigmoid 软门控: `gate = sigmoid(-k * max(collision_severity, 0))`

---

#### Issue #4: 离散事件奖励的潜在双触发

**位置**: `_reward_stairs_alternating_tread` 等事件奖励

**问题**: 一次性事件奖励使用 `/dt` 来抵消框架的 `reward * dt` 缩放。如果接触传感器噪声导致同一个踏板跳变在连续2步中被触发，会产生 2× 奖励。

**现有保护**:
- `alternating_tread_event` 每步清零 (第 1102 行)
- 稳定接触检测有确认窗口 (0.04s)

**残余风险**: 极端噪声下仍可能出现罕见的双触发。在数千环境并行训练中，少量双触发的统计影响很小。

---

### 🟡 P1 — 设计风险

#### Issue #5: 精调学习率可能过低

**位置**: `run_isaac_stability_curriculum.sh` 第 28 行

**现状**: `LEARNING_RATE=1.0e-6`, 仅解冻2层, 400次迭代。

**影响估算**:
- 总 transitions ≈ 400 × 256 × 24 = 2.46M
- 参数更新次数 ≈ 400 × 5 epochs × 4 batches = 8000
- 权重变化量级 ≈ 1e-6 × O(1) × 8000 ≈ 0.008
- 策略几乎不会发生实质性变化

**建议**: 提升到 3e-5 ~ 1e-4, 同时增大 `REFERENCE_COEFF` 到 0.50 补偿。

---

#### Issue #6: 36个奖励函数可能互相稀释

**现状**: 部分奖励权重极小 (如 `torques=-1e-5`, `dof_acc=-2.5e-7`)，在梯度中几乎可忽略。但它们增加了调参维度和潜在的奖励冲突。

**建议**: 审计每个奖励对总 return 的实际贡献比例。权重贡献 < 总 return 的 0.1% 的奖励可以安全移除。

---

#### Issue #7: Critic 特权观测不含课程级别

**现状**: Critic 接收完整地形高度场 (21 点) + 动力学参数，但不直接知道当前课程级别编号。

**影响**: Critic 必须从原始高度场隐式推断楼梯难度，增加了值函数学习的负担。

**建议**: 加入 `terrain_level / max_level` 作为标量观测或 one-hot 编码。

---

#### Issue #8: 评估样本量偏少

**现状**: `EVAL_ENVS=128`, `episodes_per_env=1` → 每级别仅 128 个 episode。

**统计波动**: 完成率标准误差 ≈ `sqrt(p(1-p)/n)` ≈ `sqrt(0.9×0.1/128)` ≈ 2.6%。但锦标赛容差为 2.5-3.0%，两者量级相当。

**影响**: 选择决策可能因统计噪声而不稳定。

**建议**: 增加到 `EVAL_ENVS=256` 或 `episodes_per_env=3`。

---

### 🟢 P2 — 潜在改进

#### Issue #9: frame_stack=5 的时序窗口可能偏短

5帧 × 50Hz = 0.1秒。对于需要规划多级楼梯的任务，这个窗口仅覆盖一个步态周期的很小一部分。基线 N2 配置使用 frame_stack=10。

---

#### Issue #10: gamma=0.99 对长 episode 的规划视野

30秒 episode × 50Hz = 1500步。gamma=0.99 的有效视野 = 100步 ≈ 2秒，可能不足以让策略学习完整的6级楼梯规划。

---

#### Issue #11: 缺少训练中途的 early stopping

精调固定运行 400 次迭代，没有基于验证性能的提前终止。如果策略在第 100 次迭代就达到最优，后 300 次可能导致过拟合或回退。

---

## 九、改进建议摘要

| 编号 | 改进 | 预期收益 | 实施成本 |
|------|------|----------|----------|
| A | 精调学习率 1e-6 → 5e-5 + reference_coeff 0.20 → 0.50 | 🟢高 | 改环境变量 |
| B | 修复 swing_clearance 掩码 (Issue #1) | 🟢高 | 改几行代码 |
| C | 地形高度偏移参数化 (Issue #2) | 🟡中 | 改1行 + 需重训 |
| D | 增加评估样本量 (Issue #8) | 🟡中 | 改环境变量 |
| E | 更细粒度楼梯课程 (5级→8级) | 🟡中 | 改配置 |
| F | 软门控替代硬切断 (Issue #3) | 🟢中 | 改几行代码 |
| G | Critic 加入课程级别 (Issue #7) | 🟡中 | 改 env 代码 |
| H | 增加 IMU 偏置漂移 + 观测延迟 | 🟡中 | 改 env 代码 |
| I | frame_stack 5→8 或 LSTM 替代 | 🔴高 | 需重训/重构 |
| J | gamma 0.99 → 0.995 (精调阶段) | 🟡中 | 改配置 |
| K | 分层奖励调度 (逐阶段启用) | 🟡中 | 改 env 代码 |
| L | 训练中早停 (rolling evaluation) | 🟡中 | 改 shell 脚本 |
