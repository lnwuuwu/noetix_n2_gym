# 替代 PPO 的强化学习算法分析

> 针对 N2 18-DOF 人形机器人楼梯攀爬任务

## 任务特征约束

在选择算法之前，先明确你的任务对算法的关键约束：

| 特征 | 要求 | 对算法的影响 |
|------|------|-------------|
| Isaac Gym GPU 并行 | 4096 环境同步采集 | 需要 on-policy 或能高效利用大 batch 的算法 |
| 18-DOF 连续动作 | 高维动作空间 | 排除离散动作算法 |
| 36 个多目标奖励 | 复杂 shaped reward | 需要稳定的多目标优化 |
| Sim2Real 部署 | 需要鲁棒策略 | 需要支持域随机化 |
| 自然步态约束 | 步态时钟 + 对称性 | 需要能整合结构化先验的框架 |
| 已有 PPO baseline | 9050 iter checkpoint | 最好能复用现有训练成果 |

---

## 一、直接替代 PPO 的 On-Policy 算法

### 1.1 DAPO (Decoupled Actor-PPO) / Dual-Clip PPO

**核心思路**: PPO 的改进变体，解耦 Actor 和 Critic 的更新频率。

```
标准 PPO:  Actor 和 Critic 共享更新节奏 (5 epochs × 4 batches)
DAPO:     Critic 更新更频繁 (10 epochs), Actor 保守更新 (3 epochs)
```

**适合你的原因**:
- 你的 Critic 有 146 维特权观测（比 Actor 的 75 维多很多），需要更多更新来拟合
- 实现改动极小：只需分离 Actor/Critic 的 epoch 数
- 完全兼容现有 Isaac Gym 流水线

**实现方式**:
```python
# 在 PPO update 中分离
for epoch in range(critic_epochs):  # 10
    update_critic(mini_batches)

for epoch in range(actor_epochs):   # 3
    update_actor(mini_batches)
```

**评估**: ⭐⭐⭐⭐⭐ 改动最小，收益确定


### 1.2 APPO (Asynchronous PPO) / Sample Factory

**核心思路**: 将采样和训练解耦为异步流水线，大幅提高 GPU 利用率。

**但**: Isaac Gym 已经是同步 GPU 并行模拟，瓶颈不在采样吞吐量而是策略质量。APPO 的主要优势（采样效率）在你的场景中收益有限。

**评估**: ⭐⭐ 不匹配你的瓶颈


### 1.3 TPPO (Trust-region PPO with Natural Gradient)

**核心思路**: 用 Fisher 信息矩阵的自然梯度替代 PPO 的 clip 机制，在参数空间而非目标空间限制更新步长。

```
PPO:   clip(ratio, 1-ε, 1+ε) × advantage
TPPO:  KL(π_old || π_new) ≤ δ, 用共轭梯度求解
```

**适合你的原因**:
- 精调阶段最需要：精确控制策略更新幅度
- 比 PPO 的 clip 更精确地限制策略偏移，与你的 reference loss 锚定策略互补

**缺点**: 共轭梯度求解增加计算开销约 30-50%

**评估**: ⭐⭐⭐ 精调阶段有价值，但增加复杂度

---

## 二、当前最热门的替代方案

### 2.1 SHAC (Short-Horizon Actor-Critic) — 可微分模拟专用

**核心思路**: 利用 Isaac Gym 的可微分物理引擎，通过物理模拟的梯度直接优化策略，不需要 RL 的间接信号。

```
PPO:   策略 → 动作 → 模拟(黑盒) → 奖励 → 策略梯度估计
SHAC:  策略 → 动作 → 模拟(可微分) → 奖励 → 解析梯度反传
```

**原理**:
```python
# SHAC 的核心循环
for t in range(short_horizon):  # H=16~32 步
    action = actor(obs)
    obs_next, reward = differentiable_sim.step(action)  # 梯度可穿透
    total_reward += gamma**t * reward

# 直接对 total_reward 求 actor 参数梯度
loss = -total_reward
loss.backward()  # 梯度穿过整个模拟器
actor_optimizer.step()
```

**显著优势**:
- 梯度信号比 PPO 的策略梯度估计方差低 **数个数量级**
- 在 Isaac Gym 人形控制任务上，收敛速度比 PPO 快 5-10×
- 论文 (Xu et al., 2022) 已在人形步行/奔跑上验证

**关键限制**:
- 需要 Isaac Gym 的可微分模式 (`torch` 后端), 你的代码用的是 `gymtorch`
- 不能处理不可微分的奖励（如你的离散事件奖励 `alternating_tread`）
- 短视野 (16-32步) 难以捕获你的长期课程规划

**对你的适用性**:
- 可以用于**动作平滑性、姿态控制、轨迹跟踪**等可微分子目标
- 离散步态事件和课程逻辑仍需 PPO 处理
- 混合方案: SHAC 处理运动学 + PPO 处理任务逻辑

**评估**: ⭐⭐⭐⭐ 潜力很大，但需要显著的代码重构


### 2.2 DreamerV3 — 基于世界模型的算法

**核心思路**: 先学一个环境的"世界模型"（预测下一步状态和奖励），然后在模型内部做"想象训练"。

```
阶段 1: 从真实交互学习世界模型
  obs, action → 世界模型 → predicted_obs', predicted_reward

阶段 2: 在世界模型内部训练策略 (无需真实模拟)
  for imagination_step in range(horizon):
      action = actor(latent_state)
      latent_state, reward = world_model.predict(latent_state, action)
  update actor to maximize imagined returns
```

**优势**:
- 样本效率极高（真实交互数据的利用率远超 PPO）
- 可以在"想象"中探索危险状态（如跌倒后恢复），而不需要真实模拟
- DreamerV3 的 symlog 归一化天然适合你的多尺度奖励（从 -10 到 +25）

**关键限制**:
- 你已经有 4096 环境 GPU 并行模拟，**采样不是瓶颈**
- 世界模型的预测误差在 18-DOF 接触丰富的楼梯环境中可能很大
- DreamerV3 原版不直接支持 Isaac Gym 的大规模并行

**对你的适用性**: 不太适合。你的场景是"模拟器免费"（GPU 并行），world model 的核心优势（节省模拟次数）不成立。

**评估**: ⭐⭐ 技术先进但不匹配你的场景


### 2.3 SAC (Soft Actor-Critic) — Off-Policy 替代

**核心思路**: 最大化"奖励 + 熵"的 off-policy 算法，用 replay buffer 重复利用过去经验。

```python
# SAC 目标
# J(π) = E[Σ γ^t (reward + α × entropy(π(·|s_t)))]
# α 自动调节探索/利用平衡
```

**优势**:
- 天然最大熵框架鼓励多样化步态探索
- Off-policy 可以重复利用过去经验，样本效率更高
- 自动温度调节 (α) 可能比你手动调的 `entropy_coef=0.002` 更好

**关键限制**:
- Off-policy 在 Isaac Gym 大规模并行中**没有优势**（on-policy 数据已经很充足）
- Replay buffer 在 GPU 内存中存储 4096 × 24 × N 步数据消耗巨大
- SAC 在高维连续控制中训练不如 PPO 稳定

**评估**: ⭐⭐⭐ 有一定价值但不是最佳选择

---

## 三、当前人形机器人领域最成功的方案

### 3.1 ⭐⭐ AMP (Adversarial Motion Priors) — 最推荐替代方案

**核心思路**: 用一个判别器（类似 GAN）替代你手工设计的 36 个步态奖励。判别器学习区分"真实人类步态"和"策略生成步态"，然后用判别器分数作为步态风格奖励。

```
你现在的方法:
  total_reward = Σ (手工权重_i × 手工奖励_i)   ← 36个需要调参

AMP 方法:
  total_reward = task_reward + style_weight × discriminator(状态转移)
  其中 discriminator 从人类参考动作中自动学习

  task_reward = forward_progress + completion + termination  ← 仅3个
  style_reward = D(s_t, s_{t+1}) ← 自动从参考数据学来
```

**参考动作数据来源**:
- 人体动捕数据 (CMU MoCap 等) → 转换到 N2 骨架
- 你已有的最佳 checkpoint 的 rollout 数据
- 甚至可以用物理仿真生成的"理想步态"数据

**对你的巨大优势**:

1. **消除 36 个奖励的调参地狱**
   - 你的 `stairs_stride_symmetry`, `stairs_arm_swing`, `stairs_forward_pitch`, `stairs_feet_yaw` 等全都可以被判别器自动学习
   - 只需保留纯任务奖励（前进进度、完成、安全终止）

2. **步态质量自动提升**
   - 判别器可以捕获人类步态中你难以用奖励函数表达的微妙特征
   - 如膝盖弯曲的时序曲线、重心转移的自然节奏

3. **解决你的精调困境**
   - 你的 stability tournament 本质上是在寻找"看起来更自然"的 checkpoint
   - AMP 的判别器直接优化"看起来自然"，不需要事后筛选

**实现方式**:
```python
class AMPDiscriminator(nn.Module):
    """判别状态转移是否来自参考动作"""
    def __init__(self, obs_dim):
        self.net = MLP([obs_dim * 2, 1024, 512, 1])  # (s_t, s_{t+1}) -> score
    
    def forward(self, s_t, s_t1):
        return self.net(torch.cat([s_t, s_t1], dim=-1))

# 训练循环
for iteration in range(max_iter):
    # 1. 正常 PPO 采集
    trajectories = env.rollout(policy)
    
    # 2. 计算 AMP 风格奖励
    with torch.no_grad():
        style_reward = discriminator(obs_t, obs_t1)  # 越像参考越高
    
    # 3. 组合奖励
    total_reward = task_reward + 0.5 * style_reward
    
    # 4. PPO 更新策略
    ppo.update(trajectories, total_reward)
    
    # 5. 更新判别器 (区分 policy 数据 vs 参考数据)
    discriminator.update(
        policy_data=(obs_t, obs_t1),
        reference_data=sample_reference_motions()
    )
```

**已有实现**:
- NVIDIA Isaac Gym 官方示例中已包含 AMP 实现
- `legged_gym` 社区有适配人形的版本
- 论文: Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation" (SIGGRAPH 2021)

**评估**: ⭐⭐⭐⭐⭐ 对你的场景收益最大


### 3.2 ASE (Adversarial Skill Embeddings) — AMP 的进化版

**核心思路**: 在 AMP 基础上加入技能潜变量 (latent skill embedding)，让策略学习一个技能空间，不同潜变量对应不同运动模式。

```
AMP:   π(a|s)           → 一种步态
ASE:   π(a|s, z)        → z 控制步态风格
       z ∈ R^d           → 不同 z = 不同楼梯策略
```

**对你的价值**:
- 可以用不同 z 编码不同楼梯高度的最优策略
- 高层控制器选择 z，底层策略执行
- 论文: Peng et al., "ASE: Large-Scale Reusable Adversarial Skill Embeddings" (SIGGRAPH 2022)

**评估**: ⭐⭐⭐⭐ AMP 的自然延伸，但复杂度更高


### 3.3 H2O / Human-to-Humanoid — 最新范式

**核心思路**: 直接从人类视频/动捕数据中学习人形机器人控制策略，通过在线重定向 (retargeting)。

**代表工作**:
- "Learning Humanoid Locomotion with Transformers" (Radosavovic et al., 2024)
- "H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation" (He et al., 2024)

**评估**: ⭐⭐⭐ 趋势方向，但需要大量人类数据，改造成本高

---

## 四、PPO 框架内的增强技术（不换算法）

如果你不想完全替换 PPO，以下技术可以在现有框架内显著提升效果：

### 4.1 Symmetric Critic (对称 Critic)

```python
# 利用 N2 的左右对称性，将 Critic 的数据量翻倍
V(s) = 0.5 * (V_raw(s) + V_raw(mirror(s)))
```

### 4.2 Asymmetric Actor-Critic with Privileged Learning

你已经在用非对称 AC（Critic 有特权观测），但可以更进一步：

```
阶段 1: 训练 Teacher (Actor 也用特权观测)  → 性能上限更高
阶段 2: 蒸馏 Student (Actor 只用可部署观测) → 部署
```

### 4.3 Multi-Objective PPO (MO-PPO)

```python
# 将 36 个奖励分为几组，为每组维护独立的值函数
# V_task(s)     → 进度/完成奖励
# V_gait(s)     → 步态/相位奖励  
# V_safety(s)   → 碰撞/终止惩罚
# V_smooth(s)   → 动作平滑奖励

# 使用 Pareto 优化或约束优化组合梯度
```

### 4.4 PopArt (Adaptive Reward Normalization)

```python
# 自动归一化不同量级的奖励
# 解决你的 torques=-1e-5 vs stairs_success=12.0 的量级差异
normalized_return = (return - running_mean) / running_std
```

---

## 五、综合推荐

### 推荐方案 1: PPO + AMP（最大收益）

```
改造量: 中等 (需加入判别器)
收益:   极高 (消除36个奖励调参, 自动学习自然步态)
风险:   低 (PPO 部分不变, AMP 是增量添加)
```

```
现有 PPO 流水线
    ├── 保留: 楼梯课程、地形、终止条件、Isaac Gym 并行
    ├── 保留: 前进进度 + 完成 + 终止 (3个任务奖励)  
    ├── 移除: 其余33个手工步态/姿态奖励
    └── 新增: AMP 判别器 (从参考动作自动学习步态风格)
```

### 推荐方案 2: 改进版 PPO（最小改动）

```
改造量: 很小
收益:   中等 (解决当前瓶颈)
风险:   极低
```

```
在现有 PPO 基础上:
    ├── 分离 Actor/Critic 更新频率 (DAPO)
    ├── 加入 PopArt 奖励归一化
    ├── 精调阶段用 Natural Gradient (TPPO)
    └── 加入 Symmetric Critic
```

### 推荐方案 3: SHAC + PPO 混合（技术前沿）

```
改造量: 大
收益:   极高 (可微分梯度 + RL 灵活性)
风险:   中等 (需要可微分模拟器支持)
```

```
SHAC 处理: 关节协调、轨迹跟踪、动作平滑 (可微分目标)
PPO 处理: 课程逻辑、离散事件、探索 (不可微分目标)
```

### 最终建议

**如果只能选一个改进: 选 AMP。**

你现在最大的痛点不是 PPO 本身性能不够，而是 36 个奖励函数的调参和互相冲突。
AMP 从根本上解决这个问题——让判别器自动从参考动作中学习"什么是好步态"。
PPO 继续负责它擅长的部分：课程学习、安全约束、大规模并行训练。

已有的 9050 checkpoint 可以直接作为 AMP 判别器的参考数据源。
