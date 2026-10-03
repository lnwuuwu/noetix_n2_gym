# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import numpy as np

from isaacgym import terrain_utils
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

def add_roughness(terrain, noise_magnitude=0.02):
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-noise_magnitude,
        max_height=noise_magnitude,
        step=0.005,
        downsampled_scale=0.075,
    )

def directional_stairs(terrain, step_width, step_height, platform_size=1.5):
    """-x 端底部平台 + 沿 +x 单向逐级升/降、贯穿整块地形的定向长楼梯（沿 y 恒定）。

    机器人在 -x 端的底部平台出生（见 N2PerceptiveEnv._reset_root_states 把楼梯格的
    出生点挪到这里），正对一整块地形长度的楼梯往 +x 爬(up)或走下(down)。

    【为什么必须有这块平台】isaacgym 的 terrain_utils.stairs_terrain 是整个地形族里
    唯一不接受 platform_size 的函数——legged_gym 的每个子地形都传 platform_size=3./4.，
    IsaacLab 的 pyramid_stairs 用 platform_width=3.0，都是为了保证"机器人出生在平地上"。
    而出生高度沿用 legged_gym 的 env_origin_z = max(中心±1m 窗口)，该式默认中心是平的；
    在单调楼梯上它会取到前方 1m 处的最高台阶，实测出生点悬空：台阶 5cm→0.20m、
    10cm→0.40m、20cm→0.80m（金字塔楼梯因为有平台恒为 0.00m）。也就是每次 reset 都在
    自由落体。加上这块平台后，出生点落在 0 高度的平地上，契约恢复。

    -x 端(轴 0 低端)紧贴的是上一难度级的同类楼梯，其顶比本块的底高，天然形成一堵
    背墙，正好挡住后退、逼机器人正面往上爬。沿 y 恒定=定向：不能沿 y 等高绕圈
    （pyramid 沿 xy 都收所以能绕，这里只沿 x 收；y 向等高绕行由世界系奖励压制，
    实测最新策略横移已降到 ~0.3m）。

    step_height>0 -> 往 +x 升高（底部平台在最低，前进=上楼，index 7）；
    step_height<0 -> 往 +x 降低（底部平台在最高，前进=下楼，index 8）。
    平台恒为 0 高度；_reset_root_states 据此把出生 z 设成"平台高度+站立高度"。
    """
    sw = max(1, int(step_width / terrain.horizontal_scale))
    sh = int(step_height / terrain.vertical_scale)
    plat = max(1, int(platform_size / terrain.horizontal_scale))
    n = terrain.height_field_raw.shape[0]
    x = np.arange(n)
    d = x - plat                                    # 超出底部平台多少像素（<=0 即平台内）
    k = np.where(d <= 0, 0, (d + sw - 1) // sw)     # 第几级台阶（平台=0，往 +x 递增）
    terrain.height_field_raw[:, :] = (k * sh)[:, None]

def _pad_edges(terrain, pad_width, pad_height):
    """EP 的最外缘细边框：只防掉出地图，不是走廊墙（见 parkour_step_terrain 注释 2）。"""
    if pad_width <= 0:
        return
    pw = max(1, int(pad_width / terrain.horizontal_scale))
    ph = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:pw, :] = ph
    terrain.height_field_raw[-pw:, :] = ph
    terrain.height_field_raw[:, :pw] = ph
    terrain.height_field_raw[:, -pw:] = ph


def step_goal_indices(num_steps, num_goals, power=2.0):
    """goal 落在第几级台阶：间隔按幂律递增，前几级仍是一级一个。

    台阶数与 goal 数必须解耦，因为 goals 表对所有地形类型形状固定(num_goals=10)，
    而楼梯想要 16~20 级才够一个真实梯段。均匀撒 goal 会让最开始几级也拉开到 2~3 级
    一个，等于起步阶段没有引导；全部一级一个又只能有 8 级。幂律两头都照顾到：
    起步密（每级都有反馈），后面疏（让梯段长起来）。

    返回 1-based 的台阶序号，严格递增，最后一个恒等于 num_steps。
    18 级 / 8 个 goal / power=2 -> [1,2,3,4,7,10,14,18]，间隔 1,1,1,3,3,4,4。
    """
    num_steps = max(num_steps, num_goals)          # 每个 goal 至少要有一级可落
    k = np.arange(1, num_goals + 1) / num_goals
    idx = np.maximum(np.round(num_steps * k ** power).astype(int), 1)
    for i in range(1, num_goals):                  # 强制严格递增，避免重合的 goal
        idx[i] = max(idx[i], idx[i - 1] + 1)
    idx[-1] = num_steps
    for i in range(num_goals - 2, -1, -1):         # 反向再收一遍，防止越过末级
        idx[i] = min(idx[i], idx[i + 1] - 1)
    return idx


def parkour_step_terrain(terrain, step_height=0.2, platform_len=2.5, num_stones=8,
                         num_steps=None, goal_power=2.0,
                         x_range=(0.2, 0.4), y_range=(-0.15, 0.15),
                         half_valid_width=(0.45, 0.5), pad_width=0.1, pad_height=0.5,
                         outside_margin=None, rng=None):
    """Extreme Parkour(arXiv:2309.14341)的 parkour_step_terrain 复刻。

    参数名与默认值取自 EP 源码本身（legged_gym/utils/terrain.py）：
        platform_len=2.5, num_stones=8, x_range=[0.2,0.4], y_range=[-0.15,0.15],
        half_valid_width=[0.45,0.5], step_height=0.2, pad_width=0.1, pad_height=0.5

    两个容易搞反的要点（第一版复刻两条都错了，这里更正）：

    1) x_range 是【每一级台阶沿 x 的长度】，只有 0.2~0.4m。配 step_height=0.2 就是
       约 34 度的真陡楼梯。第一版把整块地形平分给 6 级、每级 1.6m，那不是楼梯，
       是"每隔 1.6m 一个孤立台面"，完全不同的运动技能。

    2) 通道之外保持【0 高度的低地】，不是墙。EP 只在整块地形的最外缘加一圈
       pad_width=0.1m 宽、pad_height=0.5m 高的细边框防止掉出地图。防绕行不是靠墙
       挡住，而是靠"goal 落在障碍上 + 绕出通道就得先下台阶再爬回来"，绕行本身
       在 tracking_goal_vel 上就是亏的。第一版把通道两侧抬得比台阶还高，等于
       Robot Parkour Learning 的走廊，那是另一篇论文的方案。

    3) 【下楼梯必须换一套通道外高度】。上楼时通道高于外面的 0 高度低地，绕行要先下
       台阶再爬回来，本身就亏，所以外面保持 0 就够了。下楼时通道沉到 0 以下，那块
       0 高度低地反而变成【比通道还高的平台】——机器人爬出去沿平坦的边沿一路走到
       终点，goal 奖励照拿不误，绕行从"亏"变成"赚"。所以下楼时传 outside_margin，
       让通道外【沿 x 跟着通道一起降】，始终只比同一 x 处的通道低这么多。
       不用"一整块平底深坑"：18 级下降到 -3.3m 时，那种坑在刚出发处就是 3.7m 自由
       落体，早期策略会大量摔死、接触力惩罚和终止统计全被这一下打乱。平行下沉的沟
       靠奖励而不是靠摔死来压制绕行——通道宽 1.4~1.6m，走到外面就是 |dy|>0.7m，
       超过 goal_pass_lateral_tol(0.5m)，goal 永远不计数、指针原地卡住。

    num_steps 与 num_stones(=goal 数)解耦：见 step_goal_indices。默认 num_steps 等于
    num_stones，即退化成"一级一个 goal"的旧行为。

    返回 goals: (num_stones+2, 2)，米，相对本块地形左下角；goals[0] 在起步平台上。
    """
    rng = rng or np.random
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    nx, ny = terrain.height_field_raw.shape
    mid_y = ny // 2

    terrain.height_field_raw[:] = 0                      # 通道外恒为 0 高度的低地
    lane = np.zeros(terrain.height_field_raw.shape, dtype=bool)
    goals = np.zeros((num_stones + 2, 2))

    plat = int(platform_len / hs)
    goals[0] = [(plat * 0.5) * hs, mid_y * hs]           # 起步平台中点
    lane[:plat, :] = True                                # 起步平台整幅都算通道
    sh = int(step_height / vs)

    n_steps = int(num_steps or num_stones)
    goal_at = step_goal_indices(n_steps, num_stones, goal_power)  # 1-based 台阶序号
    slot = {int(s): gi for gi, s in enumerate(goal_at)}

    dis_x = plat
    cur_h = 0
    last_center = goals[0]
    for k in range(1, n_steps + 1):
        run = int(rng.uniform(*x_range) / hs)            # 每级台阶的踏面长度(0.2~0.4m)
        rand_y = int(rng.uniform(*y_range) / hs)
        hw = int(rng.uniform(*half_valid_width) / hs)    # 该级的通道半宽
        x0, x1 = dis_x, min(dis_x + run, nx)
        if x0 >= nx:
            break
        cur_h += sh
        y0 = max(0, mid_y + rand_y - hw)
        y1 = min(ny, mid_y + rand_y + hw)
        terrain.height_field_raw[x0:x1, y0:y1] = cur_h   # 只抬中央条带
        lane[x0:x1, y0:y1] = True
        last_center = [((x0 + x1) * 0.5) * hs, (mid_y + rand_y) * hs]
        if k in slot:
            goals[slot[k] + 1] = last_center
        dis_x = x1
    # 地形塞不下时(不该发生)剩下的 goal 兜底压在最后一级上，宁可重合也不能是 0
    for gi in range(num_stones):
        if not goals[gi + 1].any():
            goals[gi + 1] = last_center

    # 末端平台与最后一级同高，最后一个 goal 落在它上面
    hw = int(np.mean(half_valid_width) / hs)
    if dis_x < nx:
        terrain.height_field_raw[dis_x:, mid_y - hw:mid_y + hw] = cur_h
        lane[dis_x:, mid_y - hw:mid_y + hw] = True
    goals[-1] = [min(dis_x + int(0.5 / hs), nx - 1) * hs, mid_y * hs]

    if outside_margin is not None:
        # 下楼梯专用：通道外沿 x 跟着通道一起降，恒低 outside_margin（见文档要点 3）
        hf = terrain.height_field_raw
        masked = np.where(lane, hf.astype(np.int32), np.iinfo(np.int32).min)
        prof = masked.max(axis=1)                        # 每个 x 处通道自身的高度
        ok = prof > np.iinfo(np.int32).min
        if ok.any():                                     # 兜底：平台/末端平台覆盖全部 x
            prof = np.interp(np.arange(nx), np.flatnonzero(ok), prof[ok])
            off = (prof - outside_margin / vs).astype(np.int32)
            hf[:] = np.where(lane, hf, off[:, None]).astype(hf.dtype)

    _pad_edges(terrain, pad_width, pad_height)
    return goals


def parkour_hurdle_terrain(terrain, num_stones=8, platform_len=2.5, stone_len=0.3,
                           x_range=(1.2, 1.8), y_range=(-0.4, 0.4),
                           half_valid_width=(0.4, 0.8), hurdle_height_range=(0.1, 0.15),
                           pad_width=0.1, pad_height=0.5, flat=False,
                           center_flat_goals=False, rng=None):
    """EP 的 parkour_hurdle_terrain 复刻：平地上每隔 x_range 一道横跨通道的方块。

    与 step 的区别：障碍是【孤立的】，之间是平地，高度不累加。所以它练的是"迈上去
    /跨过去"，不是连续爬楼。方块沿 x 有 stone_len 的厚度(最高 0.4m)，机器人可以踩
    在顶上而不必跳过去——这对没有手臂的双足很重要。

    goal 落在【两道障碍之间】(dis_x - rand_x/2)，不是障碍上，这是 EP 的原样：
    goal 在障碍后面，想拿到就必须先过障碍，绕行由 goal 的横向位置 + 环境侧的
    goal_pass_lateral_tol 压制。

    flat=True 时完全不放障碍，只留 goal 链——EP 的 "parkour_flat"，占它训练配比的
    20%，作用是把"跟着 goal 走"这件事和地形解耦。

    x_range 取 (1.2,1.8) 而非 EP 的 (1.2,2.2)：2.5 + 8*2.2 = 20.1m 会超出 18m 的
    块长，末尾几个 goal 被挤在一起甚至重合，而重合的 goal 会让 _update_goals 一步
    连推两格、课程虚高。2.5 + 8*1.8 = 16.9m 稳稳装得下。

    返回 goals: (num_stones+2, 2)，米。
    """
    rng = rng or np.random
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    nx, ny = terrain.height_field_raw.shape
    mid_y = ny // 2

    terrain.height_field_raw[:] = 0
    goals = np.zeros((num_stones + 2, 2))

    plat = int(platform_len / hs)
    sl = max(1, int(stone_len / hs))
    hvw = int(rng.uniform(*half_valid_width) / hs)
    h_lo, h_hi = (int(h / vs) for h in hurdle_height_range)

    goals[0] = [max(plat - 1, 0) * hs, mid_y * hs]
    dis_x = plat
    for k in range(num_stones):
        rand_x = int(rng.uniform(*x_range) / hs)
        rand_y = int(rng.uniform(*y_range) / hs)
        dis_x = min(dis_x + rand_x, nx - 1)
        if not flat:
            x0, x1 = max(0, dis_x - sl // 2), min(nx, dis_x + sl // 2 + 1)
            cy = mid_y + rand_y
            terrain.height_field_raw[x0:x1, max(0, cy - hvw):min(ny, cy + hvw)] = \
                rng.randint(h_lo, max(h_lo + 1, h_hi))
        # Keep consuming rand_y even when a straight flat lane is requested.
        # That preserves the full curriculum RNG stream (and therefore every
        # later stepping-stone/roughness tile) while changing only the goal
        # semantics of the flat stage.
        goal_y = mid_y if flat and center_flat_goals else mid_y + rand_y
        goals[k + 1] = [max(dis_x - rand_x // 2, 0) * hs, goal_y * hs]

    final = min(dis_x + int(np.mean(x_range) / hs), nx - 1)
    goals[-1] = [final * hs, mid_y * hs]

    _pad_edges(terrain, pad_width, pad_height)
    return goals


def parkour_stone_terrain(terrain, num_stones=8, platform_len=2.5, stone_len=0.9,
                          stone_width=1.0, gap_range=(0.1, 0.4), y_range=(0.2, 0.4),
                          pit_depth=0.5, pad_width=0.1, pad_height=0.5, rng=None):
    """EP 的 parkour_terrain（踏石）复刻，但【侧向倾斜恒为 0】。

    EP 原版给每块踏石加了横向线性倾斜，最难时 1.0m 宽上下差 ±0.25m(约 27°)。N2 的腿
    是 hip_yaw/roll/pitch + knee + ankle【只有踝俯仰，没有踝滚转】，支撑脚下的横向
    倾斜只能靠 hip_roll 整个躯干去配平——27° 不是技能是摔跤。去掉倾斜后 EP 那句
    np.tile(np.linspace(...)) 退化成常数，踏石就是一块块平的板子，也就是通常说的
    "踏石"地形：练的是离散落脚点的选择与跨越，而不是踝部适应斜面。

    【板子尺寸随难度缩小才是落脚点任务】0.9x1.0m 是脚(0.20x0.10m)面积的 45 倍，站
    哪儿都行，那练的其实只是"跨过缝"。缩到 0.45x0.50m 时单脚周围只剩 ±0.125/±0.200m
    余量，才真的要挑落脚点。

    【这不是"一脚一块"的梅花桩，当前分辨率下也做不到】N2 实测摆动距离 0.282m、步长
    0.129m、步宽 0.199m，所以一脚一块要求石块 x 间距 ~0.14m、横向偏置 ~±0.10m。而
    0.14m 在 horizontal_scale=0.1m 下只有 1.4 个格子，高度图物理上表达不出来(脚本身
    0.20x0.10m 也才 2x1 格)。本函数生成的 x 间距是 0.5~1.0m，即每块石头上要走 4~7 步，
    练的是"缝的两侧要落准 + 横向换脚"，不是逐桩点足。真梅花桩需要把 horizontal_scale
    降到 0.025m，属于另开任务，见计划文件。

    整块地形是深 pit_depth 的坑，只有踏石和首尾平台是实的；踏石沿 y 交替偏置
    ±y_range，所以必须左右换脚而不能一条直线走过去。

    返回 goals: (num_stones+2, 2)，米。
    """
    rng = rng or np.random
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    nx, ny = terrain.height_field_raw.shape
    mid_y = ny // 2

    terrain.height_field_raw[:] = -int(pit_depth / vs)
    goals = np.zeros((num_stones + 2, 2))

    plat = int(platform_len / hs)
    # 【必须 round 而不是 int】板子尺寸本来就被 horizontal_scale=0.1m 量化，再用截断
    # 就是二次损失：0.55m 宽先 /2 截成 2 格、再 x2 变成 0.40m，比配置值少了 27%，
    # 而这一项恰恰是随难度收紧的落脚余量，误差直接吃掉一整档课程。
    sl = max(1, int(round(stone_len / hs)))
    wc = max(2, int(round(stone_width / hs)))            # 板宽(格)
    terrain.height_field_raw[:plat, :] = 0
    # 【出生点放平台中点，不是平台边缘】_reset_root_states 会在 goals[0] 上叠加
    # ±parkour_spawn_jitter(0.3m) 的 xy 抖动。放在 plat-1(=2.4m) 距平台末端只剩 0.1m，
    # 实测 6.7% 的出生点直接落到坑上、最深掉 0.52m。跨栏/平地放边缘无所谓(平台外是
    # 0 高度平地)，踏石平台外是坑，必须留够余量。plat/2 留 1.25m，远大于抖动幅度。
    goals[0] = [(plat * 0.5) * hs, mid_y * hs]

    dis_x = plat
    side = rng.randint(0, 2)                             # 交替左右
    for k in range(num_stones):
        dis_x += int(rng.uniform(*gap_range) / hs)
        x0, x1 = min(dis_x, nx - 1), min(dis_x + sl, nx)
        cy = mid_y + (1 if side else -1) * int(rng.uniform(*y_range) / hs)
        y0 = max(0, cy - wc // 2)
        terrain.height_field_raw[x0:x1, y0:min(ny, y0 + wc)] = 0
        goals[k + 1] = [((x0 + x1) * 0.5) * hs, cy * hs]
        dis_x = x1
        side = 1 - side

    if dis_x < nx:                                       # 末端平台
        terrain.height_field_raw[dis_x:, :] = 0
    goals[-1] = [min(dis_x + int(0.5 / hs), nx - 1) * hs, mid_y * hs]

    _pad_edges(terrain, pad_width, pad_height)
    return goals


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.vertical_scale = cfg.vertical_scale
        self.horizontal_scale = cfg.horizontal_scale

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

class HumanoidTerrain(Terrain):
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        super().__init__(cfg, num_robots)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        step_width = np.random.uniform(0.3, 0.45)
        discrete_obstacles_height = difficulty * 0.20
        r_height = difficulty * 0.07 # 0.07
        h_slope = difficulty * 0.15 # 0.15
        if choice < self.proportions[0]:
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[1]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[2]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, step=0.005, downsampled_scale=0.2)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[3]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[4]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=-discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[7]:
            directional_stairs(terrain, step_width, discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[8]:
            directional_stairs(terrain, step_width, -discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        else:
            pass
        return terrain


class ParkourTerrain(HumanoidTerrain):
    """Extreme Parkour 式地形：每块地形都是"中央通道 + 一串必须踩上去的 goal 路点"。

    与 HumanoidTerrain 的区别只有两点：
      1) make_terrain 按 terrain_proportions 在【parkour 专属的 5 种地形】之间分派
         （索引含义见下方 TYPES，与 HumanoidTerrain 那套 9 槽完全无关）。列=类型、
         行=难度，与 EP 同构：EP 默认就是 parkour/hurdle/flat/step/gap 各 0.2 混训
         一个策略，num_goals 对所有类型都是同一个数。
      2) 多存一个 self.goals[row, col] = (num_goals, 3) 的世界坐标路点表，
         env 侧按 terrain_levels/terrain_types 索引取用。

    【硬约束】每种地形都必须吐出恰好 num_goals 个路点：self.goals 是一个形状固定的
    数组，类型之间不能不齐。所以所有生成函数一律 num_stones = num_goals - 2。
    """

    # terrain_proportions 的索引含义（parkour 专用）
    TYPES = ['上台阶', '下台阶', '跨栏', '平地路点', '踏石']

    def __init__(self, cfg, num_robots) -> None:
        self.num_goals = int(getattr(cfg, 'num_goals', 8))
        # 必须在 super().__init__ 之前建好：父类构造函数里就会调用 curiculum()
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals, 3))
        self._pending_goals = None
        super().__init__(cfg, num_robots)

    def make_terrain(self, choice, difficulty): #TODO: add more types of terrains
        # 基类默认 terrain_length == terrain_width，把 SubTerrain 建成正方形。
        # Parkour 是长通道，必须按 add_terrain_to_map 实际写入的形状建，否则广播失败。
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.length_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        c = self.cfg
        n = self.num_goals - 2                      # 见类文档的硬约束
        common = dict(num_stones=n,
                      platform_len=getattr(c, 'parkour_platform_len', 2.5),
                      pad_width=getattr(c, 'parkour_pad_width', 0.1),
                      pad_height=getattr(c, 'parkour_pad_height', 0.5))
        p = self.proportions

        # 台阶高度随难度线性增长，这是 step 类的唯一课程维度（EP 同样只调障碍尺度）
        lo, hi = getattr(c, 'parkour_step_height_range', [0.05, 0.20])
        sh = lo + difficulty * (hi - lo)
        step_kw = dict(x_range=tuple(getattr(c, 'parkour_x_range', (0.2, 0.4))),
                       y_range=tuple(getattr(c, 'parkour_y_range', (-0.15, 0.15))),
                       half_valid_width=tuple(getattr(c, 'parkour_half_valid_width',
                                                      (0.45, 0.5))),
                       num_steps=getattr(c, 'parkour_step_count', None),
                       goal_power=getattr(c, 'parkour_step_goal_power', 2.0))

        if choice < p[0]:                                   # 0 上台阶
            self._pending_goals = parkour_step_terrain(
                terrain, step_height=sh, **step_kw, **common)
        elif choice < p[1]:                                 # 1 下台阶
            self._pending_goals = parkour_step_terrain(
                terrain, step_height=-sh,
                outside_margin=getattr(c, 'parkour_stepdown_outside_margin', 0.3),
                **step_kw, **common)
        elif choice < p[3]:                                 # 2 跨栏 / 3 平地路点
            h_lo, h_hi = getattr(c, 'parkour_hurdle_height_range', [0.10, 0.30])
            top = h_lo + difficulty * (h_hi - h_lo)
            self._pending_goals = parkour_hurdle_terrain(
                terrain,
                flat=(choice >= p[2]),
                center_flat_goals=bool(getattr(
                    c, 'parkour_flat_center_goals', False)),
                stone_len=getattr(c, 'parkour_hurdle_len', 0.3),
                x_range=tuple(getattr(c, 'parkour_hurdle_x_range', (1.2, 1.8))),
                y_range=tuple(getattr(c, 'parkour_hurdle_y_range', (-0.4, 0.4))),
                half_valid_width=tuple(getattr(c, 'parkour_hurdle_half_valid_width',
                                               (0.4, 0.8))),
                hurdle_height_range=(max(h_lo * 0.5, top - 0.05), top),
                **common)
        else:                                               # 4 踏石
            g_lo, g_hi = getattr(c, 'parkour_stone_gap_range', [0.1, 0.4])
            # 板子随难度缩小：从"站哪儿都行"退化到真正要挑落脚点
            l_lo, l_hi = getattr(c, 'parkour_stone_len_range', [0.9, 0.45])
            w_lo, w_hi = getattr(c, 'parkour_stone_width_range', [1.0, 0.5])
            self._pending_goals = parkour_stone_terrain(
                terrain,
                stone_len=l_lo + difficulty * (l_hi - l_lo),
                stone_width=w_lo + difficulty * (w_hi - w_lo),
                gap_range=(g_lo, g_lo + difficulty * (g_hi - g_lo)),
                y_range=tuple(getattr(c, 'parkour_stone_y_range', (0.2, 0.4))),
                pit_depth=getattr(c, 'parkour_stone_pit_depth', 0.5),
                **common)
        add_roughness(terrain, np.random.uniform(0.01, 0.03))
        """
        if choice < self.proportions[0]:
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[1]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[2]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, step=0.005, downsampled_scale=0.2)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[3]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[4]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=-discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[7]:
            directional_stairs(terrain, step_width, discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[8]:
            directional_stairs(terrain, step_width, -discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        else:
            pass
        """
        return terrain

    def min_goal_spacing(self):
        """本配置下相邻 goal 的最小可能间距(m)——env 侧用它校验 goal_reach_dist。"""
        c = self.cfg
        p = self.cfg.terrain_proportions
        spacing = []
        if p[0] > 0 or p[1] > 0:
            spacing.append(float(getattr(c, 'parkour_x_range', (0.2, 0.4))[0]))
        if len(p) > 3 and (p[2] > 0 or p[3] > 0):
            spacing.append(float(getattr(c, 'parkour_hurdle_x_range', (1.2, 1.8))[0]) / 2)
        if len(p) > 4 and p[4] > 0:
            # 首块踏石离起步平台最近：gap + 半块板长(取最难那档的最小板长)
            spacing.append(float(getattr(c, 'parkour_stone_gap_range', (0.1, 0.4))[0])
                           + min(getattr(c, 'parkour_stone_len_range', (0.9, 0.45))) / 2)
        return min(spacing) if spacing else 0.0

    def add_terrain_to_map(self, terrain, row, col):
        super().add_terrain_to_map(terrain, row, col)
        if self._pending_goals is None:
            return
        # 块内相对坐标 -> 世界坐标；z 取该 goal 处的地形高度
        gx = self._pending_goals[:, 0] + row * self.env_length
        gy = self._pending_goals[:, 1] + col * self.env_width
        px = np.clip((self._pending_goals[:, 0] / self.horizontal_scale).astype(int),
                     0, terrain.height_field_raw.shape[0] - 1)
        py = np.clip((self._pending_goals[:, 1] / self.horizontal_scale).astype(int),
                     0, terrain.height_field_raw.shape[1] - 1)
        gz = terrain.height_field_raw[px, py] * self.vertical_scale
        self.goals[row, col, :, 0] = gx
        self.goals[row, col, :, 1] = gy
        self.goals[row, col, :, 2] = gz
        self._pending_goals = None


class ParkourCourseTerrain(ParkourTerrain):
    """One deterministic, continuous five-stage parkour course.

    This is an evaluation terrain, deliberately separate from
    :class:`ParkourTerrain`'s 8x10 training curriculum. Both Isaac Gym and
    MuJoCo consume ``sim2sim.parkour_campaign.build_campaign`` with the same
    course settings, so the central course samples, random stream,
    quantisation, cropped stage lengths and seams are byte-for-byte identical:

        up stairs -> down stairs -> hurdle -> flat -> stepping stones

    Isaac may add a lateral viewing shoulder on each side.  In that mode the
    source tile's 0.5 m lateral pads are flattened at the original 4 m course
    edge.  A low curb following the local shoulder height is installed at the
    far outside edge of the wider Isaac-only tile.  The robot therefore keeps
    a physical safety boundary without a tall wall in either the foreground
    or background of a right-side camera.  MuJoCo's campaign height field is
    never mutated.

    Repeated stage ``goals[0]`` are spawn/retry anchors, not forward targets.
    Isaac therefore exposes one initial ``g0`` followed by ``g1..g9`` from
    each of the five stages (46 goals total).
    """

    def __init__(self, cfg, num_robots) -> None:
        # Import lazily: normal training tasks do not need the campaign helper,
        # and the helper itself intentionally has no Isaac Gym dependency.
        from sim2sim.parkour_campaign import (
            HORIZONTAL_SCALE,
            NUM_GOALS,
            NUM_ROWS,
            VERTICAL_SCALE,
            build_campaign,
        )

        source_rows = int(getattr(cfg, 'course_num_rows', NUM_ROWS))
        if source_rows != NUM_ROWS:
            raise ValueError(
                "course_num_rows=%d 与共享 campaign 生成器固定的 %d 行不一致"
                % (source_rows, NUM_ROWS))
        if not np.isclose(cfg.horizontal_scale, HORIZONTAL_SCALE):
            raise ValueError(
                "course horizontal_scale 必须为 %.3f，当前为 %.3f"
                % (HORIZONTAL_SCALE, cfg.horizontal_scale))
        if not np.isclose(cfg.vertical_scale, VERTICAL_SCALE):
            raise ValueError(
                "course vertical_scale 必须为 %.3f，当前为 %.3f"
                % (VERTICAL_SCALE, cfg.vertical_scale))

        self.campaign = build_campaign(
            row=int(getattr(cfg, 'course_row', 3)),
            seed=int(getattr(cfg, 'course_seed', 5)),
            add_roughness=bool(getattr(cfg, 'course_add_roughness', True)),
            center_flat_goals=bool(getattr(
                cfg, 'parkour_flat_center_goals', False)),
            step_x_range=getattr(cfg, 'course_step_x_range', None),
        )

        side_margin = float(getattr(cfg, 'course_side_margin', 0.0))
        if side_margin < 0.0:
            raise ValueError("course_side_margin 不能为负")
        side_margin_cells = int(round(
            side_margin / self.campaign.horizontal_scale))
        quantized_margin = (
            side_margin_cells * self.campaign.horizontal_scale)
        if not np.isclose(side_margin, quantized_margin):
            raise ValueError(
                "course_side_margin 必须是 horizontal_scale=%.3f 的整数倍"
                % self.campaign.horizontal_scale)
        self.course_side_margin = quantized_margin
        self.course_side_margin_cells = side_margin_cells
        self.course_source_width = self.campaign.width
        self.course_y_offset = (
            self.course_side_margin + self.campaign.width * 0.5)

        # The map allocator in Terrain.__init__ must use the cropped, generated
        # campaign length, not 5*18 m. Width may be expanded only in Isaac to
        # make room for an unobstructed side camera.
        cfg.terrain_length = self.campaign.length
        cfg.terrain_width = (
            self.campaign.width + 2.0 * self.course_side_margin)
        cfg.num_rows = 1
        cfg.num_cols = 1
        cfg.curriculum = False
        cfg.selected = False

        stages = self.campaign.stages
        if len(stages) != 5 or any(stage.goals.shape[0] != NUM_GOALS
                                   for stage in stages):
            raise ValueError("continuous parkour course 必须是 5 段、每段 10 个 goal")

        centred_goal_chain = np.concatenate(
            [stages[0].goals[:1]]
            + [stage.goals[1:] for stage in stages],
            axis=0,
        )
        # MuJoCo centres its hfield on y=0. Isaac's wider terrain tile occupies
        # y=[0,width], so translate the goal chain to its centre; x/z are
        # already world coordinates.
        self._course_world_goals = centred_goal_chain.copy()
        self._course_world_goals[:, 1] += self.course_y_offset
        self._course_local_goals = self._course_world_goals[:, :2].copy()
        cfg.num_goals = int(self._course_local_goals.shape[0])

        self.course_stage_names = tuple(stage.label for stage in stages)
        self.course_stage_kinds = tuple(stage.kind for stage in stages)
        self.course_stage_boundaries = np.asarray(
            [stage.origin_x for stage in stages] + [stages[-1].end_x],
            dtype=np.float64,
        )
        # goal 0 is the course spawn. Each stage then owns nine forward goals.
        self.course_stage_goal_starts = np.asarray(
            [1 + i * (NUM_GOALS - 1) for i in range(len(stages))],
            dtype=np.int64,
        )
        self.course_stage_spawn_goals = np.stack(
            [stage.goals[0] for stage in stages], axis=0)
        self.course_stage_spawn_goals[:, 1] += self.course_y_offset
        self.goal_stage_indices = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.repeat(np.arange(len(stages), dtype=np.int64), NUM_GOALS - 1),
        ))

        super().__init__(cfg, num_robots)
        # Keep the exact shared z values. Recomputing them from global x with
        # ``astype(int)`` can select an adjacent 0.1 m cell at values such as
        # 36.3 because of floating-point representation.
        self.goals[0, 0] = self._course_world_goals

    def randomized_terrain(self):
        """Install the shared campaign as the single Isaac terrain tile."""
        raw = self._isaac_course_height_field()
        terrain = terrain_utils.SubTerrain(
            "parkour_course",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        if terrain.height_field_raw.shape != raw.shape:
            raise ValueError(
                "course heightfield shape mismatch: Isaac %s, shared campaign %s"
                % (terrain.height_field_raw.shape, raw.shape))
        terrain.height_field_raw[:] = raw
        self._pending_goals = self._course_local_goals.copy()
        self.add_terrain_to_map(terrain, 0, 0)

    def _isaac_course_height_field(self):
        """Return the shared course with optional Isaac-only side shoulders."""
        source = self.campaign.height_field_raw
        margin = self.course_side_margin_cells
        if margin == 0:
            return source

        pad_width = float(getattr(
            self.cfg, 'parkour_pad_width', 0.1))
        pad = max(1, int(round(
            pad_width / self.campaign.horizontal_scale)))
        if margin < pad:
            raise ValueError(
                "course_side_margin 至少要容纳 %.3fm 的安全边界"
                % (pad * self.campaign.horizontal_scale))
        if source.shape[1] <= 2 * pad:
            raise ValueError("course 原始宽度不足以移动两侧安全边界")

        # First remove the source tile's high lateral pads.  Extend the
        # centreline sample at each x across both viewing shoulders.  On the
        # stair stages this makes the shoulders climb/descend with the same
        # stair level as the robot instead of exposing a long vertical cut
        # between the central stair lane and a low field.  The source interior
        # (and therefore the policy's central height scan) stays byte-identical.
        visible = source.copy()
        centre = source.shape[1] // 2
        shoulder_profile = visible[:, centre:centre + 1]
        visible[:, :pad] = shoulder_profile
        visible[:, -pad:] = shoulder_profile

        expanded = np.empty(
            (source.shape[0], source.shape[1] + 2 * margin),
            dtype=source.dtype)
        start = margin
        end = start + source.shape[1]
        expanded[:, start:end] = visible
        expanded[:, :start] = shoulder_profile
        expanded[:, end:] = shoulder_profile

        # Keep collision safety at the far edges with a low curb that follows
        # the local shoulder profile.  Reusing the original absolute 0.5 m pad
        # would create a 2.4 m wall beside the lifted down-stair tile.
        curb_height = float(getattr(
            self.cfg, 'course_safety_curb_height', 0.20))
        if curb_height <= 0.0:
            raise ValueError("course_safety_curb_height 必须为正")
        curb = int(round(
            curb_height / self.campaign.vertical_scale))
        expanded[:, :pad] = (
            expanded[:, pad:pad + 1].astype(np.int32) + curb
        ).astype(source.dtype)
        expanded[:, -pad:] = (
            expanded[:, -pad - 1:-pad].astype(np.int32) + curb
        ).astype(source.dtype)
        return expanded

    def min_goal_spacing(self):
        """Minimum planar spacing in the actual 46-goal course."""
        delta = np.diff(self._course_local_goals, axis=0)
        return float(np.linalg.norm(delta, axis=1).min())
