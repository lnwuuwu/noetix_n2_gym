# 导入操作系统相关功能
import os
# 导入系统相关功能
import sys
# 从humanoid模块导入根目录路径
from humanoid import LEGGED_GYM_ROOT_DIR

# 导入Isaac Gym库
import isaacgym
# 导入所有环境相关模块
from humanoid.envs import *
# 导入工具函数和类
from humanoid.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

# 导入数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch


# ---- 要测试哪块地形 ----
# _get_env_origins 按 terrain_types = arange(num_envs) // (num_envs/num_cols)
# 分配地形列，play 里 num_envs=1 于是恒为 0——机器人被永久钉死在第 0 列，绝大多数
# 地形测不到。下面两个量显式指定落在哪一格；curriculum=True 时地形网格是确定性的
# (行 -> 难度，列 -> 类型)，所以 (行,列) 能精确选中"某种地形的某个难度"。
#   ROW = None -> 列轮换一圈后自动升一档难度
#   COL = None -> 每 PLAY_TILE_STEPS 步自动换下一种地形
PLAY_TERRAIN_ROW = None      # 难度行 0..num_rows-1
PLAY_TERRAIN_COL = None      # 类型列 0..num_cols-1
PLAY_TILE_STEPS = 600        # 自动轮换时每格停留的步数


def is_parkour(task):
    """n2_parkour 用的是 ParkourTerrain，它的 terrain_proportions 索引含义与
    HumanoidTerrain 那套 9 槽完全不同(见 ParkourTerrain.TYPES)，所以 play 里凡是
    依赖 proportions 的地方都要分开处理。"""
    return str(task).startswith('n2_parkour')


def is_parkour_course(task):
    """The one-tile, five-stage straight evaluation runway."""
    return str(task) in (
        'n2_parkour_course',
        'n2_parkour_slow_stable_course',
    )


HUMANOID_TILE_NAMES = ['平地+粗糙', '离散障碍', '均匀粗糙', '上坡', '下坡',
                       '金字塔上楼梯', '金字塔下楼梯', '直行上楼梯', '直行下楼梯']


def terrain_column_names(proportions, num_cols, names=None):
    """列号 -> 地形类型名。

    复刻 make_terrain 的 elif 链，choice 取值与 Terrain.curiculum() 里的
    j/num_cols + 0.001 一致，所以打印出来的就是每一列实际生成的地形。
    names 缺省是 HumanoidTerrain 的 9 槽；parkour 传 ParkourTerrain.TYPES。
    """
    cum = [float(np.sum(proportions[:i + 1])) for i in range(len(proportions))]
    names = names or HUMANOID_TILE_NAMES
    out = []
    for j in range(num_cols):
        choice = j / num_cols + 0.001
        pick = '平地(无地形,fallback)'
        for k, c in enumerate(cum):
            if choice < c:
                pick = names[k] if k < len(names) else 'idx%d' % k
                break
        out.append(pick)
    return out


def align_policy_cfg_to_checkpoint(train_cfg, args):
    """用 checkpoint 自己的 train_cfg.json 覆盖当前的 policy 网络配置。

    网络结构一改（例如加 scan encoder、把主干从 [256,128] 扩到 [512,256,128]），
    旧 checkpoint 就再也 load 不进来：load_state_dict 会同时报 missing key
    (actor.scan_encoder.*/actor.trunk.*)、unexpected key (actor.0/2/4) 和 critic
    各层的 size mismatch。但那不是坏档，只是"用新网络去装旧权重"。

    每个 run 目录里都存了当时的 train_cfg.json（含 policy 段），直接拿它覆盖，
    play 就能忠实复现任何历史 checkpoint，不必回滚代码或手改配置。
    只影响 play；训练路径不碰。
    """
    import json
    root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    run = args.load_run if getattr(args, 'load_run', None) not in (None, -1, '-1') else None
    try:
        if run is None:
            runs = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)) and d != 'exported')
            run = runs[-1]
        cfg_path = os.path.join(root, run, 'train_cfg.json')
        saved = json.load(open(cfg_path))['policy']
    except Exception as e:
        print('[play] 未能读取 checkpoint 的 policy 配置(%s)，沿用当前配置' % e)
        return
    changed = []
    for k, v in saved.items():
        if hasattr(train_cfg.policy, k) and getattr(train_cfg.policy, k) != v:
            changed.append('%s: %s -> %s' % (k, getattr(train_cfg.policy, k), v))
        setattr(train_cfg.policy, k, v)
    # 存档里没有的键要清掉，否则会凭空启用当时不存在的结构（如 scan_encoder_dims）
    for k in ('scan_encoder_dims',):
        if k not in saved and getattr(train_cfg.policy, k, None) is not None:
            changed.append('%s: %s -> None(存档中不存在)' % (k, getattr(train_cfg.policy, k)))
            setattr(train_cfg.policy, k, None)
    print('[play] 按 %s 的 train_cfg.json 对齐网络配置' % run
          + ('：' + '; '.join(changed) if changed else '（无差异）'))


def play(args):
    """
    播放/测试函数：加载训练好的策略模型并在环境中运行以可视化结果
    
    参数:
        args: 命令行参数对象，包含运行所需的各种配置
    """
    # 获取环境和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    parkour_course = is_parkour_course(args.task)
    
    # 重写一些测试参数
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 限制环境数量为1
    env_cfg.env.num_envs = 1  # 设置环境数量为1
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10  # 设置GPU接触对的最大数量
    env_cfg.terrain.mesh_type = 'trimesh'  # 设置地形类型为
    if parkour_course:
        # The course config is already one deterministic, dynamically-sized
        # tile. Turning curriculum on would route it through the 8x10 generator.
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 0
    else:
        env_cfg.terrain.num_rows = 8  # 设置地形行数
        # parkour 保持 10 列：类型按 j/num_cols 分派，配比里最小的一档是 0.1，
        # 列数少于 10 时它会被整除抹掉(8 列时踏石那一列根本不会生成)。
        env_cfg.terrain.num_cols = 10 if is_parkour(args.task) else 8
        # 必须打开：否则 Terrain 走 randomized_terrain()，每格类型和难度都随机，既
        # 选不中也复现不了。建完环境后会立刻改回 False（见下面）。
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 5
    # 必须 9 项：7 项时 cumsum 在 index 6 就到 1.0，永远走不到 index 7/8 的直行楼梯
    if not is_parkour(args.task):
        env_cfg.terrain.terrain_proportions = [0., 0.0, 0.1, 0.0, 0.0, 0.05, 0.2, 0.2, 0.15]
    # parkour 的 proportions 用 config 里的原值(索引含义是 ParkourTerrain.TYPES)，
    # 覆盖会打乱列->类型的对应关系。
    # env_cfg.terrain.selected = True
    # env_cfg.terrain.terrain_kwargs = {'type': 'pyramid_stairs_terrain',
    #                                   'step_width': 0.30,
    #                                   'step_height': 0.16,   # 分三次改这个数
    #                                   'platform_size': 2.}
    env_cfg.noise.add_noise = False  # 关闭噪声添加
    env_cfg.domain_rand.randomize_gains = False  # 关闭增益随机化
    env_cfg.domain_rand.randomize_motor_strength = False  # 关闭电机强度随机化
    env_cfg.domain_rand.randomize_base_mass = False  # 关闭基础质量随机化
    env_cfg.domain_rand.randomize_com_displacement = False  # 关闭质心位移随机化
    env_cfg.domain_rand.randomize_friction = False  # 关闭摩擦系数随机化
    env_cfg.domain_rand.push_robots = False  # 关闭机器人推动
    env_cfg.domain_rand.disturbance = False  # 关闭干扰
    env_cfg.domain_rand.disturbance_probabilities = 0.005  # 设置干扰概率
    env_cfg.domain_rand.push_force_range = [50.0, 500.0]  # 设置推力范围
    env_cfg.domain_rand.push_torque_range = [0.0, 0.0]  # 设置扭矩范围
    env_cfg.env.episode_length_s = 300 if parkour_course else 100

    # 设置为测试模式
    env_cfg.env.test = True

    # 如果控制机器人标志为真，则进一步调整参数
    if CONTROL_ROBOT:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 确保环境数量不超过1
        env_cfg.env.episode_length_s = 300 if parkour_course else 100
        env_cfg.commands.resampling_time = [1000, 1001]  # 设置命令重采样时间
        if not parkour_course:
            env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]  # 设置x方向线速度范围
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]  # 设置y方向线速度范围
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]  # 设置偏航角速度范围

    # 准备环境
    # env: 环境对象，用于模拟和交互
    # _: 忽略第二个返回值
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 地形已经按确定性网格生成好了，这里把运行时课程更新关掉：reset_idx 里
    # `if self.cfg.terrain.curriculum: self._update_terrain_curriculum(...)`
    # 每次 reset 都会按位移改写 terrain_levels，不关掉就固定不住指定的难度行。
    env.cfg.terrain.curriculum = False

    if parkour_course:
        names = env.terrain.course_stage_names
        bounds = env.terrain.course_stage_boundaries
        print('\n[play] 连续 parkour 跑道 %.1fm，row=%d/7，seed=%d：'
              % (env.terrain.campaign.length,
                 env_cfg.terrain.course_row,
                 env_cfg.terrain.course_seed))
        for j, name in enumerate(names):
            print('       stage %d : %-8s  x=[%.1f, %.1f)m'
                  % (j + 1, name, bounds[j], bounds[j + 1]))
        print('       路点：g0 + 每段 g1..g9，共 %d 个；到终点保持画面。'
              % env.terrain.num_goals)
    elif is_parkour(args.task):
        from humanoid.utils.terrain import ParkourTerrain
        hr = getattr(env_cfg.terrain, 'parkour_step_height_range', [0.05, 0.20])
        tile_names = terrain_column_names(env_cfg.terrain.terrain_proportions,
                                          env_cfg.terrain.num_cols,
                                          ParkourTerrain.TYPES)
        print('\n[play] parkour 地形 %d 行(难度) x %d 列(类型):'
              % (env_cfg.terrain.num_rows, env_cfg.terrain.num_cols))
        for j, nm in enumerate(tile_names):
            print('       col %d : %s' % (j, nm))
        for i in range(env_cfg.terrain.num_rows):
            d = i / env_cfg.terrain.num_rows
            print('       row %d : 台阶高 %.1f cm(台阶列)' %
                  (i, 100 * (hr[0] + d * (hr[1] - hr[0]))))
    else:
        tile_names = terrain_column_names(env_cfg.terrain.terrain_proportions,
                                          env_cfg.terrain.num_cols)
        print('\n[play] 地形网格 %d 行(难度) x %d 列(类型):'
              % (env_cfg.terrain.num_rows, env_cfg.terrain.num_cols))
        for j, nm in enumerate(tile_names):
            print('       col %d : %s' % (j, nm))

    def goto_tile(row, col):
        """把机器人挪到第 row 行 / 第 col 列那一格并重置。"""
        row = int(row) % env_cfg.terrain.num_rows
        col = int(col) % env_cfg.terrain.num_cols
        env.terrain_levels[:] = row
        env.terrain_types[:] = col
        env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
        o, _ = env.reset()
        print('[play] --> row %d (难度 %.2f) / col %d : %s'
              % (row, row / env_cfg.terrain.num_rows, col, tile_names[col]))
        return o

    # 自动轮换默认从第 3 行起：难度 = row/num_rows，第 0 行难度恰好是 0，台阶高度
    # discrete_obstacles_height = difficulty*0.20 = 0，不管哪一列都是平的，测不出东西。
    # 3/8 = 0.375 -> 台阶 7.5cm，和策略训练到的 terrain_level≈3.9 大致对得上。
    cur_row = PLAY_TERRAIN_ROW if PLAY_TERRAIN_ROW is not None else 3
    cur_col = PLAY_TERRAIN_COL if PLAY_TERRAIN_COL is not None else 0
    if parkour_course:
        # Initial placement at the course spawn is a normal reset, not a
        # mid-run tile teleport.
        obs, _ = env.reset()
    else:
        obs = goto_tile(cur_row, cur_col)

    # 获取初始观测值
    obs = env.get_observations()
    
    # 加载策略
    train_cfg.runner.resume = True  # 设置为恢复模式
    # 网络结构可能已经改过，用 checkpoint 自带的 train_cfg.json 对齐，
    # 否则会用新网络去装旧权重、load_state_dict 直接报 size mismatch。
    align_policy_cfg_to_checkpoint(train_cfg, args)
    # 创建算法运行器实例
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 将策略导出为JIT模块（用于C++中运行）
    if EXPORT_POLICY:
        # 构建导出路径
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_stem = getattr(args, 'export_policy_name', None) or 'policy'
        if os.path.basename(export_stem) != export_stem or export_stem in ('.', '..'):
            raise ValueError('--export_policy_name 只能是文件名，不能包含目录')
        # 导出策略为JIT模块
        export_policy_as_jit(
            ppo_runner.alg.policy, path, ppo_runner.obs_normalizer,
            filename=export_stem + '.pt')
        # 导出策略为ONNX格式
        export_policy_as_onnx(
            ppo_runner.alg.policy, path, ppo_runner.obs_normalizer,
            filename=export_stem + '.onnx')
        print('Exported policy to: ', os.path.join(path, export_stem + '.{pt,onnx}'))

    if getattr(args, 'export_only', False):
        print('[play] export-only 完成，退出。')
        return

    # 创建日志记录器
    logger = Logger(env)
    robot_index = 0  # 用于日志记录的机器人索引
    joint_index = 1  # 用于日志记录的关节索引
    stop_state_log = 100  # 开始绘制状态前的步数
    stop_rew_log = env.max_episode_length + 1  # 开始打印平均奖励前的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 相机位置
    camera_vel = np.array([1., 1., 0.])  # 相机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 相机方向
    img_idx = 0  # 图像索引
    last_course_stage = -1
    course_finish_printed = False

    # 主循环：运行指定次数的episode
    for i in range(10*int(env.max_episode_length)):
        # 自动轮换地形格：列没写死就每 PLAY_TILE_STEPS 步换下一种地形，
        # 走完一圈且行也没写死时再升一档难度，这样一次 play 能把所有地形都过一遍
        if (not parkour_course and PLAY_TERRAIN_COL is None
                and i > 0 and i % PLAY_TILE_STEPS == 0):
            cur_col += 1
            if cur_col % env_cfg.terrain.num_cols == 0 and PLAY_TERRAIN_ROW is None:
                cur_row += 1
            obs = goto_tile(cur_row, cur_col)
            continue

        actions = policy(obs.detach())
        # parkour 训练时 vx 只在 [parkour_min_vx, lin_vel_x[1]] 采样(默认 0.3~0.8)，
        # 沿用写死的 1.0 属于分布外外推，会让 play 的表现比实际更差。
        if parkour_course:
            # N2ParkourEnv sets a conservative speed for the current stage and
            # zero at the finish; do not overwrite it with one global value.
            env._apply_course_commands()
        elif is_parkour(args.task):
            env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[1]
        else:
            env.commands[:,0] = 1.0  # 控制x方向线速度为1.0
        env.commands[:,1] = 0.0  # 控制y方向线速度为0.0
        env.commands[:,2] = 0.0  # 控制偏航角速度
        obs, _, rews, dones, infos,_,_ = env.step(actions.detach())

        if parkour_course:
            stage = int(env.course_stage_idx[0].item())
            if stage != last_course_stage:
                print('[play] --> stage %d/%d : %s，vx=%.2fm/s'
                      % (stage + 1, len(env.terrain.course_stage_names),
                         env.terrain.course_stage_names[stage],
                         env.commands[0, 0].item()))
                last_course_stage = stage
            if bool(env.course_finished[0]) and not course_finish_printed:
                print('[play] 全部通关：机器人已走完连续跑道；viewer 保持打开。')
                course_finish_printed = True
                if args.headless:
                    print('[play] headless 连续跑道验收完成，退出。')
                    break
        
        # 如果需要录制帧，则保存图像
        if RECORD_FRAMES:
            if i % 2:  # 每隔一步保存一次图像
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
                
        # 如果需要移动相机，则更新相机位置
        if parkour_course and not args.headless:
            base = env.root_states[0, :3].detach().cpu().numpy()
            env.set_camera(
                base + np.array([-3.2, -4.0, 2.4]),
                base + np.array([1.0, 0.0, 0.35]))
        elif MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # 记录状态日志
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,  # 目标关节位置
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),  # 实际关节位置
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),  # 关节速度
                    'dof_torque': env.torques[robot_index, joint_index].item(),  # 关节扭矩
                    'command_x': env.commands[robot_index, 0].item(),  # x方向命令
                    'command_y': env.commands[robot_index, 1].item(),  # y方向命令
                    'command_yaw': env.commands[robot_index, 2].item(),  # 偏航命令
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),  # 基座x方向速度
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),  # 基座y方向速度
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),  # 基座z方向速度
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),  # 基座偏航角速度
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()  # 接触力z分量
                }
            )
        # 如果达到记录步数，则绘制状态图
        # elif i==stop_state_log:
        #     logger.plot_states()
            
        # 记录奖励日志
        if  0 < i < stop_rew_log:
            if infos["episode"]:  # 如果有episode信息
                num_episodes = torch.sum(env.reset_buf).item()  # 计算完成的episode数量
                if num_episodes>0:  # 如果有完成的episode
                    logger.log_rewards(infos["episode"], num_episodes)  # 记录奖励
                    
        # 如果达到奖励记录步数，则打印奖励
        elif i==stop_rew_log:
            logger.print_rewards()
   
# 程序入口点
if __name__ == '__main__':
    # 设置全局标志
    EXPORT_POLICY = True  # 是否导出策略
    CONTROL_ROBOT = False  # 是否控制机器人
    RECORD_FRAMES = False  # 是否录制帧
    MOVE_CAMERA = False  # 是否移动相机

    # 获取命令行参数
    args = get_args()
    args.num_envs = 1  # 设置环境数量为1
    # 设置加载运行的路径
    #args.load_run = "/home/yue/noetix_n2_gym_test/noetix_n2_gym/logs/n2/1111_19-05-45_"
    if args.checkpoint is None:
        args.checkpoint = -1  # 未指定时加载最新；显式 --checkpoint 会被保留
    
    # 调用播放函数
    play(args)
