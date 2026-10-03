from humanoid.envs.n2.n2_perceptive_config import N2PerceptiveCfg, N2PerceptiveCfgPPO


class N2ParkourCfg(N2PerceptiveCfg):
    """Extreme Parkour(arXiv:2309.14341)架构的配置。

    独立于 n2_perceptive 注册成新任务，不动原有的感知任务——后者已经有一份能训到
    terrain_level 5.49 的成果，不该被这次架构实验波及。

    与 n2_perceptive 的三处根本差异（都是 EP 的要件）：
      指令：只有前进速度，lin_vel_y 和 ang_vel_yaw 都是 [0,0]
      目标：goal 路点位置，而不是速度指令方向
      地形：中央通道 + y 向 pad + goal 落在障碍上，横向绕行被几何堵死
    """

    class env(N2PerceptiveCfg.env):
        # 39 本体 + 2 到goal的delta-yaw(cos/sin) + 96 高度 = 137
        frame_stack = 10
        num_single_obs = 39 + 2 + 96
        num_observations = int(frame_stack * num_single_obs)
        # 特权观测同步 +2
        num_privileged_obs = N2PerceptiveCfg.env.num_privileged_obs + 2

    class terrain(N2PerceptiveCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        max_init_terrain_level = 0

        # 与 EP 同为 18m。台阶类只用到约 5.7m(2.5 平台 + 8 级)，其余是末端平台——EP
        # 自己也是这样。但跨栏/踏石的障碍间距 1.2~1.8m x 8 装不进 8m 的块，末尾几个
        # goal 会被挤重合，而重合的 goal 会让 _update_goals 一步连推两格、课程虚高。
        terrain_length = 18.
        terrain_width = 4.
        num_rows = 10
        num_cols = 10


        # ParkourTerrain.TYPES：[上台阶, 下台阶, 跨栏, 平地路点, 踏石]。
        terrain_proportions = [0.3, 0.3, 0.2, 0.1, 0.1]

        # ---- 台阶级数与 goal 分布(索引 0/1 共用) ----
        # 18 级正好是真实楼层的一个梯段(一层约 3m / 16~18 级)。级数与 goal 数解耦：
        # goals 表对所有地形类型形状固定(num_goals=10 -> 8 个落在台阶上)，均匀撒会让
        # 起步几级也拉开到 2~3 级一个、失去引导，全部一级一个又只能有 8 级。
        parkour_step_count = 18
        # 幂律指数：goal 落在第 1,2,3,4,7,10,14,18 级，间隔 1,1,1,3,3,4,4。
        # 起步密(每级都有反馈)、后面疏(让梯段长起来)。最小间距仍是 1 级 = 踏面下限
        # 0.2m，所以 goal_reach_dist=0.15 的约束不受影响。
        parkour_step_goal_power = 2.0

        # ---- 下台阶(索引 1) ----
        # 通道外沿 x 跟着通道一起降，恒低于同一 x 处的通道这么多(m)。
        parkour_stepdown_outside_margin = 0.3

        # ---- 跨栏 / 平地路点(索引 2/3) ----
        parkour_hurdle_len = 0.3            # 沿 x 的厚度
        parkour_hurdle_x_range = (1.2, 1.8) # EP 是 (1.2,2.2)，收窄以塞进 18m
        parkour_hurdle_y_range = (-0.4, 0.4)
        parkour_hurdle_half_valid_width = (0.4, 0.8)
        # 难度 0 -> 1 的栏高(m)。EP 顶到 0.40，这里封在 0.30：N2 髋高约 0.6m，
        # 0.4m 的栏已经是抬腿到髋，先把梯子架在够得着的高度。
        parkour_hurdle_height_range = [0.10, 0.30]

        # ---- 踏石(索引 4) ----
        # 板子尺寸随难度【缩小】(易 -> 难)。0.9x1.0m 是脚(0.20x0.10m)面积的 45 倍，
        parkour_stone_len_range = [0.9, 0.45]
        parkour_stone_width_range = [1.0, 0.5]

        parkour_stone_gap_range = [0.05, 0.20]
        parkour_stone_y_range = (0.10, 0.18)  # 左右交替偏置，逼迫换脚
        parkour_stone_pit_depth = 0.5       # EP 到 1.0m，双足摔下去代价太大，减半

        measured_points_x = [-0.40, -0.25, -0.10, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30]

        # ---- 以下取自 EP 源码 parkour_step_terrain 的默认值本身 ----
        num_goals = 10                      # 起步平台 + 8 级台阶 + 末端平台 = EP 的 num_stones=8
        parkour_platform_len = 2.5          # EP: platform_len=2.5
        parkour_x_range = (0.2, 0.4)        # EP: x_range=[0.2,0.4]，每级台阶的踏面长度
        parkour_y_range = (-0.15, 0.15)     # EP: y_range=[-0.15,0.15]
        parkour_pad_width = 0.1             # EP: pad_width=0.1，最外缘细边框
        parkour_pad_height = 0.5            # EP: pad_height=0.5

        parkour_step_height_range = [0.05, 0.20]

        parkour_half_valid_width = (0.7, 0.8)
        # 出生点抖动(m)
        parkour_spawn_jitter = 0.3
        # 课程
        parkour_goals_to_level_up = 5
        parkour_goals_to_level_down = 1

    class commands(N2PerceptiveCfg.commands):
        # 本任务整体只发前进指令，不需要按地形列区分。
        stairs_forward_only = False
        # EP 完全不发站立指令；这里留一点点，其余时间在通道里前进。
        standing_prob = 0.02
        zero_vx_prob = 0.0
        zero_wz_prob = 0.0
        # 前进速度下限(m/s)
        parkour_min_vx = 0.3

        class ranges(N2PerceptiveCfg.commands.ranges):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]      # EP: lin_vel_y=[0,0]
            ang_vel_yaw = [0.0, 0.0]    # EP: ang_vel_yaw=[0,0]

    class rewards(N2PerceptiveCfg.rewards):
        # 【必须小于最小 goal 间距】
        goal_reach_dist = 0.15
        # 越过即算到达
        goal_pass_lateral_tol = 0.5

        feet_air_time_stale = 1.0

        # ---- _reward_stumble(N2ParkourEnv 重写为连续量)的三个阈值 ----
        # 全部取自实测接触力分布(model_999，上楼梯 lvl5，自重 327N)：
        #   支撑相 Fz 中位数 207.6N / p90 381N，摆动相 Fz p90 仅 100.7N
        stumble_stance_force = 80.0     # Fz 低于此值才算"摆动中、还没承重"
        #   摆动相 Fxy p90=16.8N 是正常噪声，p99=265N 才是真撞上
        stumble_min_force = 20.0        # 噪声门限
        stumble_ref_force = 200.0       # 归一化参考，使单脚原始值落在 [0,1]

        class scales(N2PerceptiveCfg.rewards.scales):

            tracking_goal_vel = 3.5
            tracking_yaw = 0.5          # EP: tracking_yaw = 0.5
            goal_reached = 0.0          # 非 EP 项，默认关闭


            world_progress = 0.0
            world_heading = 0.0
            anti_freeze = 0.0


            contact_no_vel = -2

            # base 系速度跟踪降权：主目标已由 tracking_goal_vel 承担。
            tracking_lin_vel = 0.5
            tracking_ang_vel = 0.3

            stumble = -6.0


class N2ParkourCfgPPO(N2PerceptiveCfgPPO):
    class runner(N2PerceptiveCfgPPO.runner):
        experiment_name = 'n2_parkour'
        empirical_normalization = True

    class policy(N2PerceptiveCfgPPO.policy):
        # 与 Extreme Parkour / legged_gym 上游一致。
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        # 把 frame_stack x 96 = 960 维高度图压到 32 维再与本体感知拼接(EP 的 scan_encoder)。
        # 不启用时 actor 第一层 1370x512 占参数 91%；启用后主干输入降到 41x10+32=442。
        scan_encoder_dims = [128, 64, 32]
        # 观测布局（必须与 N2ParkourEnv.compute_observations 一致）：
        # 每帧 = [cmd3+angvel3+grav3+dofpos10+dofvel10+act10+goal2 = 41] + [高度96]
        frame_stack = 10
        # 观测布局必须与 N2ParkourEnv.compute_observations 一致：
        # 每帧 = [cmd3+angvel3+grav3+dofpos10+dofvel10+act10+goal2 = 41] + [高度96]。
        # 构造时 assert frame_stack*(n_proprio+n_scan) == num_actor_obs。
        n_proprio = 41
        n_scan = 96


class N2ParkourStabilityCfg(N2ParkourCfg):
    """Reward-only variant for low-rate stability fine-tuning.

    It is registered as a separate task so the original paper-reproduction
    configuration remains reproducible and old checkpoints keep their exact
    semantics.
    """

    class rewards(N2ParkourCfg.rewards):
        # Do not punish harmless gait oscillation around the requested speed.
        overspeed_tolerance = 0.10       # m/s
        # 0.5m/s excess gives raw penalty 1.0; larger spikes are squared.
        overspeed_reference = 0.50       # m/s
        # Bound the term before applying its negative scale.
        overspeed_max_penalty = 4.0

        class scales(N2ParkourCfg.rewards.scales):
            # At +0.7m/s above command this contributes roughly -2.9, enough to
            # counter the saturated +3.5 tracking_goal_vel reward.
            overspeed = -2.0
            # Original values were -5e-3 and -0.05 respectively.
            action_smoothness = -1.5e-2
            feet_contact_forces = -0.10


class N2ParkourStabilityCfgPPO(N2ParkourCfgPPO):
    class algorithm(N2ParkourCfgPPO.algorithm):
        # The reward landscape changed, but the policy is already competent:
        # use conservative fixed-rate fine-tuning rather than another broad
        # exploration phase.
        learning_rate = 3.0e-5
        schedule = 'fixed'
        entropy_coef = 0.003

    class runner(N2ParkourCfgPPO.runner):
        # Keep the same experiment root so --load_run can resolve model_10000.
        experiment_name = 'n2_parkour'
        empirical_normalization = True
        # Adam state is retained, but its checkpoint LR must not override the
        # deliberately lower fine-tuning LR above.
        reset_optimizer_lr_on_resume = True


class N2ParkourCourseCfg(N2ParkourCfg):
    """Deterministic five-stage evaluation course shared with MuJoCo.

    The original ``N2ParkourCfg`` remains the 8x10 mixed-terrain training
    task.  This separate config places exactly one robot on one continuous
    +x runway and can load the same ``n2_parkour`` checkpoints unchanged.
    """

    class env(N2ParkourCfg.env):
        num_envs = 1
        episode_length_s = 300
        test = True

    class terrain(N2ParkourCfg.terrain):
        course_mode = True
        course_row = 3
        course_num_rows = 8
        course_seed = 5
        course_add_roughness = True
        # Isaac-only viewing shoulders remove the high pads from the original
        # 4 m course edge.  A low curb three metres farther out remains a
        # physical boundary while a right-side camera gets an open sightline.
        course_side_margin = 3.0
        course_safety_curb_height = 0.20

        curriculum = False
        max_init_terrain_level = 0
        num_rows = 1
        num_cols = 1
        # Exact cropped length for row=3/seed=5. ParkourCourseTerrain derives
        # and re-validates this from the shared campaign before allocating.
        terrain_length = 59.1
        terrain_width = 4.0
        # g0 once, then g1..g9 for each of five stages.
        num_goals = 46
        parkour_spawn_jitter = 0.0

    class commands(N2ParkourCfg.commands):
        standing_prob = 0.0
        zero_vx_prob = 0.0
        zero_wz_prob = 0.0
        resampling_time = [1000, 1001]
        # Match the conservative per-stage MuJoCo campaign speeds.
        course_command_speeds = [0.50, 0.30, 0.50, 0.50, 0.40]

        class ranges(N2ParkourCfg.commands.ranges):
            lin_vel_x = [0.30, 0.50]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class noise(N2ParkourCfg.noise):
        add_noise = False

    class domain_rand(N2ParkourCfg.domain_rand):
        refresh_shape_props_on_reset = False
        randomize_gains = False
        randomize_motor_strength = False
        randomize_com_displacement = False
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        push_robots = False
        disturbance = False

    class viewer(N2ParkourCfg.viewer):
        # Initial right/rear elevated view; play.py follows the robot thereafter.
        pos = [-2.5, -2.5, 2.8]
        lookat = [2.0, 2.0, 0.8]


class N2ParkourCourseCfgPPO(N2ParkourCfgPPO):
    class runner(N2ParkourCfgPPO.runner):
        # Resolve and load existing parkour runs/checkpoints; never create a
        # second incompatible experiment root for this evaluation-only task.
        experiment_name = 'n2_parkour'
        empirical_normalization = True


class N2ParkourSlowStableCfg(N2ParkourCfg):
    """Fresh-training variant prioritising stable, slow, straight locomotion.

    The paper-reproduction task remains unchanged.  This task deliberately
    uses a separate log root because its self-collision physics and reward
    semantics are incompatible with the old checkpoints.
    """

    class asset(N2ParkourCfg.asset):
        # Isaac's actor filter 1 disables all self-collision.  The explicit
        # cross-leg mode creates with filter 0, then applies bit masks so only
        # left-leg versus right-leg contacts remain.  This stops leg
        # interpenetration without unstable same-chain adjacent-mesh contacts.
        self_collisions = 0
        self_collision_mode = 'cross_leg'
        require_self_collisions = True

    class sim(N2ParkourCfg.sim):
        class physx(N2ParkourCfg.sim.physx):
            # Cross-leg broadphase pairs raise the aggregate-pair demand well
            # above the old all-self-collision-disabled task.  At 2048 envs
            # PhysX reports 14,804,791 required pairs; the inherited 2**23
            # capacity silently drops interactions, while 2**24 covers it.
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 5

    class terrain(N2ParkourCfg.terrain):
        # The legacy flat task deliberately zig-zags its goals.  Keep that
        # reproduction untouched and centre only this stable variant.
        parkour_flat_center_goals = True
        parkour_spawn_jitter = 0.10
        # Require sustained progress before moving to a harder terrain row.
        parkour_goals_to_level_up = 7
        parkour_goals_to_level_down = 2

    class commands(N2ParkourCfg.commands):
        min_cmd_vel = 0.10
        parkour_min_vx = 0.20

        class ranges(N2ParkourCfg.commands.ranges):
            lin_vel_x = [0.20, 0.50]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(N2ParkourCfg.domain_rand):
        # Start with realistic but non-trivial variation.  Large 300 N pushes,
        # 0.1 friction and +/-5 kg payloads are reserved for a later robustness
        # fine-tune after the gait itself is reliable.
        refresh_shape_props_on_reset = False
        randomize_gains = True
        p_gain_range = [0.90, 1.10]
        d_gain_range = [0.90, 1.10]
        randomize_motor_strength = True
        motor_strength_range = [0.90, 1.10]
        randomize_com_displacement = True
        com_displacement_range = [-0.02, 0.02]
        randomize_friction = True
        friction_range = [0.50, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 0.20]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        push_robots = False
        disturbance = False

    class rewards(N2ParkourCfg.rewards):
        overspeed_tolerance = 0.05
        overspeed_reference = 0.30
        overspeed_max_penalty = 4.0
        feet_separation_min = 0.08
        feet_separation_max_penalty = 4.0

        class scales(N2ParkourCfg.rewards.scales):
            # Goal following remains primary, but no longer overwhelms balance,
            # heading and commanded-speed tracking once it saturates.
            tracking_goal_vel = 2.8
            tracking_lin_vel = 1.0
            tracking_yaw = 1.0
            tracking_ang_vel = 0.6

            overspeed = -1.5
            action_smoothness = -1.5e-2
            feet_contact_forces = -0.10
            feet_separation = -0.30

            # Reduce the incentive for hurried long aerial phases and allow
            # more double-support time for a visibly calmer gait.
            feet_air_time = 0.5
            feet_contact = 0.6


class N2ParkourSlowStableCfgPPO(N2ParkourCfgPPO):
    class policy(N2ParkourCfgPPO.policy):
        init_noise_std = 0.8

    class algorithm(N2ParkourCfgPPO.algorithm):
        learning_rate = 5.0e-4
        schedule = 'adaptive'
        entropy_coef = 0.008

    class runner(N2ParkourCfgPPO.runner):
        experiment_name = 'n2_parkour_slow_stable'
        empirical_normalization = True
        save_interval = 200
        max_iterations = 15001


class N2ParkourSlowStableCourseCfg(N2ParkourSlowStableCfg):
    """Deterministic five-stage course for the new stable policy."""

    class env(N2ParkourSlowStableCfg.env):
        num_envs = 1
        episode_length_s = 300
        test = True

    class terrain(N2ParkourSlowStableCfg.terrain):
        course_mode = True
        course_row = 3
        course_num_rows = 8
        course_seed = 5
        course_add_roughness = True
        # MuJoCo's interpolated hfield riser consumes one 0.1 m cell. Give
        # both evaluation engines the same one-cell-deeper 0.30/0.40 m treads.
        # Mixed-terrain training keeps the harder paper range above.
        course_step_x_range = (0.30, 0.50)
        parkour_flat_center_goals = True
        course_side_margin = 3.0
        course_safety_curb_height = 0.20

        curriculum = False
        max_init_terrain_level = 0
        num_rows = 1
        num_cols = 1
        terrain_length = 62.7
        terrain_width = 4.0
        num_goals = 46
        parkour_spawn_jitter = 0.0

    class commands(N2ParkourSlowStableCfg.commands):
        standing_prob = 0.0
        zero_vx_prob = 0.0
        zero_wz_prob = 0.0
        resampling_time = [1000, 1001]
        course_command_speeds = [0.40, 0.25, 0.40, 0.35, 0.30]

        class ranges(N2ParkourSlowStableCfg.commands.ranges):
            lin_vel_x = [0.20, 0.40]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class noise(N2ParkourSlowStableCfg.noise):
        add_noise = False

    class domain_rand(N2ParkourSlowStableCfg.domain_rand):
        refresh_shape_props_on_reset = False
        randomize_gains = False
        randomize_motor_strength = False
        randomize_com_displacement = False
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        push_robots = False
        disturbance = False

    class viewer(N2ParkourSlowStableCfg.viewer):
        pos = [-3.5, -3.0, 3.1]
        lookat = [2.0, 2.0, 0.8]


class N2ParkourSlowStableCourseCfgPPO(N2ParkourSlowStableCfgPPO):
    class runner(N2ParkourSlowStableCfgPPO.runner):
        experiment_name = 'n2_parkour_slow_stable'
        empirical_normalization = True
