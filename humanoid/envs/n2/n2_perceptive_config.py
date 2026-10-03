from humanoid.envs.n2.n2_10dof_config import N2_10dof_Cfg, N2_10dof_CfgPPO

class N2PerceptiveCfg(N2_10dof_Cfg):
    class env(N2_10dof_Cfg.env):
        # 每帧 = 本体感知 39 + 高度图 96(= measured_points_x 12 × measured_points_y 8)
        frame_stack = 10
        num_single_obs = 39 + 96
        num_observations = int(frame_stack * num_single_obs)

    class terrain(N2_10dof_Cfg.terrain):
        measure_heights = True              # blind 里本来也是 True,显式写上保险
        debug_viz = True
        curriculum = True

        # 初始地形等级
        max_init_terrain_level = 0 #10
        # [平地;离散障碍;均匀;上坡;下坡;金字塔上/下楼梯;直行上/下楼梯]，累加须为 1.0。
        terrain_proportions = [0.0, 0.15, 0.0, 0.0, 0.0, 0.3, 0.25, 0.20, 0.1]

        curriculum_up_distance = 3.2


        terrain_length = 8.
        terrain_width = 8.

        stairs_platform_size = 1.5

    class commands(N2_10dof_Cfg.commands):
        # 站立指令比例。
        standing_prob = 0.05

        stairs_forward_only = False
        # 楼梯列前进速度重采样下限(m/s)。
        stairs_min_vx = 0.25

    class domain_rand(N2_10dof_Cfg.domain_rand):

        refresh_shape_props_on_reset = False

    class noise(N2_10dof_Cfg.noise):
        class noise_scales(N2_10dof_Cfg.noise.noise_scales):
            height_measurements = 0.0       # privileged = 干净真值,必须为0

    class rewards(N2_10dof_Cfg.rewards):
        foothold_depth_tol = 0.04
        # _reward_foothold 的平滑系数：raw = 1 - exp(-k * 平均悬空深度)。
        # k=20 使旧阈值 4cm 落在 0.55 附近(动态范围中段)。
        foothold_flat_k = 20.0
        # foot sole footprint used to lay out the n sample points (metres).
        # N2 "ankle" foot ~0.20 x 0.10; tune to your collision mesh.
        foot_length = 0.20
        foot_width = 0.10
        foot_n_x = 3  # samples along length
        foot_n_y = 2  # samples along width  → n = 6 per foot

        # yaw_ref 相对实际 yaw 的泄漏钳制上限(rad)。
        world_heading_max_err = 1.57
        # _reward_stand_still = sum|dof_pos-default| + w*sum(dof_vel²)，速度项二次且无上界。
        stand_still_vel_weight = 0.05
        stand_still_max = 20.0


        # _reward_anti_freeze 的饱和阈值(m/s)：命令方向的世界系前进速度达到即拿满分。
        anti_freeze_speed = 0.15


        max_contact_force = 400.
        # 每只脚超出部分的上限(N)。基类不封顶，摔倒砸地时单步惩罚能盖过整个正项栈。
        feet_contact_force_max_excess = 200.

        # _reward_world_progress 归一化分母的下限(m/s)。min_cmd_vel 只约束三维指令模长，
        # wz 主导的指令仍可能让 |v_xy| 接近 0，故给分母兜底。
        world_progress_min_speed = 0.1

        class scales(N2_10dof_Cfg.rewards.scales):

            foothold = -0.5

            feet_air_time = 1.0

            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.8


            world_progress = 0.8
            world_heading = 1.0

            # 与 tracking_lin_vel 同量级
            anti_freeze = 0.7

            collision = -1.0

            stumble = -2.5


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'
        # 高度图占每帧 71% 且幅度远大于其余维度，基类默认 Identity 归一化不够用。
        empirical_normalization = True