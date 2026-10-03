from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2_18DofCfg(LeggedRobotCfg):
    """
    N2人形机器人的配置类
    """
    class init_state(LeggedRobotCfg.init_state):
        """初始状态配置"""
        # 机器人初始位置 [x, y, z]
        pos = [0.0, 0.0, 0.75]
        
        # 默认关节角度配置（注释掉的是只有腿部关节的版本）
        # default_joint_angles = {
        #     "L_leg_hip_yaw_joint": 0.,
        #     "L_leg_hip_roll_joint": 0.,
        #     "L_leg_hip_pitch_joint": -0.1495,
        #     "L_leg_knee_joint": 0.3215,
        #     "L_leg_ankle_joint": -0.1720,
        #     "R_leg_hip_yaw_joint": 0.,
        #     "R_leg_hip_roll_joint": 0.,
        #     "R_leg_hip_pitch_joint": -0.1495,
        #     "R_leg_knee_joint": 0.3215,
        #     "R_leg_ankle_joint": -0.1720,
        # }
        
        # 包含手臂和腿部关节的默认角度配置
        default_joint_angles = {
            # 左臂关节
            "L_arm_shoulder_pitch_joint": 0.,
            "L_arm_shoulder_roll_joint": 0.2,
            "L_arm_shoulder_yaw_joint": 0.,
            "L_arm_elbow_joint": 0.,
           
            # 左腿关节
            "L_leg_hip_yaw_joint": 0.,
            "L_leg_hip_roll_joint": 0.,
            "L_leg_hip_pitch_joint": -0.1495,
            "L_leg_knee_joint": 0.3215,
            "L_leg_ankle_joint": -0.1720,

            # 右臂关节
            "R_arm_shoulder_pitch_joint": 0.,
            "R_arm_shoulder_roll_joint": -0.2,
            "R_arm_shoulder_yaw_joint": 0.,
            "R_arm_elbow_joint": 0.,
            
            # 右腿关节
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_joint": -0.1720,
        }

    class env(LeggedRobotCfg.env):
        """环境配置"""
        # 并行环境数量
        num_envs = 4096

        # 单次观测维度（注释掉的是帧堆叠相关配置）
        frame_stack = 10
        num_single_obs = 63 #39
        num_privileged_obs = 221 #77 + 96 
        """
        num_single_obs = 39
        num_privileged_obs = 77 + 96 
        """

        # 观测空间维度（注释掉的是帧堆叠版本）
        num_observations = int(frame_stack * num_single_obs)
        # num_observations = num_single_obs
        
        # 动作空间维度（18个自由度，注释中显示之前是10）
        num_actions = 18 #10

        # 是否启用早期终止机制
        enable_early_termination = True
        
        # 终止高度阈值
        termination_height = 0.4
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        """域随机化配置"""
        # 是否随机化增益参数
        randomize_gains = True
        # P增益范围
        p_gain_range = [0.8, 1.2]
        # D增益范围
        d_gain_range = [0.8, 1.2]

        # 是否随机化电机强度
        randomize_motor_strength = True
        # 电机强度范围
        motor_strength_range = [0.8, 1.2]

        # 是否随机化质心偏移
        randomize_com_displacement = True
        # 质心偏移范围
        com_displacement_range = [-0.05, 0.05]

        # 是否随机化摩擦系数
        randomize_friction = True
        # 摩擦系数范围
        friction_range = [0.1, 2.]

        # 是否随机化恢复系数
        randomize_restitution = True
        # 恢复系数范围
        restitution_range = [0., 1.]

        # 是否随机化基础质量
        randomize_base_mass = True
        # 添加质量范围
        added_mass_range = [-5., 5.]

        # 是否启用扰动
        disturbance = True
        # 推力范围
        push_force_range = [50.0, 300.0]
        # 推扭矩范围
        push_torque_range = [25.0, 100.0]
        # 扰动概率
        disturbance_probabilities = 0.002
        # 扰动间隔（单位：dt * decimation 毫秒）
        disturbance_interval = [10, 25] # * dt * decimation ms
    
    class control(LeggedRobotCfg.control):
        """控制配置"""
        # PD控制器参数:
        # 控制类型为位置控制
        control_type = 'P'
        
        # 刚度参数（注释掉的是只有腿部关节的版本）
        # stiffness = {
        #     'leg_hip_yaw_joint': 80.0, 'leg_hip_roll_joint': 80.0, 'leg_hip_pitch_joint': 120.0,
        #     'leg_knee_joint': 120.0, 'leg_ankle_joint': 20.0
        # }
        # 阻尼参数（注释掉的是只有腿部关节的版本）
        # damping = {
        #     'leg_hip_yaw_joint': 5.0, 'leg_hip_roll_joint': 5.0, 'leg_hip_pitch_joint': 5.0,
        #     'leg_knee_joint': 5.0, 'leg_ankle_joint': 1.0
        # }
        
        # 包含手臂和腿部关节的刚度参数
        stiffness = {

            # 手臂关节刚度
            "arm_shoulder_pitch_joint": 30.0, 
            "arm_shoulder_roll_joint": 30.0, 
            "arm_shoulder_yaw_joint": 30.0, 
            "arm_elbow_joint": 30.0,

            # 腿部关节刚度
            'leg_hip_yaw_joint': 80.0, 
            'leg_hip_roll_joint': 80.0, 
            'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 
            'leg_ankle_joint': 20.0
        }
        
        # 包含手臂和腿部关节的阻尼参数
        damping = {
            # 手臂关节阻尼
            "arm_shoulder_pitch_joint": 1.0, 
            "arm_shoulder_roll_joint": 1.0, 
            "arm_shoulder_yaw_joint": 1.0, 
            "arm_elbow_joint": 1.0,

            # 腿部关节阻尼
            'leg_hip_yaw_joint': 5.0, 
            'leg_hip_roll_joint': 5.0, 
            'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 
            'leg_ankle_joint': 2.0
        }
        
        # 动作缩放因子: 目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        
        # 降采样: 每个策略DT中的控制动作更新次数 @ 仿真DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        """仿真配置"""
        # 仿真时间步长
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        """资产配置"""
        # URDF文件路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2/urdf/N2.urdf'
        # 机器人名称
        name = "Ning"
        # 足部链接名称
        foot_name = "ankle"
        
        # 对哪些部位的接触进行惩罚
        penalize_contacts_on = ["hip", "knee"]
        # 在哪些部位接触时终止仿真
        terminate_after_contacts_on = ["base"]
        # 自碰撞检测设置（1表示禁用，0表示启用）
        self_collisions = 1  
        # 是否翻转视觉附件
        flip_visual_attachments = False
        # 是否用胶囊体替换圆柱体
        replace_cylinder_with_capsule = False
        # 是否固定基座链接
        fix_base_link = False
    
    class terrain(LeggedRobotCfg.terrain):
        """地形配置"""
        # 网格类型（平面或三角网格）
        mesh_type = 'trimesh' # plane trimesh
        # 是否启用课程学习
        curriculum = False
        
        # 仅用于复杂地形:
        # 是否测量高度
        measure_heights = True
        # X方向测量点
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 1mx1.6m rectangle (without center line)
        # Y方向测量点
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
        # 静摩擦系数
        static_friction = 0.6
        # 动摩擦系数
        dynamic_friction = 0.6
        # 地形长度
        terrain_length = 4.
        # 地形宽度
        terrain_width = 4.

        # 地形行数（级别）
        num_rows = 10  
        # 地形列数（类型）
        num_cols = 10  
        # 初始地形等级
        max_init_terrain_level = 0 #10  
        # 地形比例分布 [平面; 障碍物; 均匀; 上坡; 下坡, 上楼梯, 下楼梯]
        # terrain_proportions = [0.1, 0.0, 0.05, 0.05, 0.05, 0.45, 0.2]
        terrain_proportions = [1., 0., 0., 0., 0., 0., 0.]
        # 恢复系数
        restitution = 0.
    
    class rewards:
        """奖励配置"""
        # 关节位置软限制
        soft_dof_pos_limit = 0.9
        # 基础高度目标
        base_height_target = 0.698
        # 最大接触力（超过此值会被惩罚）
        max_contact_force = 300. 
        # 是否只允许正奖励（如果为True，则负总奖励会被截断为零）
        only_positive_rewards = True 
        
        class scales:
            """奖励缩放因子"""
            # 速度跟踪奖励
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.0
            
            
            # 基础姿态奖励
            orientation = 1.0
            base_height = -30.0
            
            # 步态风格奖励
            stand_still = -0.15
            feet_air_time = 2.
            default_joint_pos = 1.0
            default_up_joint_pos = 1.1 #0.5
            #feet_clearance = 0.01
            
            # 接触奖励
            feet_contact = 1.0
            contact_no_vel = -2
            feet_contact_forces = -0.05
            
            # 能耗奖励
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_smoothness = -0.01
            
            # 其他奖励
            collision = 0.0
            dof_pos_limits = -5.0

    class noise:
        """噪声配置"""
        # 是否添加噪声
        add_noise = True
        # 噪声等级（1表示按其他值缩放）
        noise_level = 1.0    

        class noise_scales:
            """各类噪声的缩放因子"""
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.05
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1

    class commands:
        """命令配置"""
        # 是否启用课程学习
        curriculum = False
        # 最大课程学习等级
        max_curriculum = 1.
        # 命令数量（默认：lin_vel_x, lin_vel_y, ang_vel_yaw, heading）
        num_commands = 4 
        # 命令重新采样时间间隔[s]
        resampling_time = [5, 15] 
        # 是否启用航向命令模式（如果为True，则从航向误差计算角速度命令）
        heading_command = False
        # 最小命令速度
        min_cmd_vel = 0.2
        
        class ranges:
            """命令范围配置"""
            # 线速度x方向范围 [m/s]
            lin_vel_x = [-0.8, 0.8] 
            # 线速度y方向范围 [m/s]
            lin_vel_y = [-0.5, 0.5]   
            # 角速度yaw方向范围 [rad/s]
            ang_vel_yaw = [-1.0, 1.0]    
            # 航向范围
            heading = [-3.14, 3.14]

    class normalization:
        """归一化配置"""
        class obs_scales:
            """观测值缩放因子"""
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        # 观测值裁剪阈值
        clip_observations = 18.
        # 动作裁剪阈值
        clip_actions = 18.

class N2_18DofCfgPPO(LeggedRobotCfgPPO):
    """PPO算法配置类"""
    # 运行器类名
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        """策略网络配置"""
        # 类名（注释掉的是另一种配置）
        # class_name = 'ActorCritic' 

        # actor_hidden_dims = [1024, 256, 128]
        # critic_hidden_dims = [768, 256, 128]
        
        # 初始化噪声标准差（注释掉的是另一种配置）
        # init_noise_std = 1.0
        # 激活函数（可以是 elu, relu, selu, crelu, lrelu, tanh, sigmoid）

        class_name = "ActorCritic"
        init_noise_std = 1.0
        # Actor网络隐藏层维度
        actor_hidden_dims = [256, 128]
        # Critic网络隐藏层维度
        critic_hidden_dims = [256, 128]
        # 激活函数
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        """算法配置"""
        class_name = "PPO"
        # 熵系数
        entropy_coef = 0.01

        # 对称性设置（注释掉的是未使用的配置）
        # symmetry_cfg = {
        #     "use_data_augmentation": False,
        #     "use_mirror_loss": True,
        #     "data_augmentation_func": "humanoid.envs.n2.n2_sym_utils:data_augmentation_func_n2",
        #     "mirror_loss_coeff": 1.0
        # }

    class runner( LeggedRobotCfgPPO.runner ):
        """运行器配置"""
        # 最大迭代次数
        max_iterations = 20000
        # 保存间隔
        save_interval = 200 
        # 运行名称
        run_name = ''
        # 实验名称
        experiment_name = 'n2'