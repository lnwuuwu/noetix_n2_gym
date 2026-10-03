from humanoid.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class N2Env(LeggedRobot):
    """N2机器人环境类，继承自LeggedRobot基类"""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """初始化N2环境
        
        Args:
            cfg: 环境配置参数
            sim_params: 仿真参数
            physics_engine: 物理引擎类型
            sim_device: 仿真设备
            headless: 是否无头模式运行
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 根据关节数量设置不同的关节索引
        if self.num_dofs == 10:
            # 10自由度配置：左右脚踝偏航和滚动关节索引
            self.left_yaw_roll  = [0, 1]        # 左脚踝偏航和滚动关节索引
            self.right_yaw_roll  = [5, 6]       # 右脚踝偏航和滚动关节索引
            self.ankle_dof_idxs = [4, 9]        # 脚踝关节索引
        elif self.num_dofs == 18:
            # 18自由度配置：包含上下肢关节索引
            self.left_yaw_roll  = [4, 5]        # 左脚踝偏航和滚动关节索引
            self.right_yaw_roll  = [13, 14]     # 右脚踝偏航和滚动关节索引
            self.up_left_yaw_roll  = [1, 2]     # 左上肢偏航和滚动关节索引
            self.up_right_yaw_roll  = [10, 11]  # 右上肢偏航和滚动关节索引
            self.ankle_dof_idxs = [8, 17]       # 脚踝关节索引
            self.up_joint_idxs = [0,1,2,3, 9,10,11,12]
        elif self.num_dofs == 20:
            # 20自由度配置：更完整的关节索引
            self.left_yaw_roll  = [4, 5]        # 左脚踝偏航和滚动关节索引
            self.right_yaw_roll  = [14, 15]     # 右脚踝偏航和滚动关节索引
            self.up_left_yaw_roll  = [1, 2]     # 左上肢偏航和滚动关节索引
            self.up_right_yaw_roll  = [11, 12]  # 右上肢偏航和滚动关节索引
            self.ankle_dof_idxs = [8, 9, 18, 19] # 脚踝关节索引
        else:
            assert False, "Unsupported num joints"  # 不支持的关节数量

    def _get_noise_scale_vec(self, cfg):
        """ 设置用于缩放添加到观测值中的噪声的向量
            [注意]: 当更改观测结构时必须进行调整

        Args:
            cfg (Dict): 环境配置文件

        Returns:
            [torch.Tensor]: 用于乘以[-1, 1]范围内均匀分布的缩放向量
        """
        # 根据是否使用帧堆叠初始化噪声向量
        if self.cfg.env.frame_stack is not None:
            noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        else:
            noise_vec = torch.zeros_like(self.obs_buf[0])
            
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # 为不同观测值设置噪声缩放因子
        noise_vec[:3] = 0. # 命令噪声
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 角速度噪声
        noise_vec[6:9] = noise_scales.gravity * noise_level  # 重力噪声
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 关节位置噪声
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 关节速度噪声
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # 上一动作噪声

        return noise_vec

    def _init_foot(self):
        """初始化足部相关状态变量"""
        self.feet_num = len(self.feet_indices)  # 足部数量
        
        # 获取刚体状态张量
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # 重塑张量为[环境数, 刚体数, 13]的视图
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        # 提取足部状态
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        # 提取足部位置、姿态和速度
        self.feet_pos = self.feet_state[:, :, :3]      # 位置
        self.feet_quat = self.feet_state[:, :, 3:7]    # 四元数姿态
        self.feet_vel = self.feet_state[:, :, 7:10]    # 线速度
        self.last_feet_vel = self.feet_vel.clone()     # 上一时刻足部速度
        
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()  # 调用父类缓冲区初始化
        self._init_foot()        # 初始化足部状态

    def update_feet_state(self):
        """更新足部状态"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # 刷新刚体状态张量
        
        # 更新足部状态信息
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.last_feet_vel = self.feet_vel.clone()
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        """物理步骤后的回调函数"""
        self.update_feet_state()  # 更新足部状态
        return super()._post_physics_step_callback()
    
    def _resample_commands(self, env_ids):
        """ 随机选择某些环境的命令

        Args:
            env_ids (List[int]): 需要新命令的环境ID列表
        """
        # 为指定环境随机生成线速度命令
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 根据配置选择生成航向命令或偏航角速度命令
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 为部分环境设置特殊命令规则
        if len(env_ids) > 0:
            random_tensor = torch.rand_like(self.commands[env_ids, 0])
            # 20%概率设置所有命令为0
            self.commands[env_ids[random_tensor < 0.20], :] = 0
            # 10%概率只设置x方向线速度为0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.20, random_tensor < 0.30)], 0] = 0
            # 10%概率只设置偏航角速度为0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.30, random_tensor < 0.40)], 2] = 0

        # 将小命令设置为零
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > self.min_cmd_vel).unsqueeze(1)
    
    def compute_observations(self):
        """ 计算观测值 """
        # 构建基础观测缓冲区
        obs_buf = torch.cat((  
                                self.commands[:, :3] * self.commands_scale,      # 缩放后的命令
                                self.base_ang_vel  * self.obs_scales.ang_vel,    # 基座角速度
                                self.projected_gravity,                          # 投影重力
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
                                self.dof_vel * self.obs_scales.dof_vel,          # 关节速度
                                self.actions                                     # 当前动作
                            ),dim=-1)
        
        # 构建特权观测缓冲区（包含更多信息）
        self.privileged_obs_buf = torch.cat((  
                                    self.commands[:, :3] * self.commands_scale,   # 缩放后的命令
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 基座角速度
                                    self.projected_gravity,                       # 投影重力
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
                                    self.dof_vel * self.obs_scales.dof_vel,       # 关节速度
                                    self.actions,                                 # 当前动作
                                    self.base_lin_vel * self.obs_scales.lin_vel,  # 基座线速度
                                    self.payload * 0.5,                           # 负载信息
                                    self.friction_coeffs,                         # 摩擦系数
                                    self.restitution_coeffs,                      # 恢复系数
                                    self.Kp_factors,                              # Kp因子
                                    self.Kd_factors,                              # Kd因子
                                    self.motor_strength,                          # 电机强度
                                    self.contacts                                 # 接触信息
                                ),dim=-1)
        
        # 如果配置了地形高度测量，添加高度信息到特权观测
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # 根据需要添加噪声
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        
        # 处理帧堆叠
        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            # 堆叠历史观测
            obs_buf_all = torch.stack([self.obs_history[i]
                                    for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        else:
            self.obs_buf = obs_now
    

# ================================================ Rewards ================================================== #
    def _reward_tracking_lin_vel(self):
        """线性速度跟踪奖励"""
        # 计算xy轴线速度命令跟踪误差
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 站立命令时的线速度误差
        stand_lin_vel_error = torch.norm(-self.base_lin_vel[self.standing_cmd, :2], dim=1)
        # 指数衰减奖励
        rew = torch.exp(-lin_vel_error * 5)
        rew[self.standing_cmd] = torch.exp(-stand_lin_vel_error * 5)
        return rew
    
    def _reward_tracking_ang_vel(self):
        """角速度跟踪奖励"""
        # 计算偏航角速度命令跟踪误差
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * 5)
    
    def _reward_low_speed(self):
        """
        根据机器人速度相对于命令速度的奖励或惩罚。
        检查机器人移动速度过慢、过快或达到期望速度，
        以及移动方向是否与命令匹配。
        """
        # 计算速度和命令的绝对值用于比较
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # 定义期望速度范围的标准
        speed_too_low = absolute_speed < 0.5 * absolute_command   # 速度过低
        speed_too_high = absolute_speed > 1.2 * absolute_command  # 速度过高
        speed_desired = ~(speed_too_low | speed_too_high)         # 速度在期望范围内

        # 初始化奖励张量
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # 根据条件分配奖励
        reward[speed_too_low] = -1.0    # 速度过低惩罚
        reward[speed_too_high] = 0.     # 速度过高无奖励
        reward[speed_desired] = 1.2     # 速度在期望范围内奖励
        return reward * (self.commands[:, 0].abs() > self.min_cmd_vel)
    
    def _reward_feet_contact(self):
        """足部接触奖励"""
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 接触滤波器
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        # 单足接触
        single_feet_contact = torch.logical_xor(contact_filt[:, 0], contact_filt[:, 1]) 
        # 双足接触
        both_feet_contact = torch.logical_and(contact_filt[:, 0], contact_filt[:, 1])

        # 更新双足接触时间
        self.feet_both_contact_time[both_feet_contact] += self.dt
        self.feet_both_contact_time *= both_feet_contact
        
        # 奖励条件：单足接触或双足接触时间小于0.2秒，或站立命令
        rew_filter = torch.logical_or(single_feet_contact, self.feet_both_contact_time < 0.2)
        rew_filter = torch.logical_or(rew_filter, self.standing_cmd)
        rew[rew_filter] = 1.0
        return rew
    
    def _reward_contact_no_vel(self):
        """无速度接触惩罚"""
        # 惩罚有接触但无速度的情况
        contact_feet_vel = self.feet_vel * self.contacts.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_default_joint_pos(self):
        """默认关节位置奖励"""
        # 计算关节位置与默认位置的偏差
        joint_diff = self.dof_pos - self.default_dof_pos

        # 计算左右脚踝偏航和滚动关节的偏差
        left_yaw_roll = joint_diff[:, self.left_yaw_roll]
        right_yaw_roll = joint_diff[:, self.right_yaw_roll]

        # 计算偏航和滚动关节的范数
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        # 指数衰减奖励减去偏差惩罚
        rew = torch.exp(-yaw_roll * 100) - 0.05 * torch.norm(joint_diff[:, self.left_yaw_roll + self.right_yaw_roll], dim=1)
        # 奖励条件：y或偏航命令非零，或站立命令
        rew_filter = torch.logical_or(torch.any(torch.abs(self.commands[:, [1, 2]]) > self.min_cmd_vel, dim=1), self.standing_cmd) 
        
        rew[rew_filter] = 1.0
        return rew
    
    def _reward_default_up_joint_pos(self):
        """上肢默认关节位置奖励"""
        # 计算关节位置与默认位置的偏差
        joint_diff = self.dof_pos - self.default_dof_pos
        joint_diff = joint_diff[:, self.up_joint_idxs]
        # 指数衰减奖励减去偏差惩罚
        rew = torch.exp(-torch.norm(joint_diff, dim=1) * 5) - 0.05 * torch.norm(joint_diff, dim=1)
        return rew
    
    def _reward_stand_still(self):
        """静止站立惩罚"""
        # 惩罚零命令时的运动
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 对于站立命令，惩罚关节位置偏差和关节速度
        rew[self.standing_cmd] = torch.sum(torch.abs(self.dof_pos[self.standing_cmd] - self.default_dof_pos), dim=1) 
        rew[self.standing_cmd] += torch.sum(torch.square(self.dof_vel[self.standing_cmd]), dim=1)
        return rew 
    
    def _reward_center_of_mass(self):
        """质心奖励"""
        # 获取左右足接触状态
        left_contact, right_contact = self.contacts[:, 0], self.contacts[:, 1]
        # 计算总接触数（防止除零）
        total_contact = (torch.sum(self.contacts, dim=-1)).clamp(min=1e-5)

        # 计算接触点的空间位置
        p_csp = (left_contact.unsqueeze(-1) * self.feet_pos[:, 0] +
                 right_contact.unsqueeze(-1) * self.feet_pos[:, 1]) / total_contact.unsqueeze(-1)
        # 计算质心与根状态的差异
        diff = p_csp[:, :2] - self.root_states[:, :2].clone()

        # 指数衰减奖励
        rew = torch.exp(-torch.norm(diff, dim=1).pow(2) * 50)
        rew[total_contact == 0] = 0.0  # 无接触时奖励为0
        return rew

    def _reward_feet_orientation(self):        
        """足部姿态奖励"""
        # 提取足部四元数姿态
        feet_quat = self.feet_state[:, :, 3:7]
        foot_quat_1 = feet_quat[:, 0, :]  # 左足
        foot_quat_2 = feet_quat[:, 1, :]  # 右足
        
        # 获取欧拉角（仅使用俯仰角）
        lfoot_pitch = get_euler_xyz(foot_quat_1)[1]
        rfoot_pitch = get_euler_xyz(foot_quat_2)[1]
        
        # 规范化角度到[-π, π]范围
        lfoot_pitch[lfoot_pitch > torch.pi] -= 2*torch.pi
        lfoot_pitch[lfoot_pitch < -torch.pi] += 2*torch.pi
        rfoot_pitch[rfoot_pitch > torch.pi] -= 2*torch.pi
        rfoot_pitch[rfoot_pitch < -torch.pi] += 2*torch.pi
        
        # 组合足部姿态
        feet_orient = torch.cat((lfoot_pitch.unsqueeze(1), rfoot_pitch.unsqueeze(1)), dim=-1)
        # 指数衰减奖励
        return torch.exp(-torch.norm(feet_orient * self.contacts, dim=1) * 20)
    #
    # def _reward_feet_clearance(self):
    #     # 每只脚相对其脚下地形的高度,仅在摆动相
    #     feet_z = self.feet_pos[:, :, 2] - self.terrain_h.unsqueeze(-1)  # 地形相对
    #     swing = ~self.contacts  # 未接触的脚
    #     target = 0.18
    #     # 摆动时鼓励接近目标间隙;忽略支撑脚
    #     err = torch.square(torch.clamp(target - feet_z, min=0.0)) * swing
    #     return torch.exp(-err.sum(dim=1) * 50)