import torch.nn.functional as F
import math
import copy
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
import torch
from pynput.keyboard import Listener, Key
import yaml

import matplotlib.pyplot as plt

class cmd:
    def __init__(self):
        self.cmd = np.array([0., 0., 0.],dtype=np.float32)
    def cmd_swtich(self, key_input):
        if key_input == Key.up:
            self.cmd[0] += 0.1
        elif key_input == Key.down:
            self.cmd[0] -= 0.1
        elif key_input == Key.home:
            self.cmd[1] += 0.1
        elif key_input == Key.end:
            self.cmd[1] -= 0.1
        elif key_input == Key.insert:
            self.cmd[2] += 0.1
        elif key_input == Key.delete:
            self.cmd[2] -= 0.1
        elif key_input == Key.f1:
            self.cmd[:] = 0.
        print(f"Moved to ({self.cmd[0]}, {self.cmd[1]}, {self.cmd[2]})")

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """

    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{cfg}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_single_obs = config["num_single_obs"]
        frame_stack = config["frame_stack"]
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)

    # load policy
    policy = torch.jit.load(policy_path)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("joint_names:", joint_names)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuator_names:", actuator_names)

    defaut_dof_pos = default_angles
    data.qpos[7:] = defaut_dof_pos

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((num_actions), dtype=np.double)
    action = np.zeros((num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(frame_stack):
        hist_obs.append(np.zeros([1, num_single_obs], dtype=np.double))

    count_lowlevel = 0
    L_foot_force_list = []
    R_foot_force_list = []

    # Per-actuator effort limits read from the yaml (matches Isaac torque
    # clipping at legged_robot.py:468). Absent -> no clipping.
    # tau_limit = np.array(config["tau_limit"], dtype=np.float32) if "tau_limit" in config else None

    for _ in tqdm(range(int(simulation_duration / simulation_dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-num_actions:]
        dq = dq[-num_actions:]

        if count_lowlevel % control_decimation == 0:
            obs = np.zeros([1, num_single_obs], dtype=np.float32)

            obs[0, :3] = command.cmd * cmd_scale
            obs[0, 3:6] = omega * ang_vel_scale
            obs[0, 6:9] = gvec[:3]
            obs[0, 9:9 + num_actions] = (q - defaut_dof_pos) * dof_pos_scale
            obs[0, 9 + num_actions:9 + num_actions * 2] = dq * dof_vel_scale
            obs[0, 9 + num_actions * 2:9 + num_actions * 3] = action

            hist_obs.append(obs)
            hist_obs.popleft()

            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single_obs : (i + 1) * num_single_obs] = hist_obs[i][0, :]
            policy_input = torch.tensor(model_input)
            
            action[:] = policy(policy_input)[0].detach().numpy()

            target_q = (action * action_scale) + defaut_dof_pos
        
        L_leg_foot_force = data.sensor('L_leg_foot_force')
        R_leg_foot_force = data.sensor('R_leg_foot_force')

        if _ % 10 == 0:
            print("Current linear velocity x: ", v[0], " Command linear velocity x", command.cmd[0])

        L_foot_force_list.append(copy.copy(L_leg_foot_force.data[2]))
        R_foot_force_list.append(copy.copy(R_leg_foot_force.data[2]))

        target_dq = np.zeros((num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, kps,
                        target_dq, dq, kds)  # Calc torques
        # if tau_limit is not None:
        #     tau = np.clip(tau, -tau_limit, tau_limit)  # match Isaac effort-limit clipping
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1


    viewer.close()


if __name__ == '__main__':
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_18dof.yaml", help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_single_obs = config["num_single_obs"]
        frame_stack = config["frame_stack"]
    
    command = cmd()
    listener = Listener(on_press=command.cmd_swtich)
    listener.start()
    run_mujoco(config_file)
