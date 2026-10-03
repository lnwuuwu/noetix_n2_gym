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
    base_xyz = data.qpos[:3].astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_xyz)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def init_height_points(measured_points_x, measured_points_y):
    '''Grid of (dx, dy) height-scan sample points in the base frame.
    x outer / y inner, matching legged_robot._init_height_points().'''
    grid_x, grid_y = np.meshgrid(measured_points_x, measured_points_y, indexing='ij')
    return np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)  # (S, 2)

def terrain_height_at(
        model, data, points_xy, ray_start_z=10.0, max_body_skips=None):
    '''Terrain height at arbitrary world (x, y) points via a downward raycast.

    The MuJoCo scenes wired up for sim2sim (e.g. N2_10dof.xml) are NOT
    flat-plane only -- they include a real staircase built from stacked box
    geoms. The previous version of this function assumed flat-plane and
    always returned 0, silently feeding a flat height-map into both the
    debug markers and the actual policy observation (get_height_scan) even
    while standing on stairs.

    A geom directly under a sample point may belong to the robot itself.
    Isaac Gym's height scan only sees the static terrain mesh. Parkour
    sim2sim therefore moves robot geoms to group 1; when that convention is
    present, raycasts exclude group 1 in one operation. Other callers fall
    back to repeated body skipping, with enough attempts for every body.
    '''
    n = points_xy.shape[0]
    heights = np.zeros(n, dtype=np.double)
    geomid = np.zeros(1, dtype=np.int32)
    robot_geoms = model.geom_bodyid != 0
    ray_group = 1
    grouped_robot = (
        np.any(robot_geoms)
        and np.all(model.geom_group[robot_geoms] == ray_group)
        and not np.any(
            (model.geom_bodyid == 0)
            & (model.geom_group == ray_group)))
    geomgroup = None
    if grouped_robot:
        geomgroup = np.ones(6, dtype=np.uint8)
        geomgroup[ray_group] = 0
    if max_body_skips is None:
        max_body_skips = model.nbody + 1

    for i in range(n):
        x, y = points_xy[i]
        pnt = np.array([x, y, ray_start_z], dtype=np.float64)
        vec = np.array([0., 0., -1.], dtype=np.float64)
        if geomgroup is not None:
            dist = mujoco.mj_ray(
                model, data, pnt, vec, geomgroup, 1, -1, geomid)
            if dist >= 0 and geomid[0] >= 0:
                heights[i] = pnt[2] - dist
            continue

        bodyexclude = -1
        terrain_hit = False
        for _ in range(max_body_skips):
            dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, bodyexclude, geomid)
            if dist < 0 or geomid[0] < 0:
                break
            z_hit = pnt[2] - dist
            if model.geom_bodyid[geomid[0]] == 0:  # worldbody geom == terrain
                heights[i] = z_hit
                terrain_hit = True
                break
            bodyexclude = int(model.geom_bodyid[geomid[0]])
            pnt = np.array([x, y, z_hit - 1e-4], dtype=np.float64)
        if not terrain_hit:
            heights[i] = 0.0
    return heights

def get_height_scan(model, data, base_xyz, quat, height_points, base_height_offset, height_clip, height_measurements_scale):
    '''Yaw-rotates + translates the base-frame height_points grid into world
    space, samples terrain height under each point, and returns the scaled
    relative-height observation. Mirrors N2PerceptiveEnv.compute_observations()
    and humanoid.utils.math.quat_apply_yaw (quat is [x, y, z, w]; the yaw-only
    rotation angle is 2*atan2(qz, qw), independent of roll/pitch).'''
    yaw = 2. * math.atan2(quat[2], quat[3])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = base_xyz[0] + height_points[:, 0] * cos_yaw - height_points[:, 1] * sin_yaw
    world_y = base_xyz[1] + height_points[:, 0] * sin_yaw + height_points[:, 1] * cos_yaw
    terrain_h = terrain_height_at(model, data, np.stack([world_x, world_y], axis=-1))
    heights = np.clip(base_xyz[2] - base_height_offset - terrain_h, -height_clip, height_clip)
    return heights * height_measurements_scale

def get_height_points_world(model, data, base_xyz, quat, height_points):
    '''World-space (x, y, z) of every height-scan sample point, z = the
    sampled terrain height under it. Same yaw-rotation as get_height_scan,
    kept separate since debug markers want raw terrain height (not the
    base-relative/clipped/scaled obs value). Mirrors legged_robot._draw_debug_vis.'''
    yaw = 2. * math.atan2(quat[2], quat[3])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = base_xyz[0] + height_points[:, 0] * cos_yaw - height_points[:, 1] * sin_yaw
    world_y = base_xyz[1] + height_points[:, 0] * sin_yaw + height_points[:, 1] * cos_yaw
    world_z = terrain_height_at(model, data, np.stack([world_x, world_y], axis=-1))
    return np.stack([world_x, world_y, world_z], axis=-1)  # (S, 3)

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

        height_measurements_scale = config["height_measurements_scale"]
        base_height_offset = config["base_height_offset"]
        height_clip = config["height_clip"]
        measured_points_x = config["measured_points_x"]
        measured_points_y = config["measured_points_y"]

        # Draw the height-scan grid as yellow spheres, matching
        # legged_robot._draw_debug_vis (terrain.debug_viz) in Isaac Gym.
        debug_viz = bool(config.get("debug_viz", True))

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)

    # load policy
    policy = torch.jit.load(policy_path)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("joint_names:", joint_names)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuator_names:", actuator_names)

    height_points = init_height_points(measured_points_x, measured_points_y)
    num_height_points = height_points.shape[0]
    assert num_single_obs == 9 + num_actions * 3 + num_height_points, \
        f"num_single_obs ({num_single_obs}) doesn't match 9 + 3*num_actions + height points ({9 + num_actions * 3 + num_height_points})"
    height_marker_world = np.zeros((num_height_points, 3))  # refreshed at control rate, redrawn every render()

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
    tau_limit = np.array(config["tau_limit"], dtype=np.float32) if "tau_limit" in config else None

    for _ in tqdm(range(int(simulation_duration / simulation_dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_xyz = get_obs(data)
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
            obs[0, 9 + num_actions * 3:] = get_height_scan(
                model, data, base_xyz, quat, height_points, base_height_offset, height_clip, height_measurements_scale)
            if debug_viz:
                height_marker_world[:] = get_height_points_world(model, data, base_xyz, quat, height_points)

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
        if tau_limit is not None:
            tau = np.clip(tau, -tau_limit, tau_limit)  # match Isaac effort-limit clipping
        data.ctrl = tau

        mujoco.mj_step(model, data)
        if debug_viz:
            # markers are cleared every render() call, so re-add them each frame
            for px, py, pz in height_marker_world:
                viewer.add_marker(pos=[px, py, pz], size=[0.02, 0.02, 0.02],
                                   rgba=[1, 1, 0, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                   label="")  # mjvGeom.label isn't cleared by default -> stale text otherwise
        viewer.render()
        count_lowlevel += 1


    viewer.close()


if __name__ == '__main__':
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_perceptive.yaml", help="config file name in the config folder")
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
