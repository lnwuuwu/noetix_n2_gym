import math
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.utils.stairs_terrain import terrain_height_at_x
import torch
from pynput.keyboard import Listener, Key
import yaml

def load_mujoco_model(xml_path, stair_cfg=None):
    """Load MJCF, optionally replacing its direct-child boxes with training stairs."""
    if stair_cfg is None:
        return mujoco.MjModel.from_xml_path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody: {}".format(xml_path))

    # The repository's 18-DoF MJCF already contains a historical staircase.
    # Remove only world-level boxes; robot collision geoms are nested in bodies.
    # It also contains two coincident planes, which would duplicate contacts.
    friction = str(stair_cfg.get("friction", "0.8 0.005 0.0001"))
    seen_ground_plane = False
    for geom in list(worldbody.findall("geom")):
        if geom.attrib.get("type") == "box":
            worldbody.remove(geom)
        elif geom.attrib.get("type") == "plane":
            if seen_ground_plane:
                worldbody.remove(geom)
            else:
                geom.set("friction", friction)
                seen_ground_plane = True
    if not seen_ground_plane:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "n2_stair_ground",
                "type": "plane",
                "size": "0 0 1",
                "friction": friction,
            },
        )

    start_x = float(stair_cfg["start_x"])
    step_width = float(stair_cfg["step_width"])
    step_height = float(stair_cfg["step_height"])
    num_steps = int(stair_cfg["num_steps"])
    half_width = 0.5 * float(stair_cfg.get("stair_width", 2.0))

    for index in range(num_steps):
        height = (index + 1) * step_height
        center_x = start_x + (index + 0.5) * step_width
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "n2_stair_{:02d}".format(index + 1),
                "type": "box",
                "size": "{} {} {}".format(
                    0.5 * step_width, half_width, 0.5 * height
                ),
                "pos": "{} 0 {}".format(center_x, 0.5 * height),
                "friction": friction,
                "rgba": "0.55 0.58 0.62 1",
            },
        )

    top_length = float(stair_cfg.get("top_platform_length", 1.5))
    top_height = num_steps * step_height
    top_start = start_x + num_steps * step_width
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "n2_stair_top",
            "type": "box",
            "size": "{} {} {}".format(
                0.5 * top_length, half_width, 0.5 * top_height
            ),
            "pos": "{} 0 {}".format(
                top_start + 0.5 * top_length, 0.5 * top_height
            ),
            "friction": friction,
            "rgba": "0.55 0.58 0.62 1",
        },
    )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".stairs.xml",
            prefix="n2_",
            dir=os.path.dirname(xml_path),
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            tree.write(temporary_file, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(temporary_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def resolve_joint_layout(model, joint_order):
    """Map named policy joints to MuJoCo qpos/qvel/control addresses."""
    qpos_indices = []
    qvel_indices = []
    control_indices = []
    for joint_name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError("MuJoCo joint not found: {}".format(joint_name))
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(actuator_matches) != 1:
            raise ValueError(
                "Expected one actuator for {}, found {}".format(
                    joint_name, len(actuator_matches)
                )
            )
        control_indices.append(int(actuator_matches[0]))
    return qpos_indices, qvel_indices, control_indices


def stair_height_observations(base_position, quat_xyzw, height_cfg, stair_cfg):
    """Reproduce Isaac Gym's compact yaw-rotated forward height scan."""
    x, y, z, w = quat_xyzw
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    world_x_positions = []
    for point_x in height_cfg["points_x"]:
        for point_y in height_cfg["points_y"]:
            world_x = base_position[0] + cos_yaw * point_x - sin_yaw * point_y
            world_x_positions.append(world_x)
    stair_heights = terrain_height_at_x(
        np.asarray(world_x_positions),
        start_x=float(stair_cfg["start_x"]),
        step_width=float(stair_cfg["step_width"]),
        step_height=float(stair_cfg["step_height"]),
        num_steps=int(stair_cfg["num_steps"]),
    )
    heights = np.clip(
        base_position[2]
        - float(height_cfg.get("sensor_offset", 0.5))
        - stair_heights,
        -1.0,
        1.0,
    ) * float(height_cfg["scale"])
    return np.asarray(heights, dtype=np.float32)

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
        clip_observations = float(config.get("clip_observations", np.inf))
        clip_actions = float(config.get("clip_actions", np.inf))

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_single_obs = config["num_single_obs"]
        frame_stack = config["frame_stack"]
        joint_order = config.get("joint_order")
        torque_limits = np.asarray(
            config.get("torque_limits", [np.inf] * num_actions), dtype=np.float32
        )
        stair_cfg = config.get("stairs")
        height_cfg = config.get("height_measurements")
    
    model = load_mujoco_model(xml_path, stair_cfg)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)

    # load policy
    policy = torch.jit.load(policy_path)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("joint_names:", joint_names)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuator_names:", actuator_names)

    if joint_order is None:
        joint_order = joint_names[-num_actions:]
    if len(joint_order) != num_actions:
        raise ValueError("joint_order length must equal num_actions")
    qpos_indices, qvel_indices, control_indices = resolve_joint_layout(model, joint_order)
    print("policy_joint_order:", joint_order)

    defaut_dof_pos = default_angles
    data.qpos[qpos_indices] = defaut_dof_pos

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((num_actions), dtype=np.double)
    action = np.zeros((num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(frame_stack):
        hist_obs.append(np.zeros([1, num_single_obs], dtype=np.double))

    count_lowlevel = 0

    for _ in tqdm(range(int(simulation_duration / simulation_dt)), desc="Simulating..."):

        # Obtain an observation
        _, _, quat, v, omega, gvec = get_obs(data)
        q = data.qpos[qpos_indices].copy()
        dq = data.qvel[qvel_indices].copy()

        if count_lowlevel % control_decimation == 0:
            obs = np.zeros([1, num_single_obs], dtype=np.float32)

            obs[0, :3] = command.cmd * cmd_scale
            obs[0, 3:6] = omega * ang_vel_scale
            obs[0, 6:9] = gvec[:3]
            obs[0, 9:9 + num_actions] = (q - defaut_dof_pos) * dof_pos_scale
            obs[0, 9 + num_actions:9 + num_actions * 2] = dq * dof_vel_scale
            obs[0, 9 + num_actions * 2:9 + num_actions * 3] = action

            proprio_size = 9 + num_actions * 3
            if height_cfg is not None:
                height_obs = stair_height_observations(
                    data.qpos[:3], quat, height_cfg, stair_cfg
                )
                if proprio_size + len(height_obs) != num_single_obs:
                    raise ValueError(
                        "Height observation layout {} + {} != {}".format(
                            proprio_size, len(height_obs), num_single_obs
                        )
                    )
                obs[0, proprio_size:] = height_obs
            elif proprio_size != num_single_obs:
                raise ValueError(
                    "num_single_obs={} requires an explicit extra observation provider".format(
                        num_single_obs
                    )
                )

            hist_obs.append(obs)
            hist_obs.popleft()

            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single_obs : (i + 1) * num_single_obs] = hist_obs[i][0, :]
            model_input = np.clip(
                model_input, -clip_observations, clip_observations
            )
            policy_input = torch.tensor(model_input)
            
            action[:] = np.clip(
                policy(policy_input)[0].detach().numpy(),
                -clip_actions,
                clip_actions,
            )

            target_q = (action * action_scale) + defaut_dof_pos
        
        if _ % max(1, int(1.0 / simulation_dt)) == 0:
            print("Current linear velocity x: ", v[0], " Command linear velocity x", command.cmd[0])

        target_dq = np.zeros((num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, kps,
                        target_dq, dq, kds)  # Calc torques
        tau = np.clip(tau, -torque_limits, torque_limits)
        data.ctrl[:] = 0.0
        data.ctrl[control_indices] = tau

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
    command.cmd[:] = np.asarray(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
    print("initial_command:", command.cmd)
    listener = Listener(on_press=command.cmd_swtich)
    listener.start()
    run_mujoco(config_file)
