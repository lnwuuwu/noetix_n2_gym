"""n2_parkour 的 MuJoCo sim2sim。

与 sim2sim_perceptive.py 的区别只有一处，但很关键：**Parkour 策略的目标是 goal
路点的位置，不是速度指令的方向**，观测里多了到当前 goal 的 [cos dpsi, sin dpsi]
两维（num_single_obs 135 -> 137）。所以这里必须自己维护一套 goal 路点，并复刻
N2ParkourEnv._update_goals 的推进逻辑。

【重要前提，先说清楚】
Isaac Gym 里 goal 是地形生成时就知道的真值，MuJoCo 这边没有这个"上帝视角"。
本脚本从场景几何里推算路点（沿 +x 每隔 goal_spacing 取一个，z 用向下射线打到的
地形高度），目的是**验证策略本身**——它到底会不会朝目标爬楼梯。
这不是一条部署路径：真机上没有任何东西会告诉机器人 goal 在哪。Extreme Parkour
自己的解法是再训一个以深度相机为输入的学生策略去蒸馏这个教师策略，本仓库尚未实现。
所以：本脚本用于评估，不要据此认为 parkour 策略已经可部署。

其余部分（PD 控制、高度扫描、观测缩放、帧堆叠）与 perceptive 完全一致，直接复用
sim2sim_perceptive 里的实现，避免两份代码各自漂移。
"""
import os
import math
import time
import itertools

# Allow --check_only over a headless SSH session. Interactive runs under
# NoMachine/X11 still use the normal pynput backend.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")

import numpy as np
import mujoco
try:
    import mujoco_viewer
except ImportError:
    # Headless validation/tests and the browser streamer do not need the
    # optional GLFW viewer package.
    mujoco_viewer = None
from tqdm import tqdm
from collections import deque
from humanoid import LEGGED_GYM_ROOT_DIR
import torch
from pynput.keyboard import Listener, Key
import yaml

# 复用 perceptive 侧已经调通的实现：观测提取、PD、高度扫描射线、键盘指令
from sim2sim_perceptive import (
    cmd, get_obs, pd_control, init_height_points,
    terrain_height_at, get_height_scan, get_height_points_world,
)
from parkour_reset import (
    RESET_FALL, RESET_LOW_CLEARANCE, RESET_STAGE_CLEAR, RESET_STUCK,
    RESET_SUMMIT, RESET_TILT, choose_campaign_transition,
    choose_up_only_reset, update_forward_progress, update_success_hold,
    validate_course_mode,
)
from collision_modes import configure_robot_collision_masks
from camera_follow import SmoothFollowCamera
from stream_gate import establish_stream_start_wall

N2_LEG_JOINT_NAMES = [
    "L_leg_hip_yaw_joint", "L_leg_hip_roll_joint", "L_leg_hip_pitch_joint",
    "L_leg_knee_joint", "L_leg_ankle_joint",
    "R_leg_hip_yaw_joint", "R_leg_hip_roll_joint", "R_leg_hip_pitch_joint",
    "R_leg_knee_joint", "R_leg_ankle_joint",
]

CAMPAIGN_KIND_LABELS = {
    "up_stairs": "上台阶",
    "down_stairs": "下台阶",
    "hurdle": "跨栏",
    "flat": "平地路点",
    "stepping_stones": "踏石",
}

def campaign_setting(settings, kind, default):
    """Read a per-terrain setting, accepting a scalar or a kind mapping."""
    if isinstance(settings, dict):
        value = settings.get(kind, settings.get("default", default))
    elif settings is None:
        value = default
    else:
        value = settings
    return float(value)


def campaign_height_points_world(campaign, base_xyz, quat, height_points):
    """Sample the campaign height field at the policy's yaw-aligned grid."""
    yaw = 2.0 * math.atan2(quat[2], quat[3])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = (
        base_xyz[0] + height_points[:, 0] * cos_yaw
        - height_points[:, 1] * sin_yaw)
    world_y = (
        base_xyz[1] + height_points[:, 0] * sin_yaw
        + height_points[:, 1] * cos_yaw)
    points_xy = np.stack([world_x, world_y], axis=-1)
    # Do not raycast here. The Isaac Gym observation is indexed directly from
    # its height field, and campaign.sample_heights implements that exact
    # three-neighbour/min convention.
    world_z = campaign.sample_heights(points_xy)
    return np.column_stack([points_xy, world_z])


def campaign_height_scan(
        campaign, base_xyz, quat, height_points, base_height_offset,
        height_clip, height_measurements_scale):
    """Build the 96-D policy scan from the shared Isaac-compatible map."""
    points_world = campaign_height_points_world(
        campaign, base_xyz, quat, height_points)
    heights = np.clip(
        base_xyz[2] - base_height_offset - points_world[:, 2],
        -height_clip, height_clip)
    return heights * height_measurements_scale, points_world


def configure_and_validate_model(model, self_collision_mode="disabled"):
    """Apply the requested robot self-collision semantics.

    ``disabled`` preserves the historical behavior used by existing
    checkpoints. ``cross_leg`` enables only collision-geometry pairs whose
    bodies belong to opposite legs. In both modes robot/world contact remains
    enabled and visual-only geometry remains non-colliding.
    """
    active_planes = np.flatnonzero(
        (model.geom_bodyid == 0)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    if len(active_planes) != 1:
        raise ValueError(
            "sim2sim scene must contain exactly one active ground plane; "
            f"found {len(active_planes)}")

    body_names = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for body_id in range(model.nbody)]
    robot_geoms = configure_robot_collision_masks(
        model.geom_bodyid, model.geom_contype, model.geom_conaffinity,
        body_names, self_collision_mode)

    # Height rays must never see the robot. Group 1 is visible in the default
    # viewer and contains no world terrain; terrain_height_at filters it.
    model.geom_group[robot_geoms] = 1

    for body_name in ("L_leg_ankle_link", "R_leg_ankle_link"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0 or np.min(model.body_inertia[body_id]) < 0.005:
            raise ValueError(
                f"{body_name} inertia does not match the training URDF")


class Carrot:
    """
    按键：
      ← / →      offset 的世界系 y 增/减（转向）
      ↑ / ↓      offset 的世界系 x 增/减（拉远/拉近）
      Home/End   整体拉远 / 拉近（沿当前 offset 方向缩放）
      Insert/F1  把 offset 重置为"当前朝向正前方 lookahead 米"（只取一次 yaw，
                 此后就是世界系常量，不再跟着转）
    """

    def __init__(self, lookahead=1.5, step=0.15):
        self.offset = None          # 世界系 (dx, dy)，相对机器人当前位置
        self.lookahead = lookahead
        self.step = step
        self._pending = []          # 待处理按键，避免回调线程直接改状态

    def on_press(self, key):
        self._pending.append(key)

    def _apply(self, yaw):
        while self._pending:
            k = self._pending.pop(0)
            if self.offset is None:
                self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
            if k == Key.left:      self.offset[1] += self.step
            elif k == Key.right:   self.offset[1] -= self.step
            elif k == Key.up:      self.offset[0] += self.step
            elif k == Key.down:    self.offset[0] -= self.step
            elif k in (Key.home, Key.end):
                n = np.linalg.norm(self.offset)
                if n > 1e-6:
                    scale = (n + (0.1 if k == Key.home else -0.1)) / n
                    self.offset = self.offset * max(scale, 0.05)
            elif k in (Key.insert, Key.f1):
                # 只在此刻取一次 yaw，之后 offset 就是世界系常量
                self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
            else:
                continue
            print("[carrot] offset=(%+.2f, %+.2f) 世界系  |offset|=%.2f m"
                  % (self.offset[0], self.offset[1], np.linalg.norm(self.offset)))

    def update(self, base_xy, yaw, reach, model, data):
        """返回胡萝卜的世界坐标 (x, y, z)。reach 参数保留只为签名统一，此模式不用。"""
        if self.offset is None:
            self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
        self._apply(yaw)
        pos = base_xy + self.offset
        z = terrain_height_at(model, data, pos[None, :])[0]
        return np.array([pos[0], pos[1], z])


def build_goals(model, data, start_xy, cfg):
    """生成 goal 路点。

    优先用 yaml 里显式给的 goals（世界坐标 [[x,y],...]）；没有就沿 +x 自动铺：
    从 goal_first_x 起每隔 goal_spacing 一个，y 固定为出生点的 y，z 用向下射线
    取该处地形高度——等价于"把 goal 放在台阶顶面上"，这正是 Extreme Parkour 的
    parkour_step_terrain 干的事（goals[i+1] 落在第 i 级障碍上）。
    """
    explicit = cfg.get("goals")
    if explicit:
        pts = np.array(explicit, dtype=np.float64)
        if pts.shape[1] == 2:                       # 只给了 xy，z 用射线补
            z = terrain_height_at(model, data, pts)
            pts = np.concatenate([pts, z[:, None]], axis=1)
        return pts

    first = float(cfg.get("goal_first_x", 1.0))
    spacing = float(cfg.get("goal_spacing", 0.25))
    count = int(cfg.get("goal_count", 20))
    xs = first + spacing * np.arange(count)
    ys = np.full_like(xs, start_xy[1])
    z = terrain_height_at(model, data, np.stack([xs, ys], axis=-1))
    return np.stack([xs, ys, z], axis=-1)


def build_repeating_course(model, data, start_xy, cfg):
    """生成双车道闭环：一侧上楼、顶部 U 形转弯、另一侧下楼。"""
    stair_goals = build_goals(model, data, start_xy, cfg)
    if len(stair_goals) == 0:
        raise ValueError("repeat_course requires at least one stair goal")

    center_y = float(cfg.get("course_center_y", 0.0))
    lane = float(cfg.get("course_lane_offset", 0.65))
    top_center_x = float(cfg.get("top_turnaround_center_x", 7.0))
    bottom_center_x = float(cfg.get("bottom_turnaround_center_x", 0.0))
    turn_samples = int(cfg.get("turnaround_samples", 7))
    if lane <= 0:
        raise ValueError("course_lane_offset must be positive")
    if turn_samples < 3 or turn_samples % 2 == 0:
        raise ValueError("turnaround_samples must be an odd integer >= 3")
    if top_center_x <= stair_goals[-1, 0]:
        raise ValueError(
            "top_turnaround_center_x must be after the final stair goal")
    if bottom_center_x >= stair_goals[0, 0]:
        raise ValueError(
            "bottom_turnaround_center_x must be before the first stair goal")

    up_xy = stair_goals[:, :2].copy()
    up_xy[:, 1] = center_y + lane
    down_xy = up_xy[::-1].copy()
    down_xy[:, 1] = center_y - lane

    # Heading is +x on the upper lane. The top semicircle turns clockwise
    # from +x to -x; the bottom semicircle does the same from -x back to +x.
    top_theta = np.linspace(
        math.pi / 2.0, -math.pi / 2.0, turn_samples)
    top_xy = np.stack([
        top_center_x + lane * np.cos(top_theta),
        center_y + lane * np.sin(top_theta),
    ], axis=-1)
    bottom_theta = np.linspace(
        -math.pi / 2.0, -3.0 * math.pi / 2.0, turn_samples)
    bottom_xy = np.stack([
        bottom_center_x + lane * np.cos(bottom_theta),
        center_y + lane * np.sin(bottom_theta),
    ], axis=-1)

    route_xy = np.vstack([up_xy, top_xy, down_xy, bottom_xy])
    route_z = terrain_height_at(model, data, route_xy)
    phases = (
        ["UP"] * len(up_xy)
        + ["TOP TURN"] * len(top_xy)
        + ["DOWN"] * len(down_xy)
        + ["BOTTOM TURN"] * len(bottom_xy))
    return np.column_stack([route_xy, route_z]), phases


def run_mujoco(
        cfg_name, command, carrot=None, goal_mode_override=None,
        check_only=False, headless=False, duration_override=None,
        policy_path_override=None, render_fps_override=None,
        debug_viz_override=None, stream_options=None,
        course_mode_override=None, quiet_progress=False,
        action_filter_alpha_override=None, action_gain_override=None,
        kd_scale_override=None):
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{cfg_name}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = (policy_path_override or config["policy_path"]).replace(
        "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, policy_path)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = (
        float(duration_override)
        if duration_override is not None
        else config["simulation_duration"])
    run_forever = simulation_duration <= 0
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    action_filter_alpha = float(
        action_filter_alpha_override
        if action_filter_alpha_override is not None
        else config.get("action_filter_alpha", 0.5))
    action_gain = float(
        action_gain_override
        if action_gain_override is not None
        else config.get("action_gain", 0.7))
    clip_observations = float(config.get("clip_observations", 18.0))
    clip_actions = float(config.get("clip_actions", 18.0))
    kd_scale = float(
        kd_scale_override
        if kd_scale_override is not None
        else config.get("kd_scale", 1.5))
    start_height = float(config.get("start_height", 0.80))
    stop_on_final_goal = bool(config.get("stop_on_final_goal", True))
    repeat_course = bool(config.get("repeat_course", False))
    up_only_reset = bool(config.get("up_only_reset", False))
    campaign_mode = bool(config.get("campaign_mode", False))
    if course_mode_override is not None:
        repeat_course = course_mode_override == "repeat"
        up_only_reset = course_mode_override == "up_only"
        campaign_mode = course_mode_override == "campaign"
    fall_reset_height = float(config.get("fall_reset_height", 0.35))
    no_progress_reset_seconds = float(
        config.get("no_progress_reset_seconds", 10.0))
    no_progress_min_x = float(config.get("no_progress_min_x", 0.05))
    if not 0.0 < action_filter_alpha <= 1.0:
        raise ValueError("action_filter_alpha must be in (0, 1]")
    if action_gain <= 0.0 or kd_scale <= 0.0:
        raise ValueError("action_gain and kd_scale must be positive")
    if fall_reset_height < 0.0:
        raise ValueError("fall_reset_height must be non-negative")
    if no_progress_reset_seconds < 0.0:
        raise ValueError("no_progress_reset_seconds must be non-negative")
    if no_progress_min_x <= 0.0:
        raise ValueError("no_progress_min_x must be positive")
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
    goal_reach_dist = float(config.get("goal_reach_dist", 0.5))
    goal_pass_lateral_tol = float(config.get("goal_pass_lateral_tol", 0.5))
    goal_lookahead_steps = max(
        1, int(config.get("goal_lookahead_steps", 1)))
    turn_goal_lookahead_steps = max(
        1, int(config.get("turn_goal_lookahead_steps", 2)))
    downhill_command_scale = float(
        config.get("downhill_command_scale", 1.0))
    turn_command_scale = float(config.get("turn_command_scale", 0.5))
    if not 0.0 < downhill_command_scale <= 1.0:
        raise ValueError("downhill_command_scale must be in (0, 1]")
    if not 0.0 < turn_command_scale <= 1.0:
        raise ValueError("turn_command_scale must be in (0, 1]")
    nominal_command_x = float(command.cmd[0])
    goal_mode = goal_mode_override or config.get("goal_mode", "waypoints")
    validate_course_mode(
        goal_mode, repeat_course, up_only_reset, campaign_mode)
    campaign_row = int(config.get("campaign_row", 3))
    campaign_num_rows = int(config.get("campaign_num_rows", 8))
    campaign_seed = int(config.get("campaign_seed", 5))
    campaign_goal_lookahead_steps = max(
        1, int(config.get("campaign_goal_lookahead_steps", 1)))
    campaign_goal_pass_lateral_tol = float(
        config.get("campaign_goal_pass_lateral_tol", 0.5))
    campaign_goal_success_hold_seconds = float(
        config.get("campaign_goal_success_hold_seconds", 0.3))
    campaign_restart_on_finish = bool(
        config.get("campaign_restart_on_finish", False))
    campaign_reset_grace_seconds = float(
        config.get("campaign_reset_grace_seconds", 2.0))
    campaign_min_base_clearance = float(
        config.get("campaign_min_base_clearance", 0.35))
    campaign_min_upright_cos = float(
        config.get("campaign_min_upright_cos", 0.35))
    campaign_progress_min_x = float(
        config.get("campaign_progress_min_x", 0.05))
    campaign_command_speeds = config.get(
        "campaign_command_speeds", nominal_command_x)
    campaign_no_progress_seconds = config.get(
        "campaign_no_progress_seconds", no_progress_reset_seconds)
    if campaign_num_rows <= 1 or not 0 <= campaign_row < campaign_num_rows:
        raise ValueError(
            "campaign_row must be in [0, campaign_num_rows)")
    if campaign_goal_pass_lateral_tol <= 0.0:
        raise ValueError(
            "campaign_goal_pass_lateral_tol must be positive")
    if campaign_goal_success_hold_seconds < 0.0:
        raise ValueError(
            "campaign_goal_success_hold_seconds must be non-negative")
    if campaign_reset_grace_seconds < 0.0:
        raise ValueError(
            "campaign_reset_grace_seconds must be non-negative")
    if campaign_min_base_clearance < 0.0:
        raise ValueError(
            "campaign_min_base_clearance must be non-negative")
    if not 0.0 <= campaign_min_upright_cos <= 1.0:
        raise ValueError(
            "campaign_min_upright_cos must be in [0, 1]")
    if campaign_progress_min_x <= 0.0:
        raise ValueError("campaign_progress_min_x must be positive")
    debug_viz = (
        bool(debug_viz_override)
        if debug_viz_override is not None
        else bool(config.get("debug_viz", False)))
    render_fps = (
        float(render_fps_override)
        if render_fps_override is not None
        else float(config.get("render_fps", 30)))
    if render_fps <= 0:
        raise ValueError("render_fps must be positive")
    render_decimation = max(1, int(round(1.0 / (simulation_dt * render_fps))))
    actual_render_fps = 1.0 / (simulation_dt * render_decimation)
    viewer_width = int(config.get("viewer_width", 960))
    viewer_height = int(config.get("viewer_height", 540))
    camera_distance = float(config.get("camera_distance", 3.2))
    camera_azimuth_offset = float(
        config.get("camera_azimuth_offset", 90.0))
    camera_elevation = float(config.get("camera_elevation", -22.0))
    camera_lookahead = float(config.get("camera_lookahead", 0.35))
    camera_lookat_z_offset = float(
        config.get("camera_lookat_z_offset", -0.10))
    camera_follow_tau = float(config.get("camera_follow_tau", 0.18))
    camera_yaw_follow_tau = float(
        config.get("camera_yaw_follow_tau", 0.25))
    camera_snap_distance = float(
        config.get("camera_snap_distance", 2.0))
    campaign_side_margin = float(
        config.get("campaign_mujoco_side_margin", 0.0))
    campaign_remove_lateral_walls = bool(
        config.get("campaign_mujoco_remove_lateral_walls", False))
    campaign_widen_stair_shoulders = bool(
        config.get("campaign_mujoco_widen_stair_shoulders", False))
    if camera_follow_tau < 0.0 or camera_yaw_follow_tau < 0.0:
        raise ValueError("camera follow time constants must be non-negative")
    if camera_snap_distance <= 0.0:
        raise ValueError("camera_snap_distance must be positive")
    if campaign_side_margin < 0.0:
        raise ValueError(
            "campaign_mujoco_side_margin must be non-negative")
    tau_limit = np.array(config["tau_limit"], dtype=np.float32) if "tau_limit" in config else None

    campaign = None
    if campaign_mode:
        from parkour_campaign import build_campaign, build_mujoco_model

        campaign = build_campaign(config)
        if len(campaign.stages) != 5:
            raise ValueError(
                "campaign must contain exactly five continuous terrain stages; "
                f"found {len(campaign.stages)}")
        model = build_mujoco_model(
            xml_path,
            campaign,
            side_margin=campaign_side_margin,
            remove_lateral_walls=campaign_remove_lateral_walls,
            widen_stair_shoulders=campaign_widen_stair_shoulders,
        )
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    self_collision_mode = str(
        config.get("self_collision_mode", "disabled")).strip().lower()
    configure_and_validate_model(
        model, self_collision_mode=self_collision_mode)
    data = mujoco.MjData(model)
    policy = torch.jit.load(policy_path)
    policy.eval()

    height_points = init_height_points(measured_points_x, measured_points_y)
    num_height_points = height_points.shape[0]
    # 137 = 9(cmd+angvel+gravity) + 3*num_actions + 2(goal) + 高度点数
    expect = 9 + num_actions * 3 + 2 + num_height_points
    assert num_single_obs == expect, (
        f"num_single_obs ({num_single_obs}) != 9 + 3*{num_actions} + 2(goal) + "
        f"{num_height_points}(height) = {expect}")
    assert num_obs == frame_stack * num_single_obs, (
        f"num_obs ({num_obs}) != frame_stack ({frame_stack}) * "
        f"num_single_obs ({num_single_obs})")
    for name, values in (
        ("kps", kps), ("kds", kds), ("default_angles", default_angles),
    ):
        assert len(values) == num_actions, (
            f"{name} has {len(values)} entries, expected {num_actions}")
    if tau_limit is not None:
        assert len(tau_limit) == num_actions, (
            f"tau_limit has {len(tau_limit)} entries, expected {num_actions}")
    assert model.nu == num_actions, (
        f"MuJoCo model has {model.nu} actuators, policy expects {num_actions}")
    actuator_joint_names = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    ]
    assert actuator_joint_names == N2_LEG_JOINT_NAMES, (
        "MuJoCo actuator joint order does not match Isaac Gym policy order:\n"
        f"MuJoCo: {actuator_joint_names}\nExpected: {N2_LEG_JOINT_NAMES}")
    with torch.inference_mode():
        test_action = policy(torch.zeros((1, num_obs), dtype=torch.float32))
    assert tuple(test_action.shape) == (1, num_actions), (
        f"Policy output shape {tuple(test_action.shape)} != (1, {num_actions})")
    print(
        f"[parkour] 对齐检查通过：obs={num_obs} ({frame_stack}x{num_single_obs}), "
        f"actions={num_actions}, policy_dt={simulation_dt * control_decimation:.3f}s")
    print(
        f"[parkour] 稳定控制：action alpha={action_filter_alpha:.2f}, "
        f"gain={action_gain:.2f}, Kd x{kd_scale:.2f}, "
        f"clip={clip_actions:g}")
    print(
        f"[parkour] MuJoCo self-collision={str(self_collision_mode).lower()}")
    if campaign_mode:
        stage_summary = ", ".join(
            f"c{stage.col}:{CAMPAIGN_KIND_LABELS.get(stage.kind, stage.kind)}"
            for stage in campaign.stages)
        print(
            f"[campaign] Isaac Gym 顺序闯关：row "
            f"{campaign_row}/{campaign_num_rows - 1}, seed={campaign_seed}, "
            f"{stage_summary}")
        step_x_low, step_x_high = campaign.step_x_range
        quantized_min = int(
            np.nextafter(step_x_low, np.inf)
            / campaign.horizontal_scale) * campaign.horizontal_scale
        quantized_max = int(
            np.nextafter(step_x_high, -np.inf)
            / campaign.horizontal_scale) * campaign.horizontal_scale
        print(
            f"[campaign] 楼梯踏面配置=[{step_x_low:.2f}, "
            f"{step_x_high:.2f})m，0.1m网格实际="
            f"{quantized_min:.2f}/{quantized_max:.2f}m，"
            f"连续跑道总长={campaign.length:.1f}m")
        print(
            f"[campaign] MuJoCo 横向实体肩宽={campaign_side_margin:.1f}m/"
            "side（中央策略地形不变，hfield 裙边外移；"
            f"侧边高墙={'移除' if campaign_remove_lateral_walls else '保留'}；"
            f"楼梯肩部={'加宽' if campaign_widen_stair_shoulders else '原宽'}）")
    if check_only:
        print("[parkour] check-only 完成，未创建窗口、未运行仿真。")
        return
    height_marker_world = np.zeros((num_height_points, 3))

    default_dof_pos = default_angles
    data.qpos[2] = start_height
    if campaign_mode:
        first_spawn = np.asarray(
            campaign.stages[0].goals[0], dtype=np.float64)
        data.qpos[:2] = first_spawn[:2]
        data.qpos[2] = first_spawn[2] + start_height
    elif goal_mode == "waypoints" and repeat_course:
        # Start already aligned with the uphill lane. The closed route returns
        # to this same point after every bottom U-turn.
        data.qpos[1] = (
            float(config.get("course_center_y", 0.0))
            + float(config.get("course_lane_offset", 0.65)))
    elif goal_mode == "waypoints" and up_only_reset:
        # Every uphill trial uses the same center line and spawn pose.
        data.qpos[1] = float(config.get("course_center_y", 0.0))
    data.qpos[7:] = default_dof_pos
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    campaign_stage_index = 0
    campaign_stage = None
    campaign_attempt = 1
    stage_clears = 0
    campaign_finished = False
    campaign_stage_failures = None
    campaign_stage_clear_attempts = None
    if campaign_mode:
        campaign_stage_failures = [0] * len(campaign.stages)
        campaign_stage_clear_attempts = [None] * len(campaign.stages)
        campaign_stage = campaign.stages[campaign_stage_index]
        goals = np.asarray(campaign_stage.goals[1:], dtype=np.float64)
        if len(goals) == 0:
            raise ValueError("campaign stage requires at least one target goal")
        goal_phases = None
        cur_goal = 0
        cycles = 0
        command.cmd[0] = campaign_setting(
            campaign_command_speeds, campaign_stage.kind,
            nominal_command_x)
        if command.cmd[0] <= 0.0:
            raise ValueError("campaign command speeds must be positive")
        print(
            "[campaign] 连续跑道：上台阶 -> 下台阶回地面 -> "
            "跨栏 -> 平地路点 -> 踏石 -> 终点。")
        print(
            "[campaign] 通过最后一个 goal 并稳定 "
            f"{campaign_goal_success_hold_seconds:.1f}s 才切下一关；"
            "过关不传送、不清历史，失败才回当前段起点。")
        print(
            f"[campaign] 当前 row={campaign_stage.row}/"
            f"{campaign_num_rows - 1} col={campaign_stage.col} "
            f"type={CAMPAIGN_KIND_LABELS.get(campaign_stage.kind, campaign_stage.kind)} "
            f"attempt={campaign_attempt}")
    elif goal_mode == "carrot":
        goals = None
        goal_phases = None
        cur_goal = 0
        cycles = 0
        print("[parkour] goal_mode=carrot —— 用方向键牵引目标点")
        print("          <-/->  左右挪   ^/v  前后挪   Home/End  调 lookahead   Insert/F1  对齐正前方")
    else:
        if repeat_course:
            goals, goal_phases = build_repeating_course(
                model, data, data.qpos[:2].copy(), config)
            cur_goal = 0
            cycles = 0
            phase_counts = {
                phase: goal_phases.count(phase)
                for phase in ("UP", "TOP TURN", "DOWN", "BOTTOM TURN")}
            print(
                f"[parkour] goal_mode=waypoints/repeat —— "
                f"{phase_counts['UP']} 级上楼 + "
                f"{phase_counts['DOWN']} 级下楼，"
                f"{phase_counts['TOP TURN']}+"
                f"{phase_counts['BOTTOM TURN']} 个转弯点")
            print(
                "[parkour] 双车道闭环：上楼 -> 顶部 U 形转弯 -> "
                "下楼 -> 底部 U 形转弯，持续循环")
            print(
                f"[parkour] 导航前视 {goal_lookahead_steps} 个路点，"
                f"下楼速度指令 x{downhill_command_scale:.2f}")
        else:
            goals = build_goals(
                model, data, data.qpos[:2].copy(), config)
            goal_phases = None
            cur_goal = 0
            cycles = 0
            if up_only_reset:
                print(
                    f"[parkour] goal_mode=waypoints/up-only-reset —— "
                    f"{len(goals)} 个上楼路点，固定中线 "
                    f"y={goals[0,1]:+.2f} m")
                print(
                    "[parkour] 登顶、跌倒或长时间无前进时自动回到出生点；"
                    "窗口和浏览器视频流保持连接。")
                print(
                    f"[parkour] 自动恢复：z<{fall_reset_height:.2f} m，"
                    f"或 {no_progress_reset_seconds:g}s 内 x 未前进 "
                    f"{no_progress_min_x:.2f} m")
            else:
                print(
                    f"[parkour] goal_mode=waypoints —— {len(goals)} 个路点，"
                    f"x 从 {goals[0,0]:.2f} 到 {goals[-1,0]:.2f} m")
    print(f"[parkour] 到达半径 {goal_reach_dist} m")
    if run_forever:
        print("[parkour] 仿真时长=无限，按 Ctrl+C 停止。")

    if (
            not headless
            and stream_options is None
            and mujoco_viewer is None):
        raise RuntimeError(
            "interactive playback requires the optional mujoco_viewer "
            "package; use --headless or --stream otherwise")
    viewer = (
        None if headless or stream_options is not None
        else mujoco_viewer.MujocoViewer(
            model, data, width=viewer_width, height=viewer_height))
    viewer_follow = None
    if viewer is not None:
        viewer_follow = SmoothFollowCamera(
            lookahead=camera_lookahead,
            z_offset=camera_lookat_z_offset,
            azimuth_offset=camera_azimuth_offset,
            position_tau=camera_follow_tau,
            yaw_tau=camera_yaw_follow_tau,
            snap_distance=camera_snap_distance)
        viewer.cam.distance = camera_distance
        viewer.cam.azimuth = camera_azimuth_offset
        viewer.cam.elevation = camera_elevation
        print(
            f"[parkour] viewer={actual_render_fps:.1f} FPS "
            f"(每 {render_decimation} 个物理步渲染一次), "
            f"height markers={'on' if debug_viz else 'off'}")

    streamer = None
    stream_renderer = None
    stream_camera = None
    stream_follow = None
    stream_decimation = None
    stream_start_wall = None
    if stream_options is not None:
        from mjpeg_stream import MJPEGStreamer

        manual_start = bool(stream_options.get("manual_start", False))
        stream_fps = float(stream_options["fps"])
        if stream_fps <= 0:
            raise ValueError("stream FPS must be positive")
        stream_decimation = max(
            1, int(round(1.0 / (simulation_dt * stream_fps))))
        stream_renderer = mujoco.Renderer(
            model,
            height=int(stream_options["height"]),
            width=int(stream_options["width"]))
        stream_camera = mujoco.MjvCamera()
        stream_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        stream_camera.distance = float(
            stream_options.get("distance", camera_distance))
        stream_azimuth_offset = float(
            stream_options.get("azimuth_offset", camera_azimuth_offset))
        stream_camera.azimuth = stream_azimuth_offset
        stream_camera.elevation = float(
            stream_options.get("elevation", camera_elevation))
        stream_follow = SmoothFollowCamera(
            lookahead=camera_lookahead,
            z_offset=camera_lookat_z_offset,
            azimuth_offset=stream_azimuth_offset,
            position_tau=camera_follow_tau,
            yaw_tau=camera_yaw_follow_tau,
            snap_distance=camera_snap_distance)
        streamer = MJPEGStreamer(
            host=stream_options["host"],
            port=int(stream_options["port"]),
            jpeg_quality=int(stream_options.get("jpeg_quality", 80)),
            manual_start=manual_start)
        streamer.start()
        print(
            f"[stream] HTTP server: http://{stream_options['host']}:"
            f"{stream_options['port']}")
        stream_start_wall = establish_stream_start_wall(
            streamer, manual_start=manual_start)
        if stream_start_wall is None:
            raise RuntimeError(
                "stream server stopped before playback was started")
        # Real-time pacing begins after both gates. Time spent on the ready
        # page must never make physics run fast in an attempt to catch up.
    target_q = np.zeros(num_actions, dtype=np.double)
    action = np.zeros(num_actions, dtype=np.double)
    action_initialized = False
    hist_obs = deque([np.zeros([1, num_single_obs], dtype=np.double) for _ in range(frame_stack)],
                     maxlen=frame_stack)
    count_lowlevel = 0
    reached = 0
    completed = False
    resets = 0
    failures = 0
    fall_resets = 0
    stuck_resets = 0
    low_clearance_resets = 0
    tilt_resets = 0
    best_progress_x = float(initial_qpos[0])
    last_progress_time = 0.0
    stage_started_time = 0.0
    final_goal_hold_started = None
    max_x = float(initial_qpos[0])
    best_campaign_progress_fraction = 0.0

    simulation_steps = (
        itertools.count()
        if run_forever
        else range(int(simulation_duration / simulation_dt)))
    for step in tqdm(
            simulation_steps,
            desc="Simulating...", disable=headless):
        q, dq, quat, v, omega, gvec, base_xyz = get_obs(data)
        max_x = max(max_x, float(base_xyz[0]))
        q = q[-num_actions:]
        dq = dq[-num_actions:]

        if count_lowlevel % control_decimation == 0:
            sim_time = step * simulation_dt
            summit_reached = False
            campaign_final_goal_stable = False
            if campaign_mode and campaign_finished:
                command.cmd[:] = [0.0, 0.0, 0.0]
            if up_only_reset or (campaign_mode and not campaign_finished):
                progress_threshold = (
                    campaign_progress_min_x
                    if campaign_mode else no_progress_min_x)
                best_progress_x, last_progress_time = update_forward_progress(
                    base_xyz[0], best_progress_x, last_progress_time,
                    sim_time, progress_threshold)
            # 自身偏航（与 get_height_scan 里一致：仅取 yaw 分量）
            yaw = 2.0 * math.atan2(quat[2], quat[3])
            if goal_mode == "carrot":
                cur_xyz = carrot.update(base_xyz[:2].copy(), yaw, goal_reach_dist, model, data)
                to_goal = cur_xyz[:2] - base_xyz[:2]
            else:
                progress_to_goal = goals[cur_goal, :2] - base_xyz[:2]
                if campaign_mode:
                    if not campaign_finished:
                        passed = (
                            progress_to_goal[0] < 0.0
                            and abs(progress_to_goal[1])
                            < campaign_goal_pass_lateral_tol)
                        goal_condition = (
                            np.linalg.norm(progress_to_goal)
                            < goal_reach_dist
                            or passed)
                        if (
                                goal_condition
                                and cur_goal < len(goals) - 1):
                            reached += 1
                            cur_goal += 1
                            final_goal_hold_started = None
                        elif cur_goal == len(goals) - 1:
                            (
                                final_goal_hold_started,
                                campaign_final_goal_stable,
                            ) = update_success_hold(
                                goal_condition,
                                final_goal_hold_started,
                                sim_time,
                                campaign_goal_success_hold_seconds)
                    guidance_index = min(
                        cur_goal + campaign_goal_lookahead_steps - 1,
                        len(goals) - 1)
                elif repeat_course:
                    # Generic "passed target plane" test works for both the
                    # straight stair lanes and the semicircular turn paths.
                    previous_xy = goals[(cur_goal - 1) % len(goals), :2]
                    segment = goals[cur_goal, :2] - previous_xy
                    segment_length = np.linalg.norm(segment)
                    along = (
                        np.dot(
                            base_xyz[:2] - goals[cur_goal, :2],
                            segment)
                        / max(segment_length, 1e-9))
                    cross_track = abs(
                        segment[0] * (base_xyz[1] - previous_xy[1])
                        - segment[1] * (base_xyz[0] - previous_xy[0])
                    ) / max(segment_length, 1e-9)
                    passed = (
                        along > 0.0
                        and cross_track < goal_pass_lateral_tol)
                    if (np.linalg.norm(progress_to_goal) < goal_reach_dist
                            or passed):
                        reached += 1
                        previous_phase = goal_phases[cur_goal]
                        cur_goal = (cur_goal + 1) % len(goals)
                        current_phase = goal_phases[cur_goal]
                        if cur_goal == 0:
                            cycles += 1
                            print(
                                f"\n[parkour] 完成第 {cycles} 次完整往返，"
                                "继续上楼。")
                        elif current_phase != previous_phase:
                            phase_messages = {
                                "TOP TURN": "到达顶部，开始平滑 U 形转弯。",
                                "DOWN": "顶部转弯完成，开始下楼。",
                                "BOTTOM TURN": "到达楼底，开始平滑 U 形转弯。",
                            }
                            message = phase_messages.get(current_phase)
                            if message:
                                print(f"\n[parkour] {message}")

                    current_phase = goal_phases[cur_goal]
                    lookahead_steps = (
                        turn_goal_lookahead_steps
                        if "TURN" in current_phase
                        else goal_lookahead_steps)
                    guidance_index = (
                        cur_goal + lookahead_steps - 1) % len(goals)
                    command_scale = (
                        downhill_command_scale
                        if current_phase == "DOWN"
                        else (
                            turn_command_scale
                            if "TURN" in current_phase else 1.0))
                    command.cmd[0] = nominal_command_x * command_scale
                else:
                    passed = (
                        progress_to_goal[0] < 0
                        and abs(progress_to_goal[1])
                        < goal_pass_lateral_tol)
                    if (np.linalg.norm(progress_to_goal)
                            < goal_reach_dist or passed):
                        reached += 1
                        if cur_goal < len(goals) - 1:
                            cur_goal += 1
                        else:
                            if up_only_reset:
                                summit_reached = True
                            else:
                                completed = stop_on_final_goal
                    guidance_index = min(
                        cur_goal + goal_lookahead_steps - 1,
                        len(goals) - 1)
                cur_xyz = goals[guidance_index]
                to_goal = cur_xyz[:2] - base_xyz[:2]

            if campaign_mode:
                current_campaign_progress = (
                    campaign_stage_index
                    + float(cur_goal) / max(len(goals), 1))
                best_campaign_progress_fraction = max(
                    best_campaign_progress_fraction,
                    current_campaign_progress / len(campaign.stages))

            if up_only_reset:
                reset_reason = choose_up_only_reset(
                    summit_reached=summit_reached,
                    base_height=base_xyz[2],
                    fall_reset_height=fall_reset_height,
                    seconds_since_progress=sim_time - last_progress_time,
                    no_progress_timeout=no_progress_reset_seconds)
                if reset_reason is not None:
                    resets += 1
                    if reset_reason == RESET_SUMMIT:
                        cycles += 1
                        reset_label = (
                            f"登顶成功，第 {cycles} 次上楼完成")
                    else:
                        failures += 1
                        if reset_reason == RESET_FALL:
                            fall_resets += 1
                            reset_label = (
                                f"跌倒恢复（z={base_xyz[2]:.2f} m）")
                        elif reset_reason == RESET_STUCK:
                            stuck_resets += 1
                            reset_label = (
                                f"无前进恢复（{no_progress_reset_seconds:g}s）")
                        else:  # pragma: no cover - guarded by pure helper
                            raise RuntimeError(
                                f"unknown reset reason: {reset_reason}")

                    # Reset only MuJoCo/controller episode state. The viewer,
                    # EGL renderer and MJPEG server stay alive, so a browser
                    # keeps receiving the next uphill attempt seamlessly.
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = initial_qpos
                    data.qvel[:] = initial_qvel
                    data.ctrl[:] = 0.0
                    mujoco.mj_forward(model, data)
                    action.fill(0.0)
                    target_q[:] = default_dof_pos
                    action_initialized = False
                    hist_obs.clear()
                    hist_obs.extend(
                        np.zeros(
                            [1, num_single_obs], dtype=np.double)
                        for _ in range(frame_stack))
                    command.cmd[:] = [nominal_command_x, 0.0, 0.0]
                    cur_goal = 0
                    best_progress_x = float(initial_qpos[0])
                    last_progress_time = sim_time

                    q, dq, quat, v, omega, gvec, base_xyz = get_obs(data)
                    q = q[-num_actions:]
                    dq = dq[-num_actions:]
                    yaw = 2.0 * math.atan2(quat[2], quat[3])
                    guidance_index = min(
                        goal_lookahead_steps - 1, len(goals) - 1)
                    cur_xyz = goals[guidance_index]
                    to_goal = cur_xyz[:2] - base_xyz[:2]
                    print(
                        f"\n[parkour] {reset_label}；自动重置 #{resets} "
                        f"(cycles={cycles}, failures={failures}, "
                        f"fall={fall_resets}, stuck={stuck_resets})")

            if campaign_mode and not campaign_finished:
                terrain_under_base = float(campaign.sample_heights(
                    np.asarray([base_xyz[:2]], dtype=np.float64))[0])
                relative_base_height = (
                    float(base_xyz[2]) - terrain_under_base)
                upright_cos = float(-gvec[2])
                stage_timeout = campaign_setting(
                    campaign_no_progress_seconds,
                    campaign_stage.kind, no_progress_reset_seconds)
                if stage_timeout < 0.0:
                    raise ValueError(
                        "campaign no-progress timeouts must be non-negative")
                transition = choose_campaign_transition(
                    final_goal_stable=campaign_final_goal_stable,
                    relative_base_height=relative_base_height,
                    min_base_clearance=campaign_min_base_clearance,
                    upright_cos=upright_cos,
                    min_upright_cos=campaign_min_upright_cos,
                    seconds_since_progress=sim_time - last_progress_time,
                    no_progress_timeout=stage_timeout,
                    in_reset_grace=(
                        sim_time - stage_started_time
                        < campaign_reset_grace_seconds))
                if transition == RESET_STAGE_CLEAR:
                    reached += 1
                    stage_clears += 1
                    cleared_stage = campaign_stage
                    cleared_attempt = campaign_attempt
                    campaign_stage_clear_attempts[
                        cleared_stage.index] = cleared_attempt
                    final_goal_hold_started = None
                    if campaign_stage_index == len(campaign.stages) - 1:
                        cycles += 1
                        campaign_finished = True
                        command.cmd[:] = [0.0, 0.0, 0.0]
                        # Finite evaluations return immediately. An infinite
                        # viewer/MJPEG run remains connected at FINISH with a
                        # zero command and never teleports by default.
                        completed = not run_forever
                        print(
                            f"\n[campaign] 全部通关：row={cleared_stage.row} "
                            f"最后 col={cleared_stage.col} "
                            f"type={CAMPAIGN_KIND_LABELS.get(cleared_stage.kind, cleared_stage.kind)} "
                            f"attempt={cleared_attempt}；"
                            f"stage clears={stage_clears}, "
                            f"failures={failures}")
                        if campaign_restart_on_finish:
                            print(
                                "[campaign] campaign_restart_on_finish=true "
                                "当前版本仍在终点保持；连续跑道不会自动传送。")
                    else:
                        campaign_stage_index += 1
                        campaign_stage = campaign.stages[
                            campaign_stage_index]
                        campaign_attempt = 1
                        goals = np.asarray(
                            campaign_stage.goals[1:],
                            dtype=np.float64)
                        if len(goals) == 0:
                            raise ValueError(
                                "campaign stage requires at least one "
                                "target goal")
                        cur_goal = 0
                        command.cmd[:] = [
                            campaign_setting(
                                campaign_command_speeds,
                                campaign_stage.kind,
                                nominal_command_x),
                            0.0, 0.0]
                        if command.cmd[0] <= 0.0:
                            raise ValueError(
                                "campaign command speeds must be positive")
                        best_progress_x = float(base_xyz[0])
                        last_progress_time = sim_time
                        stage_started_time = sim_time
                        guidance_index = min(
                            campaign_goal_lookahead_steps - 1,
                            len(goals) - 1)
                        cur_xyz = goals[guidance_index]
                        to_goal = cur_xyz[:2] - base_xyz[:2]
                        print(
                            f"\n[campaign] CLEAR stage "
                            f"{cleared_stage.index + 1}/"
                            f"{len(campaign.stages)} "
                            f"(row={cleared_stage.row}, "
                            f"col={cleared_stage.col}, "
                            f"type={CAMPAIGN_KIND_LABELS.get(cleared_stage.kind, cleared_stage.kind)}, "
                            f"attempt={cleared_attempt})；"
                            "连续进入下一段，不传送。")
                        print(
                            f"[campaign] START stage "
                            f"{campaign_stage.index + 1}/"
                            f"{len(campaign.stages)} "
                            f"row={campaign_stage.row} "
                            f"col={campaign_stage.col} "
                            f"type={CAMPAIGN_KIND_LABELS.get(campaign_stage.kind, campaign_stage.kind)} "
                            f"attempt={campaign_attempt} "
                            f"speed={command.cmd[0]:.2f}m/s")
                elif transition is not None:
                    failures += 1
                    resets += 1
                    campaign_stage_failures[campaign_stage.index] += 1
                    campaign_attempt += 1
                    if transition == RESET_LOW_CLEARANCE:
                        low_clearance_resets += 1
                        reason_label = (
                            "离地高度过低 "
                            f"({relative_base_height:.2f}m)")
                    elif transition == RESET_TILT:
                        tilt_resets += 1
                        reason_label = (
                            f"严重倾倒 (upright={upright_cos:.2f})")
                    elif transition == RESET_STUCK:
                        stuck_resets += 1
                        reason_label = (
                            f"{stage_timeout:g}s 无进展")
                    else:  # pragma: no cover - guarded by pure helper
                        raise RuntimeError(
                            f"unknown campaign transition: {transition}")

                    # A failed stage alone is retried. Successful boundaries
                    # above never touch simulator/controller/history state.
                    spawn = np.asarray(
                        campaign_stage.goals[0], dtype=np.float64)
                    mujoco.mj_resetData(model, data)
                    data.qpos[:2] = spawn[:2]
                    data.qpos[2] = spawn[2] + start_height
                    data.qpos[7:] = default_dof_pos
                    data.qvel[:] = 0.0
                    data.ctrl[:] = 0.0
                    mujoco.mj_forward(model, data)
                    action.fill(0.0)
                    target_q[:] = default_dof_pos
                    action_initialized = False
                    hist_obs.clear()
                    hist_obs.extend(
                        np.zeros(
                            [1, num_single_obs], dtype=np.double)
                        for _ in range(frame_stack))
                    command.cmd[:] = [
                        campaign_setting(
                            campaign_command_speeds,
                            campaign_stage.kind,
                            nominal_command_x),
                        0.0, 0.0]
                    cur_goal = 0
                    final_goal_hold_started = None
                    best_progress_x = float(spawn[0])
                    last_progress_time = sim_time
                    stage_started_time = sim_time

                    q, dq, quat, v, omega, gvec, base_xyz = get_obs(
                        data)
                    q = q[-num_actions:]
                    dq = dq[-num_actions:]
                    yaw = 2.0 * math.atan2(quat[2], quat[3])
                    guidance_index = min(
                        campaign_goal_lookahead_steps - 1,
                        len(goals) - 1)
                    cur_xyz = goals[guidance_index]
                    to_goal = cur_xyz[:2] - base_xyz[:2]
                    print(
                        f"\n[campaign] RETRY stage "
                        f"{campaign_stage.index + 1}/"
                        f"{len(campaign.stages)} "
                        f"row={campaign_stage.row} "
                        f"col={campaign_stage.col} "
                        f"type={CAMPAIGN_KIND_LABELS.get(campaign_stage.kind, campaign_stage.kind)} "
                        f"attempt={campaign_attempt}：{reason_label}；"
                        f"failures={failures}")

            if campaign_mode and campaign_finished:
                # Match N2ParkourEnv's terminal observation: once the final
                # goal is crossed, hide the now-behind waypoint instead of
                # asking a zero-speed policy to turn around toward it.
                dpsi = 0.0
            else:
                target_yaw = math.atan2(to_goal[1], to_goal[0])
                dpsi = math.atan2(
                    math.sin(target_yaw - yaw),
                    math.cos(target_yaw - yaw))

            obs = np.zeros([1, num_single_obs], dtype=np.float32)
            obs[0, :3] = command.cmd * cmd_scale
            obs[0, 3:6] = omega * ang_vel_scale
            obs[0, 6:9] = gvec[:3]
            obs[0, 9:9 + num_actions] = (q - default_dof_pos) * dof_pos_scale
            obs[0, 9 + num_actions:9 + num_actions * 2] = dq * dof_vel_scale
            obs[0, 9 + num_actions * 2:9 + num_actions * 3] = action
            # goal 两维紧跟在 actions 之后，与 N2ParkourEnv.compute_observations 的顺序一致
            obs[0, 9 + num_actions * 3]     = math.cos(dpsi)
            obs[0, 9 + num_actions * 3 + 1] = math.sin(dpsi)
            if campaign_mode:
                height_scan, campaign_points_world = (
                    campaign_height_scan(
                        campaign, base_xyz, quat, height_points,
                        base_height_offset, height_clip,
                        height_measurements_scale))
                obs[0, 9 + num_actions * 3 + 2:] = height_scan
                if debug_viz:
                    height_marker_world[:] = campaign_points_world
            else:
                obs[0, 9 + num_actions * 3 + 2:] = get_height_scan(
                    model, data, base_xyz, quat, height_points,
                    base_height_offset, height_clip,
                    height_measurements_scale)

            hist_obs.append(obs)
            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single_obs:(i + 1) * num_single_obs] = hist_obs[i][0, :]
            model_input = np.clip(
                model_input, -clip_observations, clip_observations)
            with torch.inference_mode():
                policy_action = policy(
                    torch.from_numpy(model_input))[0].numpy()
            policy_action = np.clip(
                policy_action, -clip_actions, clip_actions) * action_gain
            if action_initialized:
                action[:] = (
                    action_filter_alpha * policy_action
                    + (1.0 - action_filter_alpha) * action)
            else:
                action[:] = policy_action
                action_initialized = True
            target_q = action * action_scale + default_dof_pos

            if not quiet_progress and step % 500 == 0:
                if campaign_mode:
                    tgt = (
                        "stage %d/%d row %d col %d %s "
                        "goal %d/%d attempt %d clears %d cycles %d"
                        % (
                            campaign_stage.index + 1,
                            len(campaign.stages),
                            campaign_stage.row,
                            campaign_stage.col,
                            CAMPAIGN_KIND_LABELS.get(
                                campaign_stage.kind,
                                campaign_stage.kind),
                            cur_goal, len(goals) - 1,
                            campaign_attempt, stage_clears, cycles))
                elif goal_mode == "carrot":
                    tgt = (
                        "carrot(%.2f,%.2f)"
                        % (cur_xyz[0], cur_xyz[1]))
                else:
                    tgt = (
                        "%s goal %d/%d cycle %d"
                        % (
                            goal_phases[cur_goal]
                            if repeat_course else "UP",
                            cur_goal, len(goals) - 1, cycles))
                print(
                    f"  x={base_xyz[0]:.2f} y={base_xyz[1]:+.2f} "
                    f"z={base_xyz[2]:.2f} {tgt} "
                    f"yaw={math.degrees(yaw):+.0f}deg "
                    f"dpsi={math.degrees(dpsi):+.0f}deg vx={v[0]:+.2f} "
                    f"resets={resets} failures={failures}")

        tau = pd_control(
            target_q, q, kps,
            np.zeros(num_actions, dtype=np.double), dq, kds * kd_scale)
        if tau_limit is not None:
            tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        if viewer is not None and count_lowlevel % render_decimation == 0:
            camera_lookat, camera_azimuth = viewer_follow.update(
                data.qpos[:3], yaw,
                simulation_dt * render_decimation)
            viewer.cam.lookat[:] = camera_lookat
            # MuJoCo azimuth=90° 是朝 +x 行走时机器人的右侧（世界 -y）。
            # 跟随实际 yaw 旋转并平滑最短角度，始终在机器人右侧。
            viewer.cam.azimuth = camera_azimuth
            if debug_viz:
                if not campaign_mode:
                    height_marker_world[:] = get_height_points_world(
                        model, data, base_xyz, quat, height_points)
                for px, py, pz in height_marker_world:
                    viewer.add_marker(
                        pos=[px, py, pz], size=[0.02, 0.02, 0.02],
                        rgba=[1, 1, 0, 1],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
            # 目标可视化：红色大球=当前目标（与 play.py 配色一致），绿色小球=后续路点
            viewer.add_marker(pos=[cur_xyz[0], cur_xyz[1], cur_xyz[2] + 0.08],
                              size=[0.12, 0.12, 0.12], rgba=[1, 0, 0, 1],
                              type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
            if goals is not None:
                if repeat_course:
                    upcoming = [
                        goals[(cur_goal + offset) % len(goals)]
                        for offset in range(
                            1, min(len(goals), 21))]
                else:
                    upcoming = goals[cur_goal + 1:]
                for g in upcoming:
                    viewer.add_marker(pos=[g[0], g[1], g[2] + 0.08], size=[0.05, 0.05, 0.05],
                                      rgba=[0, 1, 0, 0.6], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
            viewer.render()

        if streamer is not None and count_lowlevel % stream_decimation == 0:
            camera_lookat, camera_azimuth = stream_follow.update(
                data.qpos[:3], yaw,
                simulation_dt * stream_decimation)
            stream_camera.lookat[:] = camera_lookat
            stream_camera.azimuth = camera_azimuth
            stream_renderer.update_scene(data, camera=stream_camera)
            frame = stream_renderer.render()
            if campaign_mode:
                goal_label = (
                    ("FINISH · " if campaign_finished else "")
                    + f"stage {campaign_stage.index + 1}/"
                    f"{len(campaign.stages)} · "
                    f"row {campaign_stage.row} · "
                    f"col {campaign_stage.col} · "
                    f"{CAMPAIGN_KIND_LABELS.get(campaign_stage.kind, campaign_stage.kind)} · "
                    f"goal {cur_goal}/{len(goals) - 1} · "
                    f"attempt {campaign_attempt} · "
                    f"clears {stage_clears} · failures {failures}")
            else:
                goal_label = (
                    "carrot" if goals is None
                    else (
                        f"{goal_phases[cur_goal] if repeat_course else 'UP'} "
                        f"{cur_goal}/{len(goals) - 1} · cycle {cycles}"
                        + (
                            f" · reset {resets} · failure {failures}"
                            if up_only_reset else "")))
            stream_status = {
                "sim_time": (step + 1) * simulation_dt,
                "x": float(data.qpos[0]),
                "y": float(data.qpos[1]),
                "z": float(data.qpos[2]),
                "camera_lookat": [
                    float(value) for value in camera_lookat],
                "camera_azimuth": float(camera_azimuth),
                "goal": goal_label,
                "reached": reached,
                "cycles": cycles,
                "resets": resets,
                "failures": failures,
            }
            if campaign_mode:
                stream_status.update({
                    "row": campaign_stage.row,
                    "col": campaign_stage.col,
                    "terrain_type": campaign_stage.kind,
                    "stage": campaign_stage.index + 1,
                    "stage_count": len(campaign.stages),
                    "goal_index": cur_goal,
                    "goal_count": len(goals),
                    "attempt": campaign_attempt,
                    "stage_clears": stage_clears,
                    "finished": campaign_finished,
                })
            streamer.publish(
                frame,
                stream_status)
        count_lowlevel += 1

        # Keep browser playback near real time without sleeping at all 500
        # physics steps. If simulation/rendering is slower, never add delay.
        if streamer is not None and count_lowlevel % control_decimation == 0:
            target_wall = (
                stream_start_wall + count_lowlevel * simulation_dt)
            delay = target_wall - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        if completed:
            if campaign_mode:
                print(
                    "\n[campaign] SUCCESS：五类连续地形全部通过，"
                    "已到达终点。")
            else:
                print(
                    f"\n[parkour] SUCCESS：已到达最终 goal "
                    f"({cur_goal}/{len(goals) - 1})")
            break

    if viewer is not None:
        viewer.close()
    if streamer is not None:
        streamer.close()
    if stream_renderer is not None:
        stream_renderer._mjr_context.free()
        stream_renderer._gl_context.free()
    if campaign_mode:
        tail = (
            f"连续闯关通过 {stage_clears}/{len(campaign.stages)} 段，"
            f"完成 {cycles} 次全程，失败 {failures} 次"
            f"（离地过低 {low_clearance_resets}、"
            f"倾倒 {tilt_resets}、无进展 {stuck_resets}），")
    elif goals is not None and up_only_reset:
        tail = (
            f"经过 {reached} 个路点，完成 {cycles} 次上楼，"
            f"自动重置 {resets} 次、失败 {failures} 次"
            f"（跌倒 {fall_resets}、无前进 {stuck_resets}），")
    elif goals is not None and repeat_course:
        tail = f"经过 {reached} 个路点，完成 {cycles} 次完整往返，"
    else:
        tail = f"到达 {reached} 个 goal，" if goals is not None else ""
    print(f"\n[parkour] 结束：{tail}最终 x={data.qpos[0]:.2f} z={data.qpos[2]:.2f}")
    max_x = max(max_x, float(data.qpos[0]))
    campaign_stage_count = (
        len(campaign.stages) if campaign_mode else None)
    if campaign_mode:
        current_campaign_progress_fraction = (
            1.0 if campaign_finished else (
                campaign_stage_index
                + float(cur_goal) / max(len(goals), 1))
                / campaign_stage_count)
        best_campaign_progress_fraction = max(
            best_campaign_progress_fraction,
            current_campaign_progress_fraction)
        stage_failure_counts = {
            stage.kind: campaign_stage_failures[stage.index]
            for stage in campaign.stages}
        stage_clear_attempts = {
            stage.kind: campaign_stage_clear_attempts[stage.index]
            for stage in campaign.stages}
        course_end_x = float(campaign.stages[-1].goals[-1, 0])
    else:
        current_campaign_progress_fraction = None
        best_campaign_progress_fraction = None
        stage_failure_counts = None
        stage_clear_attempts = None
        course_end_x = None

    return {
        "completed": completed,
        "stop_reason": (
            "success" if completed else "duration_limit"),
        "sim_steps": count_lowlevel,
        "sim_time": float(count_lowlevel * simulation_dt),
        "reached": reached,
        "cycles": cycles,
        "resets": resets,
        "failures": failures,
        "fall_resets": fall_resets,
        "stuck_resets": stuck_resets,
        "low_clearance_resets": low_clearance_resets,
        "tilt_resets": tilt_resets,
        "stage_clears": stage_clears,
        "campaign_finished": campaign_finished,
        "stage": (
            campaign_stage.index if campaign_mode else None),
        "stage_number": (
            campaign_stage.index + 1 if campaign_mode else None),
        "stage_count": campaign_stage_count,
        "terrain_type": (
            campaign_stage.kind if campaign_mode else None),
        "goal_index": (
            cur_goal if goals is not None else None),
        "goal_count": (
            len(goals) if goals is not None else None),
        "stage_attempt": (
            campaign_attempt if campaign_mode else None),
        "stage_failure_counts": stage_failure_counts,
        "stage_clear_attempts": stage_clear_attempts,
        "final_progress_fraction": current_campaign_progress_fraction,
        "best_progress_fraction": best_campaign_progress_fraction,
        "course_end_x": course_end_x,
        "max_x": max_x,
        "x": float(data.qpos[0]),
        "y": float(data.qpos[1]),
        "z": float(data.qpos[2]),
    }

def main(force_stream=False):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_parkour.yaml",
                        help="config file name in sim2sim/configs")
    parser.add_argument("--goal_mode", choices=("waypoints", "carrot"),
                        help="override goal_mode from the yaml")
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument(
        "--campaign_mode", dest="course_mode", action="store_const",
        const="campaign",
        help="run the five-stage continuous Isaac Gym terrain campaign")
    course_group.add_argument(
        "--up_only_reset", dest="course_mode", action="store_const",
        const="up_only",
        help="run the preserved climb-and-reset staircase demo")
    course_group.add_argument(
        "--repeat_course", dest="course_mode", action="store_const",
        const="repeat",
        help="run the preserved up/turn/down closed route")
    course_group.add_argument(
        "--single_course", dest="course_mode", action="store_const",
        const="single",
        help="disable campaign/repeat/reset route state machines")
    parser.add_argument("--check_only", action="store_true",
                        help="validate XML, observation dimensions, joint order and policy I/O, then exit")
    parser.add_argument("--headless", action="store_true",
                        help="run the MuJoCo simulation without creating a viewer")
    parser.add_argument("--duration", type=float,
                        help="override simulation duration in seconds")
    parser.add_argument("--policy_path",
                        help="override policy path; relative paths are resolved from the repository root")
    parser.add_argument(
        "--command_x", type=float,
        help="forward command in m/s (default: stable value from config)")
    parser.add_argument("--render_fps", type=float,
                        help="override viewer refresh rate (physics remains 500Hz)")
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug_viz", dest="debug_viz", action="store_true",
        help="draw all 96 height-scan markers (slower)")
    debug_group.add_argument(
        "--no_debug_viz", dest="debug_viz", action="store_false",
        help="hide height-scan markers (faster)")
    parser.add_argument("--stream", action="store_true",
                        help="serve an EGL-rendered MJPEG stream instead of opening a window")
    parser.add_argument("--stream_host", default="127.0.0.1",
                        help="HTTP bind address (use 127.0.0.1 with an SSH tunnel)")
    parser.add_argument("--stream_port", type=int, default=8000)
    parser.add_argument("--stream_width", type=int, default=640)
    parser.add_argument("--stream_height", type=int, default=360)
    parser.add_argument("--stream_fps", type=float, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument(
        "--manual_start", action="store_true",
        help="wait for the browser's Start button before the first physics step")
    parser.add_argument(
        "--camera_distance", type=float,
        help="stream camera distance (default: value from config)")
    parser.add_argument(
        "--camera_azimuth_offset", type=float,
        help="camera angle relative to robot yaw; 90 degrees is its right side")
    parser.add_argument(
        "--camera_elevation", type=float,
        help="stream camera elevation in degrees (negative is above)")
    # stream_parkour.py consumes this before importing MuJoCo; keeping it in
    # this parser also documents it and avoids an unknown-argument error.
    parser.add_argument("--egl_device", type=int, default=1,
                        help="EGL GPU index selected before MuJoCo import")
    parser.set_defaults(debug_viz=None)
    args = parser.parse_args()
    args.stream = bool(args.stream or force_stream)
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{args.config_file}") as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    mode = args.goal_mode or _cfg.get("goal_mode", "waypoints")
    course_mode = args.course_mode
    if args.goal_mode == "carrot" and course_mode is None:
        # An explicit interactive carrot request should not inherit the
        # campaign default from the yaml.
        course_mode = "single"

    command = cmd()
    # vy/wz are zero in parkour training; direction comes from goal yaw.
    command_x = (
        args.command_x
        if args.command_x is not None
        else float(_cfg.get(
            "sim_command_x",
            _cfg.get("ranges_lin_vel_x_max", 0.8))))
    command.cmd[:] = [command_x, 0.0, 0.0]

    if args.check_only:
        run_mujoco(
            args.config_file, command, goal_mode_override=mode,
            check_only=True, policy_path_override=args.policy_path,
            render_fps_override=args.render_fps,
            debug_viz_override=args.debug_viz,
            course_mode_override=course_mode)
        raise SystemExit(0)

    carrot = None
    if mode == "carrot":
        carrot = Carrot(lookahead=float(_cfg.get("carrot_lookahead", 1.5)),
                        step=float(_cfg.get("carrot_step", 0.15)))
        listener = (
            Listener(on_press=carrot.on_press)
            if not args.headless and not args.stream else None)
    else:
        listener = (
            Listener(on_press=command.cmd_swtich)
            if not args.headless and not args.stream else None)
    if listener is not None:
        listener.start()
    stream_options = None
    if args.stream:
        stream_options = {
            "host": args.stream_host,
            "port": args.stream_port,
            "width": args.stream_width,
            "height": args.stream_height,
            "fps": args.stream_fps,
            "jpeg_quality": args.jpeg_quality,
            "manual_start": args.manual_start,
            "distance": (
                args.camera_distance
                if args.camera_distance is not None
                else float(_cfg.get("camera_distance", 3.2))),
            "azimuth_offset": (
                args.camera_azimuth_offset
                if args.camera_azimuth_offset is not None
                else float(_cfg.get("camera_azimuth_offset", 90.0))),
            "elevation": (
                args.camera_elevation
                if args.camera_elevation is not None
                else float(_cfg.get("camera_elevation", -22.0))),
        }
    try:
        run_mujoco(
            args.config_file, command, carrot, goal_mode_override=mode,
            headless=args.headless or args.stream,
            duration_override=args.duration,
            policy_path_override=args.policy_path,
            render_fps_override=args.render_fps,
            debug_viz_override=args.debug_viz,
            stream_options=stream_options,
            course_mode_override=course_mode)
    except KeyboardInterrupt:
        print("\n[parkour] 用户停止仿真。")


# TODO: add more terrains and align height map
if __name__ == '__main__':
    main()
