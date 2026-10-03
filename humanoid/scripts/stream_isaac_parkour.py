#!/usr/bin/env python3
"""Stream the Isaac Gym continuous parkour course to a web browser.

No native viewer or X11 display is created.  A GPU camera sensor follows the
robot and publishes RGB frames through the same lightweight MJPEG server used
by the MuJoCo evaluator.
"""

import math
import os
import sys
import time


REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymutil  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid.envs import *  # noqa: E402,F401,F403 - task registration
from humanoid.utils.task_registry import task_registry  # noqa: E402
from sim2sim.mjpeg_stream import MJPEGStreamer  # noqa: E402
from sim2sim.stream_gate import wait_for_stream_start  # noqa: E402


COURSE_TASK = "n2_parkour_course"
COURSE_TASKS = (
    COURSE_TASK,
    "n2_parkour_slow_stable_course",
)
MAX_PACING_SLEEP = 0.05


def _resolve_policy_path(path, task=COURSE_TASK):
    if path is None:
        if task == "n2_parkour_slow_stable_course":
            path = os.path.join(
                REPOSITORY_ROOT, "logs", "n2_parkour_slow_stable",
                "exported", "policies", "policy_best_stable.pt")
        else:
            path = os.path.join(
                REPOSITORY_ROOT, "logs", "n2_parkour", "exported",
                "policies", "policy_11600_stability.pt")
    elif not os.path.isabs(path):
        path = os.path.join(REPOSITORY_ROOT, path)
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到导出策略: {}".format(path))
    return path


def _validate_args(args):
    if args.task not in COURSE_TASKS:
        raise ValueError(
            "本脚本只支持连续跑道任务: {}".format(
                ", ".join(COURSE_TASKS)))
    if not 1 <= args.stream_port <= 65535:
        raise ValueError("--stream_port 必须在 [1, 65535]")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("相机宽高必须为正数")
    if not 1.0 <= args.stream_fps <= 50.0:
        raise ValueError("--stream_fps 必须在 [1, 50]")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality 必须在 [1, 100]")
    if args.camera_rear <= 0 or args.camera_side < 0 or args.camera_up <= 0:
        raise ValueError("相机 rear/up 必须为正，side 不能为负")
    if args.camera_boundary_clearance <= 0:
        raise ValueError("--camera_boundary_clearance 必须为正")
    if args.play_steps < 0:
        raise ValueError("--play_steps 不能为负")
    if args.manual_start and args.start_immediately:
        raise ValueError(
            "--manual_start 与 --start_immediately 不能同时使用")


def _yaw_from_xyzw(quat):
    x, y, z, w = (float(value) for value in quat)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z))


def _sleep_until(deadline, clock=time.monotonic, sleeper=time.sleep):
    """Sleep to an absolute wall-clock deadline without accumulating drift."""
    while True:
        remaining = deadline - clock()
        if remaining <= 0.0:
            return
        # Short chunks keep Ctrl-C responsive even if a future policy timestep
        # or a temporarily stalled clock produces an unexpectedly long wait.
        sleeper(min(remaining, MAX_PACING_SLEEP))


def _realtime_deadline(wall_start, sim_start, sim_time):
    """Map an absolute simulation timestamp to its 1x wall-clock deadline."""
    return wall_start + max(0.0, sim_time - sim_start)


def _measured_rtf(wall_start, sim_start, sim_time, wall_time):
    sim_elapsed = max(0.0, sim_time - sim_start)
    wall_elapsed = max(0.0, wall_time - wall_start)
    if wall_elapsed <= 1e-9:
        return 0.0
    return sim_elapsed / wall_elapsed


def _wait_for_playback_start(streamer, args):
    """Apply the selected browser/start policy before the first sim step."""
    if args.start_immediately:
        return True
    return wait_for_stream_start(
        streamer, manual_start=args.manual_start)


def _follow_camera_pose(root, args, lateral_bounds=None):
    """Return a true robot-right/rear pose following root XYZ in real time."""
    root = np.asarray(root, dtype=np.float64)
    yaw = _yaw_from_xyzw(root[3:7])
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    # For yaw=0, robot-right is world -Y.
    right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0])
    position = (
        root[:3]
        - args.camera_rear * forward
        + args.camera_side * right
        + np.asarray([0.0, 0.0, args.camera_up]))
    target = (
        root[:3]
        + args.camera_lookahead * forward
        + np.asarray([0.0, 0.0, args.camera_target_up]))
    if lateral_bounds is not None:
        low, high = (float(value) for value in lateral_bounds)
        if not low < high:
            raise ValueError("相机横向边界必须满足 low < high")
        position[1] = np.clip(position[1], low, high)
    return position, target


def _set_follow_camera(env, args):
    root = env.root_states[0, :7].detach().cpu().numpy()
    lateral_bounds = None
    if getattr(env.cfg.terrain, 'course_mode', False):
        width = float(env.terrain.env_width)
        clearance = float(args.camera_boundary_clearance)
        if width <= 2.0 * clearance:
            raise ValueError(
                "course 宽度 %.3f 无法容纳相机边界余量 %.3f"
                % (width, clearance))
        lateral_bounds = (clearance, width - clearance)
    position, target = _follow_camera_pose(
        root, args, lateral_bounds=lateral_bounds)
    env.gym.set_camera_location(
        env.camera_handle, env.envs[0],
        gymapi.Vec3(*position), gymapi.Vec3(*target))
    return position, target


def _course_status(
        env, attempts, failures, rtf=0.0,
        camera_position=None, camera_target=None):
    stage = int(env.course_stage_idx[0].item())
    finished = bool(env.course_finished[0].item())
    global_goal = int(env.cur_goal_idx[0].item())
    goal_start = int(env.terrain.course_stage_goal_starts[stage])
    local_goal = max(0, global_goal - goal_start + 1)
    campaign_stage = env.terrain.campaign.stages[stage]
    clears = len(env.terrain.course_stage_names) if finished else stage
    root = env.root_states[0]
    label = (
        ("FINISH · " if finished else "")
        + "stage {}/{} · row {} · col {} · {} · goal {}/9 · "
          "attempt {} · clears {} · failures {}".format(
              stage + 1, len(env.terrain.course_stage_names),
              env.cfg.terrain.course_row, campaign_stage.col,
              env.terrain.course_stage_names[stage], local_goal,
              attempts[stage], clears, failures))
    status = {
        "sim_time": float(env.gym.get_sim_time(env.sim)),
        "rtf": float(rtf),
        "x": float(root[0].item()),
        "z": float(root[2].item()),
        "goal": label,
        "reached": global_goal,
        "cycles": int(finished),
        "resets": failures,
        "failures": failures,
        "row": int(env.cfg.terrain.course_row),
        "col": int(campaign_stage.col),
        "terrain_type": env.terrain.course_stage_kinds[stage],
        "stage": stage + 1,
        "stage_count": len(env.terrain.course_stage_names),
        "goal_index": local_goal,
        "goal_count": 10,
        "attempt": attempts[stage],
        "stage_clears": clears,
        "finished": finished,
    }
    if camera_position is not None and camera_target is not None:
        status.update({
            "camera_x": float(camera_position[0]),
            "camera_y": float(camera_position[1]),
            "camera_z": float(camera_position[2]),
            "camera_target_x": float(camera_target[0]),
            "camera_target_y": float(camera_target[1]),
            "camera_target_z": float(camera_target[2]),
        })
    return status


def stream(args):
    _validate_args(args)
    policy_path = _resolve_policy_path(args.policy_path, task=args.task)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.enable_camera_sensors = True
    env_cfg.env.camera_width = args.camera_width
    env_cfg.env.camera_height = args.camera_height
    env_cfg.env.camera_horizontal_fov = args.camera_fov

    # No native viewer is created.  enable_camera_sensors keeps only the
    # graphics device required by render_all_camera_sensors().
    args.headless = True
    args.num_envs = 1
    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg)
    if env.graphics_device_id < 0 or env.camera_handle is None:
        raise RuntimeError(
            "Isaac Gym 离屏图形设备/相机没有成功创建")

    policy = torch.jit.load(policy_path, map_location=env.device)
    policy.eval()
    obs, _ = env.reset()
    with torch.inference_mode():
        probe = policy(obs.detach())
    if tuple(probe.shape) != (1, env.num_actions):
        raise ValueError(
            "策略输出维度 {}，期望 (1, {})".format(
                tuple(probe.shape), env.num_actions))

    streamer = MJPEGStreamer(
        host=args.stream_host,
        port=args.stream_port,
        jpeg_quality=args.jpeg_quality,
        manual_start=args.manual_start,
        page_title="N2 Isaac Gym Live",
        heading="N2 Isaac Gym 连续跑道实时画面")
    streamer.start()
    print(
        "[isaac-stream] task={} obs={} actions={} graphics={} camera={}x{}"
        .format(
            args.task, obs.shape[-1], env.num_actions,
            env.graphics_device_id, args.camera_width,
            args.camera_height))
    print("[isaac-stream] policy={}".format(policy_path))
    print(
        "[isaac-stream] camera=robot-right rear={:.2f} side={:.2f} "
        "up={:.2f}; XYZ follows root every rendered frame".format(
            args.camera_rear, args.camera_side, args.camera_up))
    print(
        "[isaac-stream] HTTP server: http://{}:{}"
        .format(args.stream_host, args.stream_port))

    attempts = [1] * len(env.terrain.course_stage_names)
    failures = 0
    previous_stage = int(env.course_stage_idx[0].item())
    frame_period = 1.0 / args.stream_fps
    next_frame_time = 0.0
    step = 0

    try:
        if not _wait_for_playback_start(streamer, args):
            return

        # Anchor pacing only when simulation playback really begins.  In the
        # browser-gated modes this deliberately happens after the client and,
        # for manual mode, after the explicit start click.  Time spent waiting
        # can therefore never make playback catch up.
        sim_start = float(env.gym.get_sim_time(env.sim))
        wall_start = time.monotonic()
        print("[isaac-stream] 墙钟实时节流已启用（目标 RTF=1.00）。")

        while args.play_steps == 0 or step < args.play_steps:
            env._apply_course_commands()
            with torch.inference_mode():
                actions = policy(obs.detach())
            obs, _, _, _, _, termination_ids, _ = env.step(
                actions.detach())

            stage = int(env.course_stage_idx[0].item())
            if stage != previous_stage:
                print(
                    "[isaac-stream] CLEAR stage {}/{}；START stage {}/{} {}"
                    .format(
                        previous_stage + 1,
                        len(env.terrain.course_stage_names),
                        stage + 1,
                        len(env.terrain.course_stage_names),
                        env.terrain.course_stage_names[stage]))
                previous_stage = stage
            if len(termination_ids) > 0:
                failures += int(len(termination_ids))
                attempts[stage] += int(len(termination_ids))
                print(
                    "[isaac-stream] RESET stage {}/{} attempt={} failures={}"
                    .format(
                        stage + 1, len(env.terrain.course_stage_names),
                        attempts[stage], failures))

            sim_time = float(env.gym.get_sim_time(env.sim))
            _sleep_until(
                _realtime_deadline(
                    wall_start, sim_start, sim_time))
            rtf = _measured_rtf(
                wall_start, sim_start, sim_time, time.monotonic())
            if sim_time + 1e-9 >= next_frame_time:
                env.gym.fetch_results(env.sim, True)
                camera_position, camera_target = _set_follow_camera(env, args)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                raw = env.gym.get_camera_image(
                    env.sim, env.envs[0], env.camera_handle,
                    gymapi.IMAGE_COLOR)
                rgba = np.asarray(raw, dtype=np.uint8).reshape(
                    args.camera_height, args.camera_width, 4)
                streamer.publish(
                    np.ascontiguousarray(rgba[:, :, :3]),
                    _course_status(
                        env, attempts, failures, rtf=rtf,
                        camera_position=camera_position,
                        camera_target=camera_target))
                while next_frame_time <= sim_time + 1e-9:
                    next_frame_time += frame_period

            step += 1
    except KeyboardInterrupt:
        print("\n[isaac-stream] 用户停止仿真。")
    finally:
        streamer.close()


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": COURSE_TASK},
        {"name": "--policy_path", "type": str, "default": None},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--stream_host", "type": str, "default": "127.0.0.1"},
        {"name": "--stream_port", "type": int, "default": 18081},
        {"name": "--camera_width", "type": int, "default": 640},
        {"name": "--camera_height", "type": int, "default": 360},
        {"name": "--camera_fov", "type": float, "default": 75.0},
        {"name": "--stream_fps", "type": float, "default": 25.0},
        {"name": "--jpeg_quality", "type": int, "default": 85},
        {"name": "--camera_rear", "type": float, "default": 2.6},
        {"name": "--camera_side", "type": float, "default": 3.4},
        {"name": "--camera_up", "type": float, "default": 2.8},
        {"name": "--camera_lookahead", "type": float, "default": 0.8},
        {"name": "--camera_target_up", "type": float, "default": 0.25},
        {
            "name": "--camera_boundary_clearance",
            "type": float,
            "default": 0.35,
        },
        {"name": "--play_steps", "type": int, "default": 0},
        {
            "name": "--start_immediately",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--manual_start",
            "action": "store_true",
            "default": False,
        },
    ]
    args = gymutil.parse_arguments(
        description="N2 Isaac Gym browser stream",
        custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.sim_device_id)
    return args


if __name__ == "__main__":
    stream(get_args())
