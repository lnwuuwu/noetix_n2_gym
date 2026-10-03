#!/usr/bin/env python3
"""Deterministic, headless Isaac Gym acceptance test for parkour checkpoints.

The evaluator keeps the course, perturbation seeds, commands and metrics fixed
across checkpoints.  It runs faster than real time without enabling the
training-only action-delay randomisation, and writes one machine-readable JSON
report that can be regenerated from the recorded command.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import random
import time

# Isaac Gym must be imported before torch.
from isaacgym import gymutil  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from humanoid import LEGGED_GYM_ROOT_DIR  # noqa: E402
from humanoid.envs import *  # noqa: E402,F401,F403 - task registration
from humanoid.utils.helpers import update_class_from_dict  # noqa: E402
from humanoid.utils.task_registry import task_registry  # noqa: E402


DEFAULT_TASK = "n2_parkour_slow_stable_course"
DEFAULT_EXPERIMENT = "n2_parkour_slow_stable"
DEFAULT_CHECKPOINTS = "3000,3400,3800,4000,4200"


def _parse_checkpoint_numbers(value):
    numbers = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        number = int(token)
        if number < 0:
            raise ValueError("checkpoint numbers must be non-negative")
        numbers.append(number)
    if not numbers:
        raise ValueError("--checkpoints must contain at least one number")
    if len(set(numbers)) != len(numbers):
        raise ValueError("--checkpoints contains duplicates")
    return numbers


def _resolve_run_dir(value, experiment):
    path = Path(value)
    if not path.is_absolute():
        path = (
            Path(LEGGED_GYM_ROOT_DIR) / "logs" / experiment / path
        )
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError("run directory does not exist: {}".format(path))
    if not (path / "train_cfg.json").is_file():
        raise FileNotFoundError(
            "run directory has no train_cfg.json: {}".format(path))
    return path


def _cross_track_error(point, previous_goal, current_goal):
    """Distance from an XY point to its active route segment."""
    segment = current_goal - previous_goal
    denom = float(np.dot(segment, segment))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point - current_goal))
    alpha = float(np.dot(point - previous_goal, segment) / denom)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    projection = previous_goal + alpha * segment
    return float(np.linalg.norm(point - projection))


def _rms(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _percentile(values, percentile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _round_metrics(record):
    """Keep reports compact without losing useful acceptance precision."""
    rounded = {}
    for key, value in record.items():
        if isinstance(value, float) and _finite(value):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _reset_course(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.course_stage_idx.zero_()
    env.course_finished.zero_()
    env.cur_goal_idx.fill_(1)
    env.reached_goals.zero_()
    observations, _ = env.reset()
    env._apply_course_commands()
    return observations


def _evaluate_trial(env, policy, checkpoint, trial, seed, max_sim_seconds):
    observations = _reset_course(env, seed)
    max_steps = int(math.ceil(max_sim_seconds / float(env.dt)))
    route = np.asarray(
        env.terrain._course_world_goals[:, :2], dtype=np.float64)

    resets = 0
    height_falls = 0
    contact_terminations = 0
    episode_timeouts = 0
    max_goal = 1
    max_stage = 0
    completed = False

    cross_track = []
    flat_cross_track = []
    flat_yaw_degrees = []
    flat_world_vy = []
    roll_degrees = []
    pitch_degrees = []
    speed_errors = []
    foot_separations = []
    off_route_steps = 0

    wall_start = time.monotonic()
    steps = 0
    for steps in range(1, max_steps + 1):
        goal_before = int(env.cur_goal_idx[0].item())
        max_goal = max(max_goal, goal_before)

        with torch.inference_mode():
            actions = policy(observations.detach())
        env._apply_course_commands()
        observations, _, _, dones, _, termination_ids, _ = env.step(
            actions.detach())

        stage = int(env.course_stage_idx[0].item())
        goal = int(env.cur_goal_idx[0].item())
        max_stage = max(max_stage, stage)
        max_goal = max(max_goal, goal)

        point = env.root_states[0, :2].detach().cpu().numpy()
        active_goal = min(max(goal, 1), len(route) - 1)
        route_error = _cross_track_error(
            point, route[active_goal - 1], route[active_goal])
        cross_track.append(route_error)
        off_route_steps += int(route_error > 0.50)

        roll = math.degrees(float(env.base_euler_xyz[0, 0].item()))
        pitch = math.degrees(float(env.base_euler_xyz[0, 1].item()))
        yaw = math.degrees(float(env.base_euler_xyz[0, 2].item()))
        roll_degrees.append(roll)
        pitch_degrees.append(pitch)
        speed_errors.append(abs(
            float(env.base_lin_vel[0, 0].item())
            - float(env.commands[0, 0].item())))

        yaw_radians = math.radians(yaw)
        feet = env.feet_pos[0, :, :2].detach().cpu().numpy()
        foot_delta = feet[0] - feet[1]
        robot_left = np.asarray(
            [-math.sin(yaw_radians), math.cos(yaw_radians)])
        foot_separations.append(float(np.dot(foot_delta, robot_left)))

        if stage == 3:
            flat_cross_track.append(route_error)
            flat_yaw_degrees.append(yaw)
            flat_world_vy.append(
                float(env.root_states[0, 8].item()))

        if int(dones[0].item()):
            resets += 1
            timed_out = bool(env.time_out_buf[0].item())
            fell_low = bool(
                getattr(env, "fallen_buf", torch.zeros_like(dones))[0]
                .item())
            episode_timeouts += int(timed_out)
            height_falls += int(fell_low and not timed_out)
            contact_terminations += int(not timed_out and not fell_low)

        if bool(env.course_finished[0].item()):
            completed = True
            break

    wall_seconds = time.monotonic() - wall_start
    sim_seconds = steps * float(env.dt)
    cleared_stages = 5 if completed else max_stage
    crossing_steps = sum(value < 0.0 for value in foot_separations)

    return _round_metrics({
        "checkpoint": int(checkpoint),
        "trial": int(trial),
        "seed": int(seed),
        "completed": bool(completed),
        "cleared_stages": int(cleared_stages),
        "furthest_stage": int(max_stage + 1),
        "furthest_goal": int(max_goal),
        "sim_seconds": float(sim_seconds),
        "wall_seconds": float(wall_seconds),
        "realtime_factor": (
            float(sim_seconds / wall_seconds) if wall_seconds > 0 else None),
        "resets": int(resets),
        "height_falls": int(height_falls),
        "contact_terminations": int(contact_terminations),
        "episode_timeouts": int(episode_timeouts),
        "cross_track_rms_m": _rms(cross_track),
        "cross_track_p95_m": _percentile(cross_track, 95),
        "cross_track_max_m": (
            max(cross_track) if cross_track else None),
        "off_route_fraction_gt_0_50m": (
            float(off_route_steps / len(cross_track))
            if cross_track else None),
        "flat_cross_track_rms_m": _rms(flat_cross_track),
        "flat_cross_track_p95_m": _percentile(flat_cross_track, 95),
        "flat_yaw_rms_deg": _rms(flat_yaw_degrees),
        "flat_world_vy_rms_mps": _rms(flat_world_vy),
        "roll_rms_deg": _rms(roll_degrees),
        "roll_abs_max_deg": (
            max(map(abs, roll_degrees)) if roll_degrees else None),
        "pitch_rms_deg": _rms(pitch_degrees),
        "pitch_abs_max_deg": (
            max(map(abs, pitch_degrees)) if pitch_degrees else None),
        "forward_speed_mae_mps": (
            float(np.mean(speed_errors)) if speed_errors else None),
        "foot_separation_min_m": (
            min(foot_separations) if foot_separations else None),
        "foot_crossing_fraction": (
            float(crossing_steps / len(foot_separations))
            if foot_separations else None),
    })


def _summarise(checkpoint, trials):
    successes = [trial for trial in trials if trial["completed"]]

    def mean(key, records=trials):
        values = [
            float(record[key]) for record in records
            if _finite(record.get(key))
        ]
        return float(np.mean(values)) if values else None

    return _round_metrics({
        "checkpoint": int(checkpoint),
        "trials": len(trials),
        "successes": len(successes),
        "success_rate": float(len(successes) / len(trials)),
        "mean_resets": mean("resets"),
        "mean_sim_seconds_to_finish": mean("sim_seconds", successes),
        "mean_cleared_stages": mean("cleared_stages"),
        "mean_cross_track_rms_m": mean("cross_track_rms_m"),
        "mean_flat_cross_track_rms_m": mean("flat_cross_track_rms_m"),
        "mean_roll_rms_deg": mean("roll_rms_deg"),
        "mean_forward_speed_mae_mps": mean("forward_speed_mae_mps"),
        "mean_foot_crossing_fraction": mean("foot_crossing_fraction"),
    })


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(str(temporary), str(path))


def evaluate(args):
    if args.task != DEFAULT_TASK:
        raise ValueError(
            "this evaluator currently supports only {}".format(DEFAULT_TASK))
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_sim_seconds <= 0:
        raise ValueError("--max_sim_seconds must be positive")
    if not 0 <= args.course_row <= 7:
        raise ValueError("--course_row must be in [0, 7]")
    if not 0 <= args.course_seed <= 2**32 - 1:
        raise ValueError("--course_seed must be in [0, 2**32 - 1]")

    checkpoints = _parse_checkpoint_numbers(args.checkpoints)
    run_dir = _resolve_run_dir(args.run_dir, args.experiment)
    checkpoint_paths = {
        number: run_dir / "model_{}.pt".format(number)
        for number in checkpoints
    }
    missing = [str(path) for path in checkpoint_paths.values()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing checkpoint(s): {}".format(", ".join(missing)))

    saved_cfg = json.loads(
        (run_dir / "train_cfg.json").read_text(encoding="utf-8"))
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    for section in ("policy", "algorithm", "runner"):
        if section in saved_cfg:
            update_class_from_dict(
                getattr(train_cfg, section), saved_cfg[section])

    # Fixed deterministic course and no deployment-time randomisation.
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = max(
        float(args.max_sim_seconds) + 1.0, 301.0)
    env_cfg.terrain.course_row = int(args.course_row)
    env_cfg.terrain.course_seed = int(args.course_seed)
    env_cfg.terrain.course_add_roughness = True
    env_cfg.terrain.parkour_flat_center_goals = True
    env_cfg.terrain.parkour_spawn_jitter = 0.0
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    for name in (
            "randomize_gains", "randomize_motor_strength",
            "randomize_com_displacement", "randomize_friction",
            "randomize_restitution", "randomize_base_mass",
            "push_robots", "disturbance"):
        setattr(env_cfg.domain_rand, name, False)
    # One robot with cross-leg contacts needs far fewer pairs than training.
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**16

    # Preserve env.test=True (no action-delay randomisation) while skipping the
    # legacy viewer-oriented real-time sleep in LeggedRobot.step().
    legged_robot_module = importlib.import_module(
        "humanoid.envs.base.legged_robot")
    legged_robot_module.time.sleep = lambda _seconds: None

    args.headless = True
    args.num_envs = 1
    args.seed = int(args.eval_seed)
    args.max_iterations = None
    args.resume = False
    args.experiment_name = None
    args.run_name = None
    args.load_run = None
    args.checkpoint = None
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.sim_device_id)

    env = None
    report_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else run_dir / "evaluation" / "course_acceptance.json")
    report = {
        "schema_version": 1,
        "task": args.task,
        "run_dir": str(run_dir),
        "checkpoints": checkpoints,
        "settings": {
            "course_row": int(args.course_row),
            "course_seed": int(args.course_seed),
            "course_add_roughness": True,
            "center_flat_goals": True,
            "trials": int(args.trials),
            "trial_seed_base": int(args.eval_seed),
            "max_sim_seconds": float(args.max_sim_seconds),
            "sim_device": args.sim_device,
            "rl_device": args.rl_device,
            "command_speeds_mps": [0.40, 0.25, 0.40, 0.35, 0.30],
            "off_route_threshold_m": 0.50,
        },
        "results": [],
        "summaries": [],
    }

    try:
        env, _ = task_registry.make_env(
            name=args.task, args=args, env_cfg=env_cfg)
        train_cfg.runner.resume = False
        runner, _ = task_registry.make_alg_runner(
            env=env, args=args, train_cfg=train_cfg, log_root=None)

        print(
            "[acceptance] run={} checkpoints={} trials={} row={} "
            "course_seed={} device={}".format(
                run_dir.name, checkpoints, args.trials,
                args.course_row, args.course_seed, args.sim_device))
        for checkpoint in checkpoints:
            runner.load(
                str(checkpoint_paths[checkpoint]), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            checkpoint_trials = []
            for trial in range(args.trials):
                result = _evaluate_trial(
                    env, policy, checkpoint, trial,
                    int(args.eval_seed) + trial,
                    float(args.max_sim_seconds))
                checkpoint_trials.append(result)
                report["results"].append(result)
                print("EVAL_RESULT " + json.dumps(
                    result, ensure_ascii=False, sort_keys=True))
                _write_report(report_path, report)
            summary = _summarise(checkpoint, checkpoint_trials)
            report["summaries"].append(summary)
            print("EVAL_SUMMARY " + json.dumps(
                summary, ensure_ascii=False, sort_keys=True))
            _write_report(report_path, report)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)

    print("[acceptance] JSON report: {}".format(report_path))
    return report


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": DEFAULT_TASK},
        {"name": "--experiment", "type": str,
         "default": DEFAULT_EXPERIMENT},
        {"name": "--run_dir", "type": str, "required": True},
        {"name": "--checkpoints", "type": str,
         "default": DEFAULT_CHECKPOINTS},
        {"name": "--trials", "type": int, "default": 1},
        {"name": "--eval_seed", "type": int, "default": 20260728},
        {"name": "--max_sim_seconds", "type": float, "default": 300.0},
        {"name": "--course_row", "type": int, "default": 3},
        {"name": "--course_seed", "type": int, "default": 5},
        {"name": "--output_json", "type": str, "default": None},
        {"name": "--rl_device", "type": str, "default": "cuda:1"},
    ]
    return gymutil.parse_arguments(
        description="Headless parkour checkpoint acceptance",
        custom_parameters=custom_parameters)


if __name__ == "__main__":
    evaluate(get_args())
