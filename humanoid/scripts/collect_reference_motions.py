#!/usr/bin/env python3
"""Collect curated, episode-bounded AMP reference motions from Isaac Gym."""

from __future__ import annotations

import json
import math
import os
import sys

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if not sys.path or os.path.realpath(sys.path[0]) != _REPOSITORY_ROOT:
    sys.path.insert(0, _REPOSITORY_ROOT)

# Isaac Gym Preview 4 must load before torch.
import isaacgym  # noqa: F401

import torch

from humanoid.algo.amp.observations import (
    AMP_FEATURE_VERSION,
    AMP_FULL_FRAME_DIM,
    build_amp_full_frame,
    mirror_amp_full_frames,
)
from humanoid.envs import *  # noqa: F401,F403
from humanoid.utils.helpers import parse_humanoid_args
from humanoid.utils.task_registry import task_registry


def _disable_randomization(env_cfg):
    env_cfg.noise.add_noise = False
    # ``env.test=True`` sleeps in real time in this fork.  Keep fast headless
    # stepping and explicitly disable every stochastic actuator disturbance.
    env_cfg.env.test = False
    for name in (
        "randomize_rigid_shape_props_on_reset",
        "randomize_gains",
        "randomize_motor_strength",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_friction",
        "randomize_restitution",
        "push_robots",
        "disturbance",
        "action_delay",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)
    if hasattr(env_cfg.env, "randomize_gait_phase"):
        env_cfg.env.randomize_gait_phase = False


def _set_fixed_level(env, level):
    env.terrain_levels[:] = int(level)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types
    ]


def _episode_quality(actions):
    action_tensor = torch.stack(actions)
    if action_tensor.shape[0] < 2:
        action_rate = float("inf")
        action_accel = float("inf")
    else:
        delta = torch.diff(action_tensor, dim=0)
        action_rate = float(torch.sqrt(torch.mean(delta.square())).item())
        if delta.shape[0] < 2:
            action_accel = action_rate
        else:
            action_accel = float(
                torch.sqrt(torch.mean(torch.diff(delta, dim=0).square())).item()
            )
    return action_rate, action_accel


def _episode_metrics(env, env_id, actions):
    action_rate, action_accel = _episode_quality(actions)
    metrics = {
        "completion": float(
            env.last_episode_completion[env_id].item()
        ),
        "top_reached": float(
            env.last_episode_top_reached[env_id].item()
        ),
        "fall": float(env.last_episode_fall[env_id].item()),
        "path_failure": float(
            env.last_episode_path_failure[env_id].item()
        ),
        "climb_height_m": float(
            env.last_episode_climb_height[env_id].item()
        ),
        "max_lateral_deviation_m": float(
            env.last_episode_max_lateral_deviation[env_id].item()
        ),
        "max_yaw_deviation_rad": float(
            env.last_episode_max_yaw_deviation[env_id].item()
        ),
        "final_lateral_position_m": float(
            env.last_episode_final_lateral_position[env_id].item()
        ),
        "mean_left_swing_length_m": float(
            env.last_episode_mean_left_swing_length[env_id].item()
        ),
        "mean_right_swing_length_m": float(
            env.last_episode_mean_right_swing_length[env_id].item()
        ),
        "action_rate_rms": action_rate,
        "action_accel_rms": action_accel,
    }
    metrics["stride_imbalance_m"] = abs(
        metrics["mean_left_swing_length_m"]
        - metrics["mean_right_swing_length_m"]
    )
    metrics["quality_score"] = (
        10.0 * metrics["completion"]
        + metrics["climb_height_m"]
        - 2.0 * metrics["max_lateral_deviation_m"]
        - 0.5 * abs(metrics["final_lateral_position_m"])
        - 0.5 * metrics["stride_imbalance_m"]
        - 0.25 * action_rate
        - 0.10 * action_accel
    )
    return metrics


def _is_curated(metrics, frame_count, args):
    return (
        frame_count >= args.min_frames
        and metrics["completion"] >= args.min_completion
        and metrics["top_reached"] >= args.min_completion
        and metrics["fall"] < 0.5
        and metrics["path_failure"] < 0.5
        and metrics["max_lateral_deviation_m"]
        <= args.max_lateral_deviation
        and metrics["max_yaw_deviation_rad"] <= args.max_yaw_deviation
        and metrics["stride_imbalance_m"] <= args.max_stride_imbalance
        and math.isfinite(metrics["action_rate_rms"])
        and math.isfinite(metrics["action_accel_rms"])
    )


def _write_motion(path, frames, frame_duration, metrics, mirrored):
    payload = {
        "Frames": frames.tolist(),
        "MotionWeight": 1.0,
        "FrameDuration": float(frame_duration),
        "AMPFeatureVersion": AMP_FEATURE_VERSION,
        "Metadata": {
            **metrics,
            "mirrored": bool(mirrored),
            "frame_count": int(frames.shape[0]),
        },
    }
    with open(path, "w") as stream:
        json.dump(payload, stream, separators=(",", ":"))


def collect(args):
    if args.task != "n2_stairs_walk":
        raise ValueError("AMP collection supports n2_stairs_walk only")
    if args.num_envs is None or args.num_envs < 1:
        raise ValueError("--num_envs must be positive")
    if args.num_steps < 1 or args.max_motions < 1:
        raise ValueError("--num_steps and --max_motions must be positive")
    if not 0 <= args.fixed_terrain_level <= 4:
        raise ValueError("--fixed_terrain_level must be in [0, 4]")

    args.model_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            "Checkpoint does not exist: " + args.model_path
        )

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.fixed_level = int(args.fixed_terrain_level)
    env_cfg.terrain.level_mix = []
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [
        float(args.command_speed),
        float(args.command_speed),
    ]
    _disable_randomization(env_cfg)
    train_cfg.runner_class_name = "OnPolicyRunner"

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    args.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        log_root=None,
        load_optimizer=False,
    )
    policy = runner.get_inference_policy(device=env.device)

    _set_fixed_level(env, args.fixed_terrain_level)
    observations, _ = env.reset()
    _set_fixed_level(env, args.fixed_terrain_level)
    frame_duration = float(getattr(env, "dt", 0.02))
    episode_frames = [[] for _ in range(env.num_envs)]
    episode_actions = [[] for _ in range(env.num_envs)]
    candidates = []
    completed = 0
    rejected = 0
    candidate_target = max(args.max_motions * 3, args.max_motions)

    print(
        "[AMP] Collecting at level={} speed={:.3f} with {} envs".format(
            args.fixed_terrain_level, args.command_speed, env.num_envs
        )
    )
    for step in range(args.num_steps):
        env.commands[:, 0] = float(args.command_speed)
        env.commands[:, 1:3] = 0.0
        frames = build_amp_full_frame(env).detach().cpu()
        if frames.shape[1] != AMP_FULL_FRAME_DIM:
            raise RuntimeError("AMP collector produced the wrong frame width")

        with torch.inference_mode():
            actions = policy(observations.detach())
        actions_cpu = actions.detach().cpu()
        for env_id in range(env.num_envs):
            episode_frames[env_id].append(frames[env_id])
            episode_actions[env_id].append(actions_cpu[env_id])

        (
            observations,
            _,
            _,
            _,
            _,
            termination_ids,
            _,
        ) = env.step(actions.detach())
        for env_id in termination_ids.tolist():
            completed += 1
            metrics = _episode_metrics(
                env, env_id, episode_actions[env_id]
            )
            frame_count = len(episode_frames[env_id])
            if _is_curated(metrics, frame_count, args):
                candidates.append(
                    (
                        metrics,
                        torch.stack(episode_frames[env_id]),
                    )
                )
                print(
                    "[AMP] accepted episode={} frames={} completion={:.0%} "
                    "lateral={:.3f}m stride={:.3f}m smooth={:.4f}".format(
                        completed,
                        frame_count,
                        metrics["completion"],
                        metrics["max_lateral_deviation_m"],
                        metrics["stride_imbalance_m"],
                        metrics["action_rate_rms"],
                    )
                )
            else:
                rejected += 1
            episode_frames[env_id] = []
            episode_actions[env_id] = []

        if len(candidates) >= candidate_target:
            break
        if step and step % 500 == 0:
            print(
                "[AMP] step={}/{} completed={} accepted={} rejected={}".format(
                    step, args.num_steps, completed, len(candidates), rejected
                )
            )

    if not candidates:
        raise RuntimeError(
            "No episode passed the AMP quality gate. Do not train on failed "
            "or cross-episode reference data; evaluate the checkpoint and "
            "relax only a justified collection threshold."
        )

    candidates.sort(key=lambda item: item[0]["quality_score"], reverse=True)
    selected = candidates[: args.max_motions]
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    manifest = (
        os.path.abspath(os.path.expanduser(args.manifest))
        if args.manifest
        else os.path.join(output_dir, "manifest.txt")
    )
    os.makedirs(os.path.dirname(manifest), exist_ok=True)

    written = []
    for index, (metrics, frames) in enumerate(selected):
        source_path = os.path.join(
            output_dir, "motion_{:03d}.json".format(index)
        )
        _write_motion(
            source_path, frames, frame_duration, metrics, mirrored=False
        )
        written.append(os.path.abspath(source_path))

        if not args.no_mirror_augmentation:
            mirrored = mirror_amp_full_frames(
                frames.to(env.device), env.mirror_actions
            ).cpu()
            mirror_path = os.path.join(
                output_dir, "motion_{:03d}_mirrored.json".format(index)
            )
            _write_motion(
                mirror_path,
                mirrored,
                frame_duration,
                metrics,
                mirrored=True,
            )
            written.append(os.path.abspath(mirror_path))

    with open(manifest, "w") as stream:
        stream.write("\n".join(written) + "\n")
    print(
        "[AMP] Wrote {} curated motion file(s) from {} accepted episodes".format(
            len(written), len(candidates)
        )
    )
    print("N2_AMP_MOTION_MANIFEST={}".format(manifest))


def main():
    args = parse_humanoid_args(
        [
            {
                "name": "--model_path",
                "type": str,
                "required": True,
                "help": "Policy checkpoint used to collect expert episodes.",
            },
            {
                "name": "--output_dir",
                "type": str,
                "default": "humanoid/amp_data/stair_climb",
            },
            {"name": "--manifest", "type": str, "default": None},
            {"name": "--num_steps", "type": int, "default": 6000},
            {"name": "--max_motions", "type": int, "default": 24},
            {"name": "--min_frames", "type": int, "default": 100},
            {"name": "--min_completion", "type": float, "default": 1.0},
            {
                "name": "--max_lateral_deviation",
                "type": float,
                "default": 0.16,
            },
            {
                "name": "--max_yaw_deviation",
                "type": float,
                "default": 0.35,
            },
            {
                "name": "--max_stride_imbalance",
                "type": float,
                "default": 0.12,
            },
            {"name": "--command_speed", "type": float, "default": 0.18},
            {"name": "--fixed_terrain_level", "type": int, "default": 4},
            {
                "name": "--no_mirror_augmentation",
                "action": "store_true",
                "default": False,
            },
        ]
    )
    collect(args)


if __name__ == "__main__":
    main()
