"""Deterministic, multi-environment checkpoint evaluation on every stair level."""

import csv
import json
import os
from datetime import datetime

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 - task registration side effects
from humanoid.utils import get_args, task_registry


def _parse_levels(args, step_heights):
    if args.step_heights:
        requested = [float(value) for value in args.step_heights.split(",")]
        levels = []
        for height in requested:
            differences = [abs(height - configured) for configured in step_heights]
            level = differences.index(min(differences))
            if differences[level] > 1.0e-6:
                raise ValueError(
                    "Requested step height {} is not configured; choices are {}".format(
                        height, step_heights
                    )
                )
            levels.append(level)
    else:
        levels = [int(value) for value in args.terrain_levels.split(",")]
    levels = list(dict.fromkeys(levels))
    invalid = [level for level in levels if not 0 <= level < len(step_heights)]
    if invalid:
        raise ValueError(
            "Invalid terrain levels {}; valid range is [0, {}]".format(
                invalid, len(step_heights) - 1
            )
        )
    return levels


def _disable_randomization(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.env.test = True
    for name in (
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


def _set_level(env, level):
    env.terrain_levels[:] = level
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types
    ]


def _evaluate_level(env, policy, level, command_speed, episodes_per_env):
    _set_level(env, level)
    obs, _ = env.reset()
    episode_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    records = []
    maximum_steps = int(env.max_episode_length) * episodes_per_env * 3

    for _ in range(maximum_steps):
        env.commands[:, 0] = command_speed
        env.commands[:, 1:3] = 0.0
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, _, _, termination_ids, _ = env.step(actions.detach())

        if len(termination_ids) > 0:
            unfinished = episode_counts[termination_ids] < episodes_per_env
            selected_ids = termination_ids[unfinished]
            for env_id in selected_ids.tolist():
                records.append(
                    {
                        "success": float(env.last_episode_success[env_id].item()),
                        "top_reached": float(
                            env.last_episode_top_reached[env_id].item()
                        ),
                        "forward_distance_m": float(
                            env.last_episode_forward_progress[env_id].item()
                        ),
                        "climb_height_m": float(
                            env.last_episode_climb_height[env_id].item()
                        ),
                        "survival_time_s": float(
                            env.last_episode_survival_time[env_id].item()
                        ),
                        "fall": float(env.last_episode_fall[env_id].item()),
                        "stall": float(env.last_episode_stall[env_id].item()),
                        "first_step": float(
                            env.last_episode_first_step[env_id].item()
                        ),
                        "max_foot_height_m": float(
                            env.last_episode_max_foot_height[env_id].item()
                        ),
                        "mean_forward_speed_m_s": float(
                            env.last_episode_mean_forward_speed[env_id].item()
                        ),
                        "mean_command_error_m_s": float(
                            env.last_episode_mean_command_error[env_id].item()
                        ),
                        "phase_contact_match": float(
                            env.last_episode_phase_contact_match[env_id].item()
                        ),
                        "sagittal_foot_phase_match": float(
                            env.last_episode_sagittal_foot_phase_match[
                                env_id
                            ].item()
                        ),
                        "double_flight_fraction": float(
                            env.last_episode_double_flight_fraction[env_id].item()
                        ),
                        "max_lateral_deviation_m": float(
                            env.last_episode_max_lateral_deviation[env_id].item()
                        ),
                        "max_yaw_deviation_rad": float(
                            env.last_episode_max_yaw_deviation[env_id].item()
                        ),
                        "path_failure": float(
                            env.last_episode_path_failure[env_id].item()
                        ),
                        "alternating_tread_count": float(
                            env.last_episode_alternating_tread_count[
                                env_id
                            ].item()
                        ),
                        "alternating_tread_rate": float(
                            env.last_episode_alternating_tread_rate[
                                env_id
                            ].item()
                        ),
                        "repeated_lead_rate": float(
                            env.last_episode_repeated_lead_rate[env_id].item()
                        ),
                        "same_tread_join_rate": float(
                            env.last_episode_same_tread_join_rate[
                                env_id
                            ].item()
                        ),
                        "skipped_tread_rate": float(
                            env.last_episode_skipped_tread_rate[env_id].item()
                        ),
                        "max_sagittal_foot_separation_m": float(
                            env.last_episode_max_sagittal_foot_separation[
                                env_id
                            ].item()
                        ),
                        "mean_swing_knee_flexion_rad": float(
                            env.last_episode_mean_swing_knee_flexion[
                                env_id
                            ].item()
                        ),
                        "mean_arm_swing_match": float(
                            env.last_episode_mean_arm_swing_match[env_id].item()
                        ),
                        "gait_frequency_hz": float(
                            env.last_episode_gait_frequency[env_id].item()
                        ),
                    }
                )
            episode_counts[selected_ids] += 1

        if torch.all(episode_counts >= episodes_per_env):
            break
    else:
        raise RuntimeError(
            "Evaluation did not collect {} episodes per environment at level {}".format(
                episodes_per_env, level
            )
        )

    count = float(len(records))
    if count == 0:
        raise RuntimeError("Evaluation produced no completed episodes")

    def mean(key):
        return sum(record[key] for record in records) / count

    return {
        "terrain_level": level,
        "step_height_m": float(env.cfg.terrain.step_heights[level]),
        "command_speed_m_s": command_speed,
        "episodes": int(count),
        "success_rate": mean("success"),
        "mean_forward_distance_m": mean("forward_distance_m"),
        "mean_climb_height_m": mean("climb_height_m"),
        "mean_survival_time_s": mean("survival_time_s"),
        "fall_rate": mean("fall"),
        "top_reached_rate": mean("top_reached"),
        "stall_rate": mean("stall"),
        "first_step_rate": mean("first_step"),
        "mean_max_foot_height_m": mean("max_foot_height_m"),
        "mean_forward_speed_m_s": mean("mean_forward_speed_m_s"),
        "mean_command_error_m_s": mean("mean_command_error_m_s"),
        "mean_phase_contact_match": mean("phase_contact_match"),
        "mean_sagittal_foot_phase_match": mean(
            "sagittal_foot_phase_match"
        ),
        "mean_double_flight_fraction": mean("double_flight_fraction"),
        "mean_max_lateral_deviation_m": mean("max_lateral_deviation_m"),
        "mean_max_yaw_deviation_rad": mean("max_yaw_deviation_rad"),
        "path_failure_rate": mean("path_failure"),
        "mean_alternating_tread_count": mean("alternating_tread_count"),
        "mean_alternating_tread_rate": mean("alternating_tread_rate"),
        "mean_repeated_lead_rate": mean("repeated_lead_rate"),
        "mean_same_tread_join_rate": mean("same_tread_join_rate"),
        "mean_skipped_tread_rate": mean("skipped_tread_rate"),
        "mean_max_sagittal_foot_separation_m": mean(
            "max_sagittal_foot_separation_m"
        ),
        "mean_swing_knee_flexion_rad": mean(
            "mean_swing_knee_flexion_rad"
        ),
        "mean_arm_swing_match": mean("mean_arm_swing_match"),
        "mean_gait_frequency_hz": mean("gait_frequency_hz"),
    }


def evaluate(args):
    if args.task not in (
        "n2_stairs",
        "n2_stairs_robust",
        "n2_stairs_walk",
    ):
        raise ValueError("eval_stairs.py only supports n2_stairs tasks")
    if args.num_envs is None:
        args.num_envs = 128
    if args.episodes_per_env < 1:
        raise ValueError("--episodes_per_env must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    step_heights = list(env_cfg.terrain.step_heights)
    levels = _parse_levels(args, step_heights)
    command_min = env_cfg.commands.ranges.lin_vel_x[0]
    allowed_max = min(
        env_cfg.commands.max_curriculum,
        min(
            env_cfg.commands.initial_max_speed
            + env_cfg.commands.speed_per_terrain_level * level
            for level in levels
        ),
    )
    if not command_min <= args.command_speed <= allowed_max:
        raise ValueError(
            "--command_speed {:.3f} is outside training range [{:.3f}, {:.3f}]".format(
                args.command_speed, command_min, allowed_max
            )
        )

    _disable_randomization(env_cfg)
    # ``env.test=True`` throttles every low-level physics step to wall-clock
    # time. Randomization is already disabled explicitly, so keep batch
    # headless evaluation unthrottled.
    env_cfg.env.test = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.fixed_level = levels[0]
    env_cfg.commands.resampling_time = [1000, 1001]
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [args.command_speed, args.command_speed]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
        load_optimizer=False,
    )
    policy = runner.get_inference_policy(device=env.device)

    summaries = []
    for level in levels:
        result = _evaluate_level(
            env,
            policy,
            level,
            args.command_speed,
            args.episodes_per_env,
        )
        summaries.append(result)
        print(
            "level={terrain_level} height={step_height_m:.2f}m "
            "success={success_rate:.1%} distance={mean_forward_distance_m:.3f}m "
            "climb={mean_climb_height_m:.3f}m first_step={first_step_rate:.1%} "
            "speed={mean_forward_speed_m_s:.3f}m/s "
            "lateral={mean_max_lateral_deviation_m:.3f}m "
            "flight={mean_double_flight_fraction:.1%} "
            "alternate={mean_alternating_tread_rate:.1%} "
            "join={mean_same_tread_join_rate:.1%} "
            "footphase={mean_sagittal_foot_phase_match:.1%} "
            "stride={mean_max_sagittal_foot_separation_m:.3f}m "
            "knee={mean_swing_knee_flexion_rad:.2f}rad "
            "arm={mean_arm_swing_match:.2f} "
            "gait={mean_gait_frequency_hz:.2f}Hz "
            "fall={fall_rate:.1%}".format(
                **result
            )
        )

    output_path = args.output
    if output_path is None:
        output_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "evaluations",
        )
        output_path = os.path.join(
            output_dir,
            "stairs_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        )
    if not output_path.lower().endswith(".csv"):
        output_path += ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(
            {
                "task": args.task,
                "load_run": args.load_run,
                "checkpoint": args.checkpoint,
                "checkpoint_path": train_cfg.runner.resume_path,
                "seed": train_cfg.seed,
                "num_envs": args.num_envs,
                "episodes_per_env": args.episodes_per_env,
                "summaries": summaries,
            },
            json_file,
            indent=2,
        )
    print("Saved evaluation CSV: {}".format(output_path))
    print("Saved evaluation JSON: {}".format(json_path))


if __name__ == "__main__":
    extra_parameters = [
        {
            "name": "--terrain_levels",
            "type": str,
            "default": "0,1,2,3,4",
            "help": "Comma-separated curriculum rows to evaluate.",
        },
        {
            "name": "--step_heights",
            "type": str,
            "default": None,
            "help": "Optional comma-separated exact heights, e.g. 0.02,0.06,0.10.",
        },
        {
            "name": "--command_speed",
            "type": float,
            "default": 0.25,
            "help": "Forward command in m/s, within the baseline training range.",
        },
        {
            "name": "--episodes_per_env",
            "type": int,
            "default": 2,
            "help": "Completed episodes collected from each parallel environment.",
        },
        {
            "name": "--output",
            "type": str,
            "default": None,
            "help": "CSV output path; a JSON summary is written beside it.",
        },
    ]
    evaluate(get_args(extra_parameters))
