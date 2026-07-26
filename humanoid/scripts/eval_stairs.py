"""Deterministic, multi-environment checkpoint evaluation on every stair level."""

import csv
import json
import os
import sys
from datetime import datetime

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if not sys.path or os.path.realpath(sys.path[0]) != _REPOSITORY_ROOT:
    sys.path.insert(0, _REPOSITORY_ROOT)

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 - task registration side effects
from humanoid.utils.helpers import parse_humanoid_args
from humanoid.utils.policy_symmetry import make_reflection_blended_policy
from humanoid.utils.residual_policy import (
    configure_residual_policy,
    residual_metadata_from_checkpoint,
    resolve_checkpoint_path,
)
from humanoid.utils.task_registry import task_registry


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
    previous_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    previous_action_delta = torch.zeros_like(previous_actions)
    has_previous_action = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    has_previous_delta = torch.zeros_like(has_previous_action)
    action_rate_square_sum = torch.zeros(
        env.num_envs, dtype=torch.float, device=env.device
    )
    action_accel_square_sum = torch.zeros_like(action_rate_square_sum)
    action_rate_sample_count = torch.zeros_like(action_rate_square_sum)
    action_accel_sample_count = torch.zeros_like(action_rate_square_sum)
    actor_symmetry_square_sum = torch.zeros_like(action_rate_square_sum)
    actor_symmetry_sample_count = torch.zeros_like(action_rate_square_sum)
    # Columns are [right swing / left support, left swing / right support].
    # The second phase is the one reported as visibly unstable by the user.
    phase_action_rate_square_sum = torch.zeros(
        env.num_envs, 2, dtype=torch.float, device=env.device
    )
    phase_action_accel_square_sum = torch.zeros_like(
        phase_action_rate_square_sum
    )
    phase_action_rate_sample_count = torch.zeros_like(
        phase_action_rate_square_sum
    )
    phase_action_accel_sample_count = torch.zeros_like(
        phase_action_rate_square_sum
    )
    phase_roll_rate_square_sum = torch.zeros_like(
        phase_action_rate_square_sum
    )
    phase_lateral_velocity_square_sum = torch.zeros_like(
        phase_action_rate_square_sum
    )
    phase_body_sample_count = torch.zeros_like(
        phase_action_rate_square_sum
    )

    for _ in range(maximum_steps):
        env.commands[:, 0] = command_speed
        env.commands[:, 1:3] = 0.0
        with torch.inference_mode():
            actions = policy(obs.detach())
            mirrored_observations = env.mirror_observations(obs.detach())
            mirrored_policy_actions = policy(mirrored_observations)
            reflected_actions = env.mirror_actions(
                mirrored_policy_actions
            )
        actor_symmetry_square_sum += torch.mean(
            torch.square(actions - reflected_actions), dim=1
        )
        actor_symmetry_sample_count += 1.0
        action_delta = actions - previous_actions
        action_accel = action_delta - previous_action_delta
        action_rate_square = torch.mean(torch.square(action_delta), dim=1)
        action_accel_square = torch.mean(torch.square(action_accel), dim=1)
        action_rate_square_sum += (
            action_rate_square
            * has_previous_action.float()
        )
        action_accel_square_sum += (
            action_accel_square
            * has_previous_delta.float()
        )
        action_rate_sample_count += has_previous_action.float()
        action_accel_sample_count += has_previous_delta.float()
        desired_contacts = env.desired_contacts
        phase_mask = torch.stack(
            (
                desired_contacts[:, 0] & ~desired_contacts[:, 1],
                desired_contacts[:, 1] & ~desired_contacts[:, 0],
            ),
            dim=1,
        )
        phase_rate_mask = phase_mask & has_previous_action.unsqueeze(1)
        phase_accel_mask = phase_mask & has_previous_delta.unsqueeze(1)
        phase_action_rate_square_sum += (
            action_rate_square.unsqueeze(1) * phase_rate_mask.float()
        )
        phase_action_accel_square_sum += (
            action_accel_square.unsqueeze(1) * phase_accel_mask.float()
        )
        phase_action_rate_sample_count += phase_rate_mask.float()
        phase_action_accel_sample_count += phase_accel_mask.float()
        phase_roll_rate_square_sum += (
            torch.square(env.base_ang_vel[:, 0]).unsqueeze(1)
            * phase_mask.float()
        )
        phase_lateral_velocity_square_sum += (
            torch.square(env.base_lin_vel[:, 1]).unsqueeze(1)
            * phase_mask.float()
        )
        phase_body_sample_count += phase_mask.float()
        previous_actions[:] = actions
        previous_action_delta[:] = action_delta
        has_previous_delta |= has_previous_action
        has_previous_action[:] = True
        obs, _, _, _, _, termination_ids, _ = env.step(actions.detach())

        if len(termination_ids) > 0:
            unfinished = episode_counts[termination_ids] < episodes_per_env
            selected_ids = termination_ids[unfinished]
            for env_id in selected_ids.tolist():
                records.append(
                    {
                        "success": float(env.last_episode_success[env_id].item()),
                        "completion": float(
                            env.last_episode_completion[env_id].item()
                        ),
                        "curriculum_completion": float(
                            env.last_episode_curriculum_completion[
                                env_id
                            ].item()
                        ),
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
                        "paired_lead_advances": float(
                            env.last_episode_paired_lead_advances[
                                env_id
                            ].item()
                        ),
                        "paired_trailing_joins": float(
                            env.last_episode_paired_trailing_joins[
                                env_id
                            ].item()
                        ),
                        "paired_sequence_rate": float(
                            env.last_episode_paired_sequence_rate[
                                env_id
                            ].item()
                        ),
                        "paired_join_coverage": float(
                            env.last_episode_paired_join_coverage[
                                env_id
                            ].item()
                        ),
                        "paired_premature_rate": float(
                            env.last_episode_paired_premature_rate[
                                env_id
                            ].item()
                        ),
                        "paired_lead_switch_rate": float(
                            env.last_episode_paired_lead_switch_rate[
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
                        "same_tread_support_fraction": float(
                            env.last_episode_same_tread_support_fraction[
                                env_id
                            ].item()
                        ),
                        "lower_leg_collision_fraction": float(
                            env.last_episode_lower_leg_collision_fraction[
                                env_id
                            ].item()
                        ),
                        "foot_riser_collision_fraction": float(
                            env.last_episode_foot_riser_collision_fraction[
                                env_id
                            ].item()
                        ),
                        "swing_timeout_fraction": float(
                            env.last_episode_swing_timeout_fraction[
                                env_id
                            ].item()
                        ),
                        "mean_base_behind_support": float(
                            env.last_episode_mean_base_behind_support[
                                env_id
                            ].item()
                        ),
                        "mean_foot_pitch_error": float(
                            env.last_episode_mean_foot_pitch_error[
                                env_id
                            ].item()
                        ),
                        "max_swing_duration": float(
                            env.last_episode_max_swing_duration[env_id].item()
                        ),
                        "left_tread_advances": float(
                            env.last_episode_left_tread_advances[env_id].item()
                        ),
                        "right_tread_advances": float(
                            env.last_episode_right_tread_advances[env_id].item()
                        ),
                        "final_lateral_position_m": float(
                            env.last_episode_final_lateral_position[
                                env_id
                            ].item()
                        ),
                        "mean_left_swing_length_m": float(
                            env.last_episode_mean_left_swing_length[
                                env_id
                            ].item()
                        ),
                        "mean_right_swing_length_m": float(
                            env.last_episode_mean_right_swing_length[
                                env_id
                            ].item()
                        ),
                        "mean_left_foot_inward_error_m": float(
                            env.last_episode_mean_left_foot_inward_error[
                                env_id
                            ].item()
                        ),
                        "mean_right_foot_inward_error_m": float(
                            env.last_episode_mean_right_foot_inward_error[
                                env_id
                            ].item()
                        ),
                        "mean_left_foot_lateral_position_m": float(
                            env.last_episode_mean_left_foot_lateral_position[
                                env_id
                            ].item()
                        ),
                        "mean_right_foot_lateral_position_m": float(
                            env.last_episode_mean_right_foot_lateral_position[
                                env_id
                            ].item()
                        ),
                        "faststair_planner_valid_fraction": float(
                            env.last_episode_faststair_planner_valid_fraction[
                                env_id
                            ].item()
                        ),
                        "faststair_foothold_error_m": float(
                            env.last_episode_faststair_foothold_error[
                                env_id
                            ].item()
                        ),
                        "faststair_edge_margin_m": float(
                            env.last_episode_faststair_edge_margin[
                                env_id
                            ].item()
                        ),
                        "actual_sole_support_fraction": float(
                            env.last_episode_actual_sole_support_fraction[
                                env_id
                            ].item()
                        ),
                        "action_rate_rms": float(
                            torch.sqrt(
                                action_rate_square_sum[env_id]
                                / torch.clamp(
                                    action_rate_sample_count[env_id], min=1.0
                                )
                            ).item()
                        ),
                        "action_accel_rms": float(
                            torch.sqrt(
                                action_accel_square_sum[env_id]
                                / torch.clamp(
                                    action_accel_sample_count[env_id], min=1.0
                                )
                            ).item()
                        ),
                        "actor_symmetry_error_rms": float(
                            torch.sqrt(
                                actor_symmetry_square_sum[env_id]
                                / torch.clamp(
                                    actor_symmetry_sample_count[env_id],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "right_swing_action_rate_rms": float(
                            torch.sqrt(
                                phase_action_rate_square_sum[env_id, 0]
                                / torch.clamp(
                                    phase_action_rate_sample_count[env_id, 0],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "right_swing_action_accel_rms": float(
                            torch.sqrt(
                                phase_action_accel_square_sum[env_id, 0]
                                / torch.clamp(
                                    phase_action_accel_sample_count[env_id, 0],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "left_swing_action_rate_rms": float(
                            torch.sqrt(
                                phase_action_rate_square_sum[env_id, 1]
                                / torch.clamp(
                                    phase_action_rate_sample_count[env_id, 1],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "left_swing_action_accel_rms": float(
                            torch.sqrt(
                                phase_action_accel_square_sum[env_id, 1]
                                / torch.clamp(
                                    phase_action_accel_sample_count[env_id, 1],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "right_swing_roll_rate_rms": float(
                            torch.sqrt(
                                phase_roll_rate_square_sum[env_id, 0]
                                / torch.clamp(
                                    phase_body_sample_count[env_id, 0],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "left_swing_roll_rate_rms": float(
                            torch.sqrt(
                                phase_roll_rate_square_sum[env_id, 1]
                                / torch.clamp(
                                    phase_body_sample_count[env_id, 1],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "right_swing_lateral_velocity_rms": float(
                            torch.sqrt(
                                phase_lateral_velocity_square_sum[env_id, 0]
                                / torch.clamp(
                                    phase_body_sample_count[env_id, 0],
                                    min=1.0,
                                )
                            ).item()
                        ),
                        "left_swing_lateral_velocity_rms": float(
                            torch.sqrt(
                                phase_lateral_velocity_square_sum[env_id, 1]
                                / torch.clamp(
                                    phase_body_sample_count[env_id, 1],
                                    min=1.0,
                                )
                            ).item()
                        ),
                    }
                )
            episode_counts[selected_ids] += 1
            previous_actions[termination_ids] = 0.0
            previous_action_delta[termination_ids] = 0.0
            has_previous_action[termination_ids] = False
            has_previous_delta[termination_ids] = False
            action_rate_square_sum[termination_ids] = 0.0
            action_accel_square_sum[termination_ids] = 0.0
            action_rate_sample_count[termination_ids] = 0.0
            action_accel_sample_count[termination_ids] = 0.0
            actor_symmetry_square_sum[termination_ids] = 0.0
            actor_symmetry_sample_count[termination_ids] = 0.0
            phase_action_rate_square_sum[termination_ids] = 0.0
            phase_action_accel_square_sum[termination_ids] = 0.0
            phase_action_rate_sample_count[termination_ids] = 0.0
            phase_action_accel_sample_count[termination_ids] = 0.0
            phase_roll_rate_square_sum[termination_ids] = 0.0
            phase_lateral_velocity_square_sum[termination_ids] = 0.0
            phase_body_sample_count[termination_ids] = 0.0

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
        "completion_rate": mean("completion"),
        "curriculum_completion_rate": mean("curriculum_completion"),
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
        "mean_paired_lead_advances": mean("paired_lead_advances"),
        "mean_paired_trailing_joins": mean("paired_trailing_joins"),
        "mean_paired_sequence_rate": mean("paired_sequence_rate"),
        "mean_paired_join_coverage": mean("paired_join_coverage"),
        "mean_paired_premature_rate": mean("paired_premature_rate"),
        "mean_paired_lead_switch_rate": mean(
            "paired_lead_switch_rate"
        ),
        "mean_skipped_tread_rate": mean("skipped_tread_rate"),
        "mean_max_sagittal_foot_separation_m": mean(
            "max_sagittal_foot_separation_m"
        ),
        "mean_swing_knee_flexion_rad": mean(
            "mean_swing_knee_flexion_rad"
        ),
        "mean_arm_swing_match": mean("mean_arm_swing_match"),
        "mean_gait_frequency_hz": mean("gait_frequency_hz"),
        "mean_same_tread_support_fraction": mean(
            "same_tread_support_fraction"
        ),
        "mean_lower_leg_collision_fraction": mean(
            "lower_leg_collision_fraction"
        ),
        "mean_foot_riser_collision_fraction": mean(
            "foot_riser_collision_fraction"
        ),
        "mean_swing_timeout_fraction": mean("swing_timeout_fraction"),
        "mean_base_behind_support": mean("mean_base_behind_support"),
        "mean_foot_pitch_error": mean("mean_foot_pitch_error"),
        "mean_max_swing_duration": mean("max_swing_duration"),
        "mean_left_tread_advances": mean("left_tread_advances"),
        "mean_right_tread_advances": mean("right_tread_advances"),
        "mean_final_lateral_position_m": mean(
            "final_lateral_position_m"
        ),
        "mean_left_swing_length_m": mean("mean_left_swing_length_m"),
        "mean_right_swing_length_m": mean("mean_right_swing_length_m"),
        "mean_left_foot_inward_error_m": mean(
            "mean_left_foot_inward_error_m"
        ),
        "mean_right_foot_inward_error_m": mean(
            "mean_right_foot_inward_error_m"
        ),
        "mean_left_foot_lateral_position_m": mean(
            "mean_left_foot_lateral_position_m"
        ),
        "mean_right_foot_lateral_position_m": mean(
            "mean_right_foot_lateral_position_m"
        ),
        "mean_faststair_planner_valid_fraction": mean(
            "faststair_planner_valid_fraction"
        ),
        "mean_faststair_foothold_error_m": mean(
            "faststair_foothold_error_m"
        ),
        "mean_faststair_edge_margin_m": mean(
            "faststair_edge_margin_m"
        ),
        "mean_actual_sole_support_fraction": mean(
            "actual_sole_support_fraction"
        ),
        "mean_action_rate_rms": mean("action_rate_rms"),
        "mean_action_accel_rms": mean("action_accel_rms"),
        "mean_actor_symmetry_error_rms": mean(
            "actor_symmetry_error_rms"
        ),
        "mean_right_swing_action_rate_rms": mean(
            "right_swing_action_rate_rms"
        ),
        "mean_right_swing_action_accel_rms": mean(
            "right_swing_action_accel_rms"
        ),
        "mean_left_swing_action_rate_rms": mean(
            "left_swing_action_rate_rms"
        ),
        "mean_left_swing_action_accel_rms": mean(
            "left_swing_action_accel_rms"
        ),
        "mean_right_swing_roll_rate_rms": mean(
            "right_swing_roll_rate_rms"
        ),
        "mean_left_swing_roll_rate_rms": mean(
            "left_swing_roll_rate_rms"
        ),
        "mean_right_swing_lateral_velocity_rms": mean(
            "right_swing_lateral_velocity_rms"
        ),
        "mean_left_swing_lateral_velocity_rms": mean(
            "left_swing_lateral_velocity_rms"
        ),
    }


def evaluate(args):
    if args.task not in (
        "n2_stairs",
        "n2_stairs_robust",
        "n2_stairs_walk",
        "n2_faststair",
    ):
        raise ValueError("eval_stairs.py only supports n2_stairs tasks")
    if args.num_envs is None:
        args.num_envs = 128
    if args.episodes_per_env < 1:
        raise ValueError("--episodes_per_env must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.contact_phase_reset:
        env_cfg.env.contact_phase_reset = True
    if args.resume:
        checkpoint_path = resolve_checkpoint_path(args, train_cfg)
        policy_metadata = residual_metadata_from_checkpoint(
            checkpoint_path
        )
        if configure_residual_policy(
            env_cfg, train_cfg, policy_metadata
        ):
            print(
                "Detected residual policy checkpoint: {}".format(
                    checkpoint_path
                )
            )
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
    policy = make_reflection_blended_policy(
        policy, env, args.policy_symmetry_blend
    )
    print(
        "Policy reflection blend: {:.3f}".format(
            args.policy_symmetry_blend
        )
    )

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
            "success={success_rate:.1%} completion={completion_rate:.1%} "
            "curriculum={curriculum_completion_rate:.1%} "
            "distance={mean_forward_distance_m:.3f}m "
            "climb={mean_climb_height_m:.3f}m first_step={first_step_rate:.1%} "
            "speed={mean_forward_speed_m_s:.3f}m/s "
            "lateral={mean_max_lateral_deviation_m:.3f}m "
            "flight={mean_double_flight_fraction:.1%} "
            "alternate={mean_alternating_tread_rate:.1%} "
            "join={mean_same_tread_join_rate:.1%} "
            "pair=L{mean_paired_lead_advances:.2f}/"
            "J{mean_paired_trailing_joins:.2f} "
            "pair_rate={mean_paired_sequence_rate:.1%} "
            "pair_cover={mean_paired_join_coverage:.1%} "
            "premature={mean_paired_premature_rate:.1%} "
            "step-to={mean_same_tread_support_fraction:.1%} "
            "shin={mean_lower_leg_collision_fraction:.1%} "
            "riser={mean_foot_riser_collision_fraction:.1%} "
            "swing_timeout={mean_swing_timeout_fraction:.1%} "
            "footphase={mean_sagittal_foot_phase_match:.1%} "
            "stride={mean_max_sagittal_foot_separation_m:.3f}m "
            "knee={mean_swing_knee_flexion_rad:.2f}rad "
            "baseback={mean_base_behind_support:.3f} "
            "footpitch={mean_foot_pitch_error:.3f}rad "
            "swingmax={mean_max_swing_duration:.2f}s "
            "adv=L{mean_left_tread_advances:.2f}/R{mean_right_tread_advances:.2f} "
            "swing=L{mean_left_swing_length_m:.3f}/"
            "R{mean_right_swing_length_m:.3f}m "
            "inward=L{mean_left_foot_inward_error_m:.3f}/"
            "R{mean_right_foot_inward_error_m:.3f}m "
            "foot_y=L{mean_left_foot_lateral_position_m:+.3f}/"
            "R{mean_right_foot_lateral_position_m:+.3f}m "
            "final_y={mean_final_lateral_position_m:+.3f}m "
            "action_rate={mean_action_rate_rms:.3f} "
            "action_accel={mean_action_accel_rms:.3f} "
            "symerr={mean_actor_symmetry_error_rms:.3f} "
            "phase_accel=Rsw{mean_right_swing_action_accel_rms:.3f}/"
            "Lsw{mean_left_swing_action_accel_rms:.3f} "
            "phase_roll=Rsw{mean_right_swing_roll_rate_rms:.3f}/"
            "Lsw{mean_left_swing_roll_rate_rms:.3f} "
            "plan_valid={mean_faststair_planner_valid_fraction:.1%} "
            "plan_err={mean_faststair_foothold_error_m:.3f}m "
            "edge={mean_faststair_edge_margin_m:.3f}m "
            "sole={mean_actual_sole_support_fraction:.1%} "
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
                "policy_symmetry_blend": args.policy_symmetry_blend,
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
        {
            "name": "--policy_symmetry_blend",
            "type": float,
            "default": 0.0,
            "help": (
                "Inference-time mirrored-policy blend in [0, 0.5]. "
                "Use 0.5 for an exactly reflection-equivariant diagnostic."
            ),
        },
        {
            "name": "--contact_phase_reset",
            "action": "store_true",
            "default": False,
            "help": (
                "Synchronize the gait clock after physical stair landings. "
                "Used to compare a legacy Actor fairly with FastStair."
            ),
        },
    ]
    evaluate(parse_humanoid_args(extra_parameters))
