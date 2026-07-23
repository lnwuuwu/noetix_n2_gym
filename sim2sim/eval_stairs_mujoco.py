"""Headless MuJoCo sim-to-sim acceptance test for N2 stair policies.

This is deliberately an evaluator, not a second training environment.  It
reconstructs the deployable 410-D actor observation, runs the exported JIT
policy through the same PD loop, and reports contact/step metrics that expose
policies which only succeed by exploiting PhysX stair contacts.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import deque

import mujoco
import numpy as np
import torch
import yaml

from humanoid import LEGGED_GYM_ROOT_DIR

_UTILS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "humanoid", "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from stairs_terrain import (  # noqa: E402
    classify_tread_transition,
    terrain_height_at_x,
)

# When this file is executed as ``python sim2sim/eval_stairs_mujoco.py``, the
# sibling sim2sim.py is importable as the top-level ``sim2sim`` module.
from sim2sim import (  # noqa: E402
    gait_phase_observations,
    get_obs,
    load_mujoco_model,
    pd_control,
    resolve_joint_layout,
    stair_height_observations,
)


FOOT_BODIES = ("L_leg_ankle_link", "R_leg_ankle_link")
LOWER_LEG_BODIES = ("L_leg_knee_link", "R_leg_knee_link")
FOOT_SITES = ("L_leg_foot_contact_ground", "R_leg_foot_contact_ground")

PHYSICS_PRESETS = {
    # Exact historical MJCF joint/contact values, including self-collision.
    "legacy_mjcf": {
        "joint_damping": 0.001,
        "joint_armature": 0.01,
        "joint_frictionloss": 0.1,
        "contact_dim": 1,
        "contact_priority": 0,
        "disable_self_collisions": False,
    },
    # Isolate the self-collision mismatch while retaining legacy dynamics.
    "legacy_no_self": {
        "joint_damping": 0.001,
        "joint_armature": 0.01,
        "joint_frictionloss": 0.1,
        "contact_dim": 1,
        "contact_priority": 0,
        "disable_self_collisions": True,
    },
    # Preserve the stabilizing MJCF joint dynamics but align contact handling.
    "hybrid": {
        "joint_damping": 0.001,
        "joint_armature": 0.01,
        "joint_frictionloss": 0.1,
        "contact_dim": 3,
        "contact_priority": 1,
        "disable_self_collisions": True,
    },
    # Match the Isaac URDF import as closely as the MJCF permits.
    "isaac_aligned": {
        "joint_damping": 0.0,
        "joint_armature": 0.0,
        "joint_frictionloss": 0.0,
        "contact_dim": 3,
        "contact_priority": 1,
        "disable_self_collisions": True,
    },
}


def _resolve_config_path(value):
    if os.path.isabs(value):
        return value
    return os.path.join(LEGGED_GYM_ROOT_DIR, "sim2sim", "configs", value)


def _expanded_path(value):
    return os.path.abspath(
        os.path.expanduser(
            str(value).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        )
    )


def _named_ids(model, object_type, names):
    result = []
    for name in names:
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise ValueError("MuJoCo object not found: {}".format(name))
        result.append(int(object_id))
    return result


def _configure_solver(model, config):
    model.opt.timestep = float(config["simulation_dt"])
    integrator = str(config.get("integrator", "")).strip().upper()
    if integrator:
        enum_name = "mjINT_{}".format(integrator)
        if not hasattr(mujoco.mjtIntegrator, enum_name):
            raise ValueError("Unsupported MuJoCo integrator: " + integrator)
        model.opt.integrator = getattr(mujoco.mjtIntegrator, enum_name)
    model.opt.iterations = int(
        config.get("solver_iterations", model.opt.iterations)
    )
    model.opt.noslip_iterations = int(
        config.get(
            "solver_noslip_iterations", model.opt.noslip_iterations
        )
    )


class ContactLayout:
    """Resolved MuJoCo ids needed for independent contact diagnostics."""

    def __init__(self, model, num_steps):
        self.foot_bodies = _named_ids(
            model, mujoco.mjtObj.mjOBJ_BODY, FOOT_BODIES
        )
        self.lower_leg_bodies = _named_ids(
            model, mujoco.mjtObj.mjOBJ_BODY, LOWER_LEG_BODIES
        )
        self.foot_sites = _named_ids(
            model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES
        )
        self.surface_tread = {}
        self.stair_geoms = set()
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if not name:
                continue
            if name == "n2_stair_top":
                self.surface_tread[geom_id] = int(num_steps)
                self.stair_geoms.add(geom_id)
            elif name.startswith("n2_stair_"):
                suffix = name[len("n2_stair_"):]
                if suffix.isdigit():
                    self.surface_tread[geom_id] = int(suffix)
                    self.stair_geoms.add(geom_id)
            elif "ground" in name.lower():
                self.surface_tread[geom_id] = 0

        if not self.stair_geoms:
            raise ValueError("Generated MuJoCo model contains no N2 stairs")


def sample_contacts(model, data, layout, validation):
    """Return per-foot support treads and collision flags for one sim step."""
    support_tread = np.full(2, -1, dtype=np.int64)
    riser = np.zeros(2, dtype=bool)
    lower_leg = np.zeros(2, dtype=bool)
    support_normal_z = float(validation["support_normal_z"])
    contact_threshold = float(validation["contact_force_threshold"])
    riser_threshold = float(validation["riser_force_threshold"])
    shin_threshold = float(validation["lower_leg_force_threshold"])

    foot_by_body = {
        body_id: foot for foot, body_id in enumerate(layout.foot_bodies)
    }
    shin_by_body = {
        body_id: foot for foot, body_id in enumerate(layout.lower_leg_bodies)
    }

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        surface_geom = None
        robot_body = None
        if geom1 in layout.surface_tread:
            surface_geom, robot_body = geom1, body2
        elif geom2 in layout.surface_tread:
            surface_geom, robot_body = geom2, body1
        if surface_geom is None:
            continue

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_force = abs(float(force[0]))
        normal = np.asarray(contact.frame[:3], dtype=np.float64)

        if robot_body in foot_by_body:
            foot = foot_by_body[robot_body]
            if (
                normal_force >= contact_threshold
                and abs(float(normal[2])) >= support_normal_z
            ):
                support_tread[foot] = max(
                    support_tread[foot],
                    int(layout.surface_tread[surface_geom]),
                )
            if (
                surface_geom in layout.stair_geoms
                and normal_force >= riser_threshold
                and abs(float(normal[0])) > abs(float(normal[2]))
            ):
                riser[foot] = True
        elif (
            robot_body in shin_by_body
            and surface_geom in layout.stair_geoms
            and normal_force >= shin_threshold
        ):
            lower_leg[shin_by_body[robot_body]] = True

    return support_tread, riser, lower_leg


class GaitTracker:
    """Stateful, contact-based stair-over-stair event tracker."""

    def __init__(self, control_dt, stair_cfg, validation):
        self.control_dt = float(control_dt)
        self.step_height = float(stair_cfg["step_height"])
        self.num_steps = int(stair_cfg["num_steps"])
        self.stable_contact_s = float(validation["stable_contact_s"])
        self.stable_release_s = float(validation["stable_release_s"])
        self.raw_candidate = np.full(2, -1, dtype=np.int64)
        self.candidate_time = np.zeros(2, dtype=np.float64)
        self.loss_time = np.zeros(2, dtype=np.float64)
        self.stable_tread = np.full(2, -1, dtype=np.int64)
        self.accepted_tread = np.zeros(2, dtype=np.int64)
        self.pending_swing = np.zeros(2, dtype=bool)
        self.airborne_seen = np.zeros(2, dtype=bool)
        self.opposite_valid = np.zeros(2, dtype=bool)
        self.last_advanced_tread = 0
        self.last_advanced_foot = -1
        self.last_joined_tread = -1
        self.advance_count = 0
        self.alternating_count = 0
        self.repeated_count = 0
        self.join_count = 0
        self.skipped_count = 0
        self.left_advances = 0
        self.right_advances = 0
        self.control_steps = 0
        self.double_flight_steps = 0
        self.same_tread_steps = 0
        self.riser_steps = 0
        self.lower_leg_steps = 0

    def _update_stable(self, raw_tread):
        previous = self.stable_tread.copy()
        for foot in range(2):
            tread = int(raw_tread[foot])
            if tread >= 0:
                if tread == self.raw_candidate[foot]:
                    self.candidate_time[foot] += self.control_dt
                else:
                    self.raw_candidate[foot] = tread
                    self.candidate_time[foot] = self.control_dt
                self.loss_time[foot] = 0.0
                if self.candidate_time[foot] >= self.stable_contact_s:
                    self.stable_tread[foot] = tread
            else:
                self.raw_candidate[foot] = -1
                self.candidate_time[foot] = 0.0
                self.loss_time[foot] += self.control_dt
                if self.loss_time[foot] >= self.stable_release_s:
                    self.stable_tread[foot] = -1
        return previous

    def _consume_landing(self, foot, tread):
        result = classify_tread_transition(
            np.asarray([True]),
            np.asarray([tread]),
            np.asarray([foot]),
            np.asarray([self.last_advanced_tread]),
            np.asarray([self.last_advanced_foot]),
            np.asarray([self.last_joined_tread]),
        )
        advanced, alternating, repeated, joined, skipped = (
            bool(value[0]) for value in result
        )
        self.advance_count += int(advanced)
        self.alternating_count += int(alternating)
        self.repeated_count += int(repeated)
        self.join_count += int(joined)
        self.skipped_count += int(skipped)
        if advanced:
            if foot == 0:
                self.left_advances += 1
            else:
                self.right_advances += 1
            self.last_advanced_tread = int(tread)
            self.last_advanced_foot = int(foot)
        if joined:
            self.last_joined_tread = int(tread)

    def update(self, raw_tread, foot_site_z, riser, lower_leg):
        previous_stable = self._update_stable(raw_tread)
        self.control_steps += 1
        self.riser_steps += int(np.any(riser))
        self.lower_leg_steps += int(np.any(lower_leg))
        if np.all(raw_tread < 0):
            self.double_flight_steps += 1
        if (
            self.stable_tread[0] == self.stable_tread[1]
            and 0 < self.stable_tread[0] < self.num_steps
        ):
            self.same_tread_steps += 1

        true_airborne = np.asarray(
            [
                raw_tread[foot] < 0
                and foot_site_z[foot]
                - (self.accepted_tread[foot] * self.step_height + 0.045)
                > 0.015
                for foot in range(2)
            ],
            dtype=bool,
        )
        for foot in range(2):
            if (
                not self.pending_swing[foot]
                and previous_stable[foot] >= 0
                and raw_tread[foot] < 0
            ):
                self.pending_swing[foot] = True
                self.airborne_seen[foot] = False
                self.opposite_valid[foot] = self.stable_tread[1 - foot] >= 0

            if self.pending_swing[foot] and true_airborne[foot]:
                self.airborne_seen[foot] = True
            if self.pending_swing[foot] and true_airborne[1 - foot]:
                self.opposite_valid[foot] = False

        landings = []
        for foot in range(2):
            tread = int(self.stable_tread[foot])
            if (
                self.pending_swing[foot]
                and self.airborne_seen[foot]
                and tread >= 0
                and tread > self.accepted_tread[foot]
            ):
                landings.append((foot, tread, self.opposite_valid[foot]))
            if self.pending_swing[foot] and tread >= 0:
                self.pending_swing[foot] = False
                self.airborne_seen[foot] = False
                self.opposite_valid[foot] = False

        if len(landings) == 1:
            foot, tread, natural_support = landings[0]
            previous_alternating = self.alternating_count
            self._consume_landing(foot, tread)
            if not natural_support:
                self.alternating_count = previous_alternating
            self.accepted_tread[foot] = max(
                self.accepted_tread[foot], tread
            )
        elif len(landings) > 1:
            # Consume simultaneous height changes without crediting a natural
            # stair-over-stair event.
            highest = self.last_advanced_tread
            for foot, tread, _ in landings:
                self.accepted_tread[foot] = max(
                    self.accepted_tread[foot], tread
                )
                if tread > highest:
                    highest = tread
                if foot == 0:
                    self.left_advances += 1
                else:
                    self.right_advances += 1
            self.advance_count += int(highest > self.last_advanced_tread)
            self.last_advanced_tread = highest
            self.last_advanced_foot = -1
            self.last_joined_tread = -1

    def summary(self):
        event_denominator = max(1, self.advance_count + self.join_count)
        step_denominator = max(1, self.control_steps)
        return {
            "alternating_tread_count": self.alternating_count,
            "alternating_tread_rate": (
                self.alternating_count / event_denominator
            ),
            "repeated_lead_rate": self.repeated_count / event_denominator,
            "same_tread_join_rate": self.join_count / event_denominator,
            "skipped_tread_rate": self.skipped_count / event_denominator,
            "same_tread_support_fraction": (
                self.same_tread_steps / step_denominator
            ),
            "double_flight_fraction": (
                self.double_flight_steps / step_denominator
            ),
            "foot_riser_collision_fraction": (
                self.riser_steps / step_denominator
            ),
            "lower_leg_collision_fraction": (
                self.lower_leg_steps / step_denominator
            ),
            "left_tread_advances": self.left_advances,
            "right_tread_advances": self.right_advances,
        }


def _build_observation(
    config,
    data,
    quat,
    base_velocity,
    omega,
    gravity,
    q,
    dq,
    action,
    command,
    elapsed_time,
):
    num_actions = int(config["num_actions"])
    num_single_obs = int(config["num_single_obs"])
    default = np.asarray(config["default_angles"], dtype=np.float32)
    obs = np.zeros((1, num_single_obs), dtype=np.float32)
    cursor = 0
    obs[0, cursor:cursor + 3] = command * np.asarray(
        config["cmd_scale"], dtype=np.float32
    )
    cursor += 3

    phase_cfg = config.get("gait_phase")
    if phase_cfg is not None:
        obs[0, cursor:cursor + 2] = gait_phase_observations(
            elapsed_time, command[0], phase_cfg
        )
        cursor += 2
    if bool(config.get("include_base_lin_vel", False)):
        obs[0, cursor:cursor + 3] = base_velocity * float(
            config.get("lin_vel_scale", 1.0)
        )
        cursor += 3

    navigation = config.get("navigation_state")
    if navigation is not None:
        x, y, z, w = quat
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        target_yaw = float(navigation.get("target_yaw", 0.0))
        yaw_error = math.atan2(
            math.sin(yaw - target_yaw), math.cos(yaw - target_yaw)
        )
        obs[0, cursor] = (
            data.qpos[1] - float(navigation.get("center_y", 0.0))
        ) * float(navigation.get("lateral_scale", 2.0))
        obs[0, cursor + 1] = yaw_error * float(
            navigation.get("yaw_scale", 1.0)
        )
        cursor += 2

    obs[0, cursor:cursor + 3] = omega * float(config["ang_vel_scale"])
    cursor += 3
    obs[0, cursor:cursor + 3] = gravity[:3]
    cursor += 3
    obs[0, cursor:cursor + num_actions] = (
        q - default
    ) * float(config["dof_pos_scale"])
    cursor += num_actions
    obs[0, cursor:cursor + num_actions] = dq * float(
        config["dof_vel_scale"]
    )
    cursor += num_actions
    obs[0, cursor:cursor + num_actions] = action
    cursor += num_actions

    height_cfg = config.get("height_measurements")
    if height_cfg is not None:
        heights = stair_height_observations(
            data.qpos[:3], quat, height_cfg, config["stairs"]
        )
        if cursor + len(heights) != num_single_obs:
            raise ValueError(
                "Observation layout {} + {} != {}".format(
                    cursor, len(heights), num_single_obs
                )
            )
        obs[0, cursor:] = heights
    elif cursor != num_single_obs:
        raise ValueError(
            "Observation layout {} != {}".format(cursor, num_single_obs)
        )
    return np.clip(
        obs,
        -float(config.get("clip_observations", np.inf)),
        float(config.get("clip_observations", np.inf)),
    )


def run_episode(config, model, policy, seed):
    validation = config["validation"]
    stair_cfg = config["stairs"]
    control_decimation = int(config["control_decimation"])
    simulation_dt = float(config["simulation_dt"])
    control_dt = simulation_dt * control_decimation
    duration = float(validation["episode_duration"])
    num_actions = int(config["num_actions"])
    frame_stack = int(config["frame_stack"])
    num_single_obs = int(config["num_single_obs"])
    num_obs = int(config["num_obs"])

    data = mujoco.MjData(model)
    joint_order = list(config["joint_order"])
    qpos_indices, qvel_indices, control_indices = resolve_joint_layout(
        model, joint_order
    )
    default = np.asarray(config["default_angles"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    joint_noise = float(validation.get("initial_joint_noise", 0.0))
    lateral_noise = float(validation.get("initial_lateral_noise", 0.0))
    data.qpos[qpos_indices] = default + rng.uniform(
        -joint_noise, joint_noise, size=num_actions
    )
    data.qpos[1] += rng.uniform(-lateral_noise, lateral_noise)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    layout = ContactLayout(model, int(stair_cfg["num_steps"]))
    tracker = GaitTracker(control_dt, stair_cfg, validation)
    history = deque(
        [np.zeros((1, num_single_obs), dtype=np.float32) for _ in range(frame_stack)],
        maxlen=frame_stack,
    )
    command = np.asarray(config.get("cmd_init", [0.18, 0.0, 0.0]), dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_q = default.copy()
    kps = np.asarray(config["kps"], dtype=np.float64)
    kds = np.asarray(config["kds"], dtype=np.float64)
    torque_limits = np.asarray(config["torque_limits"], dtype=np.float64)
    action_scale = float(config["action_scale"])
    clip_actions = float(config.get("clip_actions", np.inf))

    completion_time = 0.0
    completed = False
    fell = False
    path_failure = False
    numerical_failure = False
    max_lateral = 0.0
    max_yaw = 0.0
    speed_sum = 0.0
    control_steps = 0
    elapsed = 0.0

    lowlevel_steps = int(math.ceil(duration / simulation_dt))
    for lowlevel_step in range(lowlevel_steps):
        _, _, quat, velocity, omega, gravity = get_obs(data)
        q = data.qpos[qpos_indices].copy()
        dq = data.qvel[qvel_indices].copy()

        if lowlevel_step % control_decimation == 0:
            obs = _build_observation(
                config,
                data,
                quat,
                velocity,
                omega,
                gravity,
                q,
                dq,
                action,
                command,
                elapsed,
            )
            history.append(obs)
            model_input = np.concatenate(list(history), axis=1)
            if model_input.shape != (1, num_obs):
                raise ValueError(
                    "Stacked observation {} != (1, {})".format(
                        model_input.shape, num_obs
                    )
                )
            with torch.inference_mode():
                output = policy(torch.from_numpy(model_input))[0]
            action[:] = np.clip(
                output.detach().cpu().numpy(), -clip_actions, clip_actions
            )
            target_q = action * action_scale + default

            raw_tread, riser, lower_leg = sample_contacts(
                model, data, layout, validation
            )
            foot_site_z = np.asarray(
                [data.site_xpos[site_id, 2] for site_id in layout.foot_sites]
            )
            tracker.update(raw_tread, foot_site_z, riser, lower_leg)
            control_steps += 1
            speed_sum += float(velocity[0])
            max_lateral = max(max_lateral, abs(float(data.qpos[1])))
            x, y, z, w = quat
            yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            max_yaw = max(max_yaw, abs(yaw))
            path_failure = path_failure or (
                max_lateral > float(validation["corridor_half_width"])
                or max_yaw > float(validation["corridor_yaw_limit"])
            )

            terrain_height = float(
                terrain_height_at_x(
                    np.asarray([data.qpos[0]]),
                    start_x=float(stair_cfg["start_x"]),
                    step_width=float(stair_cfg["step_width"]),
                    step_height=float(stair_cfg["step_height"]),
                    num_steps=int(stair_cfg["num_steps"]),
                )[0]
            )
            fell = (
                data.qpos[2] - terrain_height
                < float(validation["fall_base_height"])
                or gravity[2] > float(validation["fall_upright_z"])
            )
            on_top = (
                data.qpos[0] >= float(validation["success_x"])
                and data.qpos[2] - terrain_height
                >= float(validation["fall_base_height"])
            )
            completion_time = (
                completion_time + control_dt if on_top else 0.0
            )
            completed = completion_time >= float(
                validation["completion_dwell_s"]
            )

        tau = pd_control(target_q, q, kps, np.zeros(num_actions), dq, kds)
        tau = np.clip(tau, -torque_limits, torque_limits)
        data.ctrl[:] = 0.0
        data.ctrl[control_indices] = tau
        mujoco.mj_step(model, data)
        elapsed = (lowlevel_step + 1) * simulation_dt

        if (
            not np.all(np.isfinite(data.qpos))
            or not np.all(np.isfinite(data.qvel))
            or not np.all(np.isfinite(data.qacc))
            or np.max(np.abs(data.qacc)) > 1.0e8
        ):
            numerical_failure = True
        if fell or path_failure or completed or numerical_failure:
            break

    gait = tracker.summary()
    success = (
        completed
        and not fell
        and not path_failure
        and not numerical_failure
        and tracker.alternating_count
        >= int(validation["success_min_alternating_tread_count"])
        and gait["alternating_tread_rate"]
        >= float(validation["success_min_alternating_tread_rate"])
        and gait["same_tread_join_rate"]
        <= float(validation["success_max_same_tread_join_rate"])
        and gait["double_flight_fraction"]
        <= float(validation["success_max_double_flight_fraction"])
    )
    result = {
        "success": float(success),
        "completion": float(completed),
        "fall": float(fell),
        "path_failure": float(path_failure),
        "numerical_failure": float(numerical_failure),
        "forward_distance_m": float(data.qpos[0]),
        "climb_height_m": float(
            max(
                0.0,
                data.qpos[2]
                - float(validation.get("initial_base_height", 0.75)),
            )
        ),
        "survival_time_s": float(elapsed),
        "mean_forward_speed_m_s": speed_sum / max(1, control_steps),
        "max_lateral_deviation_m": max_lateral,
        "max_yaw_deviation_rad": max_yaw,
    }
    result.update({key: float(value) for key, value in gait.items()})
    return result


def aggregate_results(results, config, policy_path):
    keys = list(results[0].keys())
    summary = {
        "engine": "mujoco",
        "physics_preset": str(
            config.get("mujoco_physics", {}).get("preset", "configured")
        ),
        "policy_path": policy_path,
        "episodes": len(results),
        "stair_start_x_m": float(config["stairs"]["start_x"]),
        "step_height_m": float(config["stairs"]["step_height"]),
        "command_speed_m_s": float(config["cmd_init"][0]),
    }
    for key in keys:
        aggregate_key = key if key.startswith("mean_") else "mean_" + key
        summary[aggregate_key] = float(
            np.mean([episode[key] for episode in results])
        )
    for binary in (
        "success",
        "completion",
        "fall",
        "path_failure",
        "numerical_failure",
    ):
        summary[binary + "_rate"] = summary.pop("mean_" + binary)
    return summary


def write_report(output_path, summary, episodes):
    if not output_path:
        return
    output_path = os.path.abspath(output_path)
    if not output_path.lower().endswith(".csv"):
        output_path += ".csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(
            {"summary": summary, "episode_results": episodes},
            json_file,
            indent=2,
        )
    print("Saved MuJoCo CSV: " + output_path)
    print("Saved MuJoCo JSON: " + json_path)


def _apply_cli_overrides(config, args):
    if args.policy_path:
        config["policy_path"] = args.policy_path
    if args.step_height is not None:
        config["stairs"]["step_height"] = float(args.step_height)
    if args.duration is not None:
        config["validation"]["episode_duration"] = float(args.duration)
    if args.command_speed is not None:
        config["cmd_init"][0] = float(args.command_speed)
    if args.stair_start_x is not None:
        stair_start_x = float(args.stair_start_x)
        config["stairs"]["start_x"] = stair_start_x
        config["validation"]["success_x"] = (
            stair_start_x
            + float(config["stairs"]["step_width"])
            * int(config["stairs"]["num_steps"])
            + 0.20
        )
    if args.initial_joint_noise is not None:
        config["validation"]["initial_joint_noise"] = float(
            args.initial_joint_noise
        )
    if args.initial_lateral_noise is not None:
        config["validation"]["initial_lateral_noise"] = float(
            args.initial_lateral_noise
        )
    if args.physics_preset is not None:
        if args.physics_preset not in PHYSICS_PRESETS:
            raise ValueError(
                "Unknown MuJoCo physics preset: " + args.physics_preset
            )
        physics = config.setdefault("mujoco_physics", {})
        physics.update(PHYSICS_PRESETS[args.physics_preset])
        physics["preset"] = args.physics_preset
    for name in ("initial_joint_noise", "initial_lateral_noise"):
        if float(config["validation"].get(name, 0.0)) < 0.0:
            raise ValueError("--{} must be non-negative".format(name))
    return config


def evaluate(args):
    config_path = _resolve_config_path(args.config_file)
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = _apply_cli_overrides(config, args)

    policy_path = _expanded_path(config["policy_path"])
    xml_path = _expanded_path(config["xml_path"])
    if not os.path.isfile(policy_path):
        raise ValueError("JIT policy does not exist: " + policy_path)
    if not os.path.isfile(xml_path):
        raise ValueError("MJCF does not exist: " + xml_path)

    model = load_mujoco_model(
        xml_path, config["stairs"], config.get("mujoco_physics")
    )
    _configure_solver(model, config)
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    results = []
    started = time.monotonic()
    for episode in range(args.episodes):
        print(
            "Running MuJoCo episode {}/{} (seed={})...".format(
                episode + 1, args.episodes, args.seed + episode
            ),
            flush=True,
        )
        result = run_episode(config, model, policy, args.seed + episode)
        results.append(result)
        print(
            "  completion={:.0%} fall={:.0%} path={:.0%} numeric={:.0%} "
            "alternate={:.1%} join={:.1%} sim_time={:.2f}s".format(
                result["completion"],
                result["fall"],
                result["path_failure"],
                result["numerical_failure"],
                result["alternating_tread_rate"],
                result["same_tread_join_rate"],
                result["survival_time_s"],
            ),
            flush=True,
        )
    summary = aggregate_results(results, config, policy_path)
    print(
        "MuJoCo physics={physics_preset} start={stair_start_x_m:.2f}m "
        "step={step_height_m:.2f}m "
        "success={success_rate:.1%} "
        "completion={completion_rate:.1%} fall={fall_rate:.1%} "
        "numeric={numerical_failure_rate:.1%} "
        "alternate={mean_alternating_tread_rate:.1%} "
        "join={mean_same_tread_join_rate:.1%} "
        "riser={mean_foot_riser_collision_fraction:.1%} "
        "shin={mean_lower_leg_collision_fraction:.1%} "
        "distance={mean_forward_distance_m:.3f}m".format(**summary)
    )
    print("MuJoCo wall time: {:.1f}s".format(time.monotonic() - started))
    write_report(args.output, summary, results)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="n2_stairs_walk.yaml"
    )
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--step_height", type=float, default=None)
    parser.add_argument("--stair_start_x", type=float, default=None)
    parser.add_argument("--command_speed", type=float, default=None)
    parser.add_argument("--initial_joint_noise", type=float, default=None)
    parser.add_argument("--initial_lateral_noise", type=float, default=None)
    parser.add_argument(
        "--physics_preset", choices=tuple(PHYSICS_PRESETS), default=None
    )
    parser.add_argument("--physics_sweep", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        raise ValueError("--episodes must be positive")
    if arguments.physics_sweep:
        output_path = arguments.output
        for preset_name in PHYSICS_PRESETS:
            sweep_arguments = copy.copy(arguments)
            sweep_arguments.physics_sweep = False
            sweep_arguments.physics_preset = preset_name
            if output_path:
                output_root, output_extension = os.path.splitext(output_path)
                if output_extension.lower() != ".csv":
                    output_root = output_path
                sweep_arguments.output = (
                    output_root + "_" + preset_name + ".csv"
                )
            print("\n=== MuJoCo physics preset: {} ===".format(preset_name))
            evaluate(sweep_arguments)
    else:
        evaluate(arguments)
