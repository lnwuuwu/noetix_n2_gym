"""MuJoCo contact-mask policies shared by sim2sim and headless tests."""

import numpy as np


# Keep these category bits separate from MuJoCo's low default robot/world bit.
# Left geoms advertise LEFT and accept RIGHT; right geoms do the inverse.
LEFT_LEG_COLLISION_BIT = np.int32(1 << 28)
RIGHT_LEG_COLLISION_BIT = np.int32(1 << 29)
SELF_COLLISION_MODES = frozenset(("disabled", "cross_leg"))


def configure_robot_collision_masks(
        geom_bodyid, geom_contype, geom_conaffinity, body_names,
        self_collision_mode="disabled"):
    """Mutate geom masks while preserving world and visual contact semantics.

    Args:
        geom_bodyid: Per-geom MuJoCo body IDs. Body zero is the world.
        geom_contype: Writable per-geom collision category masks.
        geom_conaffinity: Writable per-geom accepted category masks.
        body_names: Body names indexed by body ID.
        self_collision_mode: ``disabled`` or ``cross_leg``.

    Returns:
        A boolean mask selecting every robot geom.
    """
    mode = str(self_collision_mode).strip().lower()
    if mode not in SELF_COLLISION_MODES:
        choices = ", ".join(sorted(SELF_COLLISION_MODES))
        raise ValueError(
            f"unknown self_collision_mode={mode!r}; expected one of: "
            f"{choices}")

    body_ids = np.asarray(geom_bodyid)
    contype = np.asarray(geom_contype)
    conaffinity = np.asarray(geom_conaffinity)
    if not (
            body_ids.shape == contype.shape == conaffinity.shape):
        raise ValueError(
            "geom_bodyid, geom_contype and geom_conaffinity must have "
            "matching shapes")
    if body_ids.size and int(np.max(body_ids)) >= len(body_names):
        raise ValueError("body_names does not cover every geom body ID")

    robot_geoms = body_ids != 0
    collision_geoms = (
        robot_geoms & ((contype != 0) | (conaffinity != 0)))
    visual_geoms = robot_geoms & ~collision_geoms

    # Keep each collision geom's original low contype. World geoms accept that
    # bit, so robot/world contact survives even though robot affinity is zero.
    conaffinity[robot_geoms] = 0

    if mode == "cross_leg":
        geom_body_names = [
            body_names[int(body_id)] or "" for body_id in body_ids]
        left_leg_geoms = np.array([
            is_collision and name.startswith("L_leg_")
            for is_collision, name in zip(
                collision_geoms, geom_body_names)
        ], dtype=bool)
        right_leg_geoms = np.array([
            is_collision and name.startswith("R_leg_")
            for is_collision, name in zip(
                collision_geoms, geom_body_names)
        ], dtype=bool)
        if not np.any(left_leg_geoms) or not np.any(right_leg_geoms):
            raise ValueError(
                "self_collision_mode='cross_leg' requires collision geoms "
                "on both L_leg_* and R_leg_* bodies")

        contype[left_leg_geoms] |= LEFT_LEG_COLLISION_BIT
        conaffinity[left_leg_geoms] = RIGHT_LEG_COLLISION_BIT
        contype[right_leg_geoms] |= RIGHT_LEG_COLLISION_BIT
        conaffinity[right_leg_geoms] = LEFT_LEG_COLLISION_BIT

    # Duplicated visual meshes start at 0/0 and must remain ray/render-only.
    if np.any(
            (contype[visual_geoms] != 0)
            | (conaffinity[visual_geoms] != 0)):
        raise ValueError("visual-only robot geoms unexpectedly became active")
    return robot_geoms
