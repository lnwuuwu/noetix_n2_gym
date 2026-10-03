"""Viewer-independent smooth camera tracking for MuJoCo playback."""

import math

import numpy as np


def _smoothing_fraction(dt, time_constant):
    if dt <= 0.0:
        raise ValueError("camera update dt must be positive")
    if time_constant < 0.0:
        raise ValueError("camera time constants must be non-negative")
    return 1.0 if time_constant == 0.0 else (
        1.0 - math.exp(-dt / time_constant))


class SmoothFollowCamera:
    """Track base XYZ and heading without frame-rate-dependent snapping."""

    def __init__(
            self, lookahead=0.35, z_offset=-0.10,
            azimuth_offset=70.0, position_tau=0.18, yaw_tau=0.25,
            snap_distance=2.0):
        if snap_distance <= 0.0:
            raise ValueError("camera snap_distance must be positive")
        if position_tau < 0.0 or yaw_tau < 0.0:
            raise ValueError("camera time constants must be non-negative")
        self.lookahead = float(lookahead)
        self.z_offset = float(z_offset)
        self.azimuth_offset = float(azimuth_offset)
        self.position_tau = float(position_tau)
        self.yaw_tau = float(yaw_tau)
        self.snap_distance = float(snap_distance)
        self.lookat = None
        self.azimuth = None

    def target(self, base_xyz, yaw):
        base_xyz = np.asarray(base_xyz, dtype=np.float64)
        if base_xyz.shape != (3,):
            raise ValueError("base_xyz must have shape (3,)")
        return np.array([
            base_xyz[0] + self.lookahead * math.cos(yaw),
            base_xyz[1] + self.lookahead * math.sin(yaw),
            base_xyz[2] + self.z_offset,
        ], dtype=np.float64)

    def update(self, base_xyz, yaw, dt):
        target = self.target(base_xyz, yaw)
        target_azimuth = (
            math.degrees(float(yaw)) + self.azimuth_offset)
        snap = (
            self.lookat is None
            or np.linalg.norm(target - self.lookat) > self.snap_distance)
        if snap:
            self.lookat = target
            self.azimuth = target_azimuth
        else:
            position_fraction = _smoothing_fraction(
                float(dt), self.position_tau)
            self.lookat += position_fraction * (target - self.lookat)

            yaw_fraction = _smoothing_fraction(float(dt), self.yaw_tau)
            delta = (
                (target_azimuth - self.azimuth + 180.0) % 360.0
                - 180.0)
            self.azimuth += yaw_fraction * delta
        return self.lookat.copy(), float(self.azimuth)
