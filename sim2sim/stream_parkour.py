#!/usr/bin/env python3
"""Launch n2_parkour with EGL rendering and browser MJPEG streaming."""

import os
import sys


def _raw_arg(name, default):
    prefix = name + "="
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
    return default


# These must be selected before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MUJOCO_EGL_DEVICE_ID", _raw_arg("--egl_device", "1"))
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from sim2sim_parkour import main  # noqa: E402


if __name__ == "__main__":
    main(force_stream=True)
