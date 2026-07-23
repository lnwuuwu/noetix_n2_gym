"""Stream a stair-policy evaluation from an off-screen Isaac Gym camera.

Isaac Gym Preview 4's interactive viewer uses Vulkan and is commonly black
inside an Xvnc desktop.  This tool deliberately creates no viewer: an Isaac
Gym camera sensor renders on the GPU and a tiny local HTTP server publishes
the frames as MJPEG.  Expose the localhost-only port through SSH to watch it
from a normal browser.
"""

import io
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if not sys.path or os.path.realpath(sys.path[0]) != _REPOSITORY_ROOT:
    sys.path.insert(0, _REPOSITORY_ROOT)

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
from isaacgym import gymapi
import numpy as np
import torch

from humanoid.envs import *  # noqa: F401,F403 - task registration side effects
from humanoid.utils.helpers import parse_humanoid_args
from humanoid.utils import task_registry


STAIR_TASKS = ("n2_stairs", "n2_stairs_robust", "n2_stairs_walk")


class _FrameStore:
    """Thread-safe latest-frame handoff; slow clients never block simulation."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.stopped = False

    def publish(self, frame):
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def wait_for_frame(self, previous_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence or self.stopped,
                timeout=5.0,
            )
            return self.frame, self.sequence, self.stopped


class _StreamHandler(BaseHTTPRequestHandler):
    frame_store = None

    def log_message(self, _format, *args):
        # Avoid one terminal line per browser refresh.
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>N2 stairs live</title>"
                "<style>html,body{margin:0;background:#111;color:#eee;"
                "font-family:sans-serif;height:100%}body{display:flex;"
                "flex-direction:column;align-items:center;justify-content:center}"
                "img{max-width:100vw;max-height:calc(100vh - 3rem);object-fit:contain}"
                "p{margin:.6rem}</style></head><body>"
                "<img src='/stream.mjpg' alt='Waiting for Isaac Gym frames'>"
                "<p>N2 stairs — live off-screen camera</p></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()

        sequence = -1
        try:
            while True:
                frame, sequence, stopped = self.frame_store.wait_for_frame(sequence)
                if frame is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        "Content-Length: {}\r\n\r\n".format(len(frame)).encode(
                            "ascii"
                        )
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                if stopped:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass


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


def _encode_jpeg(rgba, quality):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for browser streaming; run: python -m pip install Pillow"
        ) from exc

    output = io.BytesIO()
    Image.fromarray(rgba[:, :, :3], mode="RGB").save(
        output, format="JPEG", quality=quality, optimize=False
    )
    return output.getvalue()


def _episode_metrics(infos):
    episode = infos.get("episode")
    if not episode:
        return {}
    return {
        key: float(value.item()) if isinstance(value, torch.Tensor) else value
        for key, value in episode.items()
        if key.startswith("stairs_") or key == "terrain_level"
    }


def stream(args):
    if args.task not in STAIR_TASKS:
        raise ValueError(
            "stream_stairs.py only supports: {}".format(", ".join(STAIR_TASKS))
        )
    if not 0 <= args.stream_port <= 65535 or args.stream_port == 0:
        raise ValueError("--stream_port must be in [1, 65535]")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("Camera dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if not 0 <= args.terrain_level < env_cfg.terrain.num_rows:
        raise ValueError(
            "--terrain_level must be in [0, {}]".format(
                env_cfg.terrain.num_rows - 1
            )
        )

    _disable_randomization(env_cfg)
    env_cfg.env.enable_camera_sensors = True
    env_cfg.env.camera_width = args.camera_width
    env_cfg.env.camera_height = args.camera_height
    env_cfg.env.camera_horizontal_fov = args.camera_fov
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.fixed_level = args.terrain_level

    command_min, command_max = env_cfg.commands.ranges.lin_vel_x
    allowed_max = min(
        env_cfg.commands.max_curriculum,
        env_cfg.commands.initial_max_speed
        + env_cfg.commands.speed_per_terrain_level * args.terrain_level,
    )
    command_speed = args.command_speed
    if command_speed is None:
        command_speed = 0.5 * (command_min + allowed_max)
    if not command_min <= command_speed <= allowed_max:
        raise ValueError(
            "--command_speed {:.3f} is outside training range [{:.3f}, {:.3f}]".format(
                command_speed, command_min, allowed_max
            )
        )

    env_cfg.commands.resampling_time = [1000, 1001]
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [command_speed, command_speed]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    # No Vulkan viewer is created.  Headless plus enable_camera_sensors keeps
    # only the graphics device needed by render_all_camera_sensors().
    args.headless = True
    args.num_envs = 1
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
    obs = env.get_observations()

    # Camera-sensor poses are world-frame even though the sensor belongs to an
    # environment. Higher curriculum rows are shifted by terrain_length in X,
    # so use the same terrain-origin offset as play.py's viewer camera.
    camera_offset = env.env_origins[0].detach().cpu().numpy().astype(np.float64)
    camera_position = np.asarray(
        env_cfg.viewer.pos, dtype=np.float64
    ) + camera_offset
    camera_target = np.asarray(
        env_cfg.viewer.lookat, dtype=np.float64
    ) + camera_offset
    env.gym.set_camera_location(
        env.camera_handle,
        env.envs[0],
        gymapi.Vec3(*camera_position),
        gymapi.Vec3(*camera_target),
    )

    frame_store = _FrameStore()
    handler = type(
        "N2StreamHandler",
        (_StreamHandler,),
        {"frame_store": frame_store},
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.stream_port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("Browser stream is ready on the server: http://127.0.0.1:{}/".format(
        args.stream_port
    ))
    print(
        "Keep this process running, forward that port over SSH, then open the URL "
        "in your local browser. Press Ctrl-C here to stop."
    )

    completed_episodes = 0
    step = 0
    try:
        while args.play_steps <= 0 or step < args.play_steps:
            env.commands[:, 0] = command_speed
            env.commands[:, 1:3] = 0.0
            with torch.inference_mode():
                actions = policy(obs.detach())
            obs, _, _, _, infos, termination_ids, _ = env.step(actions.detach())

            # GPU PhysX results must be complete before graphics/camera access.
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            raw = env.gym.get_camera_image(
                env.sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR
            )
            rgba = np.asarray(raw, dtype=np.uint8).reshape(
                args.camera_height, args.camera_width, 4
            )
            frame_store.publish(_encode_jpeg(rgba, args.jpeg_quality))

            if len(termination_ids) > 0:
                completed_episodes += len(termination_ids)
                metrics = _episode_metrics(infos)
                if metrics:
                    print("episode {}: {}".format(completed_episodes, metrics))
            step += 1
    except KeyboardInterrupt:
        print("Stopping live stream.")
    finally:
        frame_store.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    extra_parameters = [
        {
            "name": "--command_speed",
            "type": float,
            "default": None,
            "help": "Forward command in m/s; defaults to the level's midpoint.",
        },
        {
            "name": "--terrain_level",
            "type": int,
            "default": 0,
            "help": "Stair difficulty row (0=2 cm, ..., 4=10 cm).",
        },
        {
            "name": "--play_steps",
            "type": int,
            "default": 0,
            "help": "Policy steps to stream; 0 runs until Ctrl-C.",
        },
        {
            "name": "--stream_port",
            "type": int,
            "default": 8080,
            "help": "Server localhost port forwarded through SSH.",
        },
        {
            "name": "--camera_width",
            "type": int,
            "default": 960,
            "help": "Stream image width.",
        },
        {
            "name": "--camera_height",
            "type": int,
            "default": 540,
            "help": "Stream image height.",
        },
        {
            "name": "--camera_fov",
            "type": float,
            "default": 75.0,
            "help": "Horizontal camera field of view in degrees.",
        },
        {
            "name": "--jpeg_quality",
            "type": int,
            "default": 80,
            "help": "JPEG quality from 1 to 100.",
        },
    ]
    stream(parse_humanoid_args(extra_parameters))
