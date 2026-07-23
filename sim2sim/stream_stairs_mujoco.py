"""Publish a native MuJoCo stair evaluation as a browser MJPEG stream."""

import os

# This must be selected before importing mujoco.  AutoDL NVIDIA containers
# provide headless EGL even when no X/GLFW display is available.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import io
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_stairs_mujoco import (  # noqa: E402
    PHYSICS_PRESETS,
    _configure_solver,
    load_policy,
    run_episode,
)
from sim2sim import load_mujoco_model  # noqa: E402


class FrameStore:
    """Latest-frame handoff; a slow browser never blocks MuJoCo."""

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

    def wait(self, previous_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.sequence != previous_sequence or self.stopped
                ),
                timeout=5.0,
            )
            return self.frame, self.sequence, self.stopped


class StreamHandler(BaseHTTPRequestHandler):
    frame_store = None
    checkpoint_label = ""

    def log_message(self, _format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>N2 MuJoCo stairs</title>"
                "<style>html,body{margin:0;background:#101216;color:#eee;"
                "font-family:sans-serif;height:100%}body{display:flex;"
                "flex-direction:column;align-items:center;justify-content:center}"
                "img{max-width:100vw;max-height:calc(100vh - 4rem);"
                "object-fit:contain}p{margin:.45rem}</style></head><body>"
                "<img src='/stream.mjpg' alt='Waiting for MuJoCo frames'>"
                "<p>N2 — native MuJoCo 10 cm stair evaluation</p>"
                "<p>{}</p></body></html>".format(self.checkpoint_label)
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
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store, no-cache")
        self.end_headers()
        sequence = -1
        try:
            while True:
                frame, sequence, stopped = self.frame_store.wait(sequence)
                if frame is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        "Content-Length: {}\r\n\r\n".format(
                            len(frame)
                        ).encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                if stopped:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass


def latest_native_checkpoint():
    root = ROOT / "logs_mujoco" / "n2_stairs_walk"
    best = [
        Path(path)
        for path in glob.glob(str(root / "*" / "model_best.pt"))
    ]
    if best:
        return max(best, key=lambda path: path.stat().st_mtime)
    numeric = []
    for path_string in glob.glob(str(root / "*" / "model_*.pt")):
        path = Path(path_string)
        try:
            int(path.stem[len("model_"):])
        except ValueError:
            continue
        if "smoke" not in path.parent.name:
            numeric.append(path)
    if numeric:
        checkpoint = max(
            numeric, key=lambda path: path.stat().st_mtime
        )
        print(
            "No model_best.pt yet; visualizing latest saved checkpoint: "
            + str(checkpoint),
            flush=True,
        )
        return checkpoint
    raise ValueError(
        "No native MuJoCo checkpoint exists yet. Run training first, "
        "or pass "
        "--checkpoint_path=/absolute/path/model_*.pt"
    )


def encode_jpeg(rgb, quality):
    try:
        import cv2

        bgr = cv2.cvtColor(
            np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode a JPEG frame")
        return encoded.tobytes()
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Browser streaming needs opencv-python or Pillow"
            ) from exc
        output = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(
            output,
            format="JPEG",
            quality=int(quality),
            optimize=False,
        )
        return output.getvalue()


def overlay(rgb, episode, elapsed, data, status):
    try:
        import cv2
    except ImportError:
        return rgb
    frame = np.ascontiguousarray(rgb.copy())
    text = (
        "episode={}  t={:.1f}s  x={:.2f}m  y={:+.2f}m  yaw={:+.2f}rad"
    ).format(
        episode,
        elapsed,
        float(data.qpos[0]),
        float(data.qpos[1]),
        float(status["yaw"]),
    )
    cv2.putText(
        frame,
        text,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


class LiveRenderer:
    def __init__(
        self,
        renderer,
        camera,
        frame_store,
        episode,
        fps,
        playback_speed,
        jpeg_quality,
    ):
        self.renderer = renderer
        self.camera = camera
        self.frame_store = frame_store
        self.episode = episode
        self.frame_period = 1.0 / float(fps)
        self.playback_speed = float(playback_speed)
        self.jpeg_quality = int(jpeg_quality)
        self.next_frame_time = 0.0
        self.wall_start = time.monotonic()

    def __call__(self, data, elapsed, status):
        terminal = any(
            bool(status[name])
            for name in (
                "completed",
                "fell",
                "path_failure",
                "numerical_failure",
            )
        )
        if elapsed + 1.0e-9 < self.next_frame_time and not terminal:
            return
        self.next_frame_time = elapsed + self.frame_period
        self.renderer.update_scene(data, self.camera)
        rgb = self.renderer.render()
        rgb = overlay(rgb, self.episode, elapsed, data, status)
        self.frame_store.publish(
            encode_jpeg(rgb, self.jpeg_quality)
        )
        target_wall = self.wall_start + elapsed / self.playback_speed
        delay = target_wall - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)


def load_stream_components(args):
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = ROOT / "sim2sim" / "configs" / config_path
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    config["stairs"]["step_height"] = 0.10
    config["stairs"]["start_x"] = 0.60
    config["validation"]["success_x"] = 2.60
    config["validation"]["episode_duration"] = float(args.duration)
    config["validation"]["initial_joint_noise"] = float(
        args.initial_joint_noise
    )
    config["validation"]["initial_lateral_noise"] = float(
        args.initial_lateral_noise
    )
    config["cmd_init"][0] = float(args.command_speed)
    physics = config.setdefault("mujoco_physics", {})
    physics.update(PHYSICS_PRESETS["isaac_aligned"])
    physics["preset"] = "isaac_aligned"

    checkpoint = (
        latest_native_checkpoint()
        if args.checkpoint_path == "auto"
        else Path(args.checkpoint_path).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise ValueError(
            "Checkpoint does not exist: " + str(checkpoint)
        )
    policy, _ = load_policy(
        config, checkpoint_path=str(checkpoint)
    )

    def expanded(value):
        return Path(
            str(value).replace(
                "{LEGGED_GYM_ROOT_DIR}", str(ROOT)
            )
        ).expanduser().resolve()

    xml_path = expanded(config["xml_path"])
    urdf_path = expanded(config["urdf_path"])
    model = load_mujoco_model(
        str(xml_path),
        config["stairs"],
        config["mujoco_physics"],
        urdf_path=str(urdf_path),
    )
    _configure_solver(model, config)
    model.vis.global_.offwidth = max(
        int(model.vis.global_.offwidth), int(args.camera_width)
    )
    model.vis.global_.offheight = max(
        int(model.vis.global_.offheight), int(args.camera_height)
    )
    return config, model, policy, checkpoint


def stream(args):
    if not 1 <= args.stream_port <= 65535:
        raise ValueError("--stream_port must be in [1, 65535]")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("Camera dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")
    if args.fps <= 0.0 or args.playback_speed <= 0.0:
        raise ValueError("FPS and playback speed must be positive")

    config, model, policy, checkpoint = load_stream_components(args)
    renderer = mujoco.Renderer(
        model,
        height=int(args.camera_height),
        width=int(args.camera_width),
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([1.45, 0.0, 0.72])
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.camera_azimuth)
    camera.elevation = float(args.camera_elevation)

    frame_store = FrameStore()
    handler = type(
        "N2MujocoStreamHandler",
        (StreamHandler,),
        {
            "frame_store": frame_store,
            "checkpoint_label": checkpoint.name,
        },
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", int(args.stream_port)), handler
    )
    server_thread = threading.Thread(
        target=server.serve_forever, daemon=True
    )
    server_thread.start()
    print(
        "MuJoCo browser stream: http://127.0.0.1:{}/".format(
            args.stream_port
        ),
        flush=True,
    )
    print("Checkpoint: " + str(checkpoint), flush=True)
    print(
        "Forward this port and keep the command running. "
        "Press Ctrl-C to stop only the visualization.",
        flush=True,
    )

    episode = 0
    try:
        while args.episodes <= 0 or episode < args.episodes:
            episode += 1
            callback = LiveRenderer(
                renderer,
                camera,
                frame_store,
                episode,
                args.fps,
                args.playback_speed,
                args.jpeg_quality,
            )
            result = run_episode(
                config,
                model,
                policy,
                args.seed + episode - 1,
                step_callback=callback,
            )
            print(
                "episode {}: {}".format(
                    episode,
                    json.dumps(result, sort_keys=True),
                ),
                flush=True,
            )
            if args.episode_pause > 0.0:
                time.sleep(float(args.episode_pause))
    except KeyboardInterrupt:
        print("Stopping MuJoCo visualization.", flush=True)
    finally:
        frame_store.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="n2_stairs_walk.yaml"
    )
    parser.add_argument("--checkpoint_path", default="auto")
    parser.add_argument("--stream_port", type=int, default=8080)
    parser.add_argument("--camera_width", type=int, default=960)
    parser.add_argument("--camera_height", type=int, default=540)
    parser.add_argument("--camera_distance", type=float, default=3.4)
    parser.add_argument("--camera_azimuth", type=float, default=135.0)
    parser.add_argument("--camera_elevation", type=float, default=-18.0)
    parser.add_argument("--jpeg_quality", type=int, default=82)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--episode_pause", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--command_speed", type=float, default=0.18)
    parser.add_argument("--initial_joint_noise", type=float, default=0.0)
    parser.add_argument("--initial_lateral_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    stream(parser.parse_args())
