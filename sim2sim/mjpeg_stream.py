"""Tiny dependency-light MJPEG server for live MuJoCo visualization."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    html,body { margin:0; min-height:100%; background:#111; color:#eee;
      font:14px/1.45 system-ui,sans-serif; }
    main { width:min(1100px,96vw); margin:18px auto; }
    h1 { margin:0 0 10px; font-size:20px; font-weight:600; }
    img { display:block; width:100%; aspect-ratio:16/9; object-fit:contain;
      border-radius:8px; background:#000; box-shadow:0 2px 18px #000; }
    #status { margin-top:10px; padding:9px 12px; border-radius:6px;
      background:#1d1d1d; white-space:pre-wrap; }
    #controls { display:flex; align-items:center; gap:12px; margin:10px 0; }
    #start { appearance:none; border:0; border-radius:7px; padding:10px 18px;
      color:#fff; background:#1677ff; font:600 15px system-ui,sans-serif;
      cursor:pointer; }
    #start:hover:not(:disabled) { background:#4096ff; }
    #start:disabled { cursor:default; background:#3b4b61; color:#b8c2d0; }
    #gate { color:#bbb; }
  </style>
</head>
<body><main>
  <h1>__HEADING__</h1>
  <div id="controls">
    <button id="start" type="button">开始仿真</button>
    <span id="gate">待机中；准备好录屏后再开始。</span>
  </div>
  <img src="/stream.mjpg" alt="等待仿真画面">
  <div id="status">服务已就绪，等待浏览器画面连接和手动开始……</div>
</main>
<script>
const controls = document.getElementById('controls');
const startButton = document.getElementById('start');
const gate = document.getElementById('gate');
const statusBox = document.getElementById('status');

async function requestStart() {
  startButton.disabled = true;
  gate.textContent = '正在发送开始指令……';
  try {
    const r = await fetch('/start', {method:'POST', cache:'no-store'});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await updateStatus();
  } catch (error) {
    startButton.disabled = false;
    gate.textContent = `开始失败：${error}`;
  }
}
startButton.addEventListener('click', requestStart);

async function updateStatus() {
  try {
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();
    const manual = Boolean(s.manual_start);
    const started = Boolean(s.started);
    const clients = Number(s.clients || 0);
    controls.hidden = !manual;
    startButton.disabled = started;
    startButton.textContent = started ? '仿真已开始' : '开始仿真';
    gate.textContent = started
      ? '运行中；关闭本地 SSH 终端可同时停止服务。'
      : (clients > 0
          ? '待机中；准备好录屏后点击开始。'
          : '正在建立画面连接；可先准备录屏，再点击开始。');
    if (manual && !started) {
      statusBox.textContent =
        `待命中  |  stream clients ${clients}  |  ` +
        '请点击“开始仿真”，或在服务器运行双端 start helper。';
      return;
    }
    const campaign = Number(s.stage_count || 0) > 0
      ? `\nrow ${s.row}  col ${s.col}  type ${s.terrain_type}  |  ` +
        `stage ${s.stage}/${s.stage_count}  goal ${s.goal_index}/${Math.max(Number(s.goal_count || 1) - 1, 0)}  |  ` +
        `attempt ${s.attempt}  clears ${s.stage_clears}  cycles ${s.cycles}  failures ${s.failures}`
      : '';
    statusBox.textContent =
      `sim ${Number(s.sim_time || 0).toFixed(2)} s  |  ` +
      `x ${Number(s.x || 0).toFixed(2)} m  z ${Number(s.z || 0).toFixed(2)} m  |  ` +
      `goal ${s.goal || '0/0'}  |  stream ${Number(s.stream_fps || 0).toFixed(1)} FPS` +
      campaign;
  } catch (_) {}
}
setInterval(updateStatus, 500); updateStatus();
</script></body></html>"""


class MJPEGStreamer:
    """Serve the latest RGB frame as an MJPEG stream."""

    def __init__(
            self, host="127.0.0.1", port=8000, jpeg_quality=80,
            page_title="N2 MuJoCo Live", heading="N2 MuJoCo 实时画面",
            manual_start=False):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.manual_start = bool(manual_start)
        self.index_html = (
            _INDEX_HTML
            .replace("__PAGE_TITLE__", str(page_title))
            .replace("__HEADING__", str(heading)))
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._clients = 0
        self._stopped = False
        self._started = not self.manual_start
        self._status = {}
        self._server = None
        self._thread = None
        self._last_publish_time = None
        self._measured_fps = 0.0

    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload, status=200):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    body = streamer.index_html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/status":
                    self._send_json(streamer.status())
                    return
                if path == "/start":
                    accepted = streamer.request_start()
                    payload = streamer.status()
                    payload["accepted"] = accepted
                    self._send_json(payload, status=200 if accepted else 410)
                    return
                if path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last_sequence = -1
                    with streamer._condition:
                        streamer._clients += 1
                        streamer._condition.notify_all()
                    try:
                        while True:
                            with streamer._condition:
                                streamer._condition.wait_for(
                                    lambda: streamer._stopped
                                    or streamer._sequence != last_sequence)
                                if streamer._stopped:
                                    return
                                frame = streamer._frame
                                last_sequence = streamer._sequence
                            if frame is None:
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        with streamer._condition:
                            streamer._clients = max(
                                0, streamer._clients - 1)
                    return
                self.send_error(404)

            def do_POST(self):
                path = urlsplit(self.path).path
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length > 0:
                    self.rfile.read(content_length)
                if path == "/start":
                    accepted = streamer.request_start()
                    payload = streamer.status()
                    payload["accepted"] = accepted
                    self._send_json(payload, status=200 if accepted else 410)
                    return
                self.send_error(404)

            def log_message(self, _format, *_args):
                return

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server((self.host, self.port), Handler)
        # Port zero asks the OS for an ephemeral port, which is useful for
        # tests and programmatic embedding.
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mujoco-mjpeg-http", daemon=True)
        self._thread.start()

    def wait_for_client(self):
        """Wait for an MJPEG client exactly as before manual start existed."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopped or self._clients > 0)
            return not self._stopped

    def wait_for_start(self, timeout=None):
        """Wait until ``request_start``/HTTP ``/start`` opens the gate.

        Returns ``False`` when the streamer is closed first or when an
        optional timeout expires. Legacy instances (``manual_start=False``)
        begin with the gate open and therefore return immediately.
        """
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._stopped or self._started, timeout=timeout)
            return bool(ready and self._started and not self._stopped)

    def request_start(self):
        """Open the start gate once; safe to call from any HTTP/thread."""
        with self._condition:
            if self._stopped:
                return False
            self._started = True
            self._condition.notify_all()
            return True

    def status(self):
        """Return a coherent status snapshot, including before the first frame."""
        with self._condition:
            payload = dict(self._status)
            payload["stream_fps"] = self._measured_fps
            payload["manual_start"] = self.manual_start
            payload["started"] = self._started
            payload["waiting_for_start"] = bool(
                self.manual_start and not self._started and not self._stopped)
            payload["clients"] = self._clients
            payload["stopped"] = self._stopped
            return payload

    def publish(self, rgb_frame, status):
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("failed to encode MuJoCo frame as JPEG")
        now = time.monotonic()
        with self._condition:
            if self._last_publish_time is not None:
                instant_fps = 1.0 / max(
                    now - self._last_publish_time, 1e-6)
                self._measured_fps = (
                    0.9 * self._measured_fps + 0.1 * instant_fps
                    if self._measured_fps else instant_fps)
            self._last_publish_time = now
            self._frame = encoded.tobytes()
            self._status = dict(status)
            self._sequence += 1
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
