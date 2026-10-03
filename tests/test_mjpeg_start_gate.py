import io
import json
import threading
import unittest
from unittest import mock

import sim2sim.mjpeg_stream as mjpeg_module
from sim2sim.mjpeg_stream import MJPEGStreamer


class FakeHTTPServer:
    """Capture the generated handler without opening a real socket."""

    def __init__(self, _address, handler_class):
        self.server_address = ("127.0.0.1", 43123)
        self.handler_class = handler_class

    def serve_forever(self):
        return

    def shutdown(self):
        return

    def server_close(self):
        return


class MJPEGStartGateTest(unittest.TestCase):

    def setUp(self):
        self._streamers = []

    def tearDown(self):
        for streamer in reversed(self._streamers):
            streamer.close()

    def streamer(self, **kwargs):
        streamer = MJPEGStreamer(host="127.0.0.1", port=0, **kwargs)
        self._streamers.append(streamer)
        return streamer

    def start(self, streamer):
        with mock.patch.object(
                mjpeg_module, "ThreadingHTTPServer", FakeHTTPServer):
            streamer.start()

    def request(self, streamer, path, method="GET"):
        handler = object.__new__(streamer._server.handler_class)
        handler.path = path
        handler.headers = {"Content-Length": "0"}
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        response = {"status": None, "headers": {}}
        handler.send_response = lambda status: response.update(status=status)
        handler.send_header = (
            lambda name, value: response["headers"].update({name: value}))
        handler.end_headers = lambda: None
        handler.send_error = (
            lambda status: response.update(status=status))

        getattr(handler, "do_{}".format(method))()
        return response["status"], response["headers"], handler.wfile.getvalue()

    def json_request(self, streamer, path, method="GET"):
        status, headers, body = self.request(streamer, path, method)
        self.assertEqual(headers["Content-Type"], "application/json")
        return status, json.loads(body.decode("utf-8"))

    def test_legacy_gate_is_open_and_wait_for_client_is_unchanged(self):
        streamer = self.streamer()
        self.assertTrue(streamer.wait_for_start(timeout=0.01))

        result = []
        waiter = threading.Thread(
            target=lambda: result.append(streamer.wait_for_client()))
        waiter.start()
        with streamer._condition:
            streamer._clients = 1
            streamer._condition.notify_all()
        waiter.join(timeout=1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [True])
        self.assertTrue(streamer.status()["started"])

    def test_manual_gate_blocks_then_request_start_releases_it(self):
        streamer = self.streamer(manual_start=True)
        entered = threading.Event()
        result = []

        def wait():
            entered.set()
            result.append(streamer.wait_for_start(timeout=2))

        waiter = threading.Thread(target=wait)
        waiter.start()
        self.assertTrue(entered.wait(timeout=1))
        self.assertTrue(waiter.is_alive())

        self.assertTrue(streamer.request_start())
        self.assertTrue(streamer.request_start())  # idempotent
        waiter.join(timeout=1)

        self.assertEqual(result, [True])
        status = streamer.status()
        self.assertTrue(status["started"])
        self.assertFalse(status["waiting_for_start"])

    def test_close_releases_a_waiter_with_false(self):
        streamer = self.streamer(manual_start=True)
        result = []
        waiter = threading.Thread(
            target=lambda: result.append(streamer.wait_for_start(timeout=2)))
        waiter.start()
        streamer.close()
        waiter.join(timeout=1)

        self.assertEqual(result, [False])
        self.assertFalse(streamer.request_start())

    def test_initial_status_is_explicit_before_any_frame(self):
        streamer = self.streamer(manual_start=True)
        self.start(streamer)

        status_code, payload = self.json_request(streamer, "/status")

        self.assertEqual(status_code, 200)
        self.assertTrue(payload["manual_start"])
        self.assertFalse(payload["started"])
        self.assertTrue(payload["waiting_for_start"])
        self.assertEqual(payload["clients"], 0)
        self.assertEqual(payload["stream_fps"], 0.0)
        self.assertFalse(payload["stopped"])

    def test_get_and_post_start_are_supported_and_idempotent(self):
        for method in ("GET", "POST"):
            with self.subTest(method=method):
                streamer = self.streamer(manual_start=True)
                self.start(streamer)

                status_code, payload = self.json_request(
                    streamer, "/start", method=method)
                second_code, second = self.json_request(
                    streamer, "/start", method=method)

                self.assertEqual(status_code, 200)
                self.assertTrue(payload["accepted"])
                self.assertTrue(payload["started"])
                self.assertFalse(payload["waiting_for_start"])
                self.assertEqual(second_code, 200)
                self.assertTrue(second["started"])

    def test_ready_page_has_start_button_and_fixed_16_by_9_canvas(self):
        streamer = self.streamer(manual_start=True)
        self.start(streamer)

        status, headers, body = self.request(streamer, "/")
        page = body.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn('id="start"', page)
        self.assertIn("开始仿真", page)
        self.assertIn("待机", page)
        self.assertIn("aspect-ratio:16/9", page)
        self.assertIn("fetch('/start', {method:'POST'", page)


if __name__ == "__main__":
    unittest.main()
