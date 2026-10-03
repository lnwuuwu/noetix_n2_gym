import unittest

from sim2sim.stream_gate import (
    establish_stream_start_wall,
    wait_for_stream_start,
)


class FakeStreamer:

    def __init__(self, client=True, start=True):
        self.client = client
        self.start = start
        self.calls = []

    def wait_for_client(self):
        self.calls.append("client")
        return self.client

    def wait_for_start(self):
        self.calls.append("start")
        return self.start


class StreamGateTest(unittest.TestCase):

    def test_manual_start_waits_for_client_then_start_gate(self):
        streamer = FakeStreamer()
        self.assertTrue(wait_for_stream_start(streamer, manual_start=True))
        self.assertEqual(streamer.calls, ["client", "start"])

    def test_manual_start_never_reaches_gate_without_client(self):
        streamer = FakeStreamer(client=False)
        self.assertFalse(wait_for_stream_start(streamer, manual_start=True))
        self.assertEqual(streamer.calls, ["client"])

    def test_manual_start_propagates_closed_start_gate(self):
        streamer = FakeStreamer(start=False)
        self.assertFalse(wait_for_stream_start(streamer, manual_start=True))
        self.assertEqual(streamer.calls, ["client", "start"])

    def test_legacy_browser_gate_remains_compatible(self):
        streamer = FakeStreamer()
        self.assertTrue(wait_for_stream_start(streamer, manual_start=False))
        self.assertEqual(streamer.calls, ["client"])

    def test_pacing_epoch_is_created_only_after_manual_start(self):
        streamer = FakeStreamer()

        def clock():
            streamer.calls.append("clock")
            return 123.5

        wall_start = establish_stream_start_wall(
            streamer, manual_start=True, clock=clock)
        self.assertEqual(wall_start, 123.5)
        self.assertEqual(streamer.calls, ["client", "start", "clock"])

    def test_closed_gate_does_not_create_pacing_epoch(self):
        streamer = FakeStreamer(start=False)

        def clock():
            streamer.calls.append("clock")
            return 123.5

        wall_start = establish_stream_start_wall(
            streamer, manual_start=True, clock=clock)
        self.assertIsNone(wall_start)
        self.assertEqual(streamer.calls, ["client", "start"])


if __name__ == "__main__":
    unittest.main()
