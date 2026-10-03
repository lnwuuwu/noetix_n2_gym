"""Shared start policy for browser-backed simulation streams."""

import time


def wait_for_stream_start(streamer, manual_start=False):
    """Wait for the browser and, optionally, its explicit start signal.

    The caller deliberately creates its real-time pacing anchor only after
    this function returns.  Time spent looking at the ready page can therefore
    never make the simulator race to catch up.
    """
    if manual_start:
        print("[stream] 等待浏览器连接……")
    else:
        print("[stream] 等待浏览器连接；连接后才开始仿真……")
    if not streamer.wait_for_client():
        return False

    if manual_start:
        print("[stream] 浏览器已连接；请点击网页上的“开始仿真”。")
        if not streamer.wait_for_start():
            return False
        print("[stream] 已收到手动开始信号，开始实时仿真。")
    else:
        print("[stream] 浏览器已连接，开始实时仿真。")
    return True


def establish_stream_start_wall(
        streamer, manual_start=False, clock=time.monotonic):
    """Open all selected gates, then establish the real-time pacing epoch."""
    if not wait_for_stream_start(streamer, manual_start=manual_start):
        return None
    return clock()
