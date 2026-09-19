"""
The ONE owner of the physical webcam for the whole application.

Every other subsystem that needs a camera frame -- presence detection, the
on-demand "look at me" vision tool -- reads from THIS manager's cached
latest frame instead of opening its own cv2.VideoCapture. That is the fix
for two long-standing problems in earlier iterations of this feature:
"camera is not opening all the time" (opening/closing the device on every
poll is unreliable on several webcam drivers) and "two components request
camera access" (two independent VideoCapture handles on the same physical
index fight each other on most backends).

Responsibilities (and nothing more):
  * Open the camera ONCE, in a dedicated background thread.
  * Keep reading frames into an in-memory "latest frame" slot. Nothing is
    ever written to disk, displayed in a window, or streamed anywhere by
    this module -- it is a frame *source*, not a viewer.
  * Reconnect automatically if the camera disconnects or reads start
    failing (unplugged, OS suspended it, driver hiccup, etc).
  * Release the device only when `stop()` is called at application shutdown.

Nothing here is Gemini-, presence-, or UI-specific on purpose: this module
doesn't know or care who's asking for a frame.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

RECONNECT_BACKOFF = 3.0    # seconds between reconnect attempts after the device is lost
READ_FAIL_LIMIT   = 8      # consecutive read failures before we assume the device is gone
CAPTURE_FPS_CAP   = 12.0   # a presence sensor + snapshot source, not a video call -- no need for more


def is_available() -> bool:
    """True if OpenCV is importable. Doesn't touch the camera itself."""
    return _CV2


class CameraManager:
    """Single background reader for the physical webcam."""

    def __init__(self, logger: Callable[[str], None] = print):
        self._logger          = logger
        self._thread: threading.Thread | None = None
        self._running         = False
        self._cap             = None
        self._frame_lock       = threading.Lock()
        self._latest_frame     = None   # last successfully read BGR numpy frame
        self._latest_frame_ts  = 0.0
        self._connected        = False
        self._stop_evt         = threading.Event()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Idempotent -- safe to call more than once."""
        if self._running:
            return True
        if not _CV2:
            self._logger("Camera: OpenCV not available -- camera disabled.")
            return False
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CameraManagerThread")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Only meant to be called once, at application shutdown. Signals the
        reader thread and waits for it, so the device is released by the same
        thread that reads it (releasing a VideoCapture while another thread is
        inside read() can hard-crash on some Windows drivers)."""
        self._running = False
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._release()

    def get_latest_frame(self, max_age: float = 2.0):
        """A COPY of the most recent frame, or None if there isn't one yet
        or it's older than max_age seconds (camera likely stalled/reconnecting)."""
        with self._frame_lock:
            frame, ts = self._latest_frame, self._latest_frame_ts
        if frame is None:
            return None
        if max_age and (time.monotonic() - ts) > max_age:
            return None
        return frame.copy()

    def get_snapshot_jpeg(self, max_age: float = 2.0) -> Optional[bytes]:
        """A single JPEG-encoded frame, ready to hand to a vision model.
        None if no sufficiently fresh frame is available."""
        frame = self.get_latest_frame(max_age=max_age)
        if frame is None or not _CV2:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        return buf.tobytes()

    # -- internals -------------------------------------------------------------

    def _open(self):
        from actions.screen_processor import _get_camera_index, _cv2_backend
        index   = _get_camera_index()
        backend = _cv2_backend()
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always hand out the NEWEST frame
        except Exception:
            pass
        for _ in range(8):   # let auto-exposure/focus settle once, right after opening
            cap.read()
        return cap

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._connected = False

    def _loop(self) -> None:
        fail_streak  = 0
        warned       = False
        min_interval = 1.0 / CAPTURE_FPS_CAP

        while self._running:
            start = time.monotonic()
            try:
                if self._cap is None:
                    self._cap = self._open()
                    if self._cap is None:
                        if not warned:
                            warned = True
                            self._logger("Camera: not available yet -- retrying quietly in the background.")
                        if self._stop_evt.wait(RECONNECT_BACKOFF):
                            break
                        continue
                    self._connected = True
                    warned = False
                    self._logger("Camera: connected.")

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail_streak += 1
                    if fail_streak >= READ_FAIL_LIMIT:
                        self._logger("Camera: lost connection -- reconnecting.")
                        self._release()
                        fail_streak = 0
                        if self._stop_evt.wait(RECONNECT_BACKOFF):
                            break
                    else:
                        time.sleep(0.03)
                    continue

                fail_streak = 0
                with self._frame_lock:
                    self._latest_frame    = frame
                    self._latest_frame_ts = time.monotonic()
            except Exception as e:
                self._logger(f"Camera: loop error -- {e}")
                self._release()
                if self._stop_evt.wait(RECONNECT_BACKOFF):
                    break

            elapsed = time.monotonic() - start
            time.sleep(max(0.0, min_interval - elapsed))

        # Reader thread owns the device: it is the one that lets go of it.
        self._release()
