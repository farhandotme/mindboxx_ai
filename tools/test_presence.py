"""
tools/test_presence.py -- standalone presence-pipeline smoke test.

Tests ONLY:
    camera + human detection + arrival + departure

Does NOT require:
    Gemini              (no API key, no network call, no Live session)
    the UI               (no PyQt6, no window)
    a visible camera preview  (nothing is ever displayed or saved)

This exists because item 20 of the kiosk spec requires presence detection to
be independently testable: the camera → human-detector → arrival/departure
pipeline is the single most safety/UX-critical part of the "always-on office
kiosk" behaviour (it's what decides whether AUREX ever speaks at all), so it
must be checkable in isolation, without booting the whole app or paying for
a Gemini connection just to find out whether the webcam and the detector are
working.

Usage:
    python tools/test_presence.py
    AUREX_PRESENCE_DEBUG=1 python tools/test_presence.py   # verbose per-frame detail
    python tools/test_presence.py --grace 1.0               # shorter departure grace

Ctrl+C to stop -- the camera is released cleanly on exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running this file directly (`python tools/test_presence.py`) from
# anywhere, by making sure the project root (the parent of this tools/ dir)
# is on sys.path, the same way main.py's package-relative imports expect.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.camera_manager import CameraManager, is_available as camera_is_available
from core.presence import PresenceDetector, is_available as presence_is_available, \
    unavailable_reason as presence_unavailable_reason


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Presence pipeline smoke test (no Gemini, no UI).")
    parser.add_argument("--grace", type=float, default=None,
                         help="Departure grace period in seconds (default: PresenceDetector's own default, ~1.5s).")
    parser.add_argument("--hits", type=int, default=None,
                         help="Consecutive-ish arrival confirm hits inside the sliding window (default: 3).")
    args = parser.parse_args()

    if not camera_is_available():
        _log("CAMERA: unavailable -- OpenCV (cv2) is not installed. Run: pip install opencv-python")
        return 1
    if not presence_is_available():
        _log(f"HUMAN_DETECTOR: unavailable -- {presence_unavailable_reason()}")
        return 1

    camera = CameraManager(logger=_log)
    if not camera.start():
        _log("CAMERA: failed to start.")
        return 1

    # Wait (briefly) for the very first frame so "CAMERA: connected" prints
    # before we start waiting on presence -- purely cosmetic ordering.
    for _ in range(40):   # up to ~4s
        if camera.connected:
            break
        time.sleep(0.1)
    if not camera.connected:
        _log("CAMERA: not connected yet -- still retrying in the background. "
             "(This tool will keep waiting; check your camera permissions/index if this never clears.)")

    arrivals = 0
    departures = 0

    def on_arrived():
        nonlocal arrivals
        arrivals += 1
        _log("HUMAN_PRESENT = true")
        _log(f"ARRIVAL EVENT: triggered (#{arrivals})")

    def on_left():
        nonlocal departures
        departures += 1
        _log("HUMAN_PRESENT = false")
        _log(f"DEPARTURE EVENT: triggered (#{departures})")

    kwargs = {}
    if args.grace is not None:
        kwargs["leave_confirm_seconds"] = args.grace
    if args.hits is not None:
        kwargs["arrive_confirm_hits"] = args.hits

    detector = PresenceDetector(
        camera=camera,
        on_arrived=on_arrived,
        on_left=on_left,
        logger=_log,
        **kwargs,
    )
    if not detector.start():
        _log("HUMAN_DETECTOR: failed to start.")
        camera.stop()
        return 1

    _log(f"HUMAN_PRESENT = {str(detector.present).lower()}")
    _log("Watching for a human in front of the camera. Ctrl+C to stop.")
    _log("(No preview window, nothing saved -- this is detection-only, as required.)")

    try:
        while True:
            time.sleep(1.0)
            s = detector.last_status
            result = detector.get_result()
            _log(
                f"HUMAN_PRESENT = {str(result['human_present']).lower()} "
                f"| confidence={result['confidence']} stage={s.get('stage')} "
                f"brightness={s.get('brightness')} arrivals={arrivals} departures={departures}"
            )
    except KeyboardInterrupt:
        _log("Ctrl+C received -- shutting down.")
    finally:
        detector.stop()
        camera.stop()
        _log("CAMERA: released. Goodbye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
