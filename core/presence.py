"""
Ambient presence detection -- "is there a HUMAN in front of the camera?"

This module owns NO camera hardware. It only reads the frame that
core.camera_manager.CameraManager already has cached, so it can never fight
another component for the webcam.

  * CameraManager answers  "what does the camera currently see?"
  * PresenceDetector answers "is a human actually here?"       <- this file
  * IdentityManager answers  "who is this?"

DETECTION ENGINE -- MediaPipe Tasks API (pose + face)
----------------------------------------------------------------------------
  * POSE  (PoseLandmarker) -- the PRIMARY signal. BlazePose tracks body
    landmarks (shoulders, nose, ears, eyes, ...) and reports a per-landmark
    visibility score. Because it keys off body/shoulder structure rather
    than frontal facial features, it keeps working when someone:
      - faces the camera, looks sideways, or turns partially away
      - is close to the laptop (only the upper body/shoulders in frame)
      - is farther away
      - moves naturally (polled every ~0.3s, re-detected each time)
    Confidence = the best-supported group of landmarks in that frame (see
    `_pose_confidence`): both shoulders visible, OR the head/face landmarks
    (nose/eyes/ears) visible when someone is close enough that shoulders
    fall outside the frame.

  * FACE DETECTION (FaceDetector) -- a SECONDARY signal, only run when the
    pose signal is weak or absent that frame (saves CPU, and is "useful"
    exactly when pose alone isn't conclusive: e.g. a face-filling close-up
    where the pose model doesn't get a confident lock).

  * Neither model cares about glasses -- BlazePose landmarks and the face
    detector are both robust to eyewear.

  * Sliding-window arrival (>= N hits in the last M polls, not N-in-a-row)
    plus a short departure grace period (default 1.5 s, env-overridable)
    smooths out single bad frames without adding a long lag on departure.

  * Logs "PRESENCE: analyzing" / "PRESENCE: human candidate" /
    "PRESENCE: human confirmed" / "PRESENCE: human left".

IMPORTANT -- why this isn't `mp.solutions.pose` / `mp.solutions.face_detection`
----------------------------------------------------------------------------
Those "legacy solutions" APIs are what most MediaPipe examples online still
show, but Google removed them from the `mediapipe` pip package entirely as
of the 0.10.3x / 1.0.x releases (deprecated in 0.10.0, gone by 0.11+ --
`import mediapipe as mp; mp.solutions.pose` now raises
`AttributeError: module 'mediapipe' has no attribute 'solutions'`). The
supported replacement, used here, is the **Tasks API**
(`mediapipe.tasks.python.vision.PoseLandmarker` /
`mediapipe.tasks.python.vision.FaceDetector`).

The one real consequence: Tasks-API models are NOT bundled inside the pip
wheel (the old solutions models were). `PresenceDetector.start()` downloads
two small model files the first time it runs and caches them under
`models/mediapipe/` at the project root (`pose_landmarker_lite.task`, a few
MB, and `blaze_face_short_range.tflite`, a few hundred KB) -- every run
after that is fully offline, same as before. This is a one-time model
fetch from Google's public model store, NOT sending camera frames
anywhere -- presence decisions are still made 100% locally, never by
Gemini. On a machine with no internet access at all, pre-download the two
files yourself and point `AUREX_POSE_MODEL_PATH` / `AUREX_FACE_MODEL_PATH`
at them (see `_DEFAULT_POSE_MODEL_URL` / `_DEFAULT_FACE_MODEL_URL` below
for where to get them from another machine).

Set AUREX_PRESENCE_DEBUG=1 to print what the detector sees every ~2 seconds
(confidence, stage, brightness, hit ratio, etc).

PUBLIC API is unchanged from earlier versions of this file on purpose, so
nothing outside this file (main.py's kiosk wiring, tools/test_presence.py)
needs to change: `PresenceDetector(camera, on_arrived, on_left, logger,
...)`, `.start()`, `.stop()`, `.ready`, `.present`, `.last_status`,
`.get_result()` -> `{human_present, confidence, timestamp}`.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Optional

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

_MEDIAPIPE_IMPORT_ERROR: Optional[str] = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks_python
    from mediapipe.tasks.python import vision as _mp_tasks_vision
    _MEDIAPIPE = True
except Exception as e:
    # ImportError normally, but a broken/mismatched install can raise
    # OSError (DLL load failed on Windows) or other exceptions instead --
    # catch broadly so we can report the REAL reason, not a generic
    # "unavailable".
    _MEDIAPIPE = False
    mp = None
    _mp_tasks_python = None
    _mp_tasks_vision = None
    _MEDIAPIPE_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# -- Tunables -----------------------------------------------------------------
DETECT_WIDTH                    = 480    # frame is downscaled to this width before detection
DEFAULT_ARRIVE_CONFIRM_HITS     = 3      # hits needed inside the sliding window to count as "arrived"
ARRIVE_WINDOW                   = 6      # ... the last N polls (~2 s at POLL_INTERVAL=0.3)
# "a short configurable departure grace period, for example 1-2 seconds" --
# NOT a long timeout, which is why AUREX used to keep listening to an empty
# room for many seconds after someone walked away.
DEFAULT_LEAVE_CONFIRM_SECONDS   = float(os.environ.get("AUREX_LEAVE_GRACE_SECONDS", "1.5"))

# How confident the pose/face signal needs to be, per-model, before it's
# even considered for this frame's combined score.
DEFAULT_POSE_MIN_CONFIDENCE     = float(os.environ.get("AUREX_POSE_MIN_CONF", "0.40"))
DEFAULT_FACE_MIN_CONFIDENCE     = float(os.environ.get("AUREX_FACE_MIN_CONF", "0.50"))
# The combined (pose + face) confidence needed for a single frame to count
# as "a human is in this frame" (a "hit" that feeds the sliding window).
DEFAULT_PRESENCE_THRESHOLD      = float(os.environ.get("AUREX_PRESENCE_THRESHOLD", "0.40"))
# Only bother running the (heavier) face detector when the pose signal this
# frame is below this -- i.e. exactly when the secondary signal is "useful".
FACE_STAGE_POSE_CUTOFF          = 0.60

POLL_INTERVAL                   = 0.30   # seconds between looks at the camera's latest frame
DARK_FRAME_MEAN                 = 10.0   # mean pixel value below this = black image
ANALYZING_LOG_INTERVAL          = 4.0    # how often "PRESENCE: analyzing" is logged while idle
DEBUG = os.environ.get("AUREX_PRESENCE_DEBUG", "").strip() not in ("", "0")

# BlazePose landmark indices we actually care about (same 33-point map the
# Tasks-API PoseLandmarker uses).
_NOSE        = 0
_L_EYE       = 2
_R_EYE       = 5
_L_EAR       = 7
_R_EAR       = 8
_L_SHOULDER  = 11
_R_SHOULDER  = 12

# -- Model files ----------------------------------------------------------
# Not bundled in the pip package (see module docstring) -- downloaded once
# and cached here. "lite"/"short_range" are the fast variants; plenty for a
# coarse presence signal at a ~3 fps poll rate.
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_MODELS_DIR     = Path(os.environ.get("AUREX_MODELS_DIR", str(_PROJECT_ROOT / "models" / "mediapipe")))
_DEFAULT_POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
_DEFAULT_FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
_POSE_MODEL_PATH = Path(os.environ.get("AUREX_POSE_MODEL_PATH", str(_MODELS_DIR / "pose_landmarker_lite.task")))
_FACE_MODEL_PATH = Path(os.environ.get("AUREX_FACE_MODEL_PATH", str(_MODELS_DIR / "blaze_face_short_range.tflite")))
_DOWNLOAD_TIMEOUT_SECONDS = 20


def is_available() -> bool:
    """True if OpenCV and the MediaPipe Tasks vision API are importable.
    Doesn't touch the camera, download anything, or load a model -- that's
    cheap on purpose, since main.py calls this at startup just to decide
    whether the feature can be offered at all. Model download/loading
    happens lazily in start()."""
    if not (_CV2 and _MEDIAPIPE):
        return False
    try:
        return (hasattr(_mp_tasks_vision, "PoseLandmarker")
                and hasattr(_mp_tasks_vision, "FaceDetector"))
    except Exception:
        return False


def unavailable_reason() -> str:
    """Human-readable reason `is_available()` returned False, for logging/
    diagnostics -- callers should never have to guess why detection is off."""
    if not _CV2:
        return "OpenCV (cv2) is not installed. Run: pip install opencv-python"
    if not _MEDIAPIPE:
        return (f"MediaPipe failed to import ({_MEDIAPIPE_IMPORT_ERROR}). "
                f"Run: pip install --upgrade mediapipe")
    try:
        ok = (hasattr(_mp_tasks_vision, "PoseLandmarker")
              and hasattr(_mp_tasks_vision, "FaceDetector"))
    except Exception as e:
        return (f"MediaPipe imported but checking its Tasks API raised "
                f"{type(e).__name__}: {e}. Try: pip install --upgrade --force-reinstall mediapipe")
    if not ok:
        ver = getattr(mp, "__version__", "unknown") if mp else "unknown"
        return (f"MediaPipe {ver} is installed but mediapipe.tasks.python.vision.PoseLandmarker/"
                f"FaceDetector aren't present. Try: pip install --upgrade mediapipe")
    return "unknown"


def _ensure_model(path: Path, url: str, label: str, logger: Callable[[str], None]) -> bool:
    """Make sure `path` exists, downloading it from `url` if not. Cached
    permanently once fetched -- every run after the first is offline."""
    if path.exists() and path.stat().st_size > 0:
        return True
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger(f"Presence: fetching the {label} model (one-time, cached at {path})...")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.replace(path)
        logger(f"Presence: {label} model ready.")
        return True
    except Exception as e:
        logger(f"Presence: could not fetch the {label} model -- {type(e).__name__}: {e}. "
              f"No internet on this machine? Download it manually from {url} and save it to "
              f"{path}, or point AUREX_{label.upper()}_MODEL_PATH at an existing copy.")
        try:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _pose_confidence(landmarks) -> float:
    """Best-supported "a human's upper body/head is here" signal from one
    BlazePose landmark set, in [0, 1].

    Two independent ways a real person shows up, either is enough on its
    own (this is what makes the detector NOT require facing the camera):
      1. Both shoulders visible -- true whether the person is facing the
         camera, sideways-on, or partly turned away; shoulder geometry
         doesn't depend on facing direction.
      2. Head landmarks (nose / eyes / ears) visible -- covers the "close
         to the laptop, only head/upper-upper-body in frame, shoulders
         cropped out" case.
    """
    def vis(i: int) -> float:
        try:
            return float(landmarks[i].visibility)
        except Exception:
            return 0.0

    l_sh, r_sh = vis(_L_SHOULDER), vis(_R_SHOULDER)
    # Require BOTH shoulders to have some visibility before trusting their
    # average -- one confidently-visible shoulder plus one occluded one
    # shouldn't average up to a false high score.
    shoulder_score = (l_sh + r_sh) / 2.0 if (l_sh > 0.15 and r_sh > 0.15) else 0.0

    nose_score = vis(_NOSE)
    ear_score  = (vis(_L_EAR) + vis(_R_EAR)) / 2.0
    eye_score  = (vis(_L_EYE) + vis(_R_EYE)) / 2.0
    head_score = max(nose_score, ear_score, eye_score)

    return max(shoulder_score, head_score)


class PresenceDetector:
    """Polls CameraManager's latest frame in its own thread and fires
    on_arrived() / on_left() when a human steps up to / away from the camera."""

    def __init__(self,
                 camera,                              # a core.camera_manager.CameraManager
                 on_arrived: Callable[[], None],
                 on_left: Callable[[], None],
                 logger: Callable[[str], None] = print,
                 arrive_confirm_hits: int = DEFAULT_ARRIVE_CONFIRM_HITS,
                 leave_confirm_seconds: float = DEFAULT_LEAVE_CONFIRM_SECONDS,
                 pose_min_confidence: float = DEFAULT_POSE_MIN_CONFIDENCE,
                 face_min_confidence: float = DEFAULT_FACE_MIN_CONFIDENCE,
                 presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD):
        self._camera     = camera
        self._on_arrived = on_arrived
        self._on_left    = on_left
        self._logger     = logger

        self.arrive_confirm_hits   = arrive_confirm_hits
        self.leave_confirm_seconds = leave_confirm_seconds
        self.pose_min_confidence   = pose_min_confidence
        self.face_min_confidence   = face_min_confidence
        self.presence_threshold    = presence_threshold

        self._thread: threading.Thread | None = None
        self._running  = False
        self._ready    = False
        self._present  = False        # our current belief: is a human here?
        self._pose = None             # mediapipe.tasks.python.vision.PoseLandmarker instance
        self._face = None             # mediapipe.tasks.python.vision.FaceDetector instance
        self._lock = threading.Lock()  # guards last_status (read from other threads)
        self.last_status: dict = {
            "human": False, "confidence": 0.0, "stage": None,
            "brightness": None, "timestamp": time.time(),
        }

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def present(self) -> bool:
        return self._present

    def get_result(self) -> dict:
        """The spec-required output shape:
            {human_present: bool, confidence: float, timestamp: float}
        `human_present` is the debounced/smoothed belief (self._present) --
        i.e. what the rest of AUREX should act on -- `confidence` is the
        most recent single-frame score that fed it."""
        with self._lock:
            s = dict(self.last_status)
        return {
            "human_present": bool(self._present),
            "confidence": float(s.get("confidence") or 0.0),
            "timestamp": float(s.get("timestamp") or time.time()),
        }

    def start(self) -> bool:
        """Ensure the MediaPipe Task models are on disk (downloading them
        the first time, see module docstring), load them, and spawn the
        polling thread. Idempotent."""
        if self._running:
            return True
        if not _CV2:
            self._logger("Presence: OpenCV not available -- feature disabled.")
            return False
        if not _MEDIAPIPE:
            self._logger(f"Presence: {unavailable_reason()} -- feature disabled.")
            return False

        have_pose = _ensure_model(_POSE_MODEL_PATH, _DEFAULT_POSE_MODEL_URL, "pose", self._logger)
        have_face = _ensure_model(_FACE_MODEL_PATH, _DEFAULT_FACE_MODEL_URL, "face", self._logger)

        if have_pose:
            try:
                pose_options = _mp_tasks_vision.PoseLandmarkerOptions(
                    base_options=_mp_tasks_python.BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
                    running_mode=_mp_tasks_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                )
                self._pose = _mp_tasks_vision.PoseLandmarker.create_from_options(pose_options)
            except Exception as e:
                self._logger(f"Presence: could not load the pose model -- {e}")
                self._pose = None
        if have_face:
            try:
                face_options = _mp_tasks_vision.FaceDetectorOptions(
                    base_options=_mp_tasks_python.BaseOptions(model_asset_path=str(_FACE_MODEL_PATH)),
                    running_mode=_mp_tasks_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._face = _mp_tasks_vision.FaceDetector.create_from_options(face_options)
            except Exception as e:
                self._logger(f"Presence: could not load the face model -- {e}")
                self._face = None

        if self._pose is None and self._face is None:
            self._logger("Presence: no MediaPipe model could be loaded -- feature disabled.")
            return False

        stages = []
        if self._pose is not None: stages.append("pose")
        if self._face is not None: stages.append("face")
        self._logger(f"Presence: detector stages ready -- {', '.join(stages)}.")

        self._running = True
        self._ready   = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="PresenceThread")
        self._thread.start()
        self._logger("Presence: watching for a human in front of the camera.")
        return True

    def stop(self) -> None:
        self._running = False
        self._ready   = False
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        # MediaPipe Tasks hold native resources -- release them once the
        # polling thread (the only thing that touches them) has stopped.
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None
        if self._face is not None:
            try:
                self._face.close()
            except Exception:
                pass
            self._face = None

    # -- detection ----------------------------------------------------------

    def _detect(self, mp_image) -> tuple[float, Optional[str]]:
        """Return (confidence, stage) for this frame, combining the pose
        (primary) and face-detection (secondary) signals. `stage` records
        which signal(s) actually contributed, for logging/diagnostics."""
        pose_conf = 0.0
        if self._pose is not None:
            try:
                result = self._pose.detect(mp_image)
                if result and result.pose_landmarks:
                    pose_conf = _pose_confidence(result.pose_landmarks[0])
            except Exception:
                pass

        # Face detection is the SECONDARY signal: only worth the extra CPU
        # when pose alone isn't already a confident "yes" this frame, or
        # when there's no pose model at all to rely on.
        face_conf = 0.0
        if self._face is not None and pose_conf < FACE_STAGE_POSE_CUTOFF:
            try:
                result = self._face.detect(mp_image)
                if result and result.detections:
                    face_conf = max(
                        (d.categories[0].score if d.categories else 0.0) for d in result.detections
                    )
            except Exception:
                pass

        pose_hit = pose_conf >= self.pose_min_confidence
        face_hit = face_conf >= self.face_min_confidence

        if pose_hit and face_hit:
            return min(1.0, 0.65 * pose_conf + 0.35 * face_conf), "pose+face"
        if pose_hit:
            return pose_conf, "pose"
        if face_hit:
            return face_conf, "face"
        # Neither crossed its own bar -- report the best raw score anyway
        # (useful for debug logging) but it won't count as a "hit".
        return max(pose_conf, face_conf), None

    def _prepare(self, frame):
        h, w = frame.shape[:2]
        if w > DETECT_WIDTH:
            scale = DETECT_WIDTH / float(w)
            small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(h * scale))))
        else:
            small = frame
        brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return mp_image, brightness

    def _set_present(self, value: bool) -> None:
        if value == self._present:
            return
        self._present = value
        if value:
            self._logger("PRESENCE: human confirmed")
            cb = self._on_arrived
        else:
            self._logger("PRESENCE: human left")
            cb = self._on_left
        try:
            cb()
        except Exception as e:
            self._logger(f"Presence: callback error -- {e}")

    def _loop(self) -> None:
        hits             = deque(maxlen=ARRIVE_WINDOW)
        last_seen_ts     = 0.0
        no_camera_warned = False
        dark_warned      = False
        last_debug       = 0.0
        last_analyzing   = 0.0

        while self._running:
            start = time.monotonic()
            try:
                frame = self._camera.get_latest_frame(max_age=2.0) if self._camera else None

                if frame is None:
                    if not no_camera_warned:
                        no_camera_warned = True
                        self._logger("Presence: no camera frame yet -- waiting for the camera.")
                else:
                    no_camera_warned = False
                    mp_image, brightness = self._prepare(frame)

                    if brightness < DARK_FRAME_MEAN:
                        if not dark_warned:
                            dark_warned = True
                            self._logger("Presence: the camera image is black -- lens covered, "
                                         "privacy shutter closed, or wrong camera index?")
                        confidence, stage = 0.0, None
                    else:
                        dark_warned = False
                        confidence, stage = self._detect(mp_image)

                    seen = confidence >= self.presence_threshold
                    hits_before = sum(hits)
                    hits.append(seen)
                    now = time.monotonic()
                    if seen:
                        last_seen_ts = now

                    with self._lock:
                        self.last_status = {
                            "human": self._present or seen,
                            "confidence": round(confidence, 3),
                            "stage": stage,
                            "brightness": round(brightness, 1),
                            "timestamp": time.time(),
                        }

                    # -- required internal log lines --------------------------
                    if seen and not self._present and hits_before == 0:
                        # First positive frame of a fresh build-up sequence
                        # (not yet enough hits to confirm arrival).
                        self._logger("PRESENCE: human candidate")
                    if not seen and now - last_analyzing >= ANALYZING_LOG_INTERVAL:
                        last_analyzing = now
                        self._logger("PRESENCE: analyzing")

                    if DEBUG and now - last_debug >= 2.0:
                        last_debug = now
                        self._logger(f"Presence[debug]: confidence={confidence:.3f} "
                                     f"(need >= {self.presence_threshold}) stage={stage} "
                                     f"brightness={brightness:.0f} hits={sum(hits)}/{len(hits)} "
                                     f"present={self._present}")

                    if not self._present:
                        if sum(hits) >= self.arrive_confirm_hits:
                            self._set_present(True)
                    elif (now - last_seen_ts) >= self.leave_confirm_seconds:
                        hits.clear()
                        self._set_present(False)
            except Exception as e:
                self._logger(f"Presence: loop error -- {e}")

            elapsed = time.monotonic() - start
            time.sleep(max(0.02, POLL_INTERVAL - elapsed))
