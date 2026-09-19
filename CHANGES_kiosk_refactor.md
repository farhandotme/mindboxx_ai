# AUREX kiosk refactor — what changed and why

Your complaint was specific: *"the camera is open but AUREX remains silent
when a person is visible."* Rather than rebuild the kiosk architecture from
scratch, I dug into the existing code first — most of the spec (items 1, 3,
4, 8, 9, 11, 13, 14, 15, 16, 18) was already built and wired correctly:
`CameraManager` (single camera owner), `PresenceDetector`, `KioskStateMachine`,
`IdentityManager` all existed and cooperated the way the spec describes.

The silence bug had one real, findable root cause, plus a few contributing
bugs in the same neighbourhood. Here's exactly what was wrong and what I
changed.

## 1. The actual silence bug (item 5)

`_kiosk_on_greeting()` sent the *entire* greeting through the Gemini Live
session (`_fire_greeting` → `_send_safely`). If the session wasn't connected
yet, or was reconnecting, or the network was slow, the code just logged
`"I'll greet them the moment I'm connected"` to the system log and waited —
**nothing was ever spoken.** Meanwhile `core/tts.py` — a fully working local
TTS engine (EdgeTTS / Kokoro / ElevenLabs) — existed in the repo and was
never imported anywhere in `main.py`. It was dead code sitting right next to
the bug.

**Fix:** `main.py` now builds a second, independent `TTSPlayer` in the
background at launch (`_build_greeting_tts`). The instant `GREETING` is
entered, `_speak_deterministic_greeting()` fires a fixed
"Good morning./afternoon./evening." through it — **unconditionally**, before
checking whether Gemini is even connected. Gemini's own follow-up (visual
compliment + name ask, items 6–8) is still sent right after, with a small
0.5 s handoff delay and an instruction telling it not to repeat the opener
it already heard. If Gemini never comes up, the person still hears "Good
morning." — that's the item-5 guarantee. Departure (`_kiosk_on_departing`)
now also cuts off the local opener immediately if it's still mid-sentence.

## 2. Departure grace period was 12 seconds (item 12)

`core/presence.py`'s default `leave_confirm_seconds` was `12.0`. The spec
(and acceptance test 7 — "immediately stops speaking/listening") wants
1–2 seconds. Changed the default to **1.5s**, and made it overridable via
`AUREX_LEAVE_GRACE_SECONDS` or the `PresenceDetector(leave_confirm_seconds=…)`
constructor argument.

## 3. Face-only detection (item 2)

The detector only ever looked for a face (3 Haar cascades). Someone turned
away, leaning back, or with an occluded face was invisible to it even
though a person was plainly standing there. Added two more offline/bundled
stages, tried in order (cheapest first, each only run when the previous
found nothing):

1. **Face** — frontal ×2, profile L/R (unchanged, still fastest/most common).
2. **Body** — upper-body and full-body Haar cascades, catching turned-away
   or occluded-face cases.
3. **Silhouette** — OpenCV's bundled HOG people detector, tried last
   (heaviest), catching whatever the cascades still miss.

All three are bundled with `opencv-python` — no downloads, nothing sent
off-device.

## 4. Two other things were opening the camera (item 1's absolute rule)

- `ui.py`'s camera-preview stream (`_cam_loop`) opened its **own**
  `cv2.VideoCapture` on the same index, running continuously at ~30 fps
  *concurrently* with `CameraManager`'s background reader. Two handles on
  the same physical webcam fight each other on most drivers — a very
  plausible contributor to intermittent detection flakiness. Fixed: it now
  reads from the shared `CameraManager` like everything else
  (`actions.screen_processor.get_shared_camera_manager()`).
- `actions/screen_processor.py`'s on-demand snapshot function fell back to
  opening a second `VideoCapture` handle whenever the shared manager didn't
  have a *fresh enough* frame, instead of just waiting briefly on the one
  real owner. Fixed: it now retries against the shared manager for up to
  ~1.5s before giving up, and only opens its own handle in the genuine
  edge case where no shared manager was ever registered at all (camera
  feature off/unavailable for this run).

## 5. New debug tool (item 20)

`tools/test_presence.py` — tests camera + human detection + arrival +
departure in isolation. No Gemini, no UI, no preview window.

```
python tools/test_presence.py
AUREX_PRESENCE_DEBUG=1 python tools/test_presence.py   # verbose per-frame detail
python tools/test_presence.py --grace 1.0               # override departure grace
```

## 6. Detector engine replaced with MediaPipe (follow-up task)

Item 3 above (Haar cascades: face → body → HOG silhouette) was still
unreliable in practice — cascades flicker frame-to-frame and are sensitive
to lighting/angle. `core/presence.py` now uses **MediaPipe** instead:

- **Pose** (`mp.solutions.pose`, BlazePose) is the primary signal — it
  reads shoulder/nose/eye/ear landmark *visibility*, so it doesn't care
  which way the person is facing, and still fires when only the upper
  body is in frame (close to the camera) or the person is farther away.
- **Face Detection** (`mp.solutions.face_detection`) is a secondary
  signal, only run when pose alone isn't already confident that frame.
- Both ship their models inside the `mediapipe` pip package — still fully
  local/offline, nothing sent to Gemini for presence.
- Public API (`PresenceDetector(camera, on_arrived, on_left, logger, …)`,
  `.start()/.stop()/.ready/.present/.last_status`) is unchanged, so
  `main.py`'s kiosk wiring needed no edits. Added `get_result()` →
  `{human_present, confidence, timestamp}`.
- Sliding-window arrival / short departure grace period (item 2 above) is
  unchanged.
- New requirement: `mediapipe` added to `requirements.txt`.
- `tools/test_presence.py` now prints `HUMAN_PRESENT = true` / `= false`
  (previously `HUMAN_PRESENT: true`), plus per-second confidence.

## 7. `mp.solutions` doesn't exist anymore -- ported to the Tasks API

Item 6 above was written against `mp.solutions.pose` / `mp.solutions.face_detection`
("legacy solutions"). Turns out Google removed that API from the `mediapipe`
pip package entirely as of the 0.10.3x/1.0.x releases (deprecated since
0.10.0) -- `mp.solutions.pose` now raises `AttributeError: module
'mediapipe' has no attribute 'solutions'`. `core/presence.py` now uses the
supported replacement, the **Tasks API**
(`mediapipe.tasks.python.vision.PoseLandmarker` / `.FaceDetector`).

Consequence: Tasks-API models aren't bundled in the pip wheel like the old
solutions models were. `PresenceDetector.start()` downloads two small
files the first time it runs (`pose_landmarker_lite.task`, a few MB, and
`blaze_face_short_range.tflite`, a few hundred KB) into `models/mediapipe/`
at the project root, and caches them there — every run after the first is
fully offline again, same as before. Still 100% local decision-making;
this is a one-time model download, not sending frames anywhere. Fully
offline machines can pre-place the two files there manually, or point
`AUREX_POSE_MODEL_PATH` / `AUREX_FACE_MODEL_PATH` at existing copies — see
the URLs in `core/presence.py`.

Also added `core.presence.unavailable_reason()` (and wired it into
`main.py` and `tools/test_presence.py`) so a disabled detector always
prints the *real* reason (import failure, DLL issue, version mismatch,
model-download failure, ...) instead of a generic "unavailable".

## What I did *not* touch

Items 1, 3, 4, 8, 9, 11, 13, 14, 15, 16, 18 of the spec were already
correctly implemented and are unchanged. I looked specifically for
item-10 violations (proactive/unsolicited speech) — `ProactiveEngine` in
`actions/proactive.py` exists but is never instantiated anywhere in
`main.py` (dead code, not a live path), and `manage_monitor`'s background
topic-monitoring is user-invoked add/remove/list only, with no autonomous
speech path found. Left both as-is since they aren't actually firing.

## Try it

1. `python tools/test_presence.py` first, on its own, to confirm the camera
   and detector work in isolation (item 20's whole point).
2. Then run the app normally. The very first thing you should hear on
   arrival is the deterministic "Good morning." — even before Gemini has
   said anything — with Gemini's compliment + name-ask following ~0.5s
   later once it's ready.
