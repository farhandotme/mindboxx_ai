import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen

# ─────────────────────────────────────────────────────────────────────────────

# ── Console encoding ─────────────────────────────────────────────────────────
# Status lines in this app carry emoji and arrows ("📤 file_controller → Moved:
# a.txt → Documents/"). On a non-UTF-8 console — cp1254 on a Turkish Windows,
# cp1251 on a Russian one, cp932 on a Japanese one — printing one of those
# raises UnicodeEncodeError, and because the print sits after the tool's own
# try/except, the exception escapes into the receive loop and takes the session
# down. The assistant dies on a log line.
#
# Reconfiguring costs nothing and makes the app start the same way in every
# locale. `errors="replace"` is the belt and braces — a console that genuinely
# cannot render a glyph shows a box instead of killing the process.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import numpy as np
from google import genai
from google.genai import types
from ui import AUREXUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
    search_memory, set_trim_notifier,
)

# The file-backed tools (open_app, web_search, browser_control, …) are no longer
# imported or declared here — they self-describe via a TOOL dict in their own
# actions/*.py file and are auto-discovered by core.action_loader at startup.
# Only tools that are tied to live-session state stay inline in this file
# (screen_process, close_camera, save_memory, manage_monitor, shutdown_AUREX,
# system_status).
from actions.screen_processor  import _capture_camera, _capture_screen, set_shared_camera_manager
from actions.system_monitor    import get_system_status
from actions.background_monitor import add_monitor, remove_monitor, list_monitors
from memory.config_manager     import (
    get_voice, get_wake_word_enabled, save_wake_word_enabled, get_input_device, get_output_device,
    get_presence_enabled, save_presence_enabled,
)
from core.plugin_loader        import discover_plugins
from core                      import undo as undo_stack
from core                      import confirm as confirm_gate
from core                      import audio_devices
from core.action_loader        import discover_actions
from core.wake_word            import (
    WakeWordDetector, is_ready as wake_is_ready, install_and_download as wake_install,
)
from core.presence             import PresenceDetector, is_available as presence_is_available, \
    unavailable_reason as presence_unavailable_reason
from core.camera_manager       import CameraManager, is_available as camera_is_available
from core.kiosk_state          import KioskState, KioskStateMachine
from core.identity_manager     import IdentityManager
from core.tts                  import create_tts_player, TTSPlayer

# How long the assistant stays awake with no user speech before it auto-sleeps
# again (wake-word mode only).
WAKE_SLEEP_TIMEOUT = 120.0   # seconds (2 minutes)

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
COMPANY_INFO_PATH = BASE_DIR / "config" / "company_info.txt"
LIVE_MODEL          = "models/gemini-3.1-flash-live-preview"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000 
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

# RMS below which 16-bit PCM is treated as room silence; above _LEVEL_FULL it
# reads as a full-height waveform. Tuned so ordinary speech lands mid-range and
# the bars still move for a quiet talker — language- and device-independent.
_LEVEL_FLOOR = 60.0
_LEVEL_FULL  = 2600.0

# Voice barge-in: while AUREX is talking, the mic is otherwise muted (see
# _listen_audio) so its own voice can't be picked up and echoed back to it.
# To still let the user cut in by just talking, we watch the mic level even
# during that window and require it to read as genuine speech (not a click,
# a cough, or speaker bleed) for several consecutive frames in a row before
# we treat it as an interruption. _BARGE_IN_FRAMES × CHUNK_SIZE/SEND_SAMPLE_RATE
# is roughly the reaction time — tuned to feel instant without being twitchy.
#
# Tuned against the "Interrupted — listening..." loop: laptop speakers bleed
# AUREX's own voice into the mic, and the old 3-frame / fixed-level trigger read
# that echo as the user talking over it — AUREX cut itself off, heard its own
# tail, answered it, cut itself off again. Now the trigger (a) ignores the first
# second of every reply while it measures how loud the echo is, (b) needs the mic
# to clearly beat that measured echo level, for ~0.5 s in a row, and (c) can only
# fire once per _BARGE_IN_COOLDOWN.
_BARGE_IN_LEVEL    = 0.55   # absolute floor for "the user is talking over me"
_BARGE_IN_FRAMES   = 8      # 8 x 64 ms ≈ 0.5 s of sustained loud audio
_BARGE_IN_GRACE    = 1.0    # s after a reply starts: learn the echo level, never interrupt
_BARGE_IN_MARGIN   = 1.8    # mic level must exceed measured echo peak x this
_BARGE_IN_COOLDOWN = 2.0    # s between two voice-triggered interrupts
_POST_SPEECH_MIC_HOLD  = 0.4   # s of mic silence after AUREX finishes talking (speaker tail)
_INTERRUPT_DISCARD_MAX = 6.0   # s: never keep discarding model audio longer than this


def _pcm_level(samples) -> float:
    """Map a block of int16 PCM samples to a 0.0–1.0 loudness level for the HUD
    waveform. Returns 0.0 on empty/invalid input so it can never raise."""
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(x * x)))
    except Exception:
        return 0.0
    if rms <= _LEVEL_FLOOR:
        return 0.0
    return min(1.0, (rms - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR))


def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are AUREX, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

def _load_company_info() -> str:
    """Reads config/company_info.txt, stripping blank lines and comments.
    Returns '' if the file is missing or empty (feature silently disabled)."""
    try:
        raw = COMPANY_INFO_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    lines = [
        ln.rstrip() for ln in raw.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return "\n".join(lines).strip()


_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    # ── Inline tools ─────────────────────────────────────────────────────────
    # These stay here (rather than in an actions/*.py TOOL dict) because their
    # handling is woven into live-session state — vision capture/injection,
    # camera stream, memory writes, the monitor engine, and shutdown. All other
    # tools live in their own action file and are auto-discovered by
    # core.action_loader (see AUREXLive.__init__).
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when the user says (in ANY language): close camera, stop camera, "
            "turn off camera, that's creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "AUREX checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_AUREX",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop AUREX. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": (
            "Look up a fact you have stored about the user but which is NOT in "
            "the memory block of your system prompt. "
            "The prompt lists the keys it did not have room for under "
            "'[ALSO REMEMBERED]' — if the user asks about anything named there, "
            "call this FIRST. "
            "Also call it before saying you do not know something personal, and "
            "when the user asks what you remember about them (leave query empty "
            "for everything). "
            "This is a local file search: it is instant and costs nothing."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Keyword to search for — a name, a topic, a category "
                        "(e.g. 'ayse', 'coffee', 'projects'). "
                        "Leave empty to list everything stored."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": (
            "Reverse the last change YOU made to this computer — a file you "
            "moved, renamed, created or wrote, or a setting you changed such as "
            "volume, brightness, dark mode or WiFi. "
            "Call this whenever the user says undo, revert, take it back, put it "
            "back, cancel that, or tells you that you did the wrong thing, in ANY "
            "language. "
            "Use action='list' when they ask what can be undone. "
            "This only covers your own actions — it is not the Ctrl+Z of whatever "
            "application is on screen (that is computer_settings with action 'undo')."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "undo (default) — reverse the last change | list — show what can be undone",
                },
            },
            "required": [],
        },
    },
]

class _ReconnectSignal(Exception):
    """Raised inside the session TaskGroup to force a clean, voluntary reconnect
    (e.g. the user picked a new voice — the voice is fixed at connect time, so
    the session must be rebuilt).

    Carries `keep_context`: True for an ordinary rebuild, where the stored
    resumption handle is replayed and the conversation continues; False when the
    new session must genuinely start clean (see the voice-change note in
    _on_voice_change)."""

    def __init__(self, keep_context: bool = True):
        super().__init__()
        self.keep_context = keep_context


def _is_reconnect_signal(exc: BaseException) -> bool:
    """True if `exc` is a _ReconnectSignal, or a(n) (Base)ExceptionGroup that
    wraps one — TaskGroup bundles child exceptions into a group."""
    if isinstance(exc, _ReconnectSignal):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_reconnect_signal(sub) for sub in exc.exceptions)
    return False


def _keep_context_of(exc: BaseException) -> bool:
    """Read `keep_context` off a reconnect signal, unwrapping the group the
    TaskGroup put it in. Defaults to True: an unexpected shape must not silently
    wipe the conversation."""
    if isinstance(exc, _ReconnectSignal):
        return getattr(exc, "keep_context", True)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if _is_reconnect_signal(sub):
                return _keep_context_of(sub)
    return True


class AUREXLive:
    def __init__(self, ui: AUREXUI):
        self.ui             = ui
        self._asst_name     = "JARVI    S"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                     = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._barge_in_streak      = 0       # consecutive loud mic frames while AUREX is speaking
        self._echo_peak            = 0.0     # measured loudness of AUREX's own voice inside the mic
        self._speaking_since       = 0.0     # monotonic time the current reply started playing
        self._interrupted_at       = 0.0     # when the discard-incoming-audio flag was armed
        self._last_interrupt_ts    = 0.0     # last accepted interrupt (voice cooldown)
        self._mic_hold_until       = 0.0     # mic frames are not sent before this (speaker tail)
        self._loop_thread_id       = None    # thread id of the asyncio loop
        self._greeting_pending     = False   # someone arrived before the Live session was ready
        self._shutdown_started     = False
        self._turn_open            = False   # model has started a reply and turn_complete hasn't arrived yet
        self._expect_reply_until   = 0.0     # we just sent something that will get a reply — until this time
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_voice_change   = self._on_voice_change     # voice picker → rebuild session
        self.ui.on_audio_device_change = self._on_audio_device_change
        self._reconnect_event: asyncio.Event | None = None
        self._reconnect_keep = True   # False → next rebuild drops the resumption handle

        # ── Session resumption ─────────────────────────────────────────
        # The server issues a resumption handle every few seconds and reissues
        # it as the conversation moves on. Before this, session_resumption was
        # switched ON in the config and the update was never read, so the handle
        # was thrown away and EVERY reconnect — a dropped packet, a voice change,
        # switching microphone — started an empty session. "Unlimited sessions"
        # leaked through exactly this hole.
        #
        # Deliberately in RAM only, never written to disk. Persisting it would
        # make a fresh launch continue yesterday's conversation, which sounds
        # appealing but breaks the session-summary flow: _save_session_summary
        # runs at shutdown and the morning briefing pops it the next day. A
        # conversation that never ends never produces a summary, and the
        # "yesterday we talked about…" line silently disappears.
        self._resume_handle: str | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # proactive audio; auto-disabled if the server rejects it

        _base_dir = Path(__file__).resolve().parent
        _inline_names = {t["name"] for t in TOOL_DECLARATIONS}

        # File-backed tools: every actions/*.py with a TOOL dict, discovered the
        # same way plugins are. Reserved names = the inline tools above, so an
        # action can never shadow one.
        self._action_registry = discover_actions(
            actions_dir=_base_dir / "actions",
            reserved_names=_inline_names,
            logger=lambda msg: print(f"[Actions] {msg}"),
        )

        # Plugins must not collide with either an inline tool or a discovered action.
        _core_names = _inline_names | self._action_registry.names()
        self._plugin_registry = discover_plugins(
            plugins_dir=_base_dir / "plugins",
            core_tool_names=_core_names,
            logger=lambda msg: (print(f"[Plugins] {msg}"), self.ui.write_log(f"SYS: {msg}")),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.get_plugin_settings = self._plugin_registry.settings_schemas  # ⚙ settings tab
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

        # ── Kiosk state machine ──────────────────────────────────────────────
        # The single source of truth for AUREX's kiosk behaviour:
        #   IDLE -> APPROACHING -> GREETING -> CONVERSATION -> DEPARTING -> IDLE
        # Nothing else (presence, wake word, the UI) flips _awake / mic /
        # speaking directly — they only ever request a transition here, and
        # the on_enter handlers below are what actually wake, greet, gate the
        # mic, interrupt speech, and put AUREX back to sleep. See core/kiosk_state.py.
        self._kiosk = KioskStateMachine(
            logger=lambda m: (print(f"[Kiosk] {m}"), self.ui.write_log(f"SYS: {m}"))
        )
        self._kiosk.on_enter(KioskState.APPROACHING,  self._kiosk_on_approaching)
        self._kiosk.on_enter(KioskState.GREETING,     self._kiosk_on_greeting)
        self._kiosk.on_enter(KioskState.CONVERSATION, self._kiosk_on_conversation)
        self._kiosk.on_enter(KioskState.DEPARTING,    self._kiosk_on_departing)
        self.ui.kiosk_get_state = lambda: self._kiosk.state.value   # for logs/settings only

        # Who's currently at the desk — separate from "is someone here"
        # (that's presence detection). See core/identity_manager.py.
        self._identity = IdentityManager(
            logger=lambda m: (print(f"[Identity] {m}"), self.ui.write_log(f"SYS: {m}"))
        )

        # ── Deterministic local greeting voice (item 5) ──────────────────────
        # The very FIRST thing AUREX says to an arriving person must NEVER
        # depend on the Gemini Live session — not on it being connected, not
        # on the model deciding to respond, not on the network being fast.
        # This is a second, completely independent TTS engine (same engine
        # choices — EdgeTTS/Kokoro/ElevenLabs — as the rest of the app uses
        # via core/tts.py) whose ONLY job is to speak a fixed "Good
        # morning."/"Good afternoon."/etc. the instant a human is confirmed
        # present. It is built once, in the background, at launch — well
        # before anyone can walk up — so it's warm the first time it's
        # needed. See _build_greeting_tts / _speak_deterministic_greeting
        # and _kiosk_on_greeting below.
        self._greeting_tts: TTSPlayer | None = None
        self._greeting_tts_lock = threading.Lock()
        self._pending_deterministic_greeting: str = ""
        threading.Thread(
            target=self._build_greeting_tts, daemon=True, name="GreetingTTSInit"
        ).start()

        # ── Camera: one owner for the whole app ──────────────────────────────
        # Opened once, held open, read continuously in the background; both
        # presence detection and the on-demand vision tool read its cached
        # latest frame instead of opening their own VideoCapture. Started in
        # run() (once, independent of Gemini reconnects) and stopped only at
        # process shutdown. See core/camera_manager.py.
        self._camera: CameraManager | None = None
        if camera_is_available():
            self._camera = CameraManager(
                logger=lambda m: (print(f"[Camera] {m}"), self.ui.write_log(f"SYS: {m}"))
            )
            set_shared_camera_manager(self._camera)

        # ── Wake word (voice trigger) ─────────────────────────────────────────
        # Optional, alongside the camera: "Hey AUREX" also drives the SAME
        # kiosk state machine as a person walking up does — one path in, one
        # set of consequences (see _on_wake_detected below).
        self._wake_enabled     = get_wake_word_enabled()
        self._wake_detector: WakeWordDetector | None = None
        self._wake_sleep_timeout = WAKE_SLEEP_TIMEOUT
        # UI control surface for the Wake Word settings section.
        self.ui.wake_is_ready    = wake_is_ready          # () -> bool
        self.ui.wake_get_state   = self._wake_state       # () -> dict
        self.ui.on_wake_toggle   = self._ui_wake_toggle   # (enable: bool) -> str
        self.ui.on_wake_manual   = self._ui_wake_manual   # () -> toggle awake/asleep
        self.ui.on_wake_install  = self._ui_wake_install  # () -> (ok, msg)

        # ── Ambient presence sensor ──────────────────────────────────────────
        # "Someone just walked up to the desk" — runs locally, hidden, never
        # displayed or saved (see core/presence.py). Feeds the kiosk state
        # machine above; the camera itself is owned by CameraManager, not by
        # this detector.
        self._presence_enabled  = get_presence_enabled() and presence_is_available() and self._camera is not None
        if get_presence_enabled() and not presence_is_available():
            print(f"[Presence] disabled -- {presence_unavailable_reason()}")
        self._presence_detector: PresenceDetector | None = None
        self.ui.presence_get_state = self._presence_state      # () -> dict, for UI/log use
        self.ui.on_presence_toggle = self._ui_presence_toggle  # (enable: bool) -> str

        # Kiosk mode is "on" the moment either sensor can wake AUREX up.
        # In that mode _awake starts False (silent, mic gated) and ONLY the
        # kiosk state machine's GREETING/DEPARTING handlers ever flip it —
        # see _kiosk_on_greeting / _kiosk_on_departing below.
        # With BOTH off, kiosk choreography is skipped entirely and AUREX
        # behaves like a classic always-on desktop assistant (unchanged
        # default behaviour for anyone not using the kiosk features).
        self._kiosk_mode = self._wake_enabled or self._presence_enabled
        self._awake = not self._kiosk_mode

    # ── Wake word: state machine ─────────────────────────────────────────────

    def _wake_state(self) -> dict:
        # A loaded, running detector is definitively ready; otherwise fall back
        # to the cheap on-disk model-file check (no Model construction).
        ready = bool(self._wake_detector and self._wake_detector.ready) or wake_is_ready()
        return {"enabled": self._wake_enabled, "awake": self._awake, "ready": ready}

    def _ensure_wake_detector(self) -> bool:
        """Load the detector once (model loads on first start). Idempotent."""
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                on_detect=self._on_wake_detected,
                logger=lambda m: (print(f"[Wake] {m}"), self.ui.write_log(f"SYS: {m}")),
            )
        if not self._wake_detector.ready:
            return self._wake_detector.start()
        return True

    def _on_wake_detected(self) -> None:
        """Called from the detector thread when 'Hey AUREX' is heard. Feeds
        the SAME kiosk state machine a camera arrival does — see
        _kiosk_on_approaching/_kiosk_on_greeting below for what happens next.
        A no-op if we're already past IDLE (e.g. mid-conversation)."""
        self._kiosk.transition(KioskState.APPROACHING, "wake word heard")

    def wake(self, reason: str = "wake word") -> None:
        """Internal primitive — only ever called from within a kiosk
        on_enter handler (see _kiosk_on_greeting). Flips the mic/UI to
        listening; nothing outside the kiosk state machine should call this
        directly."""
        if self._awake:
            return
        self._awake = True
        self._last_user_speech = time.monotonic()   # start the auto-sleep clock now
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        self.ui.write_log(f"SYS: Awake — {reason}.")

    def sleep(self, reason: str = "timeout") -> None:
        """Internal primitive — only ever called from within a kiosk
        on_enter handler (see _kiosk_on_departing). Gates the mic back off."""
        if not self._awake:
            return
        self._awake = False
        self.set_speaking(False)
        self.ui.set_state("SLEEPING")
        self.ui.write_log(f"SYS: Sleeping — {reason}.")

    async def _run_sleep_watch(self) -> None:
        """
        Silence-timeout fallback for voice-only wake mode (wake word ON,
        presence camera not driving departure). When the camera IS active,
        PresenceDetector's own departure debounce is authoritative — walking
        away is what ends the conversation, not a clock — so this stays out
        of the way entirely in that case.
        """
        while True:
            await asyncio.sleep(5)
            if not self._wake_enabled or self._presence_enabled or not self._awake:
                continue
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if (time.monotonic() - self._last_user_speech) > self._wake_sleep_timeout:
                self._kiosk.transition(KioskState.DEPARTING, "no speech for 2 minutes")

    # ── Wake word: UI callbacks (called from the Qt thread) ──────────────────

    def _ui_wake_toggle(self, enable: bool) -> str:
        """Enable/disable wake word from the settings UI. Returns a status token:
        'enabled' | 'disabled' | 'need_download'."""
        if enable:
            if not wake_is_ready():
                return "need_download"
            self._wake_enabled = True
            self._kiosk_mode = True
            save_wake_word_enabled(True)
            self._ensure_wake_detector()
            self._kiosk.force_idle("wake word enabled")
            self.sleep(reason="wake word enabled")
            return "enabled"
        else:
            self._wake_enabled = False
            self._kiosk_mode = self._presence_enabled
            save_wake_word_enabled(False)
            if not self._kiosk_mode:
                self._kiosk.force_idle("wake word disabled")
                self.wake(reason="wake word disabled")
            return "disabled"

    def _ui_wake_manual(self) -> None:
        """Manual sleep/wake button in the UI — goes straight through the
        kiosk state machine so it can't drift out of sync with it."""
        if not self._wake_enabled:
            return
        if self._awake:
            self._kiosk.transition(KioskState.DEPARTING, "you tapped sleep")
        else:
            self._kiosk.transition(KioskState.APPROACHING, "you tapped wake")

    def _ui_wake_install(self) -> tuple[bool, str]:
        """Download openwakeword + the model (runs in a UI worker thread)."""
        return wake_install(logger=lambda m: self.ui.write_log(f"SYS: {m}"))

    # ── Presence sensor: state machine ────────────────────────────────────────

    def _presence_state(self) -> dict:
        det = self._presence_detector
        return {
            "enabled": self._presence_enabled,
            "ready":   bool(det and det.ready),
            "present": bool(det and det.present),
        }

    def _ensure_presence_detector(self) -> bool:
        """Load the detector once and start its polling thread. Idempotent."""
        if not self._presence_enabled or self._camera is None:
            return False
        if self._presence_detector is None:
            self._presence_detector = PresenceDetector(
                camera=self._camera,
                on_arrived=self._on_person_arrived,
                on_left=self._on_person_left,
                logger=lambda m: (print(f"[Presence] {m}"), self.ui.write_log(f"SYS: {m}")),
            )
        if not self._presence_detector.ready:
            return self._presence_detector.start()
        return True

    def _on_person_arrived(self) -> None:
        """Called from the presence-detector's own background thread the
        moment someone is confirmed standing at the desk. Only asks the
        kiosk state machine for a transition — IDLE -> APPROACHING — it
        never touches _awake/wake/sleep directly (see kiosk on_enter
        handlers below for what actually happens next)."""
        self._kiosk.transition(KioskState.APPROACHING, "someone stepped up to the desk")

    def _on_person_left(self) -> None:
        """Called from the presence-detector thread once nobody's been at
        the desk for the debounce window. Picks the right target state
        depending on how far the interaction had gotten — someone who
        wandered off before the greeting even started just goes back to
        IDLE quietly; someone who was mid-conversation triggers a full,
        immediate DEPARTING cleanup (item 7: never keep talking to an
        empty room)."""
        state = self._kiosk.state
        if state == KioskState.APPROACHING:
            self._kiosk.transition(KioskState.IDLE, "left before the greeting started")
        elif state in (KioskState.GREETING, KioskState.CONVERSATION):
            self._kiosk.transition(KioskState.DEPARTING, "no longer detected at the desk")

    def _ui_presence_toggle(self, enable: bool) -> str:
        """Settings-drawer toggle, mirrors _ui_wake_toggle. Returns a status
        token: 'enabled' | 'disabled' | 'unavailable'."""
        if enable:
            if not presence_is_available() or self._camera is None:
                return "unavailable"
            self._presence_enabled = True
            self._kiosk_mode = True
            save_presence_enabled(True)
            self._ensure_presence_detector()
            self._kiosk.force_idle("presence sensor enabled")
            self.sleep(reason="presence sensor enabled")
            return "enabled"
        else:
            self._presence_enabled = False
            self._kiosk_mode = self._wake_enabled
            save_presence_enabled(False)
            if self._presence_detector:
                self._presence_detector.stop()
            if not self._kiosk_mode:
                self._kiosk.force_idle("presence sensor disabled")
                self.wake(reason="presence sensor disabled")
            return "disabled"

    # ── Deterministic local greeting (item 5) ─────────────────────────────────

    def _build_greeting_tts(self) -> None:
        """Builds the local TTS engine used ONLY for the guaranteed first
        greeting, in the background, so it's already warm before anyone can
        possibly walk up. Best-effort: if it fails (engine package missing,
        no internet for a first-time Kokoro/EdgeTTS use, etc.) the
        deterministic opener is silently skipped for this run rather than
        crashing anything — _speak_deterministic_greeting() just no-ops."""
        try:
            cfg = {}
            try:
                cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            except Exception:
                pass
            player = create_tts_player(cfg)
            with self._greeting_tts_lock:
                self._greeting_tts = player
            print("[Kiosk] Deterministic greeting voice ready.")
        except Exception as e:
            print(f"[Kiosk] Deterministic greeting voice unavailable — {e}")

    def _speak_deterministic_greeting(self) -> None:
        """Item 5, verbatim: 'When PERSON_ARRIVED happens, AUREX must
        immediately speak using the reliable local TTS/audio system... This
        first sentence is deterministic... It must happen even when Gemini
        is reconnecting/slow/failing.' Called synchronously from
        _kiosk_on_greeting the instant GREETING is entered — but the actual
        (blocking) TTS call runs on its own thread so it never stalls the
        presence/wake-word thread that triggered the transition.

        The chosen phrase is stashed in self._pending_deterministic_greeting
        so _fire_greeting's instruction to Gemini can tell it not to repeat
        the same opener a second time."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            text = "Good morning."
        elif 12 <= hour < 17:
            text = "Good afternoon."
        elif 17 <= hour < 22:
            text = "Good evening."
        else:
            text = "Hello."
        self._pending_deterministic_greeting = text

        def _run():
            with self._greeting_tts_lock:
                player = self._greeting_tts
            if player is None:
                # Still loading (very first arrival right after launch) or
                # failed to build — the person is not left in total silence:
                # Gemini's own greeting (see _fire_greeting) is still queued
                # right behind this and will speak as soon as it can.
                print("[Kiosk] Deterministic greeting voice not ready yet — skipping local opener this time.")
                return
            try:
                self.set_speaking(True)
                player.speak(text)
            except Exception as e:
                print(f"[Kiosk] Deterministic greeting playback error — {e}")
            finally:
                self.set_speaking(False)

        threading.Thread(target=_run, daemon=True, name="DeterministicGreeting").start()

    def _stop_deterministic_greeting(self) -> None:
        """Called from _kiosk_on_departing: if the local opener is still
        mid-sentence when the person walks away, cut it off immediately —
        same "never keep talking to an empty room" guarantee as the Gemini
        audio path (see interrupt())."""
        with self._greeting_tts_lock:
            player = self._greeting_tts
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass

    # ── Kiosk state machine: on_enter handlers ────────────────────────────────
    # These are the ONLY places _awake, mic gating, speaking, and greeting
    # speech are triggered as a consequence of presence/wake activity. Both
    # PresenceDetector and WakeWordDetector only ever call
    # self._kiosk.transition(...) — everything below is the effect, not the
    # trigger.

    def _kiosk_on_approaching(self) -> None:
        """Someone is confirmed at the desk but hasn't been greeted yet. A
        brief pause lets them settle in frame before the camera-grounded
        compliment is captured, then we move straight into GREETING."""
        time.sleep(0.3)
        self._kiosk.transition(KioskState.GREETING, "starting greeting")

    def _kiosk_on_conversation(self) -> None:
        """Purely a label — mic/UI state was already flipped by wake() in
        _kiosk_on_greeting. Kept as its own state (rather than folding into
        GREETING) so departure logic can tell "still greeting" apart from
        "already talking" if that distinction is ever needed."""
        pass

    def _session_ready(self) -> bool:
        loop = self._loop
        return bool(self.session is not None and loop is not None and loop.is_running())

    def _kiosk_on_greeting(self) -> None:
        """Someone is here: wake up and greet them.

        The greeting used to be silently lost in two ways: (1) the person was
        detected a second or two after launch — BEFORE the Live session had
        connected — so there was nothing to send it to and the state machine
        just moved on; (2) a stale "discard model audio" flag armed by the
        previous departure swallowed the reply. Now the greeting is queued
        until the session exists, and firing it always disarms that flag.

        Item 5's deterministic opener is fired FIRST and unconditionally —
        it does not wait on, or care about, _session_ready() at all. That is
        the actual guarantee: the person hears "Good morning." the instant
        they're detected, full stop, regardless of what Gemini is doing."""
        self.wake(reason="starting the greeting")
        self._speak_deterministic_greeting()
        self._greeting_pending = True
        if self._session_ready():
            self._run_on_loop(self._fire_greeting_after_local_opener())
        else:
            self.ui.write_log("SYS: Someone is here — I'll greet them the moment I'm connected.")
        self._kiosk.transition(KioskState.CONVERSATION, "greeting handed over")

    async def _fire_greeting_after_local_opener(self) -> None:
        """A small head start before handing off to Gemini, so its audio
        doesn't land on top of the local "Good morning." at the exact same
        instant (both use the system's audio output). Purely a polish delay
        — item 5 only requires the FIRST sentence be independent of Gemini,
        it says nothing against Gemini continuing naturally right after."""
        await asyncio.sleep(0.5)
        self._fire_greeting()

    async def _deliver_pending_greeting(self) -> None:
        """Runs once per (re)connect: if a person arrived while the session was
        down, greet them now."""
        await asyncio.sleep(0.8)          # let the mic/speaker tasks come up first
        if self._greeting_pending and self._awake and self._session_ready():
            self._fire_greeting()

    def _fire_greeting(self) -> None:
        """Build and send exactly ONE greeting turn. No news, no briefing.
        Runs on the asyncio loop thread."""
        if not self._greeting_pending or not self._awake:
            return                       # they already left, or it was already sent
        self._greeting_pending = False

        # Fresh turn: nothing left over from a previous visit may mute or
        # talk over this one.
        self._interrupted = False
        self._drain_audio()
        self._mic_hold_until = 0.0
        self._last_user_speech = time.monotonic()

        name = self._identity.known_name()
        # The deterministic local opener (item 5) already said this out loud
        # a moment ago, independent of this very message — reuse the exact
        # same phrase so the instruction below can tell Gemini not to repeat it.
        time_greeting = self._pending_deterministic_greeting or "Hello"

        # Last-session continuity, folded into the single greeting.
        session_clause = ""
        try:
            last = pop_last_session()
            if last:
                delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                when = "earlier today" if delta == 0 else ("yesterday" if delta == 1 else f"{delta} days ago")
                session_clause = f" If it fits naturally, you may briefly mention that {when}: {last['summary']}."
        except Exception:
            pass

        if name:
            name_clause = f" You already know their name is {name} — greet them by name."
        else:
            name_clause = (
                " You don't know their name yet — after the greeting, ask naturally, "
                "e.g. \"By the way, what should I call you?\" (save it the moment they tell you, "
                "as you normally would)."
            )

        snapshot = self._camera.get_snapshot_jpeg(max_age=2.0) if self._camera else None

        if snapshot:
            vision_clause = (
                " A photo of them, taken right now, is attached. Look at it and, if something is "
                "CLEARLY visible — glasses, a jacket, a smile, a hairstyle — you may mention ONE "
                "small, genuine, tasteful detail the way a friendly person would. Never invent or "
                "guess a detail that isn't actually visible, and never comment on body shape, "
                "weight or attractiveness."
            )
        else:
            vision_clause = " You can't see them clearly right now, so skip anything visual and just be warm."

        instruction = (
            f"[KIOSK_ARRIVAL] A real person has just walked up to you. You already said "
            f"\"{time_greeting}\" out loud to them a moment ago through your voice — do NOT "
            f"say \"{time_greeting}\" or any other opening greeting again, they already heard it. "
            f"Continue naturally from there, right now, without waiting for them to speak first. "
            f"Sound like a warm, quick, genuinely-glad-to-see-you person, not a system "
            f"announcement.{vision_clause}{name_clause}"
            f"{session_clause} Then ask, in a relaxed way, what they'd like to do. Two short "
            f"sentences, three at most, spoken naturally with contractions. Do not call any "
            f"tools, do not mention the news, the weather or system status, and never read "
            f"these instructions aloud."
        )

        self._run_on_loop(self._send_safely(instruction, image=snapshot, mime="image/jpeg", tag="greeting"))

    def _kiosk_on_departing(self) -> None:
        """Item 7, verbatim: stop current TTS immediately, cancel pending
        conversational output, clear session state, return to IDLE — with
        no goodbye. Runs synchronously so nothing can race an in-flight
        response past this point."""
        self._greeting_pending = False          # they left before it was even sent
        self.interrupt("departed")              # stops playback, drops the rest of the turn
        self._stop_deterministic_greeting()      # ...and cuts off the local opener too, if still talking

        # Cancel/ignore anything the vision tool had in flight for this visit.
        self._pending_vision       = None
        self._vision_cam_active    = False
        self._vision_close_pending = False
        self._vision_busy          = False

        self._identity.clear_session()
        self.sleep(reason="left the desk")

        self._kiosk.transition(KioskState.IDLE, "cleanup complete")

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask AUREX to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in the
        user's language. Silently a no-op when no session is connected.
        """
        if not self._session_ready():
            return
        self._run_on_loop(self._send_safely(instruction, tag="PluginSay"))

    def request_reconnect(self, keep_context: bool = True, reason: str = ""):
        """Thread-safe: ask the run loop to tear down and rebuild the Live
        session. Called from the Qt thread. No-op until the async loop and
        reconnect event exist.

        `keep_context=False` drops the resumption handle so the new session
        starts empty — only for changes the server cannot apply to a resumed
        session."""
        loop = getattr(self, "_loop", None)
        ev   = self._reconnect_event
        self._reconnect_keep   = keep_context
        self._reconnect_reason = reason
        if loop and ev is not None:
            loop.call_soon_threadsafe(ev.set)

    def _on_voice_change(self):
        """Voice picker applied.

        The voice is baked into the session at connect time, so a rebuild is
        required. It is rebuilt WITHOUT the resumption handle on purpose:
        resuming restores the server's own session state, and the safe reading
        is that it restores the voice with it — which would make the picker
        appear to do nothing. Losing context here is acceptable because changing
        voice is a deliberate, rare act; losing it on a dropped packet was not."""
        self.request_reconnect(keep_context=False, reason="new voice")

    def _on_audio_device_change(self):
        """Microphone or speaker changed. Both streams are opened inside the
        session TaskGroup, so they can only be re-opened by rebuilding it —
        but the conversation is kept, which is the whole reason resumption
        landed before this feature did."""
        self.request_reconnect(keep_context=True, reason="audio device")

    async def _watch_reconnect(self):
        """Session-scoped task: when a voluntary reconnect is requested, raise a
        signal that unwinds the TaskGroup so the run loop rebuilds the session."""
        assert self._reconnect_event is not None
        await self._reconnect_event.wait()
        self._reconnect_event.clear()
        keep   = self._reconnect_keep
        reason = getattr(self, "_reconnect_reason", "") or "settings"
        self.ui.write_log(
            f"SYS: Applying {reason} — reconnecting"
            + ("..." if keep else " (starting a fresh conversation)...")
        )
        raise _ReconnectSignal(keep_context=keep)

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        # Respect wake-word sleep: a typed command must not be answered while
        # asleep either (the sleep gate is not just for the mic). Wake first with
        # "Hey AUREX" or the WAKE NOW button.
        if self._wake_enabled and not self._awake:
            self.ui.write_log("SYS: I'm asleep — say 'Hey AUREX' or tap WAKE NOW first.")
            return
        self._run_on_loop(self._send_safely(text, tag="text command"))

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            was = self._is_speaking
            self._is_speaking = value
        now = time.monotonic()
        if value:
            if not was:                      # a NEW reply started (not every audio chunk)
                self._barge_in_streak = 0
                self._echo_peak       = 0.0
                self._speaking_since  = now
            self.ui.set_state("SPEAKING")
        else:
            if was:                          # reply just ended: ignore the speaker tail briefly
                self._mic_hold_until = now + _POST_SPEECH_MIC_HOLD
            if not self.ui.muted and (self._awake or not self._kiosk_mode):
                self.ui.set_state("LISTENING")

    def _drain_audio(self) -> int:
        """Throw away queued-but-unplayed model audio. Loop-thread only."""
        q = self.audio_in_queue
        drained = 0
        if q is not None:
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
        return drained

    def _stop_playback(self) -> int:
        """Silence AUREX right now and give the mic back to the user."""
        drained = self._drain_audio()
        self.set_speaking(False)
        self._mic_hold_until = 0.0        # the user is (about to be) talking — don't clip them
        if self._turn_done_event:
            self._turn_done_event.clear()
        return drained

    def interrupt(self, reason: str = "manual") -> None:
        """Stop AUREX mid-speech. Safe to call from ANY thread (mic callback,
        Qt thread, presence thread): the real work is always done on the
        asyncio loop, because the audio queue is not thread-safe.

        reason: "manual"   — ESC / button: always honoured.
                "barge-in" — user spoke over AUREX: honoured only if AUREX is
                             actually talking, and at most once per cooldown.
                "departed" — the person walked away: always honoured, silent."""
        loop = self._loop
        if loop is not None and loop.is_running() and threading.get_ident() != self._loop_thread_id:
            loop.call_soon_threadsafe(self._do_interrupt, reason)
        else:
            self._do_interrupt(reason)

    def _turn_in_flight(self) -> bool:
        return self._turn_open or time.monotonic() < self._expect_reply_until

    def _do_interrupt(self, reason: str = "manual") -> None:
        now = time.monotonic()
        with self._speaking_lock:
            speaking = self._is_speaking
        q = self.audio_in_queue
        has_audio = speaking or (q is not None and not q.empty())

        # Nothing to cut off -> do nothing, and above all do NOT arm the
        # "discard incoming audio" flag: with nothing to cancel, it just
        # swallowed the NEXT reply (that is how the greeting went missing).
        if reason == "barge-in":
            if not has_audio or (now - self._last_interrupt_ts) < _BARGE_IN_COOLDOWN:
                return
        elif not (has_audio or self._turn_in_flight()):
            return

        if reason != "departed":
            self._last_interrupt_ts = now
        self._interrupted    = True          # drop the rest of the cancelled turn as it streams in
        self._interrupted_at = now
        drained = self._stop_playback()

        if reason == "departed":
            return
        if drained:
            print(f"[AUREX] ✋ Interrupted — {drained} audio chunks discarded")
        self.ui.write_log("SYS: Interrupted — listening...")

    # ── Sending text / pictures into the Live session ─────────────────────────

    def _call_on_loop(self, fn, *args) -> None:
        """Run a plain function on the asyncio loop thread, from any thread."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        if threading.get_ident() == self._loop_thread_id:
            fn(*args)
        else:
            loop.call_soon_threadsafe(fn, *args)

    def _run_on_loop(self, coro) -> None:
        """Schedule a coroutine on the asyncio loop from any thread."""
        loop = self._loop
        if loop is None or not loop.is_running():
            coro.close()
            return
        if threading.get_ident() == self._loop_thread_id:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

    async def _send_to_model(self, text: str, image: bytes | None = None,
                             mime: str = "image/jpeg") -> None:
        """One user turn (optional picture + text) into the Live session.

        gemini-3.1-flash-live takes text and images ONLY through
        send_realtime_input — send_client_content is limited to seeding the
        initial history there, and using it mid-conversation gets rejected by
        the server. Older SDKs/models without those arguments fall back to it."""
        session = self.session
        if session is None:
            return
        self._expect_reply_until = time.monotonic() + 12.0
        try:
            if image:
                await session.send_realtime_input(video=types.Blob(data=image, mime_type=mime))
            await session.send_realtime_input(text=text)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"[AUREX] realtime text unavailable ({e}) — using client content instead")
            parts = []
            if image:
                parts.append({"inline_data": {"mime_type": mime, "data": image}})
            parts.append({"text": text})
            await session.send_client_content(
                turns={"role": "user", "parts": parts}, turn_complete=True
            )

    async def _send_safely(self, text: str, image: bytes | None = None,
                           mime: str = "image/jpeg", tag: str = "send") -> None:
        try:
            await self._send_to_model(text, image=image, mime=mime)
        except Exception as e:
            print(f"[AUREX] {tag} error: {e}")

    def speak(self, text: str):
        if not self._session_ready():
            return
        self._run_on_loop(self._send_safely(text, tag="speak"))

    def shutdown(self) -> None:
        """Fast, idempotent, thread-safe teardown (Ctrl+C, window close):
        release the camera + sensors, and make a best-effort session summary."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._stop_deterministic_greeting()
        for part in (self._presence_detector, self._wake_detector, self._camera):
            try:
                if part is not None:
                    part.stop()
            except Exception as e:
                print(f"[AUREX] shutdown: {e}")
        loop = self._loop
        if loop is not None and loop.is_running() and len(self._session_log) >= 3:
            try:
                asyncio.run_coroutine_threadsafe(self._save_session_summary(), loop).result(timeout=2.5)
            except Exception:
                pass

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "AUREX").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "AUREX"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: Address the user with the ordinary respectful form "
                      "for a superior in the language you are currently speaking — "
                      "\"sir\" in English, its everyday equivalent in any other "
                      "language. Never an archaic or aristocratic form, and never "
                      "the form from a different language than the one you are "
                      "speaking in this sentence.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        company_info = _load_company_info()
        company_ctx = (
            f"[COMPANY INFO]\n"
            f"{company_info}\n"
            f"Whenever the user asks about the company (who you are, what the "
            f"company does, internal contacts, policies, or anything company-"
            f"related), answer using ONLY the facts above. Never invent facts "
            f"that aren't listed here — if something isn't covered, say you "
            f"don't have that information.\n\n"
        ) if company_info else ""

        parts = [time_ctx, identity_ctx]
        if company_ctx:
            parts.append(company_ctx)
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": (
                TOOL_DECLARATIONS
                + self._action_registry.get_tool_declarations()
                + self._plugin_registry.get_tool_declarations()
            )}],
            # Hand back the handle captured from the last session_resumption
            # update. `handle=None` is exactly the old behaviour (ask for
            # handles, start fresh), so the first connect of a run is unchanged.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # Sliding-window compression: session never dies from a full context
            # window — AUREX can stay in one conversation for hours
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice()
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Proactive audio: AUREX stays silent when speech isn't addressed
            # to it (background chatter, talking to someone else in the room).
            # (Affective dialog was dropped: gemini-3.1-flash-live does not
            #  support it, and it never reliably detected tone in practice.
            #  To restore it on a 2.5 native-audio model, add back:
            #  cfg["enable_affective_dialog"] = True )
            cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)
        return types.LiveConnectConfig(**cfg)

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[AUREX] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "recall_memory":
                # Local file search: no network, no second model. Kept out of
                # the executor deliberately — it is a dictionary scan over a few
                # hundred short strings, and a thread hop would cost more than
                # the work itself.
                result = search_memory(args.get("query", ""), limit=8)

            elif name == "undo":
                if str(args.get("action", "")).lower().strip() == "list":
                    items = undo_stack.history()
                    result = ("Things I can undo, most recent first:\n"
                              + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
                              ) if items else "I have not changed anything I can undo yet."
                else:
                    result = await loop.run_in_executor(None, undo_stack.undo_last)

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE short natural sentence in the user's own language, "
                        f"telling them you are looking at their {_stall} right now. "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif name == "shutdown_AUREX":
                self.ui.write_log("SYS: Shutdown requested.")
                async def _do_shutdown():
                    await self._save_session_summary()
                    if self.session:
                        await self._send_safely("Say a brief natural goodbye to the user.", tag="goodbye")
                    await asyncio.sleep(1.5)
                    if self._camera is not None:
                        self._camera.stop()   # release the physical device cleanly (item 8)
                    if self._presence_detector is not None:
                        self._presence_detector.stop()
                    import os as _os
                    _os._exit(0)
                asyncio.create_task(_do_shutdown())

            elif self._action_registry.has(name):
                # file_processor: fall back to the currently-uploaded file when none is given
                if name == "file_processor" and not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                _ctx = {"player": self.ui, "speak": self.speak,
                        "response": None, "session_memory": None}
                r = await loop.run_in_executor(None, lambda: self._action_registry.run(name, args, _ctx))
                result = r or "Done."
                # web_search: mirror results to the on-screen content panel
                if (name == "web_search" and r
                        and not r.startswith("No results")
                        and not r.startswith("Search failed")):
                    _mode  = args.get("mode", "search")
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)

            else:
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[AUREX] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # Gemini 3.x Live rejects the old realtime_input.media_chunks field
            # (what `media=...` maps to) and closes the socket with a 1007. Send
            # mic / phone PCM through the new `audio` field instead. Queue items
            # are {"data": <bytes>, "mime_type": <str>} from _listen_audio and
            # the phone relay.
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )

    async def _listen_audio(self):
        print("[AUREX] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # ── Wake-word gate ───────────────────────────────────────────────
            # While asleep, the mic audio NEVER goes to Gemini (nothing is
            # streamed, so AUREX can't respond to speech not addressed to it and
            # nothing leaves the machine). Frames are instead handed to the local
            # detector, which runs its model in ITS OWN thread — the cost here is
            # only a queue push, so the audio path is never slowed. When wake word
            # is off (default) or we're awake, this is a single boolean check.
            if self._kiosk_mode and not self._awake:
                if self._wake_enabled:
                    det = self._wake_detector
                    if det is not None:
                        det.feed(indata)
                # In presence-only mode there's nothing to feed audio to —
                # the camera thread does its own thing — we just keep the
                # mic gated (nothing streamed to Gemini) until it wakes us.
                return
            with self._speaking_lock:
                AUREX_speaking = self._is_speaking

            if AUREX_speaking:
                # Mic is otherwise held back while AUREX talks (see module
                # note above _BARGE_IN_LEVEL) — but we still watch its level
                # so the user can interrupt just by speaking over it, not
                # only via the ESC key / Interrupt button.
                if self.ui.muted or self._phone_active:
                    return
                now   = time.monotonic()
                level = _pcm_level(indata)
                if (now - self._speaking_since) < _BARGE_IN_GRACE:
                    # First second of a reply: learn how loud our own voice is
                    # inside the mic. Never interrupt during this window.
                    self._echo_peak = max(self._echo_peak, level)
                    self._barge_in_streak = 0
                    return
                thresh = min(0.95, max(_BARGE_IN_LEVEL, self._echo_peak * _BARGE_IN_MARGIN))
                if level >= thresh:
                    self._barge_in_streak += 1
                else:
                    self._barge_in_streak = 0
                    self._echo_peak = max(level, self._echo_peak * 0.995)   # track echo drift
                if self._barge_in_streak >= _BARGE_IN_FRAMES:
                    self._barge_in_streak = 0
                    self.interrupt("barge-in")   # thread-safe: runs on the asyncio loop
                    # Forward the chunk that triggered the barge-in too, so
                    # the first word the user said isn't lost.
                    data = indata.tobytes()
                    loop.call_soon_threadsafe(
                        self.out_queue.put_nowait,
                        {"data": data, "mime_type": "audio/pcm"}
                    )
                return

            if time.monotonic() < self._mic_hold_until:
                return   # AUREX just stopped talking — don't send its speaker tail back to it

            if not self.ui.muted and not self._phone_active:
                data = indata.tobytes()
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": data, "mime_type": "audio/pcm"}
                )
                # Feed the live mic level to the HUD so the waveform reacts to
                # the user's actual voice while listening. Purely cosmetic — any
                # failure here must never disturb the mic.
                try:
                    self.ui.set_audio_level(_pcm_level(indata))
                except Exception:
                    pass

        try:
            def _open_mic(dev):
                return sd.InputStream(
                    samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    device=dev,
                    callback=callback,
                )

            # Which microphone. resolve() returns None for "system default" and
            # for a saved device that is no longer present — so a headset
            # unplugged since the last run falls back to the built-in mic
            # instead of raising on startup and taking the session with it.
            _mic_name = get_input_device()
            _mic_dev  = audio_devices.resolve(_mic_name, "input")
            if _mic_dev is not None:
                print(f"[AUREX] 🎤 Input device: {_mic_name}")
            try:
                _mic_stream = _open_mic(_mic_dev)
            except Exception as _e:
                # A device the picker listed but the driver will not open right
                # now — exclusive mode, a webcam already in use, a virtual mic
                # whose source went away. Chosen hardware failing must never
                # mean the assistant cannot hear at all.
                if _mic_dev is None:
                    raise
                print(f"[AUREX] ⚠️  Mic '{_mic_name}' failed: {_e} — using default")
                self.ui.write_log(
                    f"SYS: Microphone '{_mic_name}' unavailable — using system default."
                )
                _mic_stream = _open_mic(None)

            with _mic_stream:
                print("[AUREX] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[AUREX] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[AUREX] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    # ── Session resumption ───────────────────────────────────
                    # The server sends this periodically. `resumable` goes false
                    # while a turn is mid-flight — replaying a handle from that
                    # moment is what the flag exists to prevent — so only
                    # resumable handles are kept. This is three lines and it is
                    # the entire fix for "every reconnect forgets everything".
                    _sru = getattr(response, "session_resumption_update", None)
                    if _sru is not None:
                        if getattr(_sru, "resumable", False) and getattr(_sru, "new_handle", None):
                            if self._resume_handle is None:
                                print("[AUREX] 🔗 Session resumption armed")
                            self._resume_handle = _sru.new_handle

                    if response.data:
                        self._turn_open = True
                        if self._interrupted and (time.monotonic() - self._interrupted_at) > _INTERRUPT_DISCARD_MAX:
                            self._interrupted = False   # safety valve: a lost turn_complete must never mute AUREX for good
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if getattr(sc, "interrupted", False):
                            # The SERVER cancelled its own turn (it heard the user).
                            # Anything still queued is stale; everything that arrives
                            # from now on belongs to the NEXT turn — so also disarm
                            # our own discard flag or that next reply would be muted.
                            self._stop_playback()
                            self._interrupted = False
                            self._turn_open = False
                            self._expect_reply_until = 0.0
                            out_buf = []

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()

                        if sc.turn_complete:
                            self._turn_open = False
                            self._expect_reply_until = 0.0
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "AUREX",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                await self._send_to_model(question, image=img_b, mime=mime_t)
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until AUREX finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[AUREX] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[AUREX] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[AUREX] 🔊 Play started")

        _spk_name = get_output_device()
        _spk_dev  = audio_devices.resolve(_spk_name, "output")
        if _spk_dev is not None:
            print(f"[AUREX] 🔊 Output device: {_spk_name}")

        def _open_spk(dev):
            st = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=dev,
            )
            st.start()
            return st

        try:
            stream = _open_spk(_spk_dev)
        except Exception as _e:
            # A chosen output that the host API accepts by name but refuses to
            # open (exclusive mode, wrong sample rate, device asleep) must not
            # cost the user their voice. Fall back to the default and say so.
            if _spk_dev is None:
                raise
            print(f"[AUREX] ⚠️  Output device '{_spk_name}' failed: {_e} — using default")
            self.ui.write_log(f"SYS: Speaker '{_spk_name}' unavailable — using system default.")
            stream = _open_spk(None)

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Drive the HUD waveform from AUREX's own voice while speaking.
                try:
                    self.ui.set_audio_level(_pcm_level(
                        np.frombuffer(bytes(batch), dtype=np.int16)))
                except Exception:
                    pass

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[AUREX] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=_get_api_key())
            resp   = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-flash-latest",
                contents=prompt,
            )
            summary = (resp.text or "").strip()
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    # A remote command is deliberate control and the phone user
                    # has no desktop WAKE button — so it wakes AUREX if asleep.
                    if self._wake_enabled and not self._awake:
                        self.wake(reason="remote command")
                    await self._send_to_model(text)
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._loop_thread_id = threading.get_ident()
        self._reconnect_event = asyncio.Event()

        # ── Wire the shared core services to the interface ───────────────────
        # The confirmation gate is useless without a way to ask, and a memory
        # trim is invisible without a way to say so. Both are bound once here
        # rather than passed down through every action signature.
        confirm_gate.bind(
            show = self.ui.show_confirm,
            hide = self.ui.hide_confirm,
            log  = self.ui.write_log,
        )
        set_trim_notifier(self.ui.write_log)

        # Tell the device picker the exact rates the streams open at, from the
        # constants that actually open them — so it can never list a device that
        # cannot be opened at them.
        audio_devices.configure(SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Enumerate audio devices off-thread. The settings drawer must never pay
        # for host-API enumeration on the Qt thread.
        audio_devices.prefetch()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        # ── Kiosk bootstrap (item 12) ──────────────────────────────────────────
        # CameraManager and the presence/wake sensors are independent of the
        # Gemini connection — they start ONCE here, not inside the reconnect
        # loop below, so a network hiccup never restarts the camera or
        # re-triggers a greeting. Startup is silent: no greeting, no news,
        # just IDLE and watching.
        if self._camera is not None:
            self._camera.start()
        if self._wake_enabled:
            self._ensure_wake_detector()
        if self._presence_enabled:
            self._ensure_presence_detector()

        if self._kiosk_mode:
            self._awake = False
            self.ui.set_state("SLEEPING")
        else:
            self._awake = True
            self.ui.set_state("LISTENING")

        while True:
            try:
                print("[AUREX] Connecting...")
                self.ui.set_state("THINKING")
                _resumed_with = self._resume_handle is not None
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                # v1alpha carries proactive audio; if it gets rejected we fall
                # back to v1beta.
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1alpha" if self._enhanced_live else "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False
                    self._turn_open            = False
                    self._expect_reply_until   = 0.0

                    print("[AUREX] Connected.")
                    if _resumed_with:
                        # Say it plainly: the difference between "it reconnected"
                        # and "it reconnected and still knows what we were doing"
                        # is the whole point, and it is invisible otherwise.
                        self.ui.write_log("SYS: Reconnected — conversation restored.")
                    elif self._kiosk_mode:
                        _how = []
                        if self._wake_enabled:
                            _how.append("say 'Hey AUREX'")
                        if self._presence_enabled:
                            _how.append("walk up to the camera")
                        self.ui.write_log(f"SYS: AUREX online — idle, silent. {' or '.join(_how).capitalize()} to begin.")
                    else:
                        self.ui.write_log("SYS: AUREX online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    self._reconnect_event.clear()  # ignore requests from before this session
                    tg.create_task(self._watch_reconnect())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_sleep_watch())
                    if self._greeting_pending:
                        tg.create_task(self._deliver_pending_greeting())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                # Voluntary reconnect (voice change) — not an error. Rebuild the
                # session immediately with no backoff and no scary logs.
                if _is_reconnect_signal(e):
                    print("[AUREX] Voluntary reconnect requested.")
                    if not _keep_context_of(e):
                        # A deliberate clean slate (voice change) — drop the
                        # handle so the next connect really does start empty.
                        self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                # A resumption handle the server will not accept — expired, or
                # belonging to a session it has since dropped. Without this, the
                # same dead handle would be replayed on every retry and the
                # assistant would never come back at all: the feature meant to
                # survive a reconnect would be the thing preventing one. Drop it
                # once and let the next attempt start clean.
                if _resumed_with and (
                    "resum" in str(e).lower()
                    or "handle" in str(e).lower()
                    or "INVALID_ARGUMENT" in str(e)
                    or "NOT_FOUND" in str(e)
                ):
                    print("[AUREX] 🔗 Resumption handle rejected — starting a fresh session")
                    self.ui.write_log("SYS: Could not restore the conversation — starting fresh.")
                    self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                err_str = str(e)
                print(f"[AUREX] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Proactive audio rejected by the server (preview API drift) —
                # drop it and reconnect with the plain config.
                if self._enhanced_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "proactiv" in err_str.lower()
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                ):
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Proactive audio unavailable — reconnecting without it."
                    )
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[AUREX] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Connection failed — retrying in {_conn_backoff}s. "
                        "(a VPN may be required)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[AUREX] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    import os as _os
    import signal as _signal

    ui = AUREXUI("face.png")
    holder: dict = {"aurex": None, "stopping": False, "cleanup": None}

    def runner():
        ui.wait_for_api_key()
        AUREX = AUREXLive(ui)
        holder["aurex"] = AUREX
        try:
            asyncio.run(AUREX.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    def _start_cleanup() -> threading.Thread:
        """Release camera / sensors off the UI thread, exactly once."""
        if holder["cleanup"] is None:
            def _do():
                a = holder["aurex"]
                if a is not None:
                    try:
                        a.shutdown()
                    except Exception as e:
                        print(f"[AUREX] cleanup error: {e}")
            t = threading.Thread(target=_do, daemon=True, name="CleanupThread")
            holder["cleanup"] = t
            t.start()
        return holder["cleanup"]

    def _quit(*_):
        # First Ctrl+C: graceful. Second: immediate. A watchdog guarantees the
        # process ends even if some worker thread (audio, network) is stuck.
        if holder["stopping"]:
            print("\n🔴 Forced exit.")
            _os._exit(1)
        holder["stopping"] = True
        print("\n🔴 Shutting down... (Ctrl+C again to force)")
        wd = threading.Timer(5.0, lambda: _os._exit(0))
        wd.daemon = True
        wd.start()
        _start_cleanup()
        try:
            ui._app.quit()
        except Exception:
            _os._exit(0)

    # Qt's event loop runs in C++ and never gives Python's signal handler a
    # chance to run, which is why Ctrl+C in the terminal did nothing. The
    # heartbeat timer below wakes the interpreter every 150 ms so it can.
    for _name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        _sig = getattr(_signal, _name, None)
        if _sig is not None:
            try:
                _signal.signal(_sig, _quit)
            except (ValueError, OSError):
                pass
    from PyQt6.QtCore import QTimer
    _heartbeat = QTimer()
    _heartbeat.timeout.connect(lambda: None)
    _heartbeat.start(150)

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

    # Window closed (or _quit ran): finish cleanup, then leave for real —
    # never wait on non-daemon executor threads.
    _start_cleanup().join(timeout=3.5)
    _os._exit(0)

if __name__ == "__main__":
    main()