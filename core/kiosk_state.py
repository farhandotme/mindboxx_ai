"""
The ONE state machine that owns AUREX's kiosk behaviour.

    IDLE -> APPROACHING -> GREETING -> CONVERSATION -> DEPARTING -> IDLE

Nothing else in the application is allowed to decide, on its own, whether
AUREX is awake, listening, speaking, or asleep -- every one of those is a
*consequence* of a state transition here, wired up in main.py via
`on_enter(state, callback)`. Presence detection, wake word, and the UI only
ever ASK for a transition; they never flip audio/mic flags directly. That
is what makes it possible to reason about "can AUREX ever keep talking to
an empty room" -- the answer only depends on this one file.

Thread-safety: `transition()` and `state` are safe to call from any thread
(the presence-detector thread, the wake-word thread, or the asyncio event
loop thread all call in here). The registered on_enter callbacks run
synchronously, on whatever thread called transition() -- callbacks that need
to touch the Gemini session (which lives on the asyncio loop) are expected
to hop over with `asyncio.run_coroutine_threadsafe`, the same pattern
main.py already uses elsewhere for cross-thread calls into the session.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from enum import Enum
from typing import Callable


class KioskState(Enum):
    IDLE         = "IDLE"
    APPROACHING  = "APPROACHING"
    GREETING     = "GREETING"
    CONVERSATION = "CONVERSATION"
    DEPARTING    = "DEPARTING"


# The legal graph. IDLE -> CONVERSATION is included for the "classic" mode
# where neither wake word nor presence is enabled: AUREX just comes up ready
# to talk, with no kiosk choreography at all.
_ALLOWED: dict[KioskState, set[KioskState]] = {
    KioskState.IDLE:         {KioskState.APPROACHING, KioskState.CONVERSATION},
    KioskState.APPROACHING:  {KioskState.GREETING, KioskState.IDLE},
    KioskState.GREETING:     {KioskState.CONVERSATION, KioskState.DEPARTING},
    KioskState.CONVERSATION: {KioskState.DEPARTING},
    KioskState.DEPARTING:    {KioskState.IDLE},
}


class KioskStateMachine:
    def __init__(self, logger: Callable[[str], None] = print):
        self._state = KioskState.IDLE
        self._lock  = threading.Lock()
        self._on_enter: dict[KioskState, list[Callable[[], None]]] = defaultdict(list)
        self._logger = logger

    @property
    def state(self) -> KioskState:
        with self._lock:
            return self._state

    def on_enter(self, state: KioskState, callback: Callable[[], None]) -> None:
        """Register a callback to run every time the machine enters `state`."""
        self._on_enter[state].append(callback)

    def transition(self, new_state: KioskState, reason: str = "") -> bool:
        """Attempt to move to `new_state`. Illegal or no-op transitions are
        logged and ignored -- callers don't need to pre-check the graph
        themselves, they just ask and get told no if it doesn't make sense
        right now (e.g. two "person left" signals arriving in a race)."""
        with self._lock:
            old = self._state
            if new_state == old:
                return False
            if new_state not in _ALLOWED.get(old, set()):
                self._logger(f"Kiosk: ignored {old.value} -> {new_state.value} ({reason or 'no reason given'})")
                return False
            self._state = new_state

        suffix = f" ({reason})" if reason else ""
        self._logger(f"Kiosk: {old.value} -> {new_state.value}{suffix}")
        for cb in list(self._on_enter.get(new_state, [])):
            try:
                cb()
            except Exception as e:
                self._logger(f"Kiosk: on_enter({new_state.value}) handler failed -- {e}")
        return True

    def force_idle(self, reason: str = "recovering from a failure") -> None:
        """Escape hatch for failure handling (item 10): jump straight to IDLE
        from ANY state, bypassing the legal-transition graph. Used when a
        subsystem fails in a way that would otherwise strand the machine in
        APPROACHING/GREETING/CONVERSATION/DEPARTING forever."""
        with self._lock:
            old = self._state
            if old == KioskState.IDLE:
                return
            self._state = KioskState.IDLE
        self._logger(f"Kiosk: {old.value} -> IDLE (forced -- {reason})")
        for cb in list(self._on_enter.get(KioskState.IDLE, [])):
            try:
                cb()
            except Exception as e:
                self._logger(f"Kiosk: on_enter(IDLE) handler failed -- {e}")
