"""
IdentityManager -- answers "who is this?", kept deliberately separate from
PresenceDetector, which only ever answers "is someone here?"

Today identity resolution is simple and falls back in this order:
  1. A name already established earlier in the CURRENT session (someone
     told AUREX their name a minute ago -- don't ask again while they're
     still standing there).
  2. A name already saved in long-term memory (memory/memory_manager.py's
     identity/name field -- written automatically by the existing
     `save_memory` Gemini tool whenever someone tells AUREX their name).
  3. Unknown -- the kiosk greeting asks for it naturally.

`recognize_face()` is an intentional stub / extension point: when local
face recognition or enrollment is added later, it plugs in HERE, behind
this one method, using whatever frame the CameraManager/PresenceDetector
already have -- nothing about PresenceDetector, the kiosk state machine,
or main.py's greeting flow needs to change to support it. That is the
whole reason this is its own module instead of a couple of lines inside
PresenceDetector.
"""
from __future__ import annotations

from typing import Callable, Optional


class IdentityManager:
    def __init__(self, logger: Callable[[str], None] = print):
        self._logger = logger
        self._session_name: Optional[str] = None

    def known_name(self) -> Optional[str]:
        """Best name we currently have for whoever's at the desk, or None."""
        if self._session_name:
            return self._session_name
        try:
            from memory.memory_manager import load_memory
            memory   = load_memory()
            identity = memory.get("identity", {}) or {}
            entry    = identity.get("name", {})
            val = (entry.get("value", "") if isinstance(entry, dict) else str(entry)).strip()
            return val or None
        except Exception as e:
            self._logger(f"Identity: could not read stored name -- {e}")
            return None

    def remember_for_session(self, name: str) -> None:
        """Cache a name for the rest of this visit, without touching
        long-term memory (Gemini's own save_memory tool call handles
        persisting it if the person wants that remembered permanently)."""
        name = (name or "").strip()
        if name:
            self._session_name = name

    def clear_session(self) -> None:
        """Called on DEPARTING -> IDLE: the next visitor is a clean slate
        until they're identified again."""
        self._session_name = None

    def recognize_face(self, frame) -> Optional[str]:
        """Extension point for future local face recognition/enrollment.
        Given a raw camera frame, return a known name if it matches an
        enrolled face, else None. Not implemented yet -- always returns
        None, so identity currently falls back to session/memory/asking."""
        return None
