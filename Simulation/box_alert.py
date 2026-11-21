"""Shared flag: one drone saw the target box, the whole fleet returns to base."""

from __future__ import annotations

import threading
from typing import Any, Iterable, Optional

_lock = threading.Lock()
_found = False
_finder: Optional[str] = None
_hits: list[dict[str, Any]] = []


def reset() -> None:
    global _found, _finder, _hits
    with _lock:
        _found = False
        _finder = None
        _hits = []


def announce_box(drone_id: str, detections: list[dict[str, Any]] | None = None) -> bool:
    """First call wins. Later calls keep the original finder."""
    global _found, _finder, _hits
    with _lock:
        if _found:
            return False
        _found = True
        _finder = str(drone_id)
        _hits = list(detections or [])
        print(
            f"📦 BOX DETECTED by {drone_id} — all drones return to base camp",
            flush=True,
        )
        return True


def recall_requested() -> bool:
    with _lock:
        return _found


def finder() -> Optional[str]:
    with _lock:
        return _finder


def hits() -> list[dict[str, Any]]:
    with _lock:
        return list(_hits)


def mark_drones(drones: Iterable[Any]) -> None:
    for drone in drones:
        drone._return_home_now = True
