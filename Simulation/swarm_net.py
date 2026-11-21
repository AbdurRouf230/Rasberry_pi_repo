"""Topics between each Raspberry Pi and the ground-PC coordinator.

Drones notify. The PC decides RTB. Same ROS_DOMAIN_ID + Fast DDS discovery server.
"""

from __future__ import annotations

import json
from typing import Any, Optional

BOX_HIT_TOPIC = "/swarm/box_hit"
MISSION_CMD_TOPIC = "/mission/command"
SWARM_STATE_TOPIC = "/{ns}/swarm_state"  # ns = px4_0


def swarm_qos():
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def pack_box_hit(finder: str, hits: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(
        {
            "finder": str(finder),
            "hits": list(hits or []),
        },
        separators=(",", ":"),
    )


def unpack_box_hit(raw: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    finder = data.get("finder")
    if not finder:
        return None
    return {"finder": str(finder), "hits": list(data.get("hits") or [])}


def pack_command(command: str, finder: str = "") -> str:
    return json.dumps(
        {"command": str(command), "finder": str(finder)},
        separators=(",", ":"),
    )


def unpack_command(raw: str) -> str:
    text = (raw or "").strip()
    if text in {"return_to_base", "hold", "start", "continue"}:
        return text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.lower()
    return str(data.get("command") or "").lower()


def pack_swarm_state(
    drone: str,
    patch: int,
    x: float,
    y: float,
    altitude: float,
) -> str:
    return json.dumps(
        {
            "drone": drone,
            "patch": int(patch),
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "altitude": round(float(altitude), 2),
        },
        separators=(",", ":"),
    )
