#!/usr/bin/env python3
"""Raspberry Pi companion for the Gazebo sim on the ground PC.

The Pi does NOT run a camera here. Box YOLO uses each SITL drone's
Gazebo down-camera on the PC. This node only:
  - publishes /px4_N/swarm_state so the PC sees the Pi on WiFi
  - listens for /mission/command (RTB when a Gazebo drone finds the box)
  - --fake-hit is radio-only (no Gazebo); do not use it as the sim detector
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SIM = Path(__file__).resolve().parent.parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from swarm_net import (  # noqa: E402
    BOX_HIT_TOPIC,
    MISSION_CMD_TOPIC,
    pack_box_hit,
    pack_swarm_state,
    swarm_qos,
    unpack_command,
)


class PiCompanion(Node):
    def __init__(self, drone_id: str, namespace: str, patch: int):
        super().__init__(f"pi_companion_{namespace}")
        self.drone_id = drone_id
        self.namespace = namespace
        self.patch = patch
        self.recalled = False
        self._announced = False
        qos = swarm_qos()
        self._state_pub = self.create_publisher(
            String, f"/{namespace}/swarm_state", qos,
        )
        self._hit_pub = self.create_publisher(String, BOX_HIT_TOPIC, qos)
        self.create_subscription(String, MISSION_CMD_TOPIC, self._on_cmd, qos)
        self.create_timer(0.5, self._tick)
        self.get_logger().info(
            f"companion {drone_id} ns=/{namespace} (no Pi camera; "
            f"sim YOLO is Gazebo down-cam) listen {MISSION_CMD_TOPIC}"
        )

    def _tick(self) -> None:
        msg = String()
        msg.data = pack_swarm_state(self.drone_id, self.patch, 0.0, 0.0, 0.0)
        self._state_pub.publish(msg)

    def _on_cmd(self, msg: String) -> None:
        cmd = unpack_command(msg.data)
        if cmd in {"return_to_base", "rth", "rtl"}:
            self.recalled = True
            self.get_logger().info(f"RTB received from ground PC ({msg.data})")
        else:
            self.get_logger().info(f"command from PC: {cmd or msg.data}")

    def announce_box(self, hits: list) -> None:
        if self._announced:
            return
        self._announced = True
        msg = String()
        msg.data = pack_box_hit(self.drone_id, hits)
        self._hit_pub.publish(msg)
        self.get_logger().info(f"notified ground PC: box_hit {msg.data}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi ROS 2 link for Gazebo sim. Detection is the drone camera on the PC.",
    )
    parser.add_argument("--drone-id", default=os.environ.get("DRONE_ID", "drone0"))
    parser.add_argument("--namespace", default=os.environ.get("PX4_NS", "px4_0"))
    parser.add_argument("--patch", type=int, default=1)
    parser.add_argument(
        "--fake-hit",
        action="store_true",
        help="radio test only: send box_hit after 5 s (not the sim camera)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PiCompanion(args.drone_id, args.namespace, args.patch)
    if args.fake_hit:
        def _fake():
            time.sleep(5.0)
            node.announce_box([{"name": "box", "confidence": 0.99, "source": "yolo"}])
        threading.Thread(target=_fake, daemon=True).start()
        node.get_logger().info("radio test: fake box_hit in 5 s (not a camera)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
