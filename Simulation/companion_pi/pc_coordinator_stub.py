#!/usr/bin/env python3
"""Ground-PC stub: no Gazebo. Echo Pi swarm_state, RTB on box_hit.

Use this first to prove WiFi ROS 2. Run on the PC (Ethernet).
"""

from __future__ import annotations

import sys
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
    pack_command,
    swarm_qos,
    unpack_box_hit,
)


class CoordinatorStub(Node):
    def __init__(self):
        super().__init__("pc_coordinator_stub")
        qos = swarm_qos()
        self.cmd_pub = self.create_publisher(String, MISSION_CMD_TOPIC, qos)
        self.create_subscription(String, "/px4_0/swarm_state", self._on_state, qos)
        self.create_subscription(String, BOX_HIT_TOPIC, self._on_hit, qos)
        self._sent_rtb = False
        self.get_logger().info(
            f"PC stub waiting. sub /px4_0/swarm_state + {BOX_HIT_TOPIC}"
        )

    def _on_state(self, msg: String):
        self.get_logger().info(f"Pi swarm_state: {msg.data}")

    def _on_hit(self, msg: String):
        data = unpack_box_hit(msg.data) or {}
        self.get_logger().info(f"Pi box_hit: {msg.data}")
        if self._sent_rtb:
            return
        self._sent_rtb = True
        out = String()
        out.data = pack_command("return_to_base", data.get("finder", "drone0"))
        self.cmd_pub.publish(out)
        self.get_logger().info(f"published {MISSION_CMD_TOPIC}: {out.data}")


def main():
    rclpy.init()
    node = CoordinatorStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
