#!/usr/bin/env python3
"""Pi-side ROS 2 link check: publish swarm_state, print /mission/command."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SIM = Path(__file__).resolve().parent.parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from swarm_net import MISSION_CMD_TOPIC, pack_swarm_state, swarm_qos  # noqa: E402


class LinkCheck(Node):
    def __init__(self):
        super().__init__("pi_link_check")
        qos = swarm_qos()
        self.pub = self.create_publisher(String, "/px4_0/swarm_state", qos)
        self.create_subscription(String, MISSION_CMD_TOPIC, self._on_cmd, qos)
        self.create_timer(1.0, self._tick)
        self.n = 0
        self.get_logger().info("publishing /px4_0/swarm_state every 1 s")

    def _tick(self):
        self.n += 1
        msg = String()
        msg.data = pack_swarm_state("drone0", 1, 0.0, 0.0, 0.0)
        self.pub.publish(msg)
        self.get_logger().info(f"sent swarm_state n={self.n}")

    def _on_cmd(self, msg: String):
        self.get_logger().info(f"HEARD ground command: {msg.data}")


def main():
    rclpy.init()
    node = LinkCheck()
    end = time.time() + 30
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    print("link check finished (30 s). If the PC echoed swarm_state, WiFi ROS 2 works.")


if __name__ == "__main__":
    main()
