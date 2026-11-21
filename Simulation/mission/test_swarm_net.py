"""JSON payloads used on WiFi between the Pi and the ground PC."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import swarm_net  # noqa: E402


class SwarmNetTests(unittest.TestCase):
    def test_box_hit_roundtrip(self):
        raw = swarm_net.pack_box_hit("drone0", [{"name": "box", "confidence": 0.9}])
        data = swarm_net.unpack_box_hit(raw)
        self.assertEqual(data["finder"], "drone0")
        self.assertEqual(data["hits"][0]["name"], "box")

    def test_command_json_and_plain(self):
        raw = swarm_net.pack_command("return_to_base", "drone0")
        self.assertEqual(swarm_net.unpack_command(raw), "return_to_base")
        self.assertEqual(swarm_net.unpack_command("return_to_base"), "return_to_base")

    def test_swarm_state_has_pose(self):
        raw = swarm_net.pack_swarm_state("drone0", 3, 1.25, -4.5, 9.0)
        self.assertIn("drone0", raw)
        self.assertIn("9.0", raw)


if __name__ == "__main__":
    unittest.main()
