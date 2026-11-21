"""Acceptance checks that do not need Gazebo."""

import json
import unittest
from pathlib import Path

from geometry import shoelace
from planner import plan, replan

EXAMPLE = json.loads((Path(__file__).parent / "example_request.json").read_text(encoding="utf-8"))


def _metres_rectangle():
    # 300 m by 150 m, so three strips are an easy hand check.
    return {
        "coordinate_order": "lonlat",
        "coordinates": [
            [0.0, 0.0],
            [300.0 / 111320.0, 0.0],
            [300.0 / 111320.0, 150.0 / 110540.0],
            [0.0, 150.0 / 110540.0],
        ],
        "num_patches": 3,
        "sweep_angle_deg": 0,
    }


class PlannerTests(unittest.TestCase):
    def test_needs_four_coordinates(self):
        result = plan({"coordinates": [[0, 0], [1, 0], [1, 1]], "num_patches": 1})
        self.assertFalse(result["valid"])
        self.assertIn("at least 4", result["errors"][0])

    def test_equal_area_rectangle(self):
        result = plan(_metres_rectangle())
        self.assertTrue(result["valid"], result["errors"])
        areas = [patch["area_m2"] for patch in result["patches"]]
        self.assertEqual(len(areas), 3)
        target = sum(areas) / 3.0
        for area in areas:
            self.assertLess(abs(area - target) / target, 0.02)

    def test_same_input_same_plan(self):
        self.assertEqual(plan(EXAMPLE), plan(json.loads(json.dumps(EXAMPLE))))

    def test_camera_matches_hand_calculation(self):
        # altitude = 0.02 * 4.5 * 4000 / 6.17
        request = {
            "coordinates": EXAMPLE["coordinates"],
            "num_patches": 1,
            "sweep_angle_deg": 0,
            "target_gsd_cm": 2.0,
            "front_overlap": 0.70,
            "side_overlap": 0.65,
            "camera": EXAMPLE["camera"],
            "limits": {"min_altitude_m": 1, "max_altitude_m": 500, "max_speed_ms": 100},
        }
        result = plan(request)
        self.assertTrue(result["valid"], result["errors"])
        expected_alt = 0.02 * 4.5 * 4000 / 6.17
        self.assertAlmostEqual(result["altitude_m"], expected_alt, places=2)
        gsd_m = expected_alt * 6.17 / (4.5 * 4000)
        self.assertAlmostEqual(result["line_spacing_m"], gsd_m * 4000 * 0.35, places=1)
        self.assertAlmostEqual(result["trigger_distance_m"], gsd_m * 3000 * 0.30, places=1)

    def test_concave_u_balances(self):
        # U in metres, projected by treating metres as degrees is wrong.
        # Build the U directly through a tiny local lon/lat shift.
        origin_lon, origin_lat = 0.0, 0.0
        scale_x = 111320.0
        scale_y = 110540.0

        def ll(x, y):
            return [origin_lon + x / scale_x, origin_lat + y / scale_y]

        # (0,0)-(40,0)-(40,30)-(25,30)-(25,12)-(15,12)-(15,30)-(0,30)
        shape = [(0, 0), (40, 0), (40, 30), (25, 30), (25, 12), (15, 12), (15, 30), (0, 30)]
        result = plan({
            "coordinates": [ll(x, y) for x, y in shape],
            "num_patches": 2,
            "sweep_angle_deg": 90,
        })
        self.assertTrue(result["valid"], result["errors"])
        areas = [p["area_m2"] for p in result["patches"]]
        target = sum(areas) / 2.0
        for area in areas:
            self.assertLess(abs(area - target) / target, 0.02)
        # Hand area: 40*30 - 10*18 = 1200 - 180 = 1020
        self.assertAlmostEqual(sum(areas), 1020, delta=15)

    def test_replan_splits_dropped_patch(self):
        first = plan(_metres_rectangle())
        self.assertTrue(first["valid"], first["errors"])
        request = _metres_rectangle()
        request["drones"] = [
            {"id": "drone1", "home": [0.0, -0.0002]},
            {"id": "drone2", "home": [0.001, -0.0002]},
            {"id": "drone3", "home": [0.002, -0.0002]},
        ]
        first = plan(request)
        again = replan(request, first, ["drone1", "drone2"])
        self.assertTrue(again["valid"], again["errors"])
        self.assertEqual(len(again["patches"]), 2)
        areas = [p["area_m2"] for p in again["patches"]]
        self.assertLess(abs(areas[0] - areas[1]) / max(areas), 0.02)

    def test_patch_area_matches_shoelace(self):
        result = plan(_metres_rectangle())
        for patch in result["patches"]:
            ring = patch["coordinates_m"]
            self.assertAlmostEqual(abs(shoelace(ring)), patch["area_m2"], delta=0.05)


if __name__ == "__main__":
    unittest.main()
