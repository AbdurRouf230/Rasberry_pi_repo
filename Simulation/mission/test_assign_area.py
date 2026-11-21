"""Area assignment: one random 3–6 point field, Haiku plan, plot, then fly."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MISSION = Path(__file__).resolve().parent
for path in (str(ROOT), str(MISSION)):
    if path not in sys.path:
        sys.path.insert(0, path)

import assign_area  # noqa: E402
from survey_launch import base_spawns, prepare  # noqa: E402


def _world_base_camp():
    text = (ROOT / "agricultural_farm.world").read_text(encoding="utf-8")
    block = re.search(r'<model name="base_camp">(.*?)</model>', text, re.S)
    if not block:
        raise AssertionError("agricultural_farm.world has no base_camp model")
    pose = re.search(r"<pose>\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block.group(1))
    size = re.search(r"<size>\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block.group(1))
    if not pose or not size:
        raise AssertionError("base_camp is missing pose or size")
    return (
        (float(pose.group(1)), float(pose.group(2)), float(pose.group(3))),
        (float(size.group(1)), float(size.group(2)), float(size.group(3))),
    )


def _fake_assign(rows, drones):
    n = len(rows)
    k = max(1, len(drones))
    blocks = []
    start = 1
    for i, drone in enumerate(drones):
        if start > n:
            break
        extra = 1 if i < (n % k) else 0
        width = max(1, n // k) + extra
        last = min(n, start + width - 1)
        if i == len(drones) - 1:
            last = n
        if start <= last:
            blocks.append({
                "drone_id": drone["id"],
                "first_row": start,
                "last_row": last,
            })
        start = last + 1
    return blocks, "llm", [{"total_s": 0.01, "valid": True, "errors": [], "raw_answer": ""}]


class FitAndPickTests(unittest.TestCase):
    def test_every_catalog_area_has_3_to_6_points(self):
        for case in assign_area.CASES:
            n = len(case["coords"])
            self.assertGreaterEqual(n, 3, case["name"])
            self.assertLessEqual(n, 6, case["name"])

    def test_pick_one_area_is_from_catalog(self):
        case = assign_area.pick_one_area(rng=assign_area.random.Random(7))
        names = {item["name"] for item in assign_area.CASES}
        self.assertIn(case["name"], names)
        self.assertTrue(3 <= len(case["coords"]) <= 6)

    def test_fit_polygon_stays_inside_farm(self):
        raw = [(0, 0), (400, 0), (0, 300)]
        fitted = assign_area.fit_polygon_to_farm(raw)
        xs = [p[0] for p in fitted]
        ys = [p[1] for p in fitted]
        self.assertEqual(len(fitted), 3)
        self.assertLessEqual(max(xs), assign_area.FIELD_EAST_M)
        self.assertGreaterEqual(min(xs), -assign_area.FIELD_EAST_M)
        self.assertLessEqual(max(ys), assign_area.FIELD_NORTH_M)
        self.assertGreaterEqual(min(ys), -assign_area.FIELD_NORTH_M)


class FallbackAssignTests(unittest.TestCase):
    def test_fallback_covers_every_row_once(self):
        coords = assign_area.fit_polygon_to_farm([(0, 0), (400, 0), (400, 250), (0, 250)])
        cov = assign_area.make_rows(coords, assign_area.HFOV_DEG, min_rows=6)
        drones = assign_area.drones_for_count(6)
        blocks = assign_area.fallback_split(cov["rows"], drones)
        seen = []
        for block in blocks:
            seen.extend(range(block["first_row"], block["last_row"] + 1))
        self.assertEqual(sorted(seen), list(range(1, len(cov["rows"]) + 1)))

    def test_fallback_assigns_every_drone_a_row(self):
        coords = assign_area.fit_polygon_to_farm([(0, 0), (400, 0), (400, 250), (0, 250)])
        drones = assign_area.drones_for_count(6)
        cov = assign_area.make_rows(coords, assign_area.HFOV_DEG, min_rows=len(drones))
        blocks = assign_area.fallback_split(cov["rows"], drones)
        self.assertGreaterEqual(len(cov["rows"]), 6)
        self.assertEqual(
            [b["drone_id"] for b in blocks],
            [d["id"] for d in drones],
        )
        for block in blocks:
            self.assertGreaterEqual(block["last_row"], block["first_row"])


class BaseCampTests(unittest.TestCase):
    def test_world_pad_is_large_enough_for_six_drones(self):
        (_cx, _cy, _cz), (east_m, north_m, _z) = _world_base_camp()
        self.assertGreaterEqual(east_m, 48.0)
        self.assertGreaterEqual(north_m, 48.0)

    def test_six_spawns_sit_on_the_world_pad(self):
        (cx, cy, _cz), (east_m, north_m, _z) = _world_base_camp()
        margin = 4.0
        spawns = base_spawns(6)
        self.assertEqual(len(spawns), 6)
        xs = {round(x, 3) for x, _y, _z, _yaw in spawns}
        ys = {round(y, 3) for _x, y, _z, _yaw in spawns}
        self.assertGreaterEqual(len(xs) * len(ys), 6)
        for x, y, _z, _yaw in spawns:
            self.assertGreaterEqual(x, cx - east_m / 2.0 + margin, (x, y))
            self.assertLessEqual(x, cx + east_m / 2.0 - margin, (x, y))
            self.assertGreaterEqual(y, cy - north_m / 2.0 + margin, (x, y))
            self.assertLessEqual(y, cy + north_m / 2.0 - margin, (x, y))


class PrepareMissionTests(unittest.TestCase):
    def test_prepare_writes_plot_and_one_patch_per_drone(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "example_request.json").write_text(
                (MISSION / "example_request.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plot_dir = work / "plots"
            with patch.object(assign_area, "assign_with_llm", side_effect=_fake_assign):
                launch = prepare(
                    work,
                    num_patches=6,
                    case_name="quad_rectangle",
                    plot_dir=plot_dir,
                )
            self.assertEqual(launch["area"], "quad_rectangle")
            self.assertEqual(launch["corners"], 4)
            self.assertEqual(launch["num_drones"], 6)
            self.assertEqual(len(launch["patches"]), 6)
            self.assertEqual(len(launch["spawns"]), 6)
            self.assertLess(launch["altitude_m"], 25.0)
            (cx, cy, _cz), (east_m, north_m, _z) = _world_base_camp()
            drone_ids = [p["drone_id"] for p in launch["patches"]]
            self.assertEqual(drone_ids, [f"drone{i}" for i in range(1, 7)])
            for assigned, spawn in zip(launch["patches"], launch["spawns"]):
                self.assertTrue(assigned["waypoints_m"], assigned["drone_id"])
                self.assertGreaterEqual(spawn["x"], cx - east_m / 2.0 + 4.0)
                self.assertLessEqual(spawn["x"], cx + east_m / 2.0 - 4.0)
                self.assertGreaterEqual(spawn["y"], cy - north_m / 2.0 + 4.0)
                self.assertLessEqual(spawn["y"], cy + north_m / 2.0 - 4.0)
            pngs = list(plot_dir.glob("*.png"))
            self.assertEqual(len(pngs), 1)
            self.assertTrue(Path(launch["assignment_plot"]).is_file())
            saved = json.loads((work / "survey_launch.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["area"], "quad_rectangle")
            drone_ids = [p["drone_id"] for p in saved["patches"]]
            self.assertEqual(drone_ids, [f"drone{i}" for i in range(1, 7)])


if __name__ == "__main__":
    unittest.main()
