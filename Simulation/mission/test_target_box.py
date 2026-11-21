"""Target box: random pose inside the selected area, YOLO detect, fleet RTB."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MISSION = Path(__file__).resolve().parent
for path in (str(ROOT), str(MISSION)):
    if path not in sys.path:
        sys.path.insert(0, path)

import assign_area  # noqa: E402
import target_box  # noqa: E402
from survey_launch import prepare  # noqa: E402


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


class PointInPolygonTests(unittest.TestCase):
    def test_vertex_average_of_convex_areas_is_inside(self):
        convex = {
            "tri_right", "tri_isosceles", "tri_obtuse",
            "quad_rectangle", "quad_trapezoid", "pent_convex",
            "pent_house", "hex_regular",
        }
        for case in assign_area.CASES:
            if case["name"] not in convex:
                continue
            fitted = assign_area.fit_polygon_to_farm(case["coords"])
            xs = [p[0] for p in fitted]
            ys = [p[1] for p in fitted]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            self.assertTrue(
                target_box.point_in_polygon(cx, cy, fitted),
                case["name"],
            )

    def test_point_outside_rectangle_is_rejected(self):
        square = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        self.assertFalse(target_box.point_in_polygon(20, 0, square))
        self.assertFalse(target_box.point_in_polygon(0, 20, square))


class RandomBoxTests(unittest.TestCase):
    def test_random_box_stays_inside_every_catalog_area(self):
        for case in assign_area.CASES:
            fitted = assign_area.fit_polygon_to_farm(case["coords"])
            rng = random.Random(case["name"])
            for _ in range(8):
                spec = target_box.place_in_polygon(fitted, rng=rng)
                self.assertTrue(
                    target_box.box_fits_polygon(spec["x"], spec["y"], fitted),
                    (case["name"], spec),
                )

    def test_same_area_can_get_a_new_position_each_call(self):
        fitted = assign_area.fit_polygon_to_farm(
            [(0, 0), (400, 0), (400, 250), (0, 250)]
        )
        a = target_box.place_in_polygon(fitted, rng=random.Random(1))
        b = target_box.place_in_polygon(fitted, rng=random.Random(2))
        self.assertNotEqual((a["x"], a["y"]), (b["x"], b["y"]))


class WorldInjectTests(unittest.TestCase):
    def test_injects_target_box_model_at_pose(self):
        world = "<sdf><world name='w'>\n  </world>\n</sdf>\n"
        out = target_box.inject_into_world(world, {"x": 12.5, "y": -8.25, "z": 0.0})
        self.assertIn('<model name="target_box">', out)
        self.assertIn("12.5 -8.25 0.0", out)
        self.assertIn("tape_cross", out)
        self.assertIn("</world>", out)


class TrainedWeightsTests(unittest.TestCase):
    def test_trained_box_yolo_weights_are_on_disk(self):
        from box_detector import resolve_weights

        path = resolve_weights()
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "best.pt")


class PrepareWritesBoxTests(unittest.TestCase):
    def test_prepare_writes_target_box_inside_selected_area(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "example_request.json").write_text(
                (MISSION / "example_request.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with patch.object(assign_area, "assign_with_llm", side_effect=_fake_assign):
                with patch.object(target_box.random, "Random", return_value=random.Random(11)):
                    launch = prepare(
                        work,
                        num_patches=6,
                        case_name="tri_right",
                        plot_dir=work / "plots",
                    )
            box = launch["target_box"]
            self.assertTrue(
                target_box.box_fits_polygon(box["x"], box["y"], launch["area_coords"])
            )
            saved = json.loads((work / "survey_launch.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["target_box"]["x"], box["x"])
            self.assertEqual(saved["area"], "tri_right")


class YellowBoxDetectTests(unittest.TestCase):
    def test_color_hit_does_not_recall_the_fleet(self):
        from box_detector import detections_are_target

        hits = [{
            "name": "box", "confidence": 0.99, "bbox": [0, 0, 640, 480], "source": "color",
        }]
        self.assertFalse(detections_are_target(hits))

    def test_yolo_box_hit_recalls_the_fleet(self):
        from box_detector import detections_are_target

        hits = [{
            "name": "box", "confidence": 0.82, "bbox": [40, 50, 120, 160], "source": "yolo",
        }]
        self.assertTrue(detections_are_target(hits))

    def test_plain_green_field_is_not_a_hit(self):
        from box_detector import detections_are_target, detect_image

        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[:, :] = (40, 120, 40)
        hits = detect_image(rgb, use_yolo=False)
        self.assertFalse(detections_are_target(hits))


class CaptureFolderTests(unittest.TestCase):
    def test_folder_name_is_drone_num_until_box_found(self):
        from box_detector import captured_dir_name, mark_finder_capture_dir

        self.assertEqual(captured_dir_name(0), "drone1")
        self.assertEqual(captured_dir_name(5), "drone6")
        with tempfile.TemporaryDirectory() as raw:
            survey = Path(raw) / "drone5"
            survey.mkdir()
            (survey / "wp1.png").write_bytes(b"x")
            other = Path(raw) / "drone2"
            other.mkdir()
            marked = mark_finder_capture_dir(survey)
            self.assertEqual(marked.name, "drone5_captured")
            self.assertTrue((marked / "wp1.png").is_file())
            self.assertFalse(survey.exists())
            self.assertEqual(other.name, "drone2")
            self.assertTrue(other.is_dir())
            self.assertEqual(mark_finder_capture_dir(marked).name, "drone5_captured")

    def test_annotated_png_contains_the_bounding_box(self):
        from box_detector import save_annotated_rgb

        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        hits = [{
            "name": "box", "confidence": 0.9, "bbox": [10, 15, 40, 50], "source": "yolo",
        }]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "drone1" / "wp1_bbox.png"
            saved = save_annotated_rgb(rgb, hits, path)
            self.assertTrue(saved.is_file())
            from PIL import Image
            img = np.array(Image.open(saved).convert("RGB"))
            edge = img[15, 10:40]
            self.assertTrue(np.any(edge[:, 1] > 100), "green bbox should be drawn")


class FleetRecallTests(unittest.TestCase):
    def test_first_detection_recalls_every_drone(self):
        import box_alert

        box_alert.reset()
        drones = [type("D", (), {})() for _ in range(3)]
        self.assertFalse(box_alert.recall_requested())
        box_alert.announce_box("drone2", [{"name": "box", "confidence": 0.9}])
        box_alert.mark_drones(drones)
        self.assertTrue(box_alert.recall_requested())
        self.assertEqual(box_alert.finder(), "drone2")
        for drone in drones:
            self.assertTrue(getattr(drone, "_return_home_now"))
        box_alert.announce_box("drone1", [{"name": "box", "confidence": 0.8}])
        self.assertEqual(box_alert.finder(), "drone2")
        box_alert.reset()


if __name__ == "__main__":
    unittest.main()
