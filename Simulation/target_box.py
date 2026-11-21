"""Place the cardboard search box inside the selected survey polygon.

Size and look match the trained box YOLO (1.2 x 0.9 x 0.6 m, packing tape).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

BOX_SIZE_M = 1.2
BOX_NORTH_M = 0.9
BOX_HEIGHT_M = 0.6
BOX_Z_M = 0.0


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Even-odd ray test. poly is [(east, north), ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-15
            at_x = (xj - xi) * (y - yi) / denom + xi
            if x < at_x:
                inside = not inside
        j = i
    return inside


def box_corners(x: float, y: float, size_m: float = BOX_SIZE_M):
    h = float(size_m) / 2.0
    return (
        (x - h, y - h),
        (x + h, y - h),
        (x + h, y + h),
        (x - h, y + h),
    )


def box_fits_polygon(x: float, y: float, poly, size_m: float = BOX_SIZE_M) -> bool:
    if not point_in_polygon(x, y, poly):
        return False
    return all(point_in_polygon(cx, cy, poly) for cx, cy in box_corners(x, y, size_m))


def _centroid(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def place_in_polygon(poly, rng: random.Random | None = None, size_m: float = BOX_SIZE_M) -> dict:
    """Random pose whose full footprint stays inside poly. New point every call."""
    if len(poly) < 3:
        raise ValueError("polygon must have at least 3 corners")
    rng = rng or random.Random()
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    for size in (float(size_m), 1.6, 1.0):
        for _ in range(400):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if box_fits_polygon(x, y, poly, size):
                return _spec(x, y, size)
        cx, cy = _centroid(poly)
        if box_fits_polygon(cx, cy, poly, size):
            return _spec(cx, cy, size)
    cx, cy = _centroid(poly)
    return _spec(cx, cy, 1.0)


def _spec(x: float, y: float, size_m: float) -> dict:
    return {
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "z": BOX_Z_M,
        "size_m": round(float(size_m), 3),
        "length_m": BOX_SIZE_M,
        "width_m": BOX_NORTH_M,
        "height_m": BOX_HEIGHT_M,
    }


def sdf_model(spec: dict) -> str:
    """Cardboard box with packing tape — the object the trained YOLO looks for."""
    x, y = spec["x"], spec["y"]
    z = spec.get("z", BOX_Z_M)
    sx = float(spec.get("length_m", spec.get("size_m", BOX_SIZE_M)))
    sy = float(spec.get("width_m", BOX_NORTH_M))
    sz = float(spec.get("height_m", BOX_HEIGHT_M))
    tape_w = 0.12
    return f"""
    <model name="target_box">
      <pose>{x} {y} {z} 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <pose>0 0 {sz / 2:.3f} 0 0 0</pose>
        <collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="body">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>0.80 0.55 0.28 1</ambient><diffuse>0.85 0.60 0.30 1</diffuse><specular>0.05 0.05 0.05 1</specular></material>
        </visual>
        <visual name="tape_long">
          <pose>0 0 {sz / 2 + 0.002:.3f} 0 0 0</pose>
          <geometry><box><size>{sx + 0.004} {tape_w} 0.004</size></box></geometry>
          <material><ambient>0.15 0.12 0.10 1</ambient><diffuse>0.18 0.15 0.12 1</diffuse></material>
        </visual>
        <visual name="tape_cross">
          <pose>0 0 {sz / 2 + 0.003:.3f} 0 0 0</pose>
          <geometry><box><size>{tape_w} {sy + 0.004} 0.004</size></box></geometry>
          <material><ambient>0.15 0.12 0.10 1</ambient><diffuse>0.18 0.15 0.12 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def inject_into_world(world_xml: str, spec: dict) -> str:
    """Insert (or replace) target_box in an SDF world string."""
    xml = world_xml
    if '<model name="target_box">' in xml:
        start = xml.index('<model name="target_box">')
        end = xml.index("</model>", start) + len("</model>")
        xml = xml[:start] + xml[end:]
    marker = "</world>"
    idx = xml.rfind(marker)
    if idx < 0:
        raise ValueError("world XML has no </world>")
    return xml[:idx] + sdf_model(spec) + "\n  " + xml[idx:]


def inject_from_launch(world_xml: str, launch_path: str | Path) -> str:
    path = Path(launch_path)
    if not path.is_file():
        return world_xml
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return world_xml
    spec = data.get("target_box")
    if not spec:
        return world_xml
    return inject_into_world(world_xml, spec)
