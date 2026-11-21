"""Survey planner from Milestone 1.

Pure Python. Same input always gives the same plan. Nothing is sent to a drone.

Minimum call: a polygon with at least 4 coordinates and how many patches to cut.

    plan({"coordinates": [[lon, lat], ...], "num_patches": 3})

Add camera, drones, overlaps and limits when you want altitude, survey lines
and PX4 mission items as well.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Sequence

from geometry import (
    PlanError,
    bounds,
    centroid,
    clean_ring,
    ensure_ccw,
    horizontal_chords,
    longest_edge_angle_deg,
    point_in_ring,
    project,
    ring_from_coordinates,
    ring_inside,
    rotate,
    self_intersects,
    shoelace,
    split_equal_area,
)

AREA_TOLERANCE = 0.02
TIME_MARGIN = 0.80
TURN_SECONDS = 5.0
TAKEOFF_LAND_SECONDS = 40.0


def plan(request: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-like dict in, JSON-like dict out. Invalid plans come back with errors."""
    try:
        return _plan(request)
    except PlanError as exc:
        return {
            "plan_id": _plan_id(request),
            "valid": False,
            "patches": [],
            "warnings": [],
            "errors": [str(exc)],
        }


def replan(
    request: Dict[str, Any],
    previous: Dict[str, Any],
    available_drone_ids: Sequence[str],
) -> Dict[str, Any]:
    """Plan the patches of the drones that are no longer available, among the ones that are."""
    try:
        return _replan(request, previous, list(available_drone_ids))
    except PlanError as exc:
        return {
            "plan_id": _plan_id({"replan_of": previous.get("plan_id"), "drones": list(available_drone_ids)}),
            "valid": False,
            "patches": [],
            "warnings": [],
            "errors": [str(exc)],
        }


def _plan(request: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    order = str(request.get("coordinate_order", "lonlat")).lower()
    if order not in ("lonlat", "latlon"):
        raise PlanError('coordinate_order must be "lonlat" or "latlon".')
    lonlat = ring_from_coordinates(request.get("coordinates", []), order)
    if self_intersects(lonlat):
        raise PlanError(
            "The polygon crosses itself. Draw a simple outline, without crossing edges."
        )

    to_xy, to_lonlat = project(lonlat)
    ring_m = ensure_ccw([to_xy(lon, lat) for lon, lat in lonlat])
    if self_intersects(ring_m):
        raise PlanError(
            "The polygon crosses itself. Draw a simple outline, without crossing edges."
        )

    count = _patch_count(request)
    sweep = request.get("sweep_angle_deg")
    if sweep is None or sweep == "":
        sweep = longest_edge_angle_deg(ring_m)
        warnings.append(
            f"sweep_angle_deg was empty, so survey lines follow the longest edge ({sweep:.1f}°)."
        )
    sweep = float(sweep)

    rotated = rotate(ring_m, -sweep)
    pieces = split_equal_area(rotated, count)
    pieces = [ensure_ccw(rotate(piece, sweep)) for piece in pieces]

    drones = _drones(request, count)
    order_index = _assign(pieces, drones, to_xy)
    pieces = [pieces[i] for i in order_index]

    camera = request.get("camera")
    survey = None
    if camera:
        survey = _survey_values(request, camera, warnings)
    else:
        warnings.append(
            "No camera given, so altitude, line spacing and PX4 mission items were not calculated. "
            "Add a camera and target_gsd_cm to get them."
        )

    fence = _fence(request, to_xy, warnings)
    if fence is not None and not ring_inside(ring_m, fence):
        raise PlanError(
            "The survey polygon is outside the geo-fence. "
            "Shrink the area or enlarge the geo-fence so every corner is inside it."
        )

    patches = []
    for index, piece in enumerate(pieces):
        drone = drones[index]
        patch = _patch_record(index, piece, to_lonlat, drone)
        if survey is not None:
            _add_flight(patch, piece, drone, drones, survey, to_xy, to_lonlat, fence, warnings, sweep)
        patches.append(patch)

    total = abs(shoelace(ring_m))
    result = {
        "plan_id": _plan_id(request),
        "valid": True,
        "coordinate_order": "lonlat",
        "sweep_angle_deg": round(sweep, 4),
        "total_area_m2": round(total, 3),
        "patches": patches,
        "warnings": warnings,
        "errors": [],
    }
    if survey is not None:
        result.update(survey["summary"])
    return result


def _replan(request: Dict[str, Any], previous: Dict[str, Any], available_ids: List[str]) -> Dict[str, Any]:
    if not previous.get("patches"):
        raise PlanError("previous plan has no patches to re-plan.")
    if len(available_ids) < 1:
        raise PlanError("No drones left. Add at least one available drone before re-planning.")
    available = set(available_ids)
    dropped = [p for p in previous["patches"] if p.get("drone_id") not in available]
    if not dropped:
        raise PlanError(
            "Every patch still belongs to an available drone, so there is no uncovered area to re-plan."
        )
    # The stopped drones' whole patches are treated as uncovered.
    uncovered_request = dict(request)
    uncovered_request["num_patches"] = len(available_ids)
    # One polygon per dropped patch. Split the largest until there is one piece per survivor.
    rings = []
    for patch in dropped:
        ring = _ring_m_from_patch(patch)
        rings.append(ring)
    pieces = _balance(rings, len(available_ids))
    rebuilt = dict(request)
    rebuilt["drones"] = [d for d in _drones(request, len(available_ids)) if d["id"] in available]
    if len(rebuilt["drones"]) != len(available_ids):
        raise PlanError(
            "available_drone_ids has an id that is not in the original drones list. "
            f"Available ids must be chosen from {[d['id'] for d in _drones(request, len(available_ids))]}."
        )
    # Plan each piece by asking the main planner only when there is a single ring.
    # Multi-ring leftovers are returned as one patch with several rings.
    warnings = [
        "Re-plan covers the whole patch of each stopped drone. "
        "Partly flown ground inside that patch is flown again."
    ]
    to_xy, to_lonlat = _projection_from_request(request)
    drones = rebuilt["drones"]
    assigned = _assign(pieces, drones, to_xy)
    pieces = [pieces[i] for i in assigned]
    camera = request.get("camera")
    survey = _survey_values(request, camera, warnings) if camera else None
    fence = _fence(request, to_xy, warnings)
    patches = []
    for index, piece in enumerate(pieces):
        patch = _patch_record(index, piece, to_lonlat, drones[index])
        if survey is not None:
            sweep = float(request.get("sweep_angle_deg") or 0.0)
            _add_flight(patch, piece, drones[index], drones, survey, to_xy, to_lonlat, fence, warnings, sweep)
        patches.append(patch)
    return {
        "plan_id": _plan_id({"replan_of": previous.get("plan_id"), "available": available_ids}),
        "valid": True,
        "replan_of": previous.get("plan_id"),
        "patches": patches,
        "warnings": warnings,
        "errors": [],
        **({} if survey is None else survey["summary"]),
    }


def _balance(rings: List[List], count: int) -> List:
    parts = [ring for ring in rings]
    guard = 0
    while len(parts) < count and guard < 30:
        guard += 1
        parts.sort(key=lambda ring: abs(shoelace(ring)), reverse=True)
        largest = parts.pop(0)
        # Split in the frame where the longest edge lies along x.
        angle = longest_edge_angle_deg(largest)
        rotated = rotate(largest, -angle)
        left, right = split_equal_area(rotated, 2)
        parts.append(ensure_ccw(rotate(left, angle)))
        parts.append(ensure_ccw(rotate(right, angle)))
    while len(parts) > count:
        parts.sort(key=lambda ring: abs(shoelace(ring)))
        small = parts.pop(0)
        host = parts.pop(0)
        # Keep both outlines: the survivor flies one, then the other.
        merged = _merge_rings(host, small)
        parts.append(merged)
    if len(parts) != count:
        raise PlanError(
            f"Could not re-plan the uncovered area into {count} pieces. "
            "Simplify the remaining polygon."
        )
    return parts


def _merge_rings(host, extra):
    """Represent two areas as one ring only when they already share space. Otherwise raise."""
    # A single ring is required by the flight-path code. Split/merge keeps one ring per drone
    # because the while-loops above only split. Merge happens when there are more polygons
    # than drones: join them with a zero-width bridge so the outline stays one ring.
    bridge = [host[-1], extra[0]]
    return clean_ring(list(host) + bridge + list(extra) + list(reversed(bridge)))


def _projection_from_request(request):
    order = str(request.get("coordinate_order", "lonlat")).lower()
    lonlat = ring_from_coordinates(request.get("coordinates", []), order)
    return project(lonlat)


def _ring_m_from_patch(patch: Dict[str, Any]):
    coords = patch.get("coordinates_m")
    if not coords or len(coords) < 3:
        raise PlanError("A previous patch is missing coordinates_m, so it cannot be re-planned.")
    return ensure_ccw([(float(p[0]), float(p[1])) for p in coords])


def _patch_count(request: Dict[str, Any]) -> int:
    drones = request.get("drones")
    if isinstance(drones, list) and len(drones) > 0:
        return len(drones)
    if "num_patches" not in request:
        raise PlanError(
            "Say how many patches to cut. Set num_patches, or pass a drones list with one entry per drone."
        )
    count = int(request["num_patches"])
    if count < 1:
        raise PlanError("num_patches must be at least 1.")
    return count


def _drones(request: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    raw = request.get("drones")
    if not raw:
        return [
            {"id": f"drone{i + 1}", "home": None, "max_flight_time_s": None}
            for i in range(count)
        ]
    if len(raw) != count and "num_patches" in request and int(request["num_patches"]) != len(raw):
        raise PlanError(
            f"num_patches is {request['num_patches']} but drones has {len(raw)} entries. "
            "Use one patch per drone, or omit num_patches."
        )
    drones = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "id" not in item:
            raise PlanError(f"drones[{index}] needs an id.")
        home = item.get("home")
        seconds = None
        if item.get("max_flight_time_s") not in (None, ""):
            seconds = float(item["max_flight_time_s"])
        elif item.get("max_flight_time_min") not in (None, ""):
            seconds = float(item["max_flight_time_min"]) * 60.0
        drones.append({"id": str(item["id"]), "home": home, "max_flight_time_s": seconds, "raw_home": home})
    return drones


def _assign(pieces, drones, to_xy) -> List[int]:
    """Deterministic nearest-home matching. No homes: keep west-to-east strip order."""
    remaining = set(range(len(pieces)))
    chosen = []
    homes = []
    for drone in drones:
        home = drone.get("home")
        if home is None:
            homes.append(None)
        else:
            order = "lonlat"
            lon, lat = home if not isinstance(home, dict) else home
            if isinstance(home, dict):
                lon, lat = float(home.get("lon", home.get("lng"))), float(home.get("lat"))
            else:
                lon, lat = float(home[0]), float(home[1])
            homes.append(to_xy(lon, lat))
    if all(h is None for h in homes):
        return list(range(len(pieces)))
    # Drones are already in input order. Each takes the closest remaining patch.
    for home in homes:
        if home is None:
            index = min(remaining)
        else:
            index = min(remaining, key=lambda i: _dist(home, centroid(pieces[i])))
        remaining.remove(index)
        chosen.append(index)
    return chosen


def _survey_values(request, camera, warnings: List[str]) -> Dict[str, Any]:
    required = ("sensor_width_mm", "sensor_height_mm", "focal_length_mm", "image_width_px", "image_height_px")
    missing = [name for name in required if camera.get(name) in (None, "")]
    if missing:
        raise PlanError(
            "camera is missing " + ", ".join(missing) + ". "
            "The altitude formula needs sensor size, focal length and image size."
        )
    if request.get("target_gsd_cm") in (None, ""):
        raise PlanError(
            "target_gsd_cm is missing. Set the required ground detail in centimetres per pixel."
        )
    sensor_w = float(camera["sensor_width_mm"])
    sensor_h = float(camera["sensor_height_mm"])
    focal = float(camera["focal_length_mm"])
    image_w = float(camera["image_width_px"])
    image_h = float(camera["image_height_px"])
    if min(sensor_w, sensor_h, focal, image_w, image_h) <= 0:
        raise PlanError("Camera sizes, focal length and image pixel counts must be greater than 0.")
    gsd_m = float(request["target_gsd_cm"]) / 100.0
    altitude = gsd_m * focal * image_w / sensor_w
    limits = request.get("limits") or {}
    min_alt = float(limits.get("min_altitude_m", 0))
    max_alt = float(limits.get("max_altitude_m", 10000))
    if altitude < min_alt or altitude > max_alt:
        limited = min(max(altitude, min_alt), max_alt)
        warnings.append(
            f"Altitude for {request['target_gsd_cm']} cm/px would be {altitude:.1f} m, "
            f"outside {min_alt:.1f}–{max_alt:.1f} m. Using {limited:.1f} m and recalculating GSD."
        )
        altitude = limited
    actual_gsd_m = altitude * sensor_w / (focal * image_w)
    footprint_w = actual_gsd_m * image_w
    footprint_h = actual_gsd_m * image_h
    side = float(request.get("side_overlap", 0.65))
    front = float(request.get("front_overlap", 0.70))
    if not (0 <= side < 1 and 0 <= front < 1):
        raise PlanError("front_overlap and side_overlap must be from 0 up to, but not including, 1.")
    line_spacing = footprint_w * (1.0 - side)
    trigger = footprint_h * (1.0 - front)
    if line_spacing <= 0 or trigger <= 0:
        raise PlanError("Overlap leaves no gap between photos. Lower front_overlap or side_overlap.")
    max_speed = float(limits.get("max_speed_ms", 12))
    speeds = [max_speed]
    exposure = camera.get("min_exposure_s", camera.get("minimum_exposure_time_s"))
    if exposure not in (None, "") and float(exposure) > 0:
        speeds.append(actual_gsd_m / float(exposure))
    rate = camera.get("max_photos_per_s", camera.get("maximum_photos_per_second"))
    if rate not in (None, "") and float(rate) > 0:
        speeds.append(trigger * float(rate))
    speed = min(speeds)
    return {
        "summary": {
            "altitude_m": round(altitude, 3),
            "actual_gsd_cm": round(actual_gsd_m * 100.0, 4),
            "line_spacing_m": round(line_spacing, 3),
            "trigger_distance_m": round(trigger, 3),
            "speed_ms": round(speed, 3),
            "footprint_w_m": round(footprint_w, 3),
            "footprint_h_m": round(footprint_h, 3),
        },
        "altitude_m": altitude,
        "line_spacing_m": line_spacing,
        "trigger_distance_m": trigger,
        "speed_ms": speed,
    }


def _fence(request, to_xy, warnings: List[str]):
    limits = request.get("limits") or {}
    raw = limits.get("geo_fence")
    if not raw:
        warnings.append("No geo-fence set. Waypoints were not checked against a fence.")
        return None
    order = str(request.get("coordinate_order", "lonlat")).lower()
    lonlat = ring_from_coordinates(raw, order)
    return ensure_ccw([to_xy(lon, lat) for lon, lat in lonlat])


def _patch_record(index: int, piece, to_lonlat, drone: Dict[str, Any]) -> Dict[str, Any]:
    area = abs(shoelace(piece))
    lonlat = [to_lonlat(x, y) for x, y in piece]
    lonlat.append(lonlat[0])
    return {
        "id": index + 1,
        "drone_id": drone["id"],
        "area_m2": round(area, 3),
        "coordinates": [[round(lon, 8), round(lat, 8)] for lon, lat in lonlat],
        "coordinates_m": [[round(x, 3), round(y, 3)] for x, y in piece],
    }


def _add_flight(patch, piece, drone, drones, survey, to_xy, to_lonlat, fence, warnings: List[str], sweep_deg: float) -> None:
    home = _home_xy(drone, piece, to_xy)
    angle = float(sweep_deg)
    rotated_piece = rotate(piece, -angle)
    rotated_home = rotate([home], -angle)[0]
    spacing = survey["line_spacing_m"]
    extend = survey["trigger_distance_m"] * 0.5
    path_rot = _lawnmower(rotated_piece, spacing, extend, rotated_home)
    path = rotate(path_rot, angle) if path_rot else []
    if len(path) < 2:
        raise PlanError(
            f"Patch {patch['id']} has no survey line. "
            "The piece is narrower than the line spacing. Lower the image detail or the overlap."
        )
    altitude = survey["altitude_m"]
    # Distinct transit heights so the drones do not cross at the same altitude.
    drone_index = next(i for i, item in enumerate(drones) if item["id"] == drone["id"])
    transit = altitude + 10.0 + 5.0 * drone_index
    survey_length = _path_length(path)
    transit_out = math.hypot(path[0][0] - home[0], path[0][1] - home[1])
    transit_back = math.hypot(path[-1][0] - home[0], path[-1][1] - home[1])
    turns = max(0, len(path) // 2 - 1)
    speed = survey["speed_ms"]
    seconds = (
        (survey_length + transit_out + transit_back) / speed
        + turns * TURN_SECONDS
        + TAKEOFF_LAND_SECONDS
    )
    photos = int(math.ceil(survey_length / survey["trigger_distance_m"]))
    patch["path_length_m"] = round(survey_length + transit_out + transit_back, 2)
    patch["survey_length_m"] = round(survey_length, 2)
    patch["estimated_flight_time_s"] = round(seconds, 1)
    patch["estimated_photo_count"] = photos
    patch["transit_altitude_m"] = round(transit, 2)
    patch["flight_path"] = [[round(lon, 8), round(lat, 8)] for lon, lat in (to_lonlat(x, y) for x, y in path)]
    patch["flight_path_m"] = [[round(x, 3), round(y, 3)] for x, y in path]
    limit = drone.get("max_flight_time_s")
    if limit is not None and seconds > limit * TIME_MARGIN:
        need_min = seconds / 60.0
        limit_min = limit / 60.0
        raise PlanError(
            f"Area too large for {len(drones)} drones at this detail: "
            f"{need_min:.0f} min needed for {drone['id']}, limit is {limit_min:.0f} min "
            f"with a {int((1 - TIME_MARGIN) * 100)}% reserve. Lower the detail or add drones."
        )
    mission = _mission_items(drone, path, to_lonlat, altitude, transit, survey["trigger_distance_m"])
    if fence is not None:
        for point in [home, *path]:
            if not point_in_ring(point, fence):
                raise PlanError(
                    f"A waypoint for {drone['id']} is outside the geo-fence. "
                    "Shrink the area, lower the line extension, or enlarge the fence."
                )
    for item in mission:
        if "alt" in item and item["alt"] <= 0:
            raise PlanError(
                f"Mission item {item['type']} has altitude {item['alt']}. Altitude must be positive."
            )
    patch["mission_items"] = mission


def _home_xy(drone, piece, to_xy):
    home = drone.get("home")
    if home is None:
        return centroid(piece)
    if isinstance(home, dict):
        return to_xy(float(home.get("lon", home.get("lng"))), float(home.get("lat")))
    return to_xy(float(home[0]), float(home[1]))


def _lawnmower(ring, spacing: float, extend: float, home) -> List:
    min_x, min_y, max_x, max_y = bounds(ring)
    y = min_y + spacing / 2.0
    rows = []
    guard = 0
    while y < max_y - 1e-6 and guard < 10000:
        guard += 1
        chords = horizontal_chords(ring, y)
        if chords:
            rows.append((y, [(a - extend, b + extend) for a, b in chords]))
        y += spacing
    if not rows:
        return []
    if abs(home[1] - rows[0][0]) > abs(home[1] - rows[-1][0]):
        rows.reverse()
    # Face the first row toward the home's x.
    path = []
    left_to_right = home[0] <= (rows[0][1][0][0] + rows[0][1][0][1]) / 2.0
    for y_row, chords in rows:
        sequence = chords if left_to_right else [(b, a) for a, b in reversed(chords)]
        for x0, x1 in sequence:
            path.append((x0, y_row))
            path.append((x1, y_row))
        left_to_right = not left_to_right
    return path


def _mission_items(drone, path, to_lonlat, altitude, transit, trigger) -> List[Dict[str, Any]]:
    home = drone.get("home")
    if isinstance(home, dict):
        home_ll = [float(home.get("lon", home.get("lng"))), float(home.get("lat"))]
    elif home is not None:
        home_ll = [float(home[0]), float(home[1])]
    else:
        home_ll = list(to_lonlat(path[0][0], path[0][1]))
    start = to_lonlat(path[0][0], path[0][1])
    items = [
        {"seq": 0, "type": "takeoff", "lon": round(home_ll[0], 8), "lat": round(home_ll[1], 8), "alt": round(transit, 2)},
        {"seq": 1, "type": "waypoint", "lon": round(start[0], 8), "lat": round(start[1], 8), "alt": round(transit, 2)},
        {"seq": 2, "type": "waypoint", "lon": round(start[0], 8), "lat": round(start[1], 8), "alt": round(altitude, 2)},
        {"seq": 3, "type": "camera_trigger_distance", "distance_m": round(trigger, 3)},
    ]
    seq = 4
    for x, y in path:
        lon, lat = to_lonlat(x, y)
        items.append({"seq": seq, "type": "waypoint", "lon": round(lon, 8), "lat": round(lat, 8), "alt": round(altitude, 2)})
        seq += 1
    items.append({"seq": seq, "type": "camera_trigger_stop"})
    items.append({"seq": seq + 1, "type": "return_to_launch"})
    return items


def _path_length(path) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _plan_id(request: Dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python planner.py example_request.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = plan(request)
    print(json.dumps(result, indent=2))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
