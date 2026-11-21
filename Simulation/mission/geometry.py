"""Local geometry for the survey planner. No ROS, no network."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Ring = List[Point]

METRES_PER_DEG_LAT = 110540.0
METRES_PER_DEG_LON_EQUATOR = 111320.0


class PlanError(Exception):
    """A plan cannot be built. The message says why and what to change."""


def parse_lon_lat(point, order: str) -> Point:
    """Return (lon, lat). Lists follow coordinate_order. Dicts use lat/lon keys."""
    if isinstance(point, dict):
        lon = point.get("lon", point.get("lng", point.get("longitude")))
        lat = point.get("lat", point.get("latitude"))
        if lon is None or lat is None:
            raise PlanError(
                "Each coordinate needs lon and lat. "
                "Use [lon, lat] or {\"lon\": ..., \"lat\": ...}."
            )
        return float(lon), float(lat)
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise PlanError(
            "Each coordinate must be [lon, lat] (or [lat, lon] if coordinate_order is latlon)."
        )
    a, b = float(point[0]), float(point[1])
    if order == "latlon":
        return b, a
    return a, b


def ring_from_coordinates(raw: Sequence, order: str = "lonlat") -> List[Point]:
    """Closed-ring input is opened. At least 4 distinct corners are required."""
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise PlanError(
            "coordinates is empty. Give at least 4 corners, in order around the area."
        )
    points = [parse_lon_lat(p, order) for p in raw]
    cleaned: List[Point] = []
    for lon, lat in points:
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise PlanError(
                f"Coordinate ({lon}, {lat}) is outside lon [-180, 180] and lat [-90, 90]."
            )
        if cleaned and _close(cleaned[-1], (lon, lat), 1e-12):
            continue
        cleaned.append((lon, lat))
    if len(cleaned) >= 2 and _close(cleaned[0], cleaned[-1], 1e-12):
        cleaned.pop()
    if len(cleaned) < 4:
        raise PlanError(
            f"Need at least 4 coordinates; got {len(cleaned)}. "
            "Add corners until the area is a polygon with 4 or more points."
        )
    return cleaned


def project(lonlat: Sequence[Point]):
    """Equirectangular metres around the vertex average. Returns (to_xy, to_lonlat)."""
    lon0 = sum(p[0] for p in lonlat) / len(lonlat)
    lat0 = sum(p[1] for p in lonlat) / len(lonlat)
    scale_x = METRES_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    if abs(scale_x) < 1e-6:
        raise PlanError("Polygon is too close to a pole to project. Move the area.")

    def to_xy(lon: float, lat: float) -> Point:
        return ((lon - lon0) * scale_x, (lat - lat0) * METRES_PER_DEG_LAT)

    def to_lonlat(x: float, y: float) -> Point:
        return (lon0 + x / scale_x, lat0 + y / METRES_PER_DEG_LAT)

    return to_xy, to_lonlat


def shoelace(ring: Sequence[Point]) -> float:
    area = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def ensure_ccw(ring: Sequence[Point]) -> Ring:
    pts = list(ring)
    if shoelace(pts) < 0:
        pts.reverse()
    return pts


def clean_ring(ring: Sequence[Point], eps: float = 1e-6) -> Ring:
    out: Ring = []
    for point in ring:
        if not out or math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) > eps:
            out.append((float(point[0]), float(point[1])))
    if len(out) >= 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps:
        out.pop()
    # Drop a vertex that sits on the edge between its neighbours.
    if len(out) >= 4:
        kept: Ring = []
        n = len(out)
        for i in range(n):
            a, b, c = out[(i - 1) % n], out[i], out[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) > eps:
                kept.append(b)
        if len(kept) >= 3:
            out = kept
    return out


def self_intersects(ring: Sequence[Point]) -> bool:
    n = len(ring)
    if n < 4:
        return False
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) % n <= 1 or (i == 0 and j == n - 1):
                continue
            c, d = ring[j], ring[(j + 1) % n]
            if _segments_cross(a, b, c, d):
                return True
    return False


def point_in_ring(point: Point, ring: Sequence[Point]) -> bool:
    """Boundary counts as inside."""
    if _on_boundary(point, ring):
        return True
    x, y = point
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
            if x_cross > x:
                inside = not inside
    return inside


def ring_inside(inner: Sequence[Point], outer: Sequence[Point]) -> bool:
    return all(point_in_ring(p, outer) for p in inner)


def longest_edge_angle_deg(ring: Sequence[Point]) -> float:
    best_len = -1.0
    best = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length
            best = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return best


def rotate(ring: Sequence[Point], degrees: float) -> Ring:
    radians = math.radians(degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in ring]


def centroid(ring: Sequence[Point]) -> Point:
    area = shoelace(ring)
    if abs(area) < 1e-12:
        return (
            sum(p[0] for p in ring) / len(ring),
            sum(p[1] for p in ring) / len(ring),
        )
    cx = cy = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    cx /= 6.0 * area
    cy /= 6.0 * area
    return cx, cy


def bounds(ring: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def clip_x_le(ring: Sequence[Point], x_bound: float) -> Ring:
    return _clip_halfplane(ring, lambda p: p[0] <= x_bound + 1e-9, lambda a, b: _hit_x(a, b, x_bound))


def clip_x_ge(ring: Sequence[Point], x_bound: float) -> Ring:
    return _clip_halfplane(ring, lambda p: p[0] >= x_bound - 1e-9, lambda a, b: _hit_x(a, b, x_bound))


def area_west_of(ring: Sequence[Point], x_bound: float) -> float:
    clipped = clean_ring(clip_x_le(ring, x_bound))
    if len(clipped) < 3:
        return 0.0
    return abs(shoelace(clipped))


def split_equal_area(ring: Sequence[Point], count: int) -> List[Ring]:
    """Cut a CCW ring into `count` strips of equal area. Strips run along +x."""
    if count < 1:
        raise PlanError("num_patches must be at least 1.")
    poly = ensure_ccw(clean_ring(ring))
    if len(poly) < 3:
        raise PlanError("Polygon has no area. Check that the coordinates are not in a straight line.")
    total = abs(shoelace(poly))
    if total < 1.0:
        raise PlanError(
            f"Area too small: {total:.2f} m². The polygon needs to be larger than 1 m²."
        )
    if count == 1:
        return [poly]
    min_x, _, max_x, _ = bounds(poly)
    cuts = []
    for index in range(1, count):
        target = total * index / count
        lo, hi = min_x, max_x
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if area_west_of(poly, mid) < target:
                lo = mid
            else:
                hi = mid
        cuts.append((lo + hi) / 2.0)
    edges = [min_x - 1.0, *cuts, max_x + 1.0]
    pieces: List[Ring] = []
    for left, right in zip(edges, edges[1:]):
        piece = clean_ring(clip_x_ge(clip_x_le(poly, right), left))
        if len(piece) < 3 or abs(shoelace(piece)) < 1e-4:
            raise PlanError(
                f"Could not cut the polygon into {count} pieces. "
                "The shape may be too thin. Add drones only if each piece can keep some width, "
                "or set sweep_angle_deg to another direction."
            )
        pieces.append(ensure_ccw(piece))
    _check_area_balance(pieces, total)
    return pieces


def horizontal_chords(ring: Sequence[Point], y: float) -> List[Tuple[float, float]]:
    """Inside segments of the horizontal line at y, using the even-odd rule."""
    xs: List[float] = []
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) == (y2 > y):
            continue
        if y2 == y1:
            continue
        xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    xs.sort()
    chords = []
    for i in range(0, len(xs) - 1, 2):
        if xs[i + 1] - xs[i] > 1e-4:
            chords.append((xs[i], xs[i + 1]))
    return chords


def _check_area_balance(pieces: Sequence[Ring], total: float) -> None:
    target = total / len(pieces)
    shares = []
    for piece in pieces:
        share = abs(shoelace(piece)) / target
        shares.append(share)
        if abs(share - 1.0) > 0.02:
            listed = ", ".join(f"{s * 100:.1f}%" for s in shares)
            raise PlanError(
                "Could not divide the polygon into equal areas within ±2% "
                f"(piece shares so far: {listed}). "
                "Use a simpler outline or set sweep_angle_deg."
            )


def _clip_halfplane(
    ring: Sequence[Point],
    inside: Callable[[Point], bool],
    intersect: Callable[[Point, Point], Point],
) -> Ring:
    if not ring:
        return []
    output: Ring = []
    previous = ring[-1]
    previous_inside = inside(previous)
    for current in ring:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _hit_x(a: Point, b: Point, x_bound: float) -> Point:
    if abs(b[0] - a[0]) < 1e-15:
        return (x_bound, a[1])
    t = (x_bound - a[0]) / (b[0] - a[0])
    t = min(1.0, max(0.0, t))
    return (x_bound, a[1] + t * (b[1] - a[1]))


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    d1 = _cross(a, b, c)
    d2 = _cross(a, b, d)
    d3 = _cross(c, d, a)
    d4 = _cross(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_boundary(point: Point, ring: Sequence[Point], eps: float = 1e-6) -> bool:
    n = len(ring)
    for i in range(n):
        if _distance_to_segment(point, ring[i], ring[(i + 1) % n]) <= eps:
            return True
    return False


def _distance_to_segment(point: Point, a: Point, b: Point) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = ((point[0] - a[0]) * vx + (point[1] - a[1]) * vy) / length_sq
    t = min(1.0, max(0.0, t))
    return math.hypot(point[0] - (a[0] + t * vx), point[1] - (a[1] + t * vy))


def _close(a: Point, b: Point, eps: float) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps
