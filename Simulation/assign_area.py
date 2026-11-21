"""
Area assignment for the PX4 farm swarm with Claude Haiku. PLAN ONLY.

Each mission uses ONE survey polygon (3..6 corners). Haiku splits that
polygon into contiguous row blocks, one per drone. Code then plots the
selected area and which drone flies which patch.

Input : corner coordinates in metres (local x, y).
Output: drone patches, lawn-mower lines, and a PNG next to this file.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import target_box

# ═══ RUN SETTINGS ══════════════════════════════════════════════════════════
MODEL = "claude-haiku-4-5"
RETRIES = 1
MAX_TOKENS = 800
HFOV_DEG = 66
API_URL = "https://api.anthropic.com/v1/messages"
HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "area_assignment_plots"
FIELD_EAST_M = 90.0
FIELD_NORTH_M = 140.0
FARM_MARGIN_M = 8.0

# ═══ Drones (ids match survey_launch / PX4 drone1..) ═══════════════════════
DRONES = [
    {"id": "drone1", "battery_pct": 95, "max_flight_min": 20},
    {"id": "drone2", "battery_pct": 90, "max_flight_min": 20},
    {"id": "drone3", "battery_pct": 92, "max_flight_min": 20},
    {"id": "drone4", "battery_pct": 88, "max_flight_min": 20},
    {"id": "drone5", "battery_pct": 93, "max_flight_min": 20},
    {"id": "drone6", "battery_pct": 91, "max_flight_min": 20},
]
COLORS = {
    "drone1": "#e6194b", "drone2": "#3cb44b", "drone3": "#4363d8",
    "drone4": "#f58231", "drone5": "#911eb4", "drone6": "#42d4f4",
}

# Same 12 shapes as the Haiku POC (3, 4, 5 and 6 corners).
CASES = [
    {"name": "tri_right", "coords": [(0, 0), (400, 0), (0, 300)]},
    {"name": "tri_isosceles", "coords": [(0, 0), (450, 0), (225, 350)]},
    {"name": "tri_obtuse", "coords": [(0, 0), (500, 80), (120, 260)]},
    {"name": "quad_rectangle", "coords": [(0, 0), (400, 0), (400, 250), (0, 250)]},
    {"name": "quad_trapezoid", "coords": [(0, 0), (450, 0), (330, 280), (90, 280)]},
    {"name": "quad_irregular", "coords": [(30, 0), (380, 60), (420, 330), (0, 260)]},
    {"name": "pent_convex", "coords": [(200, 0), (400, 140), (320, 360), (80, 360), (0, 140)]},
    {"name": "pent_house", "coords": [(0, 0), (400, 0), (400, 220), (200, 340), (0, 220)]},
    {"name": "pent_notch", "coords": [(0, 0), (420, 0), (420, 300), (210, 140), (0, 300)]},
    {"name": "hex_regular", "coords": [(100, 0), (300, 0), (400, 173), (300, 346), (100, 346), (0, 173)]},
    {"name": "hex_L_shape", "coords": [(0, 0), (400, 0), (400, 150), (180, 150), (180, 320), (0, 320)]},
    {"name": "hex_field", "coords": [(0, 40), (150, 0), (380, 30), (440, 220), (260, 330), (40, 280)]},
]


def drones_for_count(count: int) -> list:
    count = max(1, int(count))
    if count <= len(DRONES):
        return [dict(d) for d in DRONES[:count]]
    extra = []
    for i in range(len(DRONES), count):
        extra.append({"id": f"drone{i + 1}", "battery_pct": 80, "max_flight_min": 20})
        COLORS.setdefault(f"drone{i + 1}", "#808080")
    return [dict(d) for d in DRONES] + extra


def pick_one_area(name: str | None = None, rng: random.Random | None = None) -> dict:
    """One catalog polygon. Name wins; otherwise a random 3–6 point case."""
    if name:
        for case in CASES:
            if case["name"] == name:
                return {"name": case["name"], "coords": list(case["coords"])}
        raise ValueError(f"Unknown area {name!r}. Choose one of {[c['name'] for c in CASES]}.")
    env_name = os.environ.get("POLLINATION_AREA_CASE", "").strip()
    if env_name:
        return pick_one_area(env_name)
    chosen = (rng or random).choice(CASES)
    return {"name": chosen["name"], "coords": list(chosen["coords"])}


def fit_polygon_to_farm(
    coords,
    east_half: float = FIELD_EAST_M,
    north_half: float = FIELD_NORTH_M,
    margin: float = FARM_MARGIN_M,
):
    """Scale and centre a 3–6 point shape so it sits inside the farm rectangle."""
    xs = [float(p[0]) for p in coords]
    ys = [float(p[1]) for p in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    target_w = 2.0 * max(1.0, east_half - margin)
    target_h = 2.0 * max(1.0, north_half - margin)
    scale = min(target_w / width, target_h / height)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return [((x - cx) * scale, (y - cy) * scale) for x, y in zip(xs, ys)]


# ═══ Geometry helpers ══════════════════════════════════════════════════════
def area_signed(p):
    return 0.5 * sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p))) * -1


def area(p):
    return abs(area_signed(p))


def rotate(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(x * c - y * s, x * s + y * c) for x, y in p]


def clip_half(p, y_lim, keep_above):
    inside = (lambda q: q[1] >= y_lim) if keep_above else (lambda q: q[1] <= y_lim)
    out = []
    for i in range(len(p)):
        a, b = p[i - 1], p[i]
        if inside(b):
            if not inside(a):
                out.append(_cross(a, b, y_lim))
            out.append(b)
        elif inside(a):
            out.append(_cross(a, b, y_lim))
    return out


def _cross(a, b, y):
    t = (y - a[1]) / (b[1] - a[1]) if abs(b[1] - a[1]) > 1e-12 else 0.0
    return (a[0] + t * (b[0] - a[0]), y)


def clip_band(p, y0, y1):
    return clip_half(clip_half(p, y0, True), y1, False)


def scanline(p, y):
    xs = []
    for i in range(len(p)):
        (x0, y0), (x1, y1) = p[i - 1], p[i]
        if (y0 <= y < y1) or (y1 <= y < y0):
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def camera_swath(hfov_deg, target_m=0.22, yolo_px=640, min_px=12, overlap=0.2):
    t = math.tan(math.radians(hfov_deg) / 2)
    alt = (target_m * yolo_px) / (min_px * 2 * t)
    return alt, 2 * alt * t * (1 - overlap)


def make_rows(coords, hfov_deg, speed_mps=5, min_rows=1):
    n = len(coords)
    i = max(range(n), key=lambda k: math.dist(coords[k], coords[(k + 1) % n]))
    (x0, y0), (x1, y1) = coords[i], coords[(i + 1) % n]
    ang = math.atan2(y1 - y0, x1 - x0)
    rp = rotate(coords, -ang)
    alt, swath = camera_swath(hfov_deg)
    ymin, ymax = min(y for _, y in rp), max(y for _, y in rp)
    height = max(ymax - ymin, 1e-6)
    if min_rows > 1:
        swath = min(swath, height / float(min_rows))
    rows, k, y = [], 1, ymin
    while y < ymax - 1e-6:
        y_top = min(y + swath, ymax)
        piece = clip_band(rp, y, y_top)
        a = area(piece) if len(piece) >= 3 else 0
        if a > 1:
            length = a / (y_top - y)
            rows.append({
                "row": k, "y0": y, "y1": y_top, "length_m": round(length, 1),
                "minutes": round((length + swath) / speed_mps / 60, 2),
            })
            k += 1
        y = y_top
    return {"angle": ang, "rot_poly": rp, "rows": rows,
            "altitude_m": round(alt, 1), "swath_m": round(swath, 1)}


def usable_minutes(d, reserve=0.2):
    return round(d["max_flight_min"] * d["battery_pct"] / 100 * (1 - reserve), 1)


# ═══ Haiku ═════════════════════════════════════════════════════════════════
SYSTEM = (
    "You are a drone swarm mission planner. A survey area is cut into parallel rows numbered 1..N "
    "(neighbouring numbers are neighbouring rows). Assign the rows to the drones. Rules: "
    "1) every row exactly once - no gaps, no overlap; "
    "2) each drone gets ONE contiguous block of rows, given as first_row and last_row (inclusive); "
    "3) the sum of a drone's row minutes must not exceed its usable_min; "
    "4) use ALL drones and split the total minutes in proportion to each drone's usable_min "
    "(more battery = more rows). "
    "Each drone's target_min is its fair share; end its block where cumulative_minutes is closest to "
    "the running sum of target_min, without going over usable_min. Reply with JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {"type": "array", "items": {
            "type": "object",
            "properties": {"drone_id": {"type": "string"},
                           "first_row": {"type": "integer"},
                           "last_row": {"type": "integer"}},
            "required": ["drone_id", "first_row", "last_row"]}},
    },
    "required": ["assignments"],
}


def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    here = Path(__file__).resolve().parent
    for path in (here / ".anthropic_key", here / "mission" / ".anthropic_key"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    for parent in here.parents:
        cand = parent / "POC" / "Qwen 9b" / "Assign area by Qwen" / ".anthropic_key"
        if cand.is_file():
            text = cand.read_text(encoding="utf-8").strip()
            if text:
                return text
    raise RuntimeError(
        "Set ANTHROPIC_API_KEY, or save the key in .anthropic_key next to assign_area.py."
    )


def claude_request(user, max_tokens, with_tool):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": user}],
    }
    if with_tool:
        body["system"] = SYSTEM
        body["tools"] = [{
            "name": "submit_assignments",
            "description": "Submit one contiguous row block per drone.",
            "input_schema": SCHEMA,
        }]
        body["tool_choice"] = {"type": "tool", "name": "submit_assignments"}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(), method="POST",
        headers={"x-api-key": api_key(), "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Claude API {e.code}: {detail}") from None
    elapsed = round(time.perf_counter() - t, 2)
    return data, elapsed


def llm_call(user):
    data, elapsed = claude_request(user, MAX_TOKENS, with_tool=True)
    usage = data.get("usage") or {}
    text, blocks = [], None
    for part in data.get("content") or []:
        if part.get("type") == "tool_use":
            blocks = (part.get("input") or {}).get("assignments")
        elif part.get("type") == "text":
            text.append(part.get("text") or "")
    raw = json.dumps({"assignments": blocks}) if blocks is not None else "".join(text)
    stats = {
        "total_s": elapsed,
        "prompt_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "thinking_chars": 0,
        "done_reason": data.get("stop_reason"),
    }
    return raw, stats


def parse_assignments(raw):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    if isinstance(data, dict) and "assignments" in data:
        return data["assignments"]
    if isinstance(data, list):
        return data
    raise ValueError("no assignments in model reply")


def check(blocks, rows, drones):
    n = len(rows)
    errs, seen = [], []
    cap = {d["id"]: usable_minutes(d) for d in drones}
    for b in blocks:
        f, l = b["first_row"], b["last_row"]
        if b["drone_id"] not in cap:
            errs.append(f"unknown drone {b['drone_id']}")
            continue
        if f > l or f < 1 or l > n:
            errs.append(f"{b['drone_id']} has invalid range {f}-{l} (rows are 1..{n})")
            continue
        seen += range(f, l + 1)
        mins = sum(r["minutes"] for r in rows[f - 1:l])
        if mins > cap[b["drone_id"]] + 1e-6:
            errs.append(f"{b['drone_id']} needs {mins:.1f} min but has {cap[b['drone_id']]} min")
    if sorted(seen) != list(range(1, n + 1)):
        missing = sorted(set(range(1, n + 1)) - set(seen))
        dup = sorted({r for r in seen if seen.count(r) > 1})
        errs.append(f"rows missing {missing[:10]} duplicated {dup[:10]}")
    ids = [b["drone_id"] for b in blocks]
    if n >= len(drones):
        unused = [d["id"] for d in drones if d["id"] not in ids]
        if unused:
            errs.append(f"drones not used: {unused}")
    if len(ids) != len(set(ids)):
        errs.append("a drone appears more than once")
    return errs


def fallback_split(rows, drones):
    """Give every drone a contiguous row block when there are enough rows."""
    n = len(rows)
    if n == 0 or not drones:
        return []
    k = min(len(drones), n)
    out = []
    start = 1
    for i, d in enumerate(drones[:k]):
        left_d = k - i
        left_r = n - start + 1
        width = max(1, left_r // left_d)
        last = start + width - 1
        out.append({"drone_id": d["id"], "first_row": start, "last_row": last})
        start = last + 1
    return out


def assign_with_llm(rows, drones):
    if not rows:
        return [], "fallback", [{"valid": False, "errors": ["no survey rows"], "total_s": 0}]
    cum, c = [], 0.0
    for r in rows:
        c += r["minutes"]
        cum.append(round(c, 2))
    user = json.dumps({
        "total_rows": len(rows),
        "row_minutes": [r["minutes"] for r in rows],
        "cumulative_minutes": cum,
        "total_minutes": round(c, 2),
        "drones": [{"id": d["id"], "usable_min": usable_minutes(d),
                    "target_min": round(c * usable_minutes(d) / max(1e-6, sum(usable_minutes(x) for x in drones)), 2)}
                   for d in drones],
    })
    try:
        api_key()
    except RuntimeError as exc:
        return fallback_split(rows, drones), "fallback", [{
            "valid": False, "errors": [str(exc)], "total_s": 0, "raw_answer": "",
        }]
    attempts = []
    for _ in range(RETRIES + 1):
        raw, stats = "", {}
        try:
            raw, stats = llm_call(user)
            blocks = parse_assignments(raw)
            errs = check(blocks, rows, drones)
        except Exception as e:
            blocks, errs = None, [f"{type(e).__name__}: {e}"]
        attempts.append({**stats, "valid": not errs, "errors": errs, "raw_answer": (raw or "")[:2000]})
        if not errs:
            return blocks, "llm", attempts
        user += f"\nYour previous answer was invalid: {errs}. Fix it."
    return fallback_split(rows, drones), "fallback", attempts


def build_patches(cov, blocks):
    rows, ang, rp = cov["rows"], cov["angle"], cov["rot_poly"]
    patches = []
    for b in sorted(blocks, key=lambda item: item["first_row"]):
        rs = rows[b["first_row"] - 1:b["last_row"]]
        if not rs:
            continue
        piece = clip_band(rp, rs[0]["y0"], rs[-1]["y1"])
        lines = []
        for r in rs:
            yc = (r["y0"] + r["y1"]) / 2
            lines += [rotate([(xa, yc), (xb, yc)], ang) for xa, xb in scanline(rp, yc)]
        mid = rs[len(rs) // 2]
        ym = (mid["y0"] + mid["y1"]) / 2
        xa, xb = max(scanline(rp, ym) or [(0, 0)], key=lambda s: s[1] - s[0])
        patches.append({
            "drone_id": b["drone_id"],
            "first_row": b["first_row"],
            "last_row": b["last_row"],
            "n_rows": len(rs),
            "est_minutes": round(sum(r["minutes"] for r in rs), 2),
            "area_m2": round(area(piece)),
            "polygon": [[round(x, 1), round(y, 1)] for x, y in rotate(piece, ang)],
            "_lines": lines,
            "_label_xy": rotate([((xa + xb) / 2, ym)], ang)[0],
        })
    return patches


def lines_to_waypoints(lines):
    path = []
    for i, segment in enumerate(lines):
        (xa, ya), (xb, yb) = segment
        if i % 2:
            xa, ya, xb, yb = xb, yb, xa, ya
        path.append([round(xa, 3), round(ya, 3)])
        path.append([round(xb, 3), round(yb, 3)])
    return path


def plot_assignment(
    case, cov, patches, source, llm_s, path_png, drones=None, base=None, base_size=None,
    target=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    drones = drones or DRONES
    fig, ax = plt.subplots(figsize=(9, 8))
    caps = {d["id"]: usable_minutes(d) for d in drones}
    for p in patches:
        col = COLORS.get(p["drone_id"], "#808080")
        if len(p["polygon"]) >= 3:
            ax.add_patch(MplPolygon(p["polygon"], closed=True, facecolor=col, alpha=0.45,
                                    edgecolor="none"))
        for (xa, ya), (xb, yb) in p["_lines"]:
            ax.plot([xa, xb], [ya, yb], color=col, linewidth=0.7, alpha=0.9)
        cx, cy = p["_label_xy"]
        ax.text(cx, cy, p["drone_id"], ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=col, alpha=0.85))
        ax.plot([], [], color=col, linewidth=8, alpha=0.6,
                label=f"{p['drone_id']}: rows {p['first_row']}-{p['last_row']}  "
                      f"{p['est_minutes']:.1f}/{caps.get(p['drone_id'], 0):.1f} min  {p['area_m2']:,} m²")
    poly = case["coords"]
    ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor="black", linewidth=2))
    for k, (x, y) in enumerate(poly, 1):
        ax.plot(x, y, "ko", markersize=5)
        ax.annotate(f"P{k} ({x:.0f}, {y:.0f})", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    farm = [
        (-FIELD_EAST_M, -FIELD_NORTH_M), (FIELD_EAST_M, -FIELD_NORTH_M),
        (FIELD_EAST_M, FIELD_NORTH_M), (-FIELD_EAST_M, FIELD_NORTH_M),
    ]
    ax.add_patch(MplPolygon(farm, closed=True, fill=False, edgecolor="#666666",
                            linewidth=1.2, linestyle="--"))
    if base is not None:
        bx, by = base
        if base_size is not None:
            half_e, half_n = float(base_size[0]) / 2.0, float(base_size[1]) / 2.0
            pad = [
                (bx - half_e, by - half_n), (bx + half_e, by - half_n),
                (bx + half_e, by + half_n), (bx - half_e, by + half_n),
            ]
            ax.add_patch(MplPolygon(
                pad, closed=True, facecolor="#c62828", alpha=0.25,
                edgecolor="#b71c1c", linewidth=1.5, label="base camp",
            ))
        ax.plot(bx, by, marker="s", color="red", markersize=8)
        ax.annotate("base", (bx, by), textcoords="offset points", xytext=(6, -10), fontsize=8)
    if target is not None:
        tx, ty = float(target["x"]), float(target["y"])
        half = float(target.get("size_m", 2.5)) / 2.0
        square = [
            (tx - half, ty - half), (tx + half, ty - half),
            (tx + half, ty + half), (tx - half, ty + half),
        ]
        ax.add_patch(MplPolygon(
            square, closed=True, facecolor="#ffeb3b", alpha=0.9,
            edgecolor="#111111", linewidth=1.5, label="YOLO target box",
        ))
        ax.plot(tx, ty, marker="D", color="#f9a825", markersize=8)
        ax.annotate("box", (tx, ty), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.08)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(
        f"{case['name']}  ({len(poly)} corners, {area(poly):,.0f} m², {len(cov['rows'])} rows)\n"
        f"{MODEL} | plan by {source.upper()} | Haiku time {llm_s:.2f} s",
        fontsize=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="Drone patches")
    fig.tight_layout()
    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path_png)


def plan_one_area(
    coords,
    drones,
    name="selected_area",
    plot_dir: Path | None = None,
    assign_fn=None,
    base=None,
    base_size=None,
):
    """Fit one polygon to the farm, ask Haiku to assign drones, plot, return patches."""
    n = len(coords)
    if not (3 <= n <= 6):
        raise ValueError(f"polygon must have 3..6 corners, got {n}")
    fitted = fit_polygon_to_farm(coords)
    cov = make_rows(fitted, HFOV_DEG, min_rows=len(drones))
    assign = assign_fn or assign_with_llm
    blocks, source, attempts = assign(cov["rows"], drones)
    want = {d["id"] for d in drones}
    got = {b["drone_id"] for b in blocks}
    if got != want:
        blocks = fallback_split(cov["rows"], drones)
        source = "fallback"
    llm_s = round(sum(a.get("total_s", 0) or 0 for a in attempts), 2)
    patches = build_patches(cov, blocks)
    target = target_box.place_in_polygon(fitted)
    cap = {d["id"]: usable_minutes(d) for d in drones}
    for p in patches:
        minutes = cap.get(p["drone_id"]) or 0.0
        p["battery_used_pct"] = round(100 * p["est_minutes"] / minutes, 1) if minutes else 0.0
        p["waypoints_m"] = lines_to_waypoints(p["_lines"])
        xs = [pt[0] for pt in p["polygon"]] or [0.0]
        ys = [pt[1] for pt in p["polygon"]] or [0.0]
        p["centroid_m"] = [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(plot_dir) if plot_dir is not None else PLOT_DIR
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stamp}_{name}.png"
    plot_assignment(
        {"name": name, "coords": fitted},
        cov, patches, source, llm_s, png, drones=drones, base=base, base_size=base_size,
        target=target,
    )
    report = {
        "case": name,
        "corners": n,
        "coords": [[round(x, 3), round(y, 3)] for x, y in fitted],
        "source": source,
        "llm_time_s": llm_s,
        "attempts": len(attempts),
        "attempt_details": attempts,
        "assignments": [{k: v for k, v in p.items() if not k.startswith("_")} for p in patches],
        "image": str(png),
        "rows": len(cov["rows"]),
        "swath_m": cov["swath_m"],
        "altitude_m": cov["altitude_m"],
        "target_box": target,
    }
    json_path = out / f"{stamp}_{name}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "name": name,
        "corners": n,
        "coords": fitted,
        "source": source,
        "llm_time_s": llm_s,
        "cov": cov,
        "patches": patches,
        "plot": str(png),
        "report": str(json_path),
        "drones": drones,
        "target_box": target,
    }


def main():
    case = pick_one_area()
    drones = drones_for_count(6)
    result = plan_one_area(
        case["coords"], drones, name=case["name"],
        base=(-150.0, -52.0), base_size=(60.0, 52.0),
    )
    print(f"Selected {result['name']} ({result['corners']} pts)")
    print(f"Plan by {result['source']} in {result['llm_time_s']} s")
    print(f"Plot: {result['plot']}")
    for p in result["patches"]:
        print(f"  {p['drone_id']}: rows {p['first_row']}-{p['last_row']}  "
              f"{p['area_m2']} m²  {len(p['waypoints_m'])} waypoints")


if __name__ == "__main__":
    main()
