"""Plan one 3–6 point farm area with Haiku, then fly from the red pad.

Each mission picks ONE catalog polygon (triangle through hexagon), scales it
onto the soybean / clover / corn rectangle, and asks Claude Haiku which
contiguous rows each drone flies. Drones spawn on the red base_camp pad
south of the barn (60 m × 52 m), cover their assigned patch, and return.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from planner import _survey_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import assign_area  # noqa: E402

WORLD_LAT = 37.7749
WORLD_LON = -122.4194
FIELD_EAST_M = assign_area.FIELD_EAST_M
FIELD_NORTH_M = assign_area.FIELD_NORTH_M
# Red ground pad base_camp in agricultural_farm.world.
BASE_EAST_M = -150.0
BASE_NORTH_M = -52.0
BASE_SIZE_EAST_M = 60.0
BASE_SIZE_NORTH_M = 52.0
BASE_EAST_GAP_M = 12.0
BASE_NORTH_GAP_M = 14.0
SPAWN_Z_M = 0.5


def prepare(
    mission_dir: Path,
    num_patches: int | None = None,
    *,
    case_name: str | None = None,
    plot_dir: Path | None = None,
    assign_fn=None,
) -> dict:
    """Pick one area, assign drones with Haiku, write launch files and a plot."""
    request_path = mission_dir / "example_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    count = int(num_patches if num_patches is not None else request.get("num_patches") or 6)
    if count < 1:
        raise RuntimeError("num_patches must be at least 1.")

    case = assign_area.pick_one_area(case_name)
    drones = assign_area.drones_for_count(count)
    plot_root = Path(plot_dir) if plot_dir is not None else ROOT / "area_assignment_plots"
    planned = assign_area.plan_one_area(
        case["coords"],
        drones,
        name=case["name"],
        plot_dir=plot_root,
        assign_fn=assign_fn,
        base=(BASE_EAST_M, BASE_NORTH_M),
        base_size=(BASE_SIZE_EAST_M, BASE_SIZE_NORTH_M),
    )

    warnings = []
    # Fly the coverage altitude the lawn-mower rows were planned at (~9 m).
    # Camera GSD in example_request.json is ~58 m and does not match those rows.
    altitude_m = planned["cov"]["altitude_m"]
    line_spacing_m = planned["cov"]["swath_m"]
    camera = request.get("camera")
    if camera:
        survey = _survey_values(request, camera, warnings)
        gsd_alt = survey["summary"]["altitude_m"]
        if abs(gsd_alt - altitude_m) > 1.0:
            warnings.append(
                f"Camera GSD altitude {gsd_alt:.1f} m unused; "
                f"flying coverage altitude {altitude_m:.1f} m over assigned patches."
            )

    by_id = {p["drone_id"]: p for p in planned["patches"]}
    missing = [d["id"] for d in drones if not by_id.get(d["id"], {}).get("waypoints_m")]
    if missing:
        raise RuntimeError(
            "Every drone must have survey waypoints before launch. Missing: "
            + ", ".join(missing)
        )
    spawns = base_spawns(count)
    patches = []
    for index, drone in enumerate(drones):
        assigned = by_id[drone["id"]]
        patches.append({
            "id": index + 1,
            "drone_id": drone["id"],
            "area_m2": assigned["area_m2"],
            "centroid_m": assigned["centroid_m"],
            "waypoints_m": list(assigned["waypoints_m"]),
            "first_row": assigned["first_row"],
            "last_row": assigned["last_row"],
        })

    launch = {
        "world": "agricultural_farm.world",
        "area": planned["name"],
        "corners": planned["corners"],
        "assignment_source": planned["source"],
        "assignment_plot": planned["plot"],
        "assignment_report": planned["report"],
        "num_drones": len(patches),
        "spawns": [{"x": x, "y": y, "z": z, "yaw": yaw} for x, y, z, yaw in spawns],
        "patches": patches,
        "altitude_m": altitude_m,
        "line_spacing_m": line_spacing_m,
        "area_coords": [[round(float(x), 3), round(float(y), 3)] for x, y in planned["coords"]],
        "target_box": planned["target_box"],
        "warnings": warnings + ([
            f"Haiku assignment used fallback ({planned['source']})."
        ] if planned["source"] != "llm" else []),
    }
    (mission_dir / "survey_launch.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    (mission_dir / "survey_spawns.txt").write_text(
        "\n".join(f"{x:.2f} {y:.2f} {z:.2f} {yaw:.2f}" for x, y, z, yaw in spawns) + "\n",
        encoding="utf-8",
    )
    return launch


def base_spawns(count: int) -> list[tuple[float, float, float, float]]:
    """Pack drones on the red pad: up to 3 along east, extra rows to the south."""
    if count < 1:
        return []
    cols = min(3, count)
    row_count = math.ceil(count / cols)
    spots = []
    for index in range(count):
        col = index % cols
        row = index // cols
        east = BASE_EAST_M + (col - (cols - 1) / 2.0) * BASE_EAST_GAP_M
        north = BASE_NORTH_M + ((row_count - 1) / 2.0 - row) * BASE_NORTH_GAP_M
        spots.append((east, north, SPAWN_Z_M, 0.0))
    return spots


def _ll(east_m: float, north_m: float) -> list[float]:
    scale_x = 111320.0 * math.cos(math.radians(WORLD_LAT))
    return [
        round(WORLD_LON + east_m / scale_x, 8),
        round(WORLD_LAT + north_m / 110540.0, 8),
    ]


if __name__ == "__main__":
    launch = prepare(Path(__file__).resolve().parent)
    print(f"Area {launch['area']} ({launch['corners']} pts)  drones={launch['num_drones']}")
    print(f"Plot {launch.get('assignment_plot')}")
    box = launch.get("target_box") or {}
    print(f"Target box ({box.get('x')}, {box.get('y')}) size {box.get('size_m')} m")
    for item in launch["patches"]:
        print(
            f"  {item['drone_id']}: {len(item['waypoints_m'])} waypoints  "
            f"{item['area_m2']} m²"
        )
