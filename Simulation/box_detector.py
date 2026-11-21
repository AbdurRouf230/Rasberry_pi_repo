"""Onboard YOLO for the Gazebo cardboard search box.

Loads the trained box class weights (genai_framework/models/box_yolo/best.pt).
A colour backup still fires if ultralytics is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

TARGET_NAMES = {
    "box",
    "yellow box",
    "cube",
    "cardboard box",
    "package",
    "suitcase",
}

HERE = Path(__file__).resolve().parent
_yolo = None
_yolo_tried = False
_yolo_kind = ""
_yolo_error = ""
_yolo_path = ""


def resolve_weights() -> Optional[Path]:
    """Trained box YOLO first, then env override, then the POC copy."""
    raw = os.environ.get("BOX_YOLO_WEIGHTS", "").strip()
    candidates = []
    if raw:
        candidates.append(Path(raw))
    candidates.extend([
        HERE.parent / "models" / "box_yolo" / "best.pt",
        HERE / "models" / "box_yolo" / "best.pt",
        Path("/mnt/e/Multi Drone/POC/Main-POC-Repo/intelleswarm-ai/genai_framework/models/box_yolo/best.pt"),
        Path(r"e:\Multi Drone\POC\Main-POC-Repo\intelleswarm-ai\genai_framework\models\box_yolo\best.pt"),
    ])
    for path in candidates:
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return None


def yolo_status() -> str:
    if _yolo is not None:
        name = Path(_yolo_path).name if _yolo_path else _yolo_kind
        return f"attached:{name}" if name else (_yolo_kind or "yolo")
    if _yolo_tried:
        return _yolo_error or "unavailable"
    return "not-loaded"


def get_yolo():
    """Load the trained box YOLO once. Camera frames go through this model."""
    global _yolo, _yolo_tried, _yolo_kind, _yolo_error, _yolo_path
    if _yolo_tried:
        return _yolo
    _yolo_tried = True
    try:
        from ultralytics import YOLO
    except Exception as exc:
        _yolo_error = f"ultralytics import failed: {exc}"
        print(f"YOLO not available ({_yolo_error}). Using colour-box backup.")
        return None
    weights = resolve_weights()
    tries = []
    if weights is not None:
        tries.append((str(weights), "box_yolo", None))
    tries.append(("yolov8n.pt", "yolov8n", None))
    for path, kind, classes in tries:
        try:
            model = YOLO(path, task="detect")
            if classes and hasattr(model, "set_classes"):
                model.set_classes(classes)
            _yolo = model
            _yolo_kind = kind
            _yolo_path = path
            print(f"YOLO attached ({kind}: {path})")
            return _yolo
        except Exception as exc:
            _yolo_error = str(exc)
            continue
    print(f"YOLO weights failed ({_yolo_error}). Using colour-box backup.")
    return None


def detect_yellow_box(rgb: np.ndarray) -> list[dict[str, Any]]:
    """HSV-style mask for the emissive yellow target cube we spawn."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return []
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    yellow = (r > 180) & (g > 150) & (b < 110) & (r + 30 > g) & (g > b + 40)
    cardboard = (r > 150) & (g > 70) & (g < 210) & (b < 130) & (r > g + 15) & (g > b)
    mask = yellow | cardboard
    ys, xs = np.where(mask)
    if xs.size < 80:
        return []
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if (x2 - x1) * (y2 - y1) < 400:
        return []
    conf = min(0.99, 0.55 + xs.size / float(rgb.shape[0] * rgb.shape[1]))
    return [{
        "name": "box",
        "confidence": round(float(conf), 3),
        "bbox": [x1, y1, x2, y2],
        "source": "color",
    }]


def _yolo_rows(rgb: np.ndarray) -> list[dict[str, Any]]:
    model = get_yolo()
    if model is None:
        return []
    try:
        # Dataset was labelled from OpenCV BGR frames.
        bgr = rgb[:, :, ::-1]
        results = model.predict(bgr, verbose=False, conf=0.5, imgsz=640)
    except Exception as exc:
        print(f"YOLO predict failed: {exc}")
        return []
    hits = []
    names = getattr(model, "names", None) or {}
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
            if isinstance(names, dict):
                label = str(names.get(cls_id, cls_id))
            else:
                label = str(names[cls_id] if 0 <= cls_id < len(names) else cls_id)
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            hits.append({
                "name": label.lower().strip(),
                "confidence": round(conf, 3),
                "bbox": xyxy,
                "source": "yolo",
            })
    return hits


def _overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def detect_image(rgb: np.ndarray, use_yolo: bool = True) -> list[dict[str, Any]]:
    """YOLO on the cardboard box. Colour is diagnostic only and never recalls the fleet."""
    if not use_yolo:
        return detect_yellow_box(rgb)
    return [
        hit for hit in _yolo_rows(rgb)
        if hit["name"] in TARGET_NAMES and hit.get("source") == "yolo"
    ]


def detections_are_target(hits: list[dict[str, Any]] | None) -> bool:
    """Recall only on a YOLO box class — not the pad colour false-positive."""
    for hit in hits or []:
        if hit.get("source") != "yolo":
            continue
        if str(hit.get("name", "")).lower() in TARGET_NAMES:
            return True
    return False


def captured_dir_name(drone_index: int) -> str:
    """Survey folder: drone1, drone2, ...  Finder is later renamed droneN_captured."""
    return f"drone{int(drone_index) + 1}"


def mark_finder_capture_dir(capture_dir: str | Path) -> Path:
    """Rename droneN -> droneN_captured for the drone that detected the box."""
    src = Path(capture_dir)
    if src.name.endswith("_captured"):
        src.mkdir(parents=True, exist_ok=True)
        return src
    dest = src.parent / f"{src.name}_captured"
    if dest.exists() and not src.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        try:
            src.rename(dest)
        except OSError:
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                target = dest / item.name
                if not target.exists():
                    item.rename(target)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    return dest


def save_annotated_rgb(rgb: np.ndarray, hits: list[dict[str, Any]] | None, path: str | Path) -> Path:
    """Write RGB image with green YOLO boxes to path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = np.ascontiguousarray(rgb)
    boxes = list(hits or [])
    try:
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for hit in boxes:
            x1, y1, x2, y2 = [int(v) for v in hit["bbox"]]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{hit.get('name', 'box')} {float(hit.get('confidence', 0)):.2f}"
            cv2.putText(
                bgr, label, (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
            )
        cv2.imwrite(str(dest), bgr)
        return dest
    except Exception:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        for hit in boxes:
            x1, y1, x2, y2 = [int(v) for v in hit["bbox"]]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text(
                (x1, max(0, y1 - 12)),
                f"{hit.get('name', 'box')} {float(hit.get('confidence', 0)):.2f}",
                fill=(0, 255, 0),
            )
        im.save(dest)
        return dest


def load_rgb(path: str | Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            from PIL import Image
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None


def detect_file(path: str | Path, use_yolo: bool = True) -> list[dict[str, Any]]:
    rgb = load_rgb(path)
    if rgb is None:
        return []
    return detect_image(rgb, use_yolo=use_yolo)
