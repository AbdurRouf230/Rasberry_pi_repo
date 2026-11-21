"""Load the trained cardboard-box YOLO and time a single predict."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
TARGET_NAMES = {
    "box",
    "yellow box",
    "cube",
    "cardboard box",
    "package",
    "suitcase",
}

_model = None
_model_path = ""


def default_imgsz() -> int:
    raw = os.environ.get("BOX_YOLO_IMGSZ", "320").strip()
    try:
        return max(160, int(raw))
    except ValueError:
        return 320


def default_conf() -> float:
    raw = os.environ.get("BOX_YOLO_CONF", "0.35").strip()
    try:
        return min(0.99, max(0.05, float(raw)))
    except ValueError:
        return 0.5


def resolve_weights() -> Path:
    raw = os.environ.get("BOX_YOLO_WEIGHTS", "").strip()
    candidates = []
    if raw:
        candidates.append(Path(raw))
    candidates.extend([
        HERE / "weights" / "best.pt",
        HERE / "weights" / "best_ncnn_model",
        Path(r"e:\Multi Drone\Merge folder\intelleswarm-ai\genai_framework\models\box_yolo\best.pt"),
        Path("/mnt/e/Multi Drone/Merge folder/intelleswarm-ai/genai_framework/models/box_yolo/best.pt"),
    ])
    for path in candidates:
        try:
            if path.exists():
                return path.resolve()
        except OSError:
            continue
    raise FileNotFoundError(
        "No YOLO weights. Put best.pt in 'Only yolo/weights/' "
        "or set BOX_YOLO_WEIGHTS."
    )


def get_model():
    global _model, _model_path
    if _model is not None:
        return _model
    from ultralytics import YOLO

    path = resolve_weights()
    _model = YOLO(str(path), task="detect")
    _model_path = str(path)
    print(f"YOLO loaded: {path}")
    print(f"classes: {getattr(_model, 'names', {})}")
    return _model


def model_path() -> str:
    return _model_path


def detect_rgb(
    rgb: np.ndarray,
    imgsz: Optional[int] = None,
    conf: Optional[float] = None,
) -> tuple[list[dict[str, Any]], float]:
    """Return (hits, infer_ms). hits are YOLO box-class only."""
    model = get_model()
    imgsz = default_imgsz() if imgsz is None else imgsz
    conf = default_conf() if conf is None else conf
    bgr = rgb[:, :, ::-1]
    t0 = time.perf_counter()
    results = model.predict(bgr, verbose=False, conf=conf, imgsz=imgsz)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    names = getattr(model, "names", None) or {}
    hits: list[dict[str, Any]] = []
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
            name = label.lower().strip()
            if name not in TARGET_NAMES:
                continue
            hits.append({
                "name": name,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                "source": "yolo",
            })
    return hits, infer_ms


def draw_hits(rgb: np.ndarray, hits: list[dict[str, Any]]) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    for hit in hits:
        x1, y1, x2, y2 = hit["bbox"]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"{hit['name']} {hit['confidence']:.2f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
