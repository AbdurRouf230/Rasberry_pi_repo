"""One detector wrapper per model. Same Hit dict: name, cls_id, confidence, bbox xyxy."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

Hit = dict[str, Any]
PredictFn = Callable[[np.ndarray, float, int], tuple[list[Hit], float]]

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _hit(name: str, cls_id: int, conf: float, xyxy: list[int]) -> Hit:
    return {
        "name": str(name).lower().strip(),
        "cls_id": int(cls_id),
        "confidence": round(float(conf), 3),
        "bbox": [int(v) for v in xyxy],
        "source": "compare",
    }


class Detector:
    def __init__(self, model_id: str, title: str, load_fn: Callable[[], PredictFn]):
        self.model_id = model_id
        self.title = title
        self._load_fn = load_fn
        self.predict: Optional[PredictFn] = None
        self.error: str = ""

    def load(self) -> None:
        self.predict = self._load_fn()
        self.error = ""

    def close(self) -> None:
        self.predict = None


def _yolo_loader(weights: str) -> Callable[[], PredictFn]:
    def load() -> PredictFn:
        from ultralytics import YOLO

        model = YOLO(weights, task="detect")
        names = getattr(model, "names", {}) or {}

        def predict(rgb: np.ndarray, conf: float, imgsz: int) -> tuple[list[Hit], float]:
            bgr = rgb[:, :, ::-1]
            t0 = time.perf_counter()
            results = model.predict(bgr, verbose=False, conf=conf, imgsz=imgsz)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            hits: list[Hit] = []
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
                    hits.append(_hit(label, cls_id, float(box.conf[0]), box.xyxy[0].tolist()))
            return hits, infer_ms

        return predict

    return load


def _ssdlite_loader() -> PredictFn:
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    categories = list(weights.meta["categories"])
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    def predict(rgb: np.ndarray, conf: float, imgsz: int) -> tuple[list[Hit], float]:
        del imgsz
        arr = np.ascontiguousarray(rgb)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model([tensor])[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        hits: list[Hit] = []
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        h, w = rgb.shape[:2]
        for box, lab, score in zip(boxes, labels, scores):
            if float(score) < conf:
                continue
            name = str(categories[int(lab)] if 0 <= int(lab) < len(categories) else lab)
            if name in {"__background__", "N/A", ""}:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            hits.append(_hit(name, int(lab), float(score), [x1, y1, x2, y2]))
        return hits, infer_ms

    return predict


def _tflite_interpreter(model_path: str):
    tried = []
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=model_path)
    except Exception as exc:
        tried.append(f"tflite_runtime: {exc}")
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteRt

        return LiteRt(model_path=model_path)
    except Exception as exc:
        tried.append(f"ai_edge_litert: {exc}")
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=model_path)
    except Exception as exc:
        tried.append(f"tensorflow: {exc}")
    raise RuntimeError(
        "EfficientDet-Lite0 needs a TFLite interpreter. Install one of:\n"
        "  python -m pip install tflite-runtime\n"
        "  python -m pip install ai-edge-litert\n"
        + "\n".join(tried)
    )


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    import urllib.request

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _efficientdet_loader() -> PredictFn:
    tflite = _download(
        "https://github.com/google-coral/test_data/raw/master/efficientdet_lite0.tflite",
        WEIGHTS / "efficientdet_lite0.tflite",
    )
    labels_path = _download(
        "https://github.com/google-coral/test_data/raw/master/coco_labels.txt",
        WEIGHTS / "coco_labels.txt",
    )
    labels = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            labels.append("")
            continue
        parts = raw.split(maxsplit=1)
        labels.append(parts[-1] if parts[0].isdigit() and len(parts) > 1 else raw)
    interpreter = _tflite_interpreter(str(tflite))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    in_h, in_w = int(inp["shape"][1]), int(inp["shape"][2])
    in_dtype = inp["dtype"]

    def predict(rgb: np.ndarray, conf: float, imgsz: int) -> tuple[list[Hit], float]:
        del imgsz
        import cv2

        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (in_w, in_h))
        if in_dtype == np.uint8:
            blob = np.expand_dims(resized.astype(np.uint8), 0)
        else:
            blob = np.expand_dims(resized.astype(np.float32) / 255.0, 0)
        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], blob)
        interpreter.invoke()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        tensors = [interpreter.get_tensor(item["index"]) for item in outs]
        boxes = scores = classes = count = None
        for tensor in tensors:
            arr = np.squeeze(tensor)
            if arr.ndim == 2 and arr.shape[-1] == 4:
                boxes = arr
            elif arr.ndim == 1 and arr.dtype in (np.float32, np.float16):
                if scores is None:
                    scores = arr
                else:
                    classes = arr
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                classes = arr
            elif arr.ndim == 0 or arr.size == 1:
                count = int(np.array(arr).reshape(-1)[0])
        if boxes is None or scores is None:
            return [], infer_ms
        if classes is None:
            classes = np.zeros(len(scores))
        n = int(count) if count is not None else len(scores)
        n = min(n, len(scores), len(boxes))
        hits: list[Hit] = []
        for i in range(n):
            score = float(scores[i])
            if score < conf:
                continue
            ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
            if max(ymin, xmin, ymax, xmax) <= 1.5:
                x1, y1, x2, y2 = xmin * w, ymin * h, xmax * w, ymax * h
            else:
                x1, y1, x2, y2 = xmin, ymin, xmax, ymax
            cls_id = int(classes[i])
            name = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
            hits.append(_hit(name, cls_id, score, [x1, y1, x2, y2]))
        return hits, infer_ms

    return predict


REGISTRY: dict[str, Detector] = {
    "yolo26n": Detector("yolo26n", "YOLO26n", _yolo_loader("yolo26n.pt")),
    "yolo11n": Detector("yolo11n", "YOLO11n", _yolo_loader("yolo11n.pt")),
    "yolov8n": Detector("yolov8n", "YOLOv8n", _yolo_loader("yolov8n.pt")),
    "ssdlite_mv3": Detector(
        "ssdlite_mv3", "MobileNetV3-SSDLite", _ssdlite_loader,
    ),
    "efficientdet_lite0": Detector(
        "efficientdet_lite0", "EfficientDet-Lite0", _efficientdet_loader,
    ),
}

ORDER = [
    "yolo26n",
    "yolo11n",
    "ssdlite_mv3",
    "yolov8n",
    "efficientdet_lite0",
]


def get_detector(model_id: str) -> Detector:
    if model_id not in REGISTRY:
        known = ", ".join(ORDER)
        raise SystemExit(f"Unknown model '{model_id}'. Use: {known}")
    return REGISTRY[model_id]
