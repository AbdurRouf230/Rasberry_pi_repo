"""Fixed labeled test set: coco128. Same images, same order, every model."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

HERE = Path(__file__).resolve().parent
TESTDATA = HERE / "testdata"
COCO128 = TESTDATA / "coco128"
IMAGE_LIST = TESTDATA / "IMAGE_LIST.txt"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def _find_images(root: Path) -> list[Path]:
    images = []
    for folder in (root / "images" / "train2017", root / "images"):
        if folder.is_dir():
            images.extend(sorted(folder.glob("*.jpg")))
            images.extend(sorted(folder.glob("*.jpeg")))
            images.extend(sorted(folder.glob("*.png")))
    # unique, stable order
    seen = set()
    out = []
    for path in images:
        key = path.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _label_for(image: Path) -> Path | None:
    stem = image.stem
    for folder in (
        COCO128 / "labels" / "train2017",
        COCO128 / "labels",
        image.parent.parent.parent / "labels" / "train2017",
    ):
        cand = folder / f"{stem}.txt"
        if cand.exists():
            return cand
    return None


def ensure_coco128() -> Path:
    TESTDATA.mkdir(parents=True, exist_ok=True)
    images = _find_images(COCO128) if COCO128.exists() else []
    if len(images) >= 100:
        _write_list(images)
        return COCO128
    zpath = TESTDATA / "coco128.zip"
    if not zpath.exists():
        print(f"downloading coco128 (same test set for every model)\n  {COCO128_URL}")
        urlretrieve(COCO128_URL, zpath)
    print(f"extracting {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(TESTDATA)
    if not COCO128.exists():
        # zip may extract to testdata/coco128-seg or nested folder
        for child in TESTDATA.iterdir():
            if child.is_dir() and (child / "images").exists():
                if child.name != "coco128":
                    child.rename(COCO128)
                break
    images = _find_images(COCO128)
    if not images:
        raise SystemExit(f"coco128 images not found under {COCO128}")
    _write_list(images)
    print(f"test set: {len(images)} images in {COCO128}")
    return COCO128


def _write_list(images: list[Path]) -> None:
    lines = [str(p.relative_to(TESTDATA)).replace("\\", "/") for p in images]
    IMAGE_LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_image_list() -> list[Path]:
    ensure_coco128()
    if IMAGE_LIST.exists():
        paths = []
        for line in IMAGE_LIST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = TESTDATA / line
            if path.exists():
                paths.append(path)
        if paths:
            return paths
    return _find_images(COCO128)


def load_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gt_xyxy(image: Path, shape: tuple[int, int]) -> list[tuple[str, list[int]]]:
    """YOLO-format labels -> (coco80_name, xyxy pixels)."""
    from detectors import COCO80

    label_path = _label_for(image)
    h, w = shape[0], shape[1]
    if label_path is None or not label_path.exists():
        return []
    boxes: list[tuple[str, list[int]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        name = COCO80[cls_id] if 0 <= cls_id < len(COCO80) else str(cls_id)
        boxes.append((name, [int(x1), int(y1), int(x2), int(y2)]))
    return boxes
