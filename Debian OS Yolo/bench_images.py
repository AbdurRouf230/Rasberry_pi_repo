#!/usr/bin/env python3
"""Run box YOLO on samples/. Scores time, hits, and hit_/miss_ labels."""

from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

from box_model import default_conf, default_imgsz, detect_rgb, draw_hits, get_model

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
RESULTS = HERE / "results"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))


def shrink(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return rgb
    scale = max_side / float(side)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        import cv2

        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(rgb).resize((new_w, new_h)))


def expected_label(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("hit_"):
        return "hit"
    if lower.startswith("miss_"):
        return "miss"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=default_imgsz())
    parser.add_argument("--conf", type=float, default=default_conf())
    parser.add_argument("--dir", type=Path, default=SAMPLES)
    parser.add_argument("--max-side", type=int, default=640, help="resize long edge before YOLO")
    args = parser.parse_args()

    files = sorted(
        p for p in args.dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ) if args.dir.is_dir() else []
    if not files:
        raise SystemExit(
            f"No images in {args.dir}. Put jpg/png files in samples/ (hit_* / miss_* names)."
        )

    RESULTS.mkdir(exist_ok=True)
    annotated = RESULTS / "image_hits"
    annotated.mkdir(exist_ok=True)
    get_model()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS / f"bench_images_{stamp}.csv"

    warm = load_rgb(files[0])
    if warm is not None:
        detect_rgb(shrink(warm, args.max_side), imgsz=args.imgsz, conf=args.conf)
        print(f"warmup {files[0].name} (not scored)")

    times: list[float] = []
    tp = fp = fn = tn = unlabeled_hit = unlabeled_miss = 0
    rows = []
    from PIL import Image

    for path in files:
        rgb = load_rgb(path)
        if rgb is None:
            print(f"skip unreadable {path.name}")
            continue
        rgb = shrink(rgb, args.max_side)
        hits, infer_ms = detect_rgb(rgb, imgsz=args.imgsz, conf=args.conf)
        times.append(infer_ms)
        pred = bool(hits)
        expect = expected_label(path.name)
        max_conf = max((h["confidence"] for h in hits), default=0.0)
        if expect == "hit":
            tp += int(pred)
            fn += int(not pred)
            tag = "TP  " if pred else "FN  "
        elif expect == "miss":
            tn += int(not pred)
            fp += int(pred)
            tag = "FP  " if pred else "TN  "
        else:
            unlabeled_hit += int(pred)
            unlabeled_miss += int(not pred)
            tag = "HIT " if pred else "miss"
        Image.fromarray(draw_hits(rgb, hits)).save(annotated / f"{path.stem}_bbox.jpg", quality=90)
        print(
            f"{tag} {path.name} {infer_ms:.1f} ms "
            f"boxes={len(hits)} conf={max_conf:.2f} expect={expect or '-'}"
        )
        rows.append([
            path.name, expect or "", int(pred), f"{infer_ms:.2f}",
            len(hits), f"{max_conf:.3f}",
        ])

    labeled = tp + fp + fn + tn
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "expected", "pred_hit", "infer_ms", "n_boxes", "max_conf"])
        writer.writerows(rows)

    print("--- summary ---")
    print(f"images={len(times)}")
    if times:
        mean = statistics.fmean(times)
        print(
            f"infer mean={mean:.1f} ms  p50={statistics.median(times):.1f}  "
            f"~{1000.0 / mean:.2f} FPS  imgsz={args.imgsz}"
        )
    if labeled:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        acc = (tp + tn) / labeled
        print(f"labeled={labeled}  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"precision={prec:.2%}  recall={rec:.2%}  accuracy={acc:.2%}")
        print("hit_* = box should be found. miss_* = empty scene (online photos).")
        print("Weights were trained on a sim box; real Wikimedia photos may miss.")
    else:
        print(f"unlabeled hits={unlabeled_hit} misses={unlabeled_miss}")
    print(f"boxed previews: {annotated}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
