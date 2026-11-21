#!/usr/bin/env python3
"""Export weights/best.pt to NCNN (faster on Pi 4B CPU than raw PyTorch)."""

from __future__ import annotations

from pathlib import Path

from box_model import default_imgsz, resolve_weights

HERE = Path(__file__).resolve().parent


def main() -> None:
    from ultralytics import YOLO

    src = resolve_weights()
    imgsz = default_imgsz()
    print(f"exporting {src} -> ncnn imgsz={imgsz}")
    model = YOLO(str(src), task="detect")
    out = model.export(format="ncnn", imgsz=imgsz)
    print(f"exported: {out}")
    print("Run benches with:")
    print(f"  export BOX_YOLO_WEIGHTS={out}")


if __name__ == "__main__":
    main()
