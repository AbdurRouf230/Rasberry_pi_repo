#!/usr/bin/env python3
"""Grab a few CSI frames. No YOLO. Use this first on the Pi."""

from __future__ import annotations

from pathlib import Path

from camera_csi import close_camera, open_camera, read_rgb

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    kind, cam = open_camera((640, 480))
    print(f"camera backend: {kind}")
    try:
        for i in range(8):
            rgb = read_rgb(kind, cam)
            if rgb is None:
                print(f"frame {i}: empty")
                continue
            print(f"frame {i}: shape={rgb.shape} dtype={rgb.dtype}")
            if i == 7:
                try:
                    from PIL import Image

                    dest = OUT / "camera_check.jpg"
                    Image.fromarray(rgb).save(dest)
                    print(f"saved {dest}")
                except Exception as exc:
                    print(f"save skipped: {exc}")
    finally:
        close_camera(kind, cam)
    print("camera ok")


if __name__ == "__main__":
    main()
