#!/usr/bin/env python3
"""Live CSI YOLO: COCO multi-class model + on-screen window.

Run on the Pi desktop (AnyDesk is fine), not a headless SSH session:

  cd ~/Rasberry_pi_repo/Debian\\ OS\\ Yolo
  source .venv/bin/activate
  python3 live_detect.py --imgsz 320 --conf 0.35

First run downloads yolov8n.pt (~6 MB, 80 COCO classes: person, car, bottle, …).
q or Esc closes the window.
"""

from __future__ import annotations

import argparse
import os
import time

from box_model import default_conf, default_imgsz, detect_rgb, draw_hits, get_model, model_path
from camera_csi import close_camera, open_camera, read_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Live multi-class YOLO with a popup window")
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Ultralytics model. yolov8n.pt = general 80-class COCO. Use weights/best.pt for the box-only model.",
    )
    parser.add_argument("--imgsz", type=int, default=default_imgsz())
    parser.add_argument("--conf", type=float, default=default_conf())
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    import cv2

    get_model(args.weights)
    kind, cam = open_camera((args.width, args.height))
    win = "YOLO live"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.width, args.height)
    except cv2.error as exc:
        raise SystemExit(
            "OpenCV cannot open a window (headless build or no DISPLAY).\n"
            "Run this on the Pi desktop / AnyDesk, and use apt OpenCV:\n"
            "  sudo apt install -y python3-opencv libgtk-3-0\n"
            f"({exc})"
        ) from exc

    print(f"backend={kind} weights={model_path()} imgsz={args.imgsz} conf={args.conf}")
    print("window: YOLO live   keys: q or Esc = quit")

    frame_i = 0
    try:
        while True:
            rgb = read_rgb(kind, cam)
            if rgb is None:
                time.sleep(0.02)
                continue
            hits, infer_ms = detect_rgb(
                rgb, imgsz=args.imgsz, conf=args.conf, all_classes=True,
            )
            frame_i += 1
            vis_rgb = draw_hits(rgb, hits)
            bgr = vis_rgb[:, :, ::-1].copy()
            fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            cv2.putText(
                bgr,
                f"{infer_ms:.0f} ms  {fps:.1f} FPS  n={len(hits)}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, bgr)
            names = ", ".join(f"{h['name']} {h['confidence']:.2f}" for h in hits) or "-"
            print(f"frame={frame_i} {infer_ms:.1f} ms boxes={len(hits)} {names}")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        close_camera(kind, cam)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
