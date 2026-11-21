#!/usr/bin/env python3
"""Live CSI YOLO: COCO multi-class, on-screen window, CSV + video + per-class JPEGs.

  cd ~/Rasberry_pi_repo/Debian\\ OS\\ Yolo
  source .venv/bin/activate
  python3 live_detect.py --imgsz 320 --conf 0.35

Saves under results/live_YYYYMMDD_HHMMSS/:
  inference.csv
  video.mp4
  <classname>/frame_XXXXX.jpg   (only when that class is detected)

q or Esc closes the window.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

from box_model import default_conf, default_imgsz, detect_rgb, draw_hits, get_model, model_path
from camera_csi import close_camera, open_camera, read_rgb

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _safe_name(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower().strip()).strip("_")
    return cleaned or "unknown"


def _open_writer(path: Path, size: tuple[int, int], fps: float):
    import cv2

    w, h = size
    fps = max(1.0, min(float(fps), 15.0))
    for fourcc, dest in (("mp4v", path), ("XVID", path.with_suffix(".avi"))):
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            return writer, dest
        writer.release()
    return None, path


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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / f"live_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inference.csv"
    video_path = out_dir / "video.mp4"

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
    print(f"save dir: {out_dir}")
    print("window: YOLO live   keys: q or Esc = quit")

    writer = None
    written_video = video_path
    frame_i = 0
    times: list[float] = []
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "infer_ms", "fps", "n_boxes", "names", "confidences", "bboxes",
    ])

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
            times.append(infer_ms)
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

            if writer is None:
                h, w = bgr.shape[:2]
                writer, written_video = _open_writer(video_path, (w, h), fps if fps >= 1 else 2.0)
                if writer is None:
                    print("WARN: could not open video writer; JPEGs + CSV still saved")
                else:
                    print(f"video: {written_video}")
            if writer is not None:
                writer.write(bgr)

            seen = set()
            for hit in hits:
                folder = _safe_name(str(hit["name"]))
                if folder in seen:
                    continue
                seen.add(folder)
                class_dir = out_dir / folder
                class_dir.mkdir(exist_ok=True)
                cv2.imwrite(
                    str(class_dir / f"{folder}_{frame_i:05d}.jpg"),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )

            names = ",".join(h["name"] for h in hits)
            confs = ",".join(f"{h['confidence']:.3f}" for h in hits)
            boxes = ";".join(
                f"{h['bbox'][0]},{h['bbox'][1]},{h['bbox'][2]},{h['bbox'][3]}" for h in hits
            )
            csv_writer.writerow([
                frame_i, f"{infer_ms:.2f}", f"{fps:.3f}", len(hits), names, confs, boxes,
            ])
            csv_file.flush()

            cv2.imshow(win, bgr)
            shown = ", ".join(f"{h['name']} {h['confidence']:.2f}" for h in hits) or "-"
            print(f"frame={frame_i} {infer_ms:.1f} ms boxes={len(hits)} {shown}")
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
        csv_file.close()
        if writer is not None:
            writer.release()
        close_camera(kind, cam)
        cv2.destroyAllWindows()

    if times:
        mean = sum(times) / len(times)
        print("--- summary ---")
        print(f"frames={len(times)} mean={mean:.1f} ms  ~{1000.0 / mean:.2f} FPS")
    print(f"csv: {csv_path}")
    print(f"video: {written_video}")
    print(f"images: {out_dir}/<classname>/")


if __name__ == "__main__":
    main()
