#!/usr/bin/env python3
"""Compare YOLO26n, YOLO11n, YOLOv8n, MobileNetV3-SSDLite, EfficientDet-Lite0.

Live: one model at a time (Pi RAM). Test-set mAP uses the same coco128 files
for every model.

  cd ~/Rasberry_pi_repo/Debian\\ OS\\ Yolo/Comparison
  source ../.venv/bin/activate
  python3 compare.py live --model yolo26n
  python3 compare.py bench
  python3 compare.py report
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT))

from dataset import load_gt_xyxy, load_image_list, load_rgb  # noqa: E402
from detectors import ORDER, get_detector  # noqa: E402
from metrics import save_graphs, score_dataset, summarize_times  # noqa: E402

RESULTS = HERE / "results"
BENCH_JSON = RESULTS / "bench.json"
LIVE_SUMMARY = RESULTS / "live_summary.csv"


def _load_parent_draw():
    from box_model import draw_hits

    return draw_hits


def cmd_live(args: argparse.Namespace) -> None:
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
    import cv2
    from camera_csi import close_camera, open_camera, read_rgb

    draw_hits = _load_parent_draw()
    det = get_detector(args.model)
    print(f"loading {det.title} ...")
    det.load()
    kind, cam = open_camera((args.width, args.height))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / f"live_{det.model_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inference.csv"
    win = f"compare {det.title}"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.width, args.height)
    except cv2.error as exc:
        raise SystemExit(
            "Need the Pi desktop / AnyDesk for the live window.\n"
            f"({exc})"
        ) from exc

    print(f"window: {win}   q/Esc = quit")
    print("Live has no ground-truth labels. mAP is computed by: python3 compare.py bench")
    times: list[float] = []
    frame_i = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "infer_ms", "fps", "n_boxes", "names"])
        try:
            while True:
                rgb = read_rgb(kind, cam)
                if rgb is None:
                    time.sleep(0.02)
                    continue
                assert det.predict is not None
                hits, infer_ms = det.predict(rgb, args.conf, args.imgsz)
                frame_i += 1
                times.append(infer_ms)
                fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
                vis = draw_hits(rgb, hits)
                bgr = vis[:, :, ::-1].copy()
                cv2.putText(
                    bgr,
                    f"{det.title}  {infer_ms:.0f} ms  {fps:.1f} FPS",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, bgr)
                writer.writerow([
                    frame_i, f"{infer_ms:.2f}", f"{fps:.3f}", len(hits),
                    ",".join(h["name"] for h in hits),
                ])
                fh.flush()
                print(
                    f"{det.model_id} frame={frame_i} {infer_ms:.1f} ms "
                    f"boxes={len(hits)}"
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                if args.max_frames and frame_i >= args.max_frames:
                    break
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            close_camera(kind, cam)
            cv2.destroyAllWindows()

    stats = summarize_times(times)
    LIVE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LIVE_SUMMARY.exists()
    with LIVE_SUMMARY.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["stamp", "model", "title", "frames", "mean_ms", "p50_ms", "fps"])
        writer.writerow([
            stamp, det.model_id, det.title, int(stats["n"]),
            f"{stats['mean_ms']:.2f}", f"{stats['p50_ms']:.2f}", f"{stats['fps']:.3f}",
        ])
    print(f"csv: {csv_path}")
    print(f"live summary: {LIVE_SUMMARY}")
    print(f"mean {stats['mean_ms']:.1f} ms  ~{stats['fps']:.2f} FPS")
    print("Next model example:  python3 compare.py live --model yolo11n")


def cmd_bench(args: argparse.Namespace) -> None:
    images = load_image_list()
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    print(f"test set: {len(images)} images (same list for every model)")
    print(f"list: {HERE / 'testdata' / 'IMAGE_LIST.txt'}")
    models = args.models or ORDER
    rows = []
    RESULTS.mkdir(parents=True, exist_ok=True)

    for model_id in models:
        det = get_detector(model_id)
        print(f"\n=== {det.title} ===")
        try:
            det.load()
        except Exception as exc:
            print(f"SKIP {model_id}: {exc}")
            rows.append({
                "model": model_id,
                "title": det.title,
                "error": str(exc),
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "fps": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "n": 0.0,
            })
            continue
        times: list[float] = []
        per_image = []
        csv_path = RESULTS / f"bench_{model_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "infer_ms", "n_pred", "n_gt", "names"])
            assert det.predict is not None
            for i, path in enumerate(images, start=1):
                rgb = load_rgb(path)
                hits, infer_ms = det.predict(rgb, args.conf, args.imgsz)
                times.append(infer_ms)
                gt = load_gt_xyxy(path, rgb.shape[:2])
                per_image.append((gt, hits))
                writer.writerow([
                    path.name, f"{infer_ms:.2f}", len(hits), len(gt),
                    ",".join(h["name"] for h in hits),
                ])
                if i % 10 == 0 or i == len(images):
                    print(f"  {i}/{len(images)}  {infer_ms:.0f} ms  pred={len(hits)}")
        det.close()
        gc.collect()
        latency = summarize_times(times)
        acc = score_dataset(per_image)
        row = {
            "model": model_id,
            "title": det.title,
            "error": "",
            **latency,
            **acc,
        }
        rows.append(row)
        print(
            f"  mean={latency['mean_ms']:.1f} ms  fps={latency['fps']:.2f}  "
            f"mAP50={acc['map50']:.3f}  mAP50-95={acc['map50_95']:.3f}  "
            f"P={acc['precision']:.3f} R={acc['recall']:.3f}"
        )

    payload = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "n_images": len(images),
        "test_set": "coco128",
        "image_list": str(HERE / "testdata" / "IMAGE_LIST.txt"),
        "models": rows,
    }
    BENCH_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = RESULTS / "bench_table.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model", "title", "n", "mean_ms", "p50_ms", "fps",
                "map50", "map50_95", "precision", "recall", "tp", "fp", "fn", "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    ok_rows = [r for r in rows if not r.get("error")]
    graphs = save_graphs(ok_rows) if ok_rows else []
    print(f"\nJSON: {BENCH_JSON}")
    print(f"table: {table}")
    for path in graphs:
        print(f"graph: {path}")


def cmd_report(_args: argparse.Namespace) -> None:
    if not BENCH_JSON.exists():
        raise SystemExit("No bench.json yet. Run: python3 compare.py bench")
    payload = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    rows = [r for r in payload["models"] if not r.get("error")]
    graphs = save_graphs(rows)
    print(f"test set: {payload.get('n_images')} images ({payload.get('test_set')})")
    print(
        f"{'model':<22} {'ms':>8} {'FPS':>7} {'mAP50':>8} {'mAP5095':>8} {'P':>6} {'R':>6}"
    )
    for row in payload["models"]:
        if row.get("error"):
            print(f"{row['title']:<22} SKIP {row['error'][:60]}")
            continue
        print(
            f"{row['title']:<22} {row['mean_ms']:8.1f} {row['fps']:7.2f} "
            f"{row['map50']:8.3f} {row['map50_95']:8.3f} "
            f"{row['precision']:6.3f} {row['recall']:6.3f}"
        )
    if LIVE_SUMMARY.exists():
        print(f"\nlive camera sessions: {LIVE_SUMMARY}")
        print(LIVE_SUMMARY.read_text(encoding="utf-8"))
    for path in graphs:
        print(f"graph: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debian OS Yolo model comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)

    live = sub.add_parser("live", help="one model, live CSI window")
    live.add_argument("--model", required=True, choices=ORDER)
    live.add_argument("--imgsz", type=int, default=320)
    live.add_argument("--conf", type=float, default=0.35)
    live.add_argument("--width", type=int, default=640)
    live.add_argument("--height", type=int, default=480)
    live.add_argument("--max-frames", type=int, default=0)
    live.set_defaults(func=cmd_live)

    bench = sub.add_parser("bench", help="all models on the same coco128 set + graphs")
    bench.add_argument("--imgsz", type=int, default=320)
    bench.add_argument("--conf", type=float, default=0.35)
    bench.add_argument("--max-images", type=int, default=0, help="0 = all coco128 images")
    bench.add_argument("--models", nargs="+", choices=ORDER, default=None)
    bench.set_defaults(func=cmd_bench)

    report = sub.add_parser("report", help="reprint table and regenerate graphs")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
