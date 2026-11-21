#!/usr/bin/env python3
"""Live CSI YOLO bench: inference ms, FPS, confidence. Ctrl+C to stop."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from box_model import default_conf, default_imgsz, detect_rgb, draw_hits, get_model, model_path
from camera_csi import close_camera, open_camera, read_rgb

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HITS = RESULTS / "hits"


def summarize(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {"n": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "fps": 0}
    ordered = sorted(ms)
    p95_i = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    mean = statistics.fmean(ms)
    return {
        "n": float(len(ms)),
        "mean_ms": mean,
        "p50_ms": statistics.median(ms),
        "p95_ms": ordered[p95_i],
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi CSI camera + box YOLO bench")
    parser.add_argument("--imgsz", type=int, default=default_imgsz())
    parser.add_argument("--conf", type=float, default=default_conf())
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = until Ctrl+C")
    parser.add_argument("--save-hits", action="store_true", default=True)
    parser.add_argument("--no-save-hits", action="store_false", dest="save_hits")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    HITS.mkdir(parents=True, exist_ok=True)
    get_model()
    kind, cam = open_camera((args.width, args.height))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS / f"bench_camera_{stamp}.csv"
    print(f"backend={kind} weights={model_path()} imgsz={args.imgsz} conf={args.conf}")
    print(f"csv={csv_path}")

    times: list[float] = []
    hits_n = 0
    miss_n = 0
    confs: list[float] = []
    frame_i = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "frame", "infer_ms", "hit", "n_boxes", "max_conf", "names",
            ])
            while True:
                rgb = read_rgb(kind, cam)
                if rgb is None:
                    print("empty frame")
                    time.sleep(0.05)
                    continue
                hits, infer_ms = detect_rgb(rgb, imgsz=args.imgsz, conf=args.conf)
                frame_i += 1
                if frame_i <= args.warmup:
                    print(f"warmup {frame_i}/{args.warmup} {infer_ms:.1f} ms")
                    continue
                times.append(infer_ms)
                hit = bool(hits)
                if hit:
                    hits_n += 1
                    confs.extend(h["confidence"] for h in hits)
                    if args.save_hits:
                        from PIL import Image

                        drawn = draw_hits(rgb, hits)
                        dest = HITS / f"hit_{stamp}_{frame_i:05d}.jpg"
                        Image.fromarray(drawn).save(dest, quality=85)
                else:
                    miss_n += 1
                max_conf = max((h["confidence"] for h in hits), default=0.0)
                names = ",".join(h["name"] for h in hits)
                writer.writerow([
                    frame_i, f"{infer_ms:.2f}", int(hit), len(hits),
                    f"{max_conf:.3f}", names,
                ])
                fh.flush()
                tag = "HIT " if hit else "miss"
                print(
                    f"{tag} frame={frame_i} {infer_ms:.1f} ms "
                    f"fps={1000.0 / infer_ms:.2f} boxes={len(hits)} conf={max_conf:.2f}"
                )
                if args.max_frames and (frame_i - args.warmup) >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        close_camera(kind, cam)

    stats = summarize(times)
    print("--- summary ---")
    print(f"frames={int(stats['n'])} hits={hits_n} miss={miss_n}")
    print(
        f"infer mean={stats['mean_ms']:.1f} ms  p50={stats['p50_ms']:.1f}  "
        f"p95={stats['p95_ms']:.1f}  ~{stats['fps']:.2f} FPS"
    )
    if confs:
        print(
            f"hit conf mean={statistics.fmean(confs):.3f}  "
            f"max={max(confs):.3f}  n={len(confs)}"
        )
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
