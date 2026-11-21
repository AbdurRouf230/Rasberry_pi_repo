"""Latency stats, IoU matching, mAP50 / mAP50-95, matplotlib comparison graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """VOC-style interpolated AP."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    gts: list[tuple[str, list[int]]],
    preds: list[dict[str, Any]],
    iou_thr: float,
) -> tuple[float, int, int, int]:
    """One-class-agnostic AP at a single IoU, matching by class name. Returns AP, TP, FP, FN."""
    by_class: dict[str, list[tuple[str, list[int]]]] = {}
    for name, box in gts:
        by_class.setdefault(name, []).append((name, box))
    pred_by: dict[str, list[dict[str, Any]]] = {}
    for hit in preds:
        pred_by.setdefault(hit["name"], []).append(hit)

    tps: list[int] = []
    fps: list[int] = []
    scores: list[float] = []
    n_gt = 0
    names = set(by_class) | set(pred_by)
    for name in names:
        gt_boxes = [b for _, b in by_class.get(name, [])]
        n_gt += len(gt_boxes)
        used = [False] * len(gt_boxes)
        ranked = sorted(pred_by.get(name, []), key=lambda h: h["confidence"], reverse=True)
        for hit in ranked:
            scores.append(float(hit["confidence"]))
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gt_boxes):
                if used[j]:
                    continue
                iou = iou_xyxy(hit["bbox"], gbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thr:
                used[best_j] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)
    fn = n_gt - sum(tps)
    if not scores:
        return (1.0 if n_gt == 0 else 0.0), 0, 0, n_gt
    order = np.argsort(-np.array(scores))
    tps_a = np.cumsum(np.array(tps)[order])
    fps_a = np.cumsum(np.array(fps)[order])
    denom = max(n_gt, 1)
    recalls = tps_a / denom
    precisions = tps_a / np.maximum(tps_a + fps_a, 1e-9)
    ap = _ap_from_pr(recalls, precisions)
    return ap, int(tps_a[-1] if len(tps_a) else 0), int(fps_a[-1] if len(fps_a) else 0), int(fn)


def summarize_times(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    arr = np.array(ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "n": float(len(arr)),
        "mean_ms": mean,
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / mean) if mean > 0 else 0.0,
    }


def _mean_class_ap(
    all_gt: list[tuple[str, list[int]]],
    all_pred: list[dict[str, Any]],
    iou_thr: float,
) -> float:
    names = sorted({name for name, _ in all_gt})
    if not names:
        return 0.0
    aps = []
    for name in names:
        gt = [(n, b) for n, b in all_gt if n == name]
        pred = [h for h in all_pred if h["name"] == name]
        ap, _, _, _ = average_precision(gt, pred, iou_thr)
        aps.append(ap)
    return float(np.mean(aps))


def score_dataset(
    per_image: list[tuple[list[tuple[str, list[int]]], list[dict[str, Any]]]],
) -> dict[str, float]:
    """mAP = mean of per-class AP over classes that appear in coco128 GT."""
    all_gt: list[tuple[str, list[int]]] = []
    all_pred: list[dict[str, Any]] = []
    for gts, preds in per_image:
        all_gt.extend(gts)
        all_pred.extend(preds)
    _, tp, fp, fn = average_precision(all_gt, all_pred, 0.5)
    map50 = _mean_class_ap(all_gt, all_pred, 0.5)
    aps = [_mean_class_ap(all_gt, all_pred, float(thr)) for thr in np.arange(0.5, 0.96, 0.05)]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "map50": float(map50),
        "map50_95": float(np.mean(aps)),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_gt": float(len(all_gt)),
        "n_pred": float(len(all_pred)),
    }


def save_graphs(rows: list[dict[str, Any]], out_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["title"] for r in rows]
    x = np.arange(len(labels))
    saved: list[Path] = []

    def _bar(values: list[float], ylabel: str, title: str, fname: str, ymax: float | None = None):
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(x, values, color="#2563eb")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    _bar([r["mean_ms"] for r in rows], "ms", "Mean inference time (same coco128 set)", "infer_ms.png")
    _bar([r["fps"] for r in rows], "FPS", "Throughput (1000 / mean_ms)", "fps.png")
    _bar(
        [r["map50"] for r in rows],
        "mAP@0.50",
        "mAP50 (same coco128 labels)",
        "map50.png",
        ymax=1.05,
    )
    _bar(
        [r["map50_95"] for r in rows],
        "mAP@0.50:0.95",
        "mAP50-95 (same coco128 labels)",
        "map50_95.png",
        ymax=1.05,
    )
    _bar(
        [r["precision"] for r in rows],
        "Precision",
        "Precision at IoU 0.50, conf threshold used in bench",
        "precision.png",
        ymax=1.05,
    )
    _bar(
        [r["recall"] for r in rows],
        "Recall",
        "Recall at IoU 0.50, conf threshold used in bench",
        "recall.png",
        ymax=1.05,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], [r["mean_ms"] for r in rows], "Inference (ms)", None),
        (axes[0, 1], [r["fps"] for r in rows], "FPS", None),
        (axes[1, 0], [r["map50"] for r in rows], "mAP50", 1.05),
        (axes[1, 1], [r["map50_95"] for r in rows], "mAP50-95", 1.05),
    ]
    for ax, values, title, ymax in panels:
        ax.bar(x, values, color="#0f766e")
        ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        if ymax is not None:
            ax.set_ylim(0, ymax)
    fig.suptitle("Same test set (coco128) — all models")
    fig.tight_layout()
    overview = out_dir / "comparison_overview.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)
    saved.append(overview)
    return saved
