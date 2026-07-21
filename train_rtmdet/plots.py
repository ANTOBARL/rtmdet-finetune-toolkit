"""Plotting utilities for RTMDet training runs.

Generates:
  - training_curves.png  : loss curves, validation mAP, learning rate schedule
  - confusion_matrix.png : per-class confusion matrix (row-normalised)

Requires matplotlib and numpy (both standard in training environments).
Everything is lazy-imported so the module can be imported without side effects
even when matplotlib is missing.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


# ── matplotlib singleton ──────────────────────────────────────────────────────

_plt: Any = None


def _get_plt() -> Any:
    global _plt
    if _plt is not None:
        return _plt
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    _plt = plt
    return plt


# ── Log parsing ───────────────────────────────────────────────────────────────

def _iter_scalars(run_dir: Path):
    """Yield all JSON records from the most recent MMEngine scalars log.

    MMEngine writes scalars to  {run_dir}/{timestamp}/vis_data/scalars.json.
    When multiple timestamp directories exist (resumed runs), the most recently
    modified file wins.
    """
    # Current layout: logs/{timestamp}/vis_data/scalars.json (see tidy_checkpoints_dir
    # in pipeline.py). Older layouts kept as fallback for runs from before that move.
    log_files = sorted(
        list(run_dir.glob("logs/*/vis_data/scalars.json"))
        or list(run_dir.glob("checkpoints/*/vis_data/scalars.json"))
        or list(run_dir.glob("*/vis_data/scalars.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if not log_files:
        return
    with log_files[-1].open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def parse_training_log(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (train_records, val_records) from the MMEngine scalars log.

    Train records contain: loss, loss_cls, loss_bbox, lr, epoch, iter, step
    Val   records contain: coco/bbox_mAP*, step  (step == epoch at eval time)
    """
    train_recs: list[dict] = []
    val_recs: list[dict] = []
    for rec in _iter_scalars(run_dir):
        if "coco/bbox_mAP" in rec:
            val_recs.append(rec)
        elif "loss" in rec:
            train_recs.append(rec)
    return train_recs, val_recs


# ── Training curves ───────────────────────────────────────────────────────────

def _find_switch_iter(train_recs: list[dict], switch_epoch: int) -> int | None:
    """Return the global iteration closest to the start of *switch_epoch*."""
    if not train_recs:
        return None
    # Training records from MMEngine carry an 'epoch' field alongside 'step'.
    for r in train_recs:
        if r.get("epoch", -1) >= switch_epoch:
            return int(r["step"])
    # Fallback: linear estimate when epoch field is absent
    max_step = max((r.get("step", 0) for r in train_recs), default=0)
    max_ep = max((r.get("epoch", 0) for r in train_recs), default=0)
    if max_ep > 0 and max_step > 0:
        return int(switch_epoch * max_step / max_ep)
    return None


def _draw_switch_line(ax: Any, x: float | int, label: str = "Stage 2\n(no mosaic)") -> None:
    """Draw a labelled vertical dashed line at *x* on *ax*."""
    ax.axvline(x, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.65, zorder=3)
    ax.text(
        x, 0.97, label,
        transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=6.5, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )


def plot_training_curves(
    train_recs: list[dict],
    val_recs: list[dict],
    out_path: Path,
    model_name: str = "",
    switch_epoch: int | None = None,
) -> Path | None:
    """Save training curves to *out_path*. Returns the path, or None if no data."""
    if not train_recs and not val_recs:
        return None

    plt = _get_plt()

    # Collect all loss keys present across records
    loss_keys: list[str] = []
    if train_recs:
        all_loss = set()
        for r in train_recs:
            all_loss.update(k for k in r if k.startswith("loss"))
        loss_keys = sorted(all_loss, key=lambda k: (k != "loss", k))

    has_loss = bool(loss_keys)
    has_val = bool(val_recs)
    has_lr = bool(train_recs and any("lr" in r for r in train_recs))

    n_plots = sum([has_loss, has_val, has_lr])
    if n_plots == 0:
        return None

    # Pre-compute switch iteration once for iter-based subplots
    switch_iter: int | None = None
    if switch_epoch is not None and train_recs:
        switch_iter = _find_switch_iter(train_recs, switch_epoch)

    fig, axes = plt.subplots(n_plots, 1, figsize=(13, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    prefix = f"{model_name} — " if model_name else ""

    # ── Loss ────────────────────────────────────────────────────────────────
    if has_loss:
        ax = axes[ax_idx]; ax_idx += 1
        steps = [r["step"] for r in train_recs]
        for key in loss_keys:
            ys = [r.get(key) for r in train_recs]
            pairs = [(x, y) for x, y in zip(steps, ys) if y is not None]
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, label=key, linewidth=2 if key == "loss" else 1, alpha=0.85)
        if switch_iter is not None:
            _draw_switch_line(ax, switch_iter)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"{prefix}Training Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Validation mAP ──────────────────────────────────────────────────────
    if has_val:
        ax = axes[ax_idx]; ax_idx += 1
        map_keys = [
            ("coco/bbox_mAP_50",  "mAP50",    "tab:blue"),
            ("coco/bbox_mAP",     "mAP50-95", "tab:orange"),
            ("coco/bbox_mAP_75",  "mAP75",    "tab:green"),
        ]
        # x-axis: step field in val records == epoch at eval time
        xs_all = [r["step"] for r in val_recs]
        plotted_any = False
        for key, label, color in map_keys:
            ys = [r.get(key) for r in val_recs]
            pairs = [(x, float(y)) for x, y in zip(xs_all, ys)
                     if y is not None and float(y) >= 0]
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, marker="o", markersize=3, label=label, color=color)
                plotted_any = True
        if switch_epoch is not None:
            _draw_switch_line(ax, switch_epoch)
        if plotted_any:
            # Mark best epoch
            best_idx = max(range(len(val_recs)),
                           key=lambda i: float(val_recs[i].get("coco/bbox_mAP", -1)))
            bx = val_recs[best_idx]["step"]
            by = float(val_recs[best_idx].get("coco/bbox_mAP", 0))
            ax.axvline(bx, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.annotate(
                f"best epoch {bx}\nmAP50-95={by:.3f}",
                xy=(bx, by), xytext=(10, -20), textcoords="offset points",
                fontsize=7, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title(f"{prefix}Validation mAP")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Learning rate ────────────────────────────────────────────────────────
    if has_lr:
        ax = axes[ax_idx]; ax_idx += 1
        steps = [r["step"] for r in train_recs]
        lrs = [r.get("lr") for r in train_recs]
        pairs = [(x, y) for x, y in zip(steps, lrs) if y is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color="tab:purple", linewidth=1.2)
        if switch_iter is not None:
            _draw_switch_line(ax, switch_iter)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{prefix}Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Confusion matrix helpers ──────────────────────────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box [x1,y1,x2,y2] against N boxes (N,4) in the same format."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a1 + a2 - inter
    return np.where(union > 0, inter / union, 0.0)


def _load_coco_gt(coco_json: Path) -> dict[int, dict]:
    """Return dict[image_id → {boxes: (N,4) xyxy float32, labels: (N,) int32}]."""
    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cat_to_label = {cat["id"]: i for i, cat in enumerate(data["categories"])}
    gt: dict[int, dict] = {img["id"]: {"boxes": [], "labels": []} for img in data["images"]}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in gt:
            continue
        x, y, w, h = ann["bbox"]
        gt[iid]["boxes"].append([x, y, x + w, y + h])
        gt[iid]["labels"].append(cat_to_label[ann["category_id"]])
    for v in gt.values():
        v["boxes"] = np.array(v["boxes"], dtype=np.float32).reshape(-1, 4)
        v["labels"] = np.array(v["labels"], dtype=np.int32)
    return gt


def _load_predictions(pkl_path: Path) -> list[dict]:
    """Load MMDet 3.x DetDataSample list saved with ``--out result.pkl``."""
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    results: list[dict] = []
    for item in raw:
        meta = getattr(item, "metainfo", {}) or {}
        img_id = meta.get("img_id") or meta.get("id")
        pi = getattr(item, "pred_instances", None)
        if pi is not None and hasattr(pi, "bboxes"):
            bboxes = _to_numpy(pi.bboxes).reshape(-1, 4).astype(np.float32)
            labels = _to_numpy(pi.labels).ravel().astype(np.int32)
            scores = _to_numpy(pi.scores).ravel().astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int32)
            scores = np.zeros(0, dtype=np.float32)
        results.append({"img_id": img_id, "bboxes": bboxes, "labels": labels, "scores": scores})
    return results


def compute_confusion_matrix(
    predictions: list[dict],
    gt: dict[int, dict],
    num_classes: int,
    iou_thr: float = 0.45,
    score_thr: float = 0.25,
) -> np.ndarray:
    """Return a (num_classes+1) × (num_classes+1) confusion matrix.

    Rows = actual class; columns = predicted class.
    Index *num_classes* represents the background class.

    Matching strategy (YOLO-style):
      - TP : prediction matched to GT by best IoU ≥ iou_thr
      - FN : unmatched GT  → row = gt_label,   col = background
      - FP : unmatched det → row = background, col = pred_label
    """
    bg = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for pred in predictions:
        img_id = pred["img_id"]
        entry = gt.get(img_id) if img_id is not None else None
        gt_boxes = entry["boxes"] if entry is not None else np.zeros((0, 4), np.float32)
        gt_labels = entry["labels"] if entry is not None else np.zeros(0, np.int32)

        mask = pred["scores"] >= score_thr
        pb = pred["bboxes"][mask]
        pl = pred["labels"][mask]

        gt_hit = np.zeros(len(gt_labels), bool)
        pd_hit = np.zeros(len(pl), bool)

        for pi_i, (pbox, plabel) in enumerate(zip(pb, pl)):
            if len(gt_boxes) == 0:
                break
            ious = _bbox_iou(pbox, gt_boxes)
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                gt_hit[best] = True
                pd_hit[pi_i] = True
                matrix[int(gt_labels[best]), int(plabel)] += 1

        for gi, gl in enumerate(gt_labels):
            if not gt_hit[gi]:
                matrix[int(gl), bg] += 1       # FN

        for pi_i, plabel in enumerate(pl):
            if not pd_hit[pi_i]:
                matrix[bg, int(plabel)] += 1   # FP

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    out_path: Path,
    normalize: bool = True,
) -> Path:
    """Save a confusion matrix heatmap to *out_path*."""
    plt = _get_plt()
    labels = list(class_names) + ["background"]
    n = len(labels)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.where(row_sums > 0, matrix.astype(float) / row_sums, 0.0)
        vmax = 1.0
        title = "Confusion Matrix (row-normalised)"
    else:
        display = matrix.astype(float)
        vmax = float(matrix.max()) or 1.0
        title = "Confusion Matrix (counts)"

    cell = max(0.55, min(1.4, 10.0 / n))
    fig, ax = plt.subplots(figsize=(max(6, n * cell + 2.5), max(5, n * cell + 1.5)))

    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fs_tick = max(5, min(10, 120 // n))
    fs_cell = max(4, min(8, 80 // n))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs_tick)
    ax.set_yticklabels(labels, fontsize=fs_tick)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11)

    thresh = vmax * 0.5
    show_text = n <= 40
    if show_text:
        for i in range(n):
            for j in range(n):
                val = display[i, j]
                if val == 0:
                    continue
                txt = f"{val:.2f}" if normalize else str(int(matrix[i, j]))
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=fs_cell, color="white" if val > thresh else "black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Per-class bar chart (from val records with classwise=True) ────────────────

def plot_classwise_map(
    val_recs: list[dict],
    class_names: list[str],
    out_path: Path,
    model_name: str = "",
) -> Path | None:
    """Bar chart of per-class mAP50 at the best epoch (requires classwise=True in CocoMetric)."""
    if not val_recs:
        return None

    # Find best epoch record
    best_rec = max(val_recs, key=lambda r: float(r.get("coco/bbox_mAP", -1)))

    # Collect per-class AP50 — MMDet logs them as  "coco/{class_name}_ap50"  or similar
    patterns = [
        lambda c: f"coco/{c}_ap50",
        lambda c: f"coco/bbox_mAP_50/{c}",
        lambda c: f"coco/bbox_{c}_ap50",
        lambda c: f"{c}_ap50",
    ]
    cls_ap50: list[float | None] = []
    for cls_name in class_names:
        val = None
        for pat in patterns:
            key = pat(cls_name)
            if key in best_rec:
                try:
                    v = float(best_rec[key])
                    val = v if v >= 0 else None
                except (TypeError, ValueError):
                    pass
                break
        cls_ap50.append(val)

    if all(v is None for v in cls_ap50):
        return None  # classwise data not in log

    plt = _get_plt()
    valid = [(name, v) for name, v in zip(class_names, cls_ap50) if v is not None]
    names, vals = zip(*sorted(valid, key=lambda t: t[1], reverse=True))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.55 + 2), 5))
    bars = ax.bar(range(len(names)), vals, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{'{}  — '.format(model_name) if model_name else ''}Per-class mAP50 (epoch {best_rec['step']})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_plots(
    run_dir: Path,
    class_names: list[str],
    coco_val_ann: Path | None = None,
    model_name: str = "",
    iou_thr: float = 0.45,
    score_thr: float = 0.25,
    switch_epoch: int | None = None,
    conf_thr: float = 0.25,
) -> list[Path]:
    """Generate all available plots and metrics reports for a training run.

    Always generated (if log exists):
      - plots/training_curves.png
      - plots/per_class_map50.png  (only when classwise=True was set in CocoMetric)

    Generated when val_predictions.pkl + COCO annotation file are present:
      - plots/confusion_matrix.png
      - metrics/threshold_metrics.txt   (P/R/F1/TP/FP/FN at conf_thr, iou_thr)

    Returns a list of paths to the files actually created.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("WARNING: matplotlib not installed — skipping plots. pip install matplotlib")
        return []

    plots_dir = run_dir / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    train_recs, val_recs = parse_training_log(run_dir)

    if train_recs or val_recs:
        path = plot_training_curves(
            train_recs, val_recs,
            plots_dir / "training_curves.png",
            model_name=model_name,
            switch_epoch=switch_epoch,
        )
        if path:
            generated.append(path)

    if val_recs:
        path = plot_classwise_map(
            val_recs, class_names,
            plots_dir / "per_class_map50.png",
            model_name=model_name,
        )
        if path:
            generated.append(path)

    pkl_path = run_dir / "eval" / "val" / "val_predictions.pkl"
    if pkl_path.is_file() and coco_val_ann is not None and coco_val_ann.is_file():
        try:
            gt = _load_coco_gt(coco_val_ann)
            preds = _load_predictions(pkl_path)

            # Confusion matrix uses iou_thr for matching and score_thr for filtering.
            # Threshold metrics use conf_thr (deployment threshold) and iou_thr.
            matrix_cm = compute_confusion_matrix(preds, gt, len(class_names), iou_thr, score_thr)
            path = plot_confusion_matrix(
                matrix_cm, class_names,
                plots_dir / "confusion_matrix.png",
                normalize=True,
            )
            generated.append(path)

            # Per-threshold metrics at the deployment operating point
            matrix_thr = compute_confusion_matrix(preds, gt, len(class_names), iou_thr, conf_thr)
            num_images = len(gt)
            from .metrics import threshold_metrics_from_matrix, format_threshold_report
            thr_metrics = threshold_metrics_from_matrix(matrix_thr, class_names, num_images)
            report = format_threshold_report(thr_metrics, class_names, conf_thr, iou_thr)
            thr_path = run_dir / "metrics" / "threshold_metrics.txt"
            thr_path.parent.mkdir(parents=True, exist_ok=True)
            thr_path.write_text(report + "\n", encoding="utf-8")
            generated.append(thr_path)

        except Exception as exc:
            print(f"WARNING: post-eval metrics generation failed: {exc}")

    return generated
