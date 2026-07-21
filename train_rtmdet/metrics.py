"""
Per-threshold detection metrics: Precision / Recall / F1 / TP / FP / FN.

COCO AP metrics (mAP50-95, mAP50, …) average over many IoU and score thresholds.
These metrics add an evaluation at a single, deployment-realistic operating point
(e.g. conf ≥ 0.25, IoU ≥ 0.50) so you can reason about false alarms and misses
directly in terms of your application constraints.

The computation reuses the confusion matrix produced by plots.compute_confusion_matrix,
which already handles conf-threshold filtering and IoU-based matching.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def threshold_metrics_from_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    num_images: int,
) -> dict[str, Any]:
    """Derive per-class and macro P / R / F1 / TP / FP / FN from a confusion matrix.

    The matrix must follow the convention used by plots.compute_confusion_matrix:
      - rows   = actual class (GT)
      - columns = predicted class
      - index  num_classes → background (unmatched)

    Derivation
    ----------
    For class c in a (C+1)×(C+1) matrix M:
      TP_c = M[c, c]
      FP_c = Σ_{r≠c} M[r, c]   (column c, all rows except c → wrong-class or spurious dets)
      FN_c = Σ_{p≠c} M[c, p]   (row c, all cols except c → missed or wrong-class GTs)

    Returns
    -------
    dict with keys:
      "classes" → {class_name: {tp, fp, fn, precision, recall, f1}}
      "macro"   → {precision, recall, f1}   (mean over classes that have ≥1 GT)
      "overall" → {tp, fp, fn, fp_per_image, fn_per_image}
    """
    n = len(class_names)

    per_class: dict[str, dict[str, Any]] = {}
    macro_p: list[float] = []
    macro_r: list[float] = []
    macro_f1: list[float] = []

    for c, name in enumerate(class_names):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum()) - tp   # all detections as class c minus TP
        fn = int(matrix[c, :].sum()) - tp   # all GT of class c minus TP

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

        # Only include in macro if the class has at least one GT instance
        if (tp + fn) > 0:
            macro_p.append(precision)
            macro_r.append(recall)
            macro_f1.append(f1)

    macro = {
        "precision": round(float(np.mean(macro_p)),  4) if macro_p  else 0.0,
        "recall":    round(float(np.mean(macro_r)),  4) if macro_r  else 0.0,
        "f1":        round(float(np.mean(macro_f1)), 4) if macro_f1 else 0.0,
    }

    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())

    overall = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "fp_per_image": round(total_fp / num_images, 3) if num_images > 0 else 0.0,
        "fn_per_image": round(total_fn / num_images, 3) if num_images > 0 else 0.0,
    }

    return {"classes": per_class, "macro": macro, "overall": overall}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_threshold_report(
    metrics: dict[str, Any],
    class_names: list[str],
    conf_thr: float,
    iou_thr: float,
) -> str:
    """Return a human-readable metrics table as a plain-text string."""
    sep = "-" * 72
    col = {"cls": 20, "tp": 6, "fp": 6, "fn": 6, "p": 11, "r": 9, "f1": 9}

    def hdr(label: str, w: int) -> str:
        return f"{label:>{w}}"

    header = (
        f"{'Class':<{col['cls']}}"
        f"{hdr('TP', col['tp'])}"
        f"{hdr('FP', col['fp'])}"
        f"{hdr('FN', col['fn'])}"
        f"{hdr('Precision', col['p'])}"
        f"{hdr('Recall', col['r'])}"
        f"{hdr('F1', col['f1'])}"
    )

    lines = [
        "",
        f"Threshold metrics  (conf ≥ {conf_thr:.2f}  |  IoU ≥ {iou_thr:.2f})",
        sep,
        header,
        sep,
    ]

    per_class = metrics["classes"]
    for name in class_names:
        if name not in per_class:
            continue
        m = per_class[name]
        lines.append(
            f"{name:<{col['cls']}}"
            f"{m['tp']:>{col['tp']}}"
            f"{m['fp']:>{col['fp']}}"
            f"{m['fn']:>{col['fn']}}"
            f"{m['precision']:>{col['p']}.4f}"
            f"{m['recall']:>{col['r']}.4f}"
            f"{m['f1']:>{col['f1']}.4f}"
        )

    mac = metrics["macro"]
    lines += [
        sep,
        (
            f"{'macro avg':<{col['cls']}}"
            f"{'':>{col['tp']}}"
            f"{'':>{col['fp']}}"
            f"{'':>{col['fn']}}"
            f"{mac['precision']:>{col['p']}.4f}"
            f"{mac['recall']:>{col['r']}.4f}"
            f"{mac['f1']:>{col['f1']}.4f}"
        ),
        sep,
    ]

    ovr = metrics["overall"]
    lines += [
        "",
        f"  Total TP : {ovr['tp']}",
        f"  Total FP : {ovr['fp']}   ({ovr['fp_per_image']:.3f} per image)",
        f"  Total FN : {ovr['fn']}   ({ovr['fn_per_image']:.3f} per image)",
    ]

    return "\n".join(lines)
