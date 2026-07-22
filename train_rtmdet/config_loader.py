from __future__ import annotations

from pathlib import Path
from typing import Any

import warnings

import yaml


_CONFIG_FILENAME = "hyperparameter_config.yaml"


def _clamp_num_gpus(requested: int) -> int:
    try:
        import torch
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        available = 0
    available = max(available, 1)  # at least 1 for CPU/non-CUDA runs
    if requested > available:
        warnings.warn(
            f"num_gpus={requested} requested but only {available} GPU(s) detected; "
            f"clamping to {available}.",
            stacklevel=3,
        )
        return available
    return requested


def _find_config_file(start: Path | None = None) -> Path:
    search = start or Path.cwd()
    candidates = [
        search / _CONFIG_FILENAME,
        Path(__file__).resolve().parent.parent / _CONFIG_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Config file '{_CONFIG_FILENAME}' not found. "
        "Place it in the project root or pass the path explicitly."
    )


def _to_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def load_pipeline_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else _find_config_file()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    default_run_dir = Path.cwd() / "runs" / "rtmdet"
    default_pkg_dir = Path.cwd() / "models" / "rtmdet"

    ds = raw.get("dataset", {})
    mdl = raw.get("model", {})
    tr = raw.get("training", {})
    pl = raw.get("pipeline", {})
    onnx = raw.get("onnx_export", {})
    paths = raw.get("paths", {})
    pre = raw.get("preprocessing", {})

    # class_names: YAML list or comma-separated string → list | None
    class_names_raw = ds.get("class_names")
    if isinstance(class_names_raw, list):
        class_names = [str(n).strip() for n in class_names_raw if str(n).strip()] or None
    elif isinstance(class_names_raw, str) and class_names_raw.strip():
        class_names = [n.strip() for n in class_names_raw.split(",") if n.strip()] or None
    else:
        class_names = None

    project_dir_raw = _to_path(paths.get("project_dir"))
    package_dir_raw = _to_path(paths.get("package_dir"))

    # class_weights: optional list of floats, one per class in class_names order.
    class_weights_raw = tr.get("class_weights")
    if isinstance(class_weights_raw, list) and class_weights_raw:
        class_weights = [float(w) for w in class_weights_raw]
    else:
        class_weights = None

    return {
        # ── dataset ──────────────────────────────────────────────────────────
        "dataset_path": _to_path(ds.get("dataset_path")),
        "class_names": class_names,
        "nc": ds.get("nc"),
        # ── model ────────────────────────────────────────────────────────────
        "model_name": str(mdl.get("model_name", "rtmdet_s_640")).strip(),
        "variant": str(mdl.get("variant", "s")).strip(),
        "imgsz": int(mdl.get("imgsz", 640)),
        "pretrained_checkpoint": _to_path(mdl.get("pretrained_checkpoint")),
        # ── training ─────────────────────────────────────────────────────────
        "epochs": int(tr.get("epochs", 200)),
        "stage2_epochs": int(tr.get("stage2_epochs", 35)),
        "batch_size": int(tr.get("batch_size", 32)),
        "val_batch_size": int(tr.get("val_batch_size", 8)),
        "workers": int(tr.get("workers", 8)),
        "val_interval": int(tr.get("val_interval", 5)),
        "base_lr": float(tr.get("base_lr", 0.001)),
        "device": str(tr.get("device", "cuda")).strip(),
        "num_gpus": _clamp_num_gpus(int(tr.get("num_gpus", 1))),
        "seed": int(tr.get("seed", 1)),
        "amp": bool(tr.get("amp", True)),
        "resume": bool(tr.get("resume", False)),
        "logger_interval": int(tr.get("logger_interval", 100)),
        "early_stopping": bool(tr.get("early_stopping", False)),
        "early_stopping_patience": int(tr.get("early_stopping_patience", 20)),
        "early_stopping_min_delta": float(tr.get("early_stopping_min_delta", 0.001)),
        "class_weights": class_weights,
        # ── pipeline ─────────────────────────────────────────────────────────
        "prepare_only": bool(pl.get("prepare_only", False)),
        "run_training": bool(pl.get("run_training", True)),
        "run_evaluation": bool(pl.get("run_evaluation", False)),
        "save_onnx_weights": bool(pl.get("save_onnx_weights", False)),
        "run_packaging": bool(pl.get("run_packaging", True)),
        "generate_plots": bool(pl.get("generate_plots", True)),
        "eval_conf_threshold": float(pl.get("eval_conf_threshold", 0.25)),
        "eval_iou_threshold": float(pl.get("eval_iou_threshold", 0.50)),
        # ── onnx export ──────────────────────────────────────────────────────
        "onnx_score_threshold": float(onnx.get("score_threshold", 0.05)),
        "onnx_iou_threshold": float(onnx.get("iou_threshold", 0.5)),
        "onnx_keep_top_k": int(onnx.get("keep_top_k", 300)),
        # ── paths ────────────────────────────────────────────────────────────
        "project_dir": project_dir_raw or default_run_dir,
        "package_dir": package_dir_raw or default_pkg_dir,
        "mmdet_root": _to_path(paths.get("mmdet_root")),
        "mmdeploy_root": _to_path(paths.get("mmdeploy_root")),
        # ── preprocessing ────────────────────────────────────────────────────
        "normalize_names": bool(pre.get("normalize_names", False)),
        "convert_segments_to_boxes": bool(pre.get("convert_segments_to_boxes", True)),
        "allow_missing_labels": bool(pre.get("allow_missing_labels", True)),
        "stop_on_validation_errors": bool(pre.get("stop_on_validation_errors", True)),
    }
