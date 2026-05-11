from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any


_CONFIG_FILENAME = "iperparameter_config.txt"


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


def _opt_str(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _opt_path(value: str) -> Path | None:
    stripped = value.strip()
    return Path(stripped).expanduser() if stripped else None


def _opt_int(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped else None


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_pipeline_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else _find_config_file()

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read(str(path), encoding="utf-8")

    default_run_dir = Path.cwd() / "runs" / "rtmdet"
    default_pkg_dir = Path.cwd() / "models" / "rtmdet"

    raw_class_names = parser.get("dataset", "class_names", fallback="").strip()
    class_names = (
        [n.strip() for n in raw_class_names.split(",") if n.strip()]
        if raw_class_names else None
    )

    return {
        # ── dataset ──────────────────────────────────────────────────────────
        "dataset_path": _opt_path(parser.get("dataset", "dataset_path", fallback="")),
        "class_names": class_names,
        "nc": _opt_int(parser.get("dataset", "nc", fallback="")),
        # ── model ────────────────────────────────────────────────────────────
        "model_name": parser.get("model", "model_name", fallback="rtmdet_s_640").strip(),
        "variant": parser.get("model", "variant", fallback="s").strip(),
        "imgsz": int(parser.get("model", "imgsz", fallback="640")),
        "pretrained_checkpoint": _opt_path(
            parser.get("model", "pretrained_checkpoint", fallback="")
        ),
        # ── training ─────────────────────────────────────────────────────────
        "epochs": int(parser.get("training", "epochs", fallback="200")),
        "stage2_epochs": int(parser.get("training", "stage2_epochs", fallback="20")),
        "batch_size": int(parser.get("training", "batch_size", fallback="32")),
        "val_batch_size": int(parser.get("training", "val_batch_size", fallback="8")),
        "workers": int(parser.get("training", "workers", fallback="8")),
        "val_interval": int(parser.get("training", "val_interval", fallback="5")),
        "base_lr": float(parser.get("training", "base_lr", fallback="0.001")),
        "device": parser.get("training", "device", fallback="cuda:0").strip(),
        "seed": int(parser.get("training", "seed", fallback="1")),
        "amp": _to_bool(parser.get("training", "amp", fallback="true")),
        "resume": _to_bool(parser.get("training", "resume", fallback="false")),
        "logger_interval": int(parser.get("training", "logger_interval", fallback="100")),
        # ── pipeline ─────────────────────────────────────────────────────────
        "prepare_only": _to_bool(parser.get("pipeline", "prepare_only", fallback="false")),
        "run_training": _to_bool(parser.get("pipeline", "run_training", fallback="true")),
        "run_evaluation": _to_bool(parser.get("pipeline", "run_evaluation", fallback="false")),
        "save_onnx_weights": _to_bool(parser.get("pipeline", "save_onnx_weights", fallback="false")),
        "run_packaging": _to_bool(parser.get("pipeline", "run_packaging", fallback="true")),
        # ── onnx export ──────────────────────────────────────────────────────
        "onnx_score_threshold": float(
            parser.get("onnx_export", "score_threshold", fallback="0.05")
        ),
        "onnx_iou_threshold": float(
            parser.get("onnx_export", "iou_threshold", fallback="0.5")
        ),
        "onnx_keep_top_k": int(
            parser.get("onnx_export", "keep_top_k", fallback="300")
        ),
        # ── paths ────────────────────────────────────────────────────────────
        "project_dir": (
            _opt_path(parser.get("paths", "project_dir", fallback="")) or default_run_dir
        ),
        "package_dir": (
            _opt_path(parser.get("paths", "package_dir", fallback="")) or default_pkg_dir
        ),
        "mmdet_root": _opt_path(parser.get("paths", "mmdet_root", fallback="")),
        "mmdeploy_root": _opt_path(parser.get("paths", "mmdeploy_root", fallback="")),
        # ── preprocessing ────────────────────────────────────────────────────
        "normalize_names": _to_bool(
            parser.get("preprocessing", "normalize_names", fallback="false")
        ),
        "convert_segments_to_boxes": _to_bool(
            parser.get("preprocessing", "convert_segments_to_boxes", fallback="true")
        ),
        "allow_missing_labels": _to_bool(
            parser.get("preprocessing", "allow_missing_labels", fallback="true")
        ),
        "stop_on_validation_errors": _to_bool(
            parser.get("preprocessing", "stop_on_validation_errors", fallback="true")
        ),
    }
