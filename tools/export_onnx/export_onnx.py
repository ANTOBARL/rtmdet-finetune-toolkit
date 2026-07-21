"""
Standalone ONNX export — convert a trained RTMDet checkpoint to ONNX.

This script is fully independent from the training pipeline.

Requirements (all set in export_config.yaml):
  base_dir          — folder containing the model files
  files.checkpoint  — .pth checkpoint filename (or absolute path)
  files.mmdet_config — MMDetection .py config filename (or absolute path)
  files.sample_image — any image from the dataset (or absolute path)
  paths.mmdeploy_root — path to the MMDeploy clone

Usage:
  1. Edit export_config.yaml
  2. python tools/export_onnx/export_onnx.py

Output (inside files.output_dir, default: <base_dir>/export_onnx/):
  end2end.onnx   — model with NMS baked in
  pipeline.json  — pre/post-processing metadata
  deploy.json    — backend metadata
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "export_config.yaml"

sys.path.insert(0, str(REPO_ROOT))

from train_rtmdet.export.onnx import run_onnx_export, validate_onnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def _resolve(filename: Any, base_dir: Path | None) -> Path | None:
    """Resolve a filename against base_dir.
    If filename is already absolute, base_dir is ignored.
    Returns None if filename is null/empty.
    """
    p = _to_path(filename)
    if p is None:
        return None
    if p.is_absolute():
        return p.resolve()
    if base_dir is None:
        raise ValueError(
            f"Cannot resolve relative path '{p}' — base_dir is not set in export_config.yaml."
        )
    return (base_dir / p).resolve()


def _infer_imgsz(config_file: Path, fallback: int = 640) -> int:
    for part in config_file.stem.split("_"):
        if part.isdigit() and int(part) >= 128:
            return int(part)
    return fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()
    files_cfg = cfg.get("files", {})
    mdl = cfg.get("model", {})
    onnx_cfg = cfg.get("onnx", {})
    paths_cfg = cfg.get("paths", {})

    base_dir = _to_path(cfg.get("base_dir"))
    if base_dir:
        base_dir = base_dir.resolve()

    # ── 4 required files ──────────────────────────────────────────────────────

    # 1. Checkpoint
    checkpoint_file = _resolve(files_cfg.get("checkpoint"), base_dir)
    if not checkpoint_file:
        raise ValueError("files.checkpoint is not set in export_config.yaml.")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # 2. MMDetection config
    mmdet_config = _resolve(files_cfg.get("mmdet_config"), base_dir)
    if not mmdet_config:
        raise ValueError("files.mmdet_config is not set in export_config.yaml.")
    if not mmdet_config.is_file():
        raise FileNotFoundError(f"MMDetection config not found: {mmdet_config}")

    # 3. Sample image
    sample_image = _resolve(files_cfg.get("sample_image"), base_dir)
    if not sample_image:
        raise ValueError("files.sample_image is not set in export_config.yaml.")
    if not sample_image.is_file():
        raise FileNotFoundError(f"Sample image not found: {sample_image}")

    # 4. Output dir
    output_raw = _resolve(files_cfg.get("output_dir"), base_dir)
    output_dir = output_raw if output_raw else (
        (base_dir / "export_onnx") if base_dir
        else (checkpoint_file.parent / "export_onnx")
    ).resolve()

    # ── optional paths ────────────────────────────────────────────────────────
    mmdeploy_root_raw = _to_path(paths_cfg.get("mmdeploy_root"))
    mmdeploy_root = (
        mmdeploy_root_raw.resolve() if mmdeploy_root_raw
        else (REPO_ROOT / "mmdeploy").resolve()
    )

    mmdet_root_raw = _to_path(paths_cfg.get("mmdet_root"))
    mmdet_root = mmdet_root_raw.resolve() if (mmdet_root_raw and mmdet_root_raw.is_dir()) else None

    # ── model / onnx params ───────────────────────────────────────────────────
    imgsz_override = mdl.get("imgsz")
    resolved_imgsz = int(imgsz_override) if imgsz_override else _infer_imgsz(mmdet_config)
    device = str(mdl.get("device", "cuda:0")).strip()
    score_threshold = float(onnx_cfg.get("score_threshold", 0.05))
    iou_threshold = float(onnx_cfg.get("iou_threshold", 0.5))
    keep_top_k = int(onnx_cfg.get("keep_top_k", 10))
    do_validate = bool(onnx_cfg.get("validate", True))

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RTMDet -> ONNX EXPORT")
    print("=" * 60)
    print(f"  Checkpoint   : {checkpoint_file}")
    print(f"  Config       : {mmdet_config}")
    print(f"  Sample image : {sample_image}")
    print(f"  Input size   : {resolved_imgsz}x{resolved_imgsz}")
    print(f"  Device       : {device}")
    print(f"  Score thr    : {score_threshold}")
    print(f"  IoU thr      : {iou_threshold}")
    print(f"  Keep top-k   : {keep_top_k}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60 + "\n")

    onnx_file = run_onnx_export(
        checkpoint_file=checkpoint_file,
        mmdet_config=mmdet_config,
        sample_image=sample_image,
        output_dir=output_dir,
        imgsz=resolved_imgsz,
        mmdeploy_root=mmdeploy_root,
        python_exe=sys.executable,
        device=device,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        keep_top_k=keep_top_k,
        mmdet_root=mmdet_root,
    )

    if do_validate:
        validate_onnx(onnx_file, resolved_imgsz)

    print("Export complete.")
    print(f"ONNX: {onnx_file}\n")


if __name__ == "__main__":
    main()
