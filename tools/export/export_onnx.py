"""
Standalone ONNX export — use this when you want to (re-)export a checkpoint
without re-running the full training pipeline.

For automatic export right after training, set save_onnx_weights = True
in iperparameter_config.txt and run finetune_rtmdet.py instead.

Output files written to work_dir:
  end2end.onnx   — full model with NMS baked in
  pipeline.json  — pre/post-processing metadata
  deploy.json    — backend metadata

Usage: edit the CONFIGURATION section below, then run this file.

Dependencies: mmdetection, mmdeploy, mmcv, mmengine, torch, onnx, onnxruntime
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from train_rtmdet.export.onnx import run_onnx_export, validate_onnx


# ---------------------------------------------------------------------------
# CONFIGURATION — edit only this section
# ---------------------------------------------------------------------------

# Trained checkpoint (.pth). Relative paths are resolved from the project root.
CHECKPOINT_PATH = r"runs\rtmdet\your_model_name\best_coco_bbox_mAP.pth"

# MMDetection config used during training.
# Leave None to auto-detect from the run manifest.
CONFIG_PATH = None

# Output folder. Leave None → <checkpoint_dir>/export_onnx/
WORK_DIR = None

# Input size. Leave None to auto-detect from the run manifest.
IMGSZ = None

# Inference device.
DEVICE = "cuda:0"

# Python executable with mmengine/mmdet installed.
# Leave None to use sys.executable.
PYTHON_EXECUTABLE = None

# Detection score threshold baked into the ONNX graph.
# Cannot be lowered at inference without re-exporting.
SCORE_THRESHOLD = 0.05

# IoU threshold for NMS baked into the ONNX graph.
IOU_THRESHOLD = 0.5

# Maximum number of detections per image (defines the output tensor size).
KEEP_TOP_K = 300

# Sample image for graph tracing. Leave None to auto-detect from the dataset.
SAMPLE_IMAGE = None

# Run a quick ONNXRuntime inference check after export.
VALIDATE_ONNX = True

# ---------------------------------------------------------------------------
# End of configuration
# ---------------------------------------------------------------------------


# tools/export/ → tools/ → repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MMDEPLOY_ROOT = REPO_ROOT / "mmdeploy"
MMDET_ROOT = REPO_ROOT / "mmdetection"


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def _find_manifest(checkpoint_file: Path) -> Path | None:
    candidate = checkpoint_file.parent / "rtmdet_pipeline_manifest.json"
    return candidate if candidate.is_file() else None


def _load_manifest(manifest_path: Path | None) -> dict | None:
    if manifest_path is None:
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_python(manifest: dict | None) -> str:
    if PYTHON_EXECUTABLE:
        exe = Path(PYTHON_EXECUTABLE)
        if not exe.is_file():
            raise FileNotFoundError(f"PYTHON_EXECUTABLE not found: {exe}")
        return str(exe)
    if manifest:
        exe_raw = manifest.get("config", {}).get("python_executable")
        if exe_raw:
            exe = Path(exe_raw)
            if exe.is_file():
                return str(exe)
    return sys.executable


def _infer_config_path(checkpoint_file: Path, manifest: dict | None) -> Path:
    if manifest:
        raw = manifest.get("mmdet_config")
        if raw:
            p = Path(raw)
            if p.is_file():
                return p.resolve()

    candidates = sorted(checkpoint_file.parent.glob("*.py"))
    if len(candidates) == 1:
        return candidates[0].resolve()

    configs_dir = checkpoint_file.parent / "_configs"
    if configs_dir.is_dir():
        candidates = sorted(configs_dir.glob("*.py"))
        if candidates:
            return max(candidates, key=lambda x: x.stat().st_mtime).resolve()

    if not candidates:
        raise FileNotFoundError(
            "MMDetection config not found automatically.\n"
            "Set CONFIG_PATH at the top of this script."
        )
    raise FileNotFoundError(
        f"Found {len(candidates)} .py files near the checkpoint: pick one and set CONFIG_PATH."
    )


def _infer_imgsz(config_file: Path, manifest: dict | None) -> int:
    if manifest:
        imgsz_raw = manifest.get("config", {}).get("imgsz")
        if imgsz_raw not in (None, "None"):
            return int(imgsz_raw)
    for part in config_file.stem.split("_"):
        if part.isdigit() and int(part) >= 128:
            return int(part)
    return 640


def _find_sample_image(manifest: dict | None, checkpoint_file: Path) -> Path:
    search_roots: list[Path] = []
    if manifest:
        dr = manifest.get("dataset_root")
        if dr:
            search_roots.append(Path(dr))
    search_roots += [REPO_ROOT / "datasets", checkpoint_file.parent, REPO_ROOT]

    for root in search_roots:
        if not root.is_dir():
            continue
        for split in ("val", "valid", "test", "train", "images"):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                imgs = sorted((root / split / "images").glob(ext))
                if not imgs:
                    imgs = sorted((root / split).glob(ext))
                if imgs:
                    return imgs[0].resolve()
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs = sorted(root.glob(ext))
            if imgs:
                return imgs[0].resolve()

    raise FileNotFoundError(
        "Cannot find a sample image for MMDeploy.\n"
        "Set SAMPLE_IMAGE = r'path/to/image.jpg' at the top of this script."
    )


def main() -> None:
    checkpoint_file = _abs(CHECKPOINT_PATH)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if checkpoint_file.suffix.lower() != ".pth":
        raise ValueError(f"File must have .pth extension: {checkpoint_file}")

    manifest = _load_manifest(_find_manifest(checkpoint_file))
    mmdet_config = _abs(CONFIG_PATH) if CONFIG_PATH else _infer_config_path(checkpoint_file, manifest)
    if not mmdet_config.is_file():
        raise FileNotFoundError(f"MMDetection config not found: {mmdet_config}")

    resolved_imgsz = int(IMGSZ) if IMGSZ else _infer_imgsz(mmdet_config, manifest)
    sample_image = _abs(SAMPLE_IMAGE) if SAMPLE_IMAGE else _find_sample_image(manifest, checkpoint_file)
    output_dir = (
        _abs(WORK_DIR) if WORK_DIR
        else (checkpoint_file.parent / "export_onnx").resolve()
    )
    python_exe = _resolve_python(manifest)
    mmdet_root = MMDET_ROOT if MMDET_ROOT.is_dir() else None

    onnx_file = run_onnx_export(
        checkpoint_file=checkpoint_file,
        mmdet_config=mmdet_config,
        sample_image=sample_image,
        output_dir=output_dir,
        imgsz=resolved_imgsz,
        mmdeploy_root=MMDEPLOY_ROOT,
        python_exe=python_exe,
        device=DEVICE,
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        keep_top_k=KEEP_TOP_K,
        mmdet_root=mmdet_root,
    )

    if VALIDATE_ONNX:
        validate_onnx(onnx_file, resolved_imgsz)

    print("Export complete.")
    print(f"ONNX: {onnx_file}\n")


if __name__ == "__main__":
    main()
