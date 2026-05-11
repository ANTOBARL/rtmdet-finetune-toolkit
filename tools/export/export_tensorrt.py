# =============================================================================
# IDLE — TensorRT export is not active in the current workflow.
#
# To use this script:
#   1. Complete the training with finetune_rtmdet.py first.
#   2. Install MMDeploy from source with TensorRT support (see README).
#   3. Set the variables in the CONFIGURATION section below.
#   4. Run this file.
#
# Requirements: MMDeploy, TensorRT SDK, TENSORRT_DIR + CUDNN_DIR env vars,
#               trtexec in PATH, MMDeploy custom ops built with MSVC/GCC.
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

from train_rtmdet.pipeline import (
    RTMDetPipelineConfig,
    resolve_deploy_config,
    run_command,
    verify_tensorrt_backend,
    verify_training_environment,
)


# ============================================================
# CONFIGURATION — edit these variables before running.
# ============================================================

# tools/export/ → tools/ → repo root
repo_root = Path(__file__).resolve().parent.parent.parent

checkpoint_path = repo_root / "runs" / "rtmdet" / "your_model_name" / "best_coco_bbox_mAP.pth"

# MMDetection config used during training.
# Leave None to auto-detect from the run manifest.
config_path = None

# Sample image for MMDeploy graph tracing.
# Leave None to auto-detect from the dataset in the run manifest.
sample_image = None

# Output folder for ONNX / TensorRT engine files.
# Leave None → <checkpoint_folder>/export_tensorrt/
work_dir = None

# Input size. Leave None to auto-detect from the run manifest.
imgsz = None

device = "cuda:0"
python_executable = sys.executable
mmdeploy_root = repo_root / "mmdeploy"

# Explicit MMDeploy deploy config. Leave None to use the FP16 static default.
deploy_config = None


# ============================================================
# Implementation
# ============================================================

def load_manifest(manifest_path: Path | None) -> dict | None:
    if manifest_path is None or not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def find_manifest(checkpoint_file: Path) -> Path | None:
    candidate = checkpoint_file.parent / "rtmdet_pipeline_manifest.json"
    return candidate if candidate.is_file() else None


def infer_config_path(checkpoint_file: Path, manifest: dict | None) -> Path:
    if manifest:
        manifest_config = manifest.get("mmdet_config")
        if manifest_config:
            path = Path(manifest_config)
            if path.is_file():
                return path.resolve()

    py_candidates = sorted(checkpoint_file.parent.glob("*.py"))
    if len(py_candidates) == 1:
        return py_candidates[0].resolve()

    if not py_candidates:
        raise FileNotFoundError(
            "MMDetection config not found near the checkpoint. Set config_path explicitly."
        )
    raise FileNotFoundError(
        "Multiple .py configs found near the checkpoint. Set config_path explicitly."
    )


def first_image_in_dir(images_dir: Path) -> Path | None:
    if not images_dir.is_dir():
        return None
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
        matches = sorted(images_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def infer_sample_image(manifest: dict | None) -> Path:
    if manifest:
        dataset_root_raw = manifest.get("dataset_root")
        if dataset_root_raw:
            dataset_root = Path(dataset_root_raw)
            for split_name in ("valid", "val", "test", "train"):
                candidate = first_image_in_dir(dataset_root / split_name / "images")
                if candidate:
                    return candidate

    raise FileNotFoundError(
        "Cannot determine a sample image automatically. Set sample_image explicitly."
    )


def infer_imgsz(config_file: Path, manifest: dict | None) -> int:
    if manifest:
        config_obj = manifest.get("config", {})
        manifest_imgsz = config_obj.get("imgsz")
        if manifest_imgsz not in (None, "None"):
            return int(manifest_imgsz)

    for part in config_file.stem.split("_"):
        if part.isdigit():
            return int(part)

    raise ValueError("Cannot determine imgsz automatically. Set imgsz explicitly.")


def build_export_only_command(
    config: RTMDetPipelineConfig,
    mmdet_config: Path,
    checkpoint_file: Path,
    sample_image_file: Path,
    output_dir: Path,
) -> list[str]:
    if not config.mmdeploy_root:
        raise ValueError("Set mmdeploy_root for TensorRT export.")

    deploy_script = Path(config.mmdeploy_root).resolve() / "tools" / "deploy.py"
    if not deploy_script.is_file():
        raise FileNotFoundError(f"MMDeploy tool not found: {deploy_script}")

    deploy_cfg = resolve_deploy_config(config)

    return [
        str(config.python_executable),
        str(deploy_script),
        str(deploy_cfg),
        str(mmdet_config),
        str(checkpoint_file),
        str(sample_image_file),
        "--work-dir", str(output_dir),
        "--device", config.device,
        "--dump-info",
    ]


def main() -> None:
    checkpoint_file = Path(checkpoint_path).resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if checkpoint_file.suffix.lower() != ".pth":
        raise ValueError(f"File must be a .pth checkpoint: {checkpoint_file}")

    manifest = load_manifest(find_manifest(checkpoint_file))
    mmdet_config = Path(config_path).resolve() if config_path else infer_config_path(checkpoint_file, manifest)
    sample_image_file = Path(sample_image).resolve() if sample_image else infer_sample_image(manifest)
    resolved_imgsz = int(imgsz) if imgsz else infer_imgsz(mmdet_config, manifest)

    model_name = checkpoint_file.parent.name
    output_dir = (
        Path(work_dir).resolve() if work_dir
        else (checkpoint_file.parent / "export_tensorrt").resolve()
    )
    mmdeploy_repo = Path(mmdeploy_root).resolve()

    config = RTMDetPipelineConfig(
        dataset_path=repo_root,
        model_name=model_name,
        imgsz=resolved_imgsz,
        project_dir=checkpoint_file.parent,
        mmdeploy_root=mmdeploy_repo,
        run_export=True,
        run_training=False,
        run_evaluation=False,
        run_packaging=False,
        python_executable=python_executable,
        device=device,
        deploy_config=Path(deploy_config).resolve() if deploy_config else None,
        sample_image=sample_image_file,
        checkpoint_for_export=checkpoint_file,
    )

    verify_training_environment(config)
    verify_tensorrt_backend(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    export_command = build_export_only_command(
        config=config,
        mmdet_config=mmdet_config,
        checkpoint_file=checkpoint_file,
        sample_image_file=sample_image_file,
        output_dir=output_dir,
    )

    print(f"Checkpoint   : {checkpoint_file}")
    print(f"Config       : {mmdet_config}")
    print(f"Sample image : {sample_image_file}")
    print(f"MMDeploy     : {mmdeploy_repo}")
    print(f"Output dir   : {output_dir}")
    run_command(export_command, cwd=mmdeploy_repo)


if __name__ == "__main__":
    main()
