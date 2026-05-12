"""
Standalone TensorRT export — IDLE (not yet active).

Complete the ONNX export first, then configure this script once the target
hardware (Jetson / desktop GPU) and TensorRT SDK version are defined.

All parameters are read from export_config.txt in the same directory.

Usage:
  1. Edit export_config.txt (set checkpoint_path or project_dir + model_name).
  2. python export_tensorrt.py
"""

from __future__ import annotations

import configparser
import json
import sys
from pathlib import Path

# Repo root is 3 levels up: export_tensorrt/ → tools/ → repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "export_config.txt"

sys.path.insert(0, str(REPO_ROOT))

from train_rtmdet.pipeline import (
    RTMDetPipelineConfig,
    resolve_deploy_config,
    run_command,
    verify_tensorrt_backend,
    verify_training_environment,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> configparser.ConfigParser:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read(str(CONFIG_FILE), encoding="utf-8")
    return parser


def _opt(value: str) -> str | None:
    s = value.strip()
    return s if s else None


def _opt_path(value: str) -> Path | None:
    s = value.strip()
    return Path(s).expanduser() if s else None


def _opt_int(value: str) -> int | None:
    s = value.strip()
    return int(s) if s else None


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _find_manifest(checkpoint_file: Path) -> Path | None:
    candidate = checkpoint_file.parent / "rtmdet_pipeline_manifest.json"
    return candidate if candidate.is_file() else None


def _load_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
    if candidates:
        raise FileNotFoundError(
            "Multiple .py configs found near the checkpoint. Set config_path in export_config.txt."
        )
    raise FileNotFoundError(
        "MMDetection config not found automatically.\n"
        "Check that checkpoint_path points to a valid training run folder."
    )


def _infer_imgsz(config_file: Path, manifest: dict | None, fallback: int = 640) -> int:
    if manifest:
        raw = manifest.get("config", {}).get("imgsz")
        if raw not in (None, "None"):
            return int(raw)
    for part in config_file.stem.split("_"):
        if part.isdigit() and int(part) >= 128:
            return int(part)
    return fallback


def _find_best_checkpoint(project_dir: Path, model_name: str) -> Path:
    run_dir = project_dir / model_name
    candidates = sorted(run_dir.glob("best_coco_bbox_mAP_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No best checkpoint found in {run_dir}.\n"
            "Set checkpoint_path in export_config.txt."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


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
        for split in ("val", "valid", "test", "train"):
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
        "Set sample_image in export_config.txt [paths]."
    )


def _build_export_command(
    mmdeploy_root: Path,
    deploy_cfg: Path,
    mmdet_config: Path,
    checkpoint_file: Path,
    sample_image: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    deploy_script = mmdeploy_root / "tools" / "deploy.py"
    if not deploy_script.is_file():
        raise FileNotFoundError(f"MMDeploy tool not found: {deploy_script}")
    return [
        sys.executable,
        str(deploy_script),
        str(deploy_cfg),
        str(mmdet_config),
        str(checkpoint_file),
        str(sample_image),
        "--work-dir", str(output_dir),
        "--device", device,
        "--dump-info",
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _load_config()

    # ── checkpoint ────────────────────────────────────────────────────────────
    checkpoint_path = _opt_path(parser.get("checkpoint", "checkpoint_path", fallback=""))
    if checkpoint_path:
        checkpoint_file = checkpoint_path.resolve()
    else:
        project_dir = _opt_path(parser.get("checkpoint", "project_dir", fallback=""))
        model_name = _opt(parser.get("checkpoint", "model_name", fallback=""))
        if not project_dir or not model_name:
            raise ValueError(
                "Set checkpoint_path (or both project_dir + model_name) in export_config.txt."
            )
        checkpoint_file = _find_best_checkpoint(project_dir, model_name)

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # ── manifest + auto-detection ─────────────────────────────────────────────
    manifest = _load_manifest(_find_manifest(checkpoint_file))
    mmdet_config = _infer_config_path(checkpoint_file, manifest)

    imgsz_override = _opt_int(parser.get("model", "imgsz", fallback=""))
    resolved_imgsz = imgsz_override if imgsz_override else _infer_imgsz(mmdet_config, manifest)

    sample_image_cfg = _opt_path(parser.get("paths", "sample_image", fallback=""))
    sample_image = (
        sample_image_cfg.resolve() if sample_image_cfg
        else _find_sample_image(manifest, checkpoint_file)
    )

    work_dir_cfg = _opt_path(parser.get("output", "work_dir", fallback=""))
    output_dir = (
        work_dir_cfg.resolve() if work_dir_cfg
        else (checkpoint_file.parent / "export_tensorrt").resolve()
    )

    mmdeploy_root_cfg = _opt_path(parser.get("paths", "mmdeploy_root", fallback=""))
    mmdeploy_root = mmdeploy_root_cfg.resolve() if mmdeploy_root_cfg else REPO_ROOT / "mmdeploy"

    mmdet_root_cfg = _opt_path(parser.get("paths", "mmdet_root", fallback=""))

    device = parser.get("model", "device", fallback="cuda:0").strip()

    deploy_config_cfg = _opt_path(parser.get("paths", "deploy_config", fallback=""))

    # ── pipeline config (needed for resolve_deploy_config / verify helpers) ───
    config = RTMDetPipelineConfig(
        dataset_path=REPO_ROOT,
        model_name=checkpoint_file.parent.name,
        imgsz=resolved_imgsz,
        project_dir=checkpoint_file.parent,
        mmdeploy_root=mmdeploy_root,
        mmdet_root=mmdet_root_cfg.resolve() if mmdet_root_cfg else None,
        run_export=True,
        run_training=False,
        run_evaluation=False,
        run_packaging=False,
        python_executable=sys.executable,
        device=device,
        deploy_config=deploy_config_cfg.resolve() if deploy_config_cfg else None,
        sample_image=sample_image,
        checkpoint_for_export=checkpoint_file,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RTMDet -> TensorRT EXPORT")
    print("=" * 60)
    print(f"  Checkpoint   : {checkpoint_file}")
    print(f"  Config       : {mmdet_config}")
    print(f"  Sample image : {sample_image}")
    print(f"  Input size   : {resolved_imgsz}x{resolved_imgsz}")
    print(f"  Device       : {device}")
    print(f"  MMDeploy     : {mmdeploy_root}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60 + "\n")

    verify_training_environment(config)
    verify_tensorrt_backend(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    deploy_cfg = resolve_deploy_config(config)

    export_command = _build_export_command(
        mmdeploy_root=mmdeploy_root,
        deploy_cfg=deploy_cfg,
        mmdet_config=mmdet_config,
        checkpoint_file=checkpoint_file,
        sample_image=sample_image,
        output_dir=output_dir,
        device=device,
    )

    run_command(export_command, cwd=mmdeploy_root)
    print(f"\nTensorRT export complete. Engine files written to:\n  {output_dir}\n")


if __name__ == "__main__":
    main()
