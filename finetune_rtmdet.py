"""
Step 2 of 2 — Fine-tune RTMDet on a custom YOLO dataset.

All hyperparameters and paths are read from iperparameter_config.txt.
Edit that file before running this script.

Usage:
    python finetune_rtmdet.py

Prerequisite:  python normalize_dataset.py  (Step 1)

Pipeline:
    1. Validate dataset structure and annotations.
    2. Convert YOLO .txt labels → COCO JSON.
    3. Generate MMDetection training config.
    4. Train (mmdetection/tools/train.py).
    5. Export best checkpoint to ONNX  [if save_onnx_weights = true].
    6. Package artifacts to models/rtmdet/<model_name>/.
"""

from __future__ import annotations

from pathlib import Path

from train_rtmdet.config_loader import load_pipeline_config
from train_rtmdet.pipeline import RTMDetPipelineConfig, run_rtmdet_pipeline


def main() -> None:
    config_path = Path(__file__).resolve().parent / "hyperparameter_config.yaml"
    cfg = load_pipeline_config(config_path)

    if cfg["dataset_path"] is None:
        raise ValueError(
            "dataset_path is not set in hyperparameter_config.yaml. "
            "Edit the dataset section before running."
        )

    pipeline_config = RTMDetPipelineConfig(
        dataset_path=cfg["dataset_path"],
        model_name=cfg["model_name"],
        variant=cfg["variant"],
        imgsz=cfg["imgsz"],
        logger_interval=cfg["logger_interval"],
        epochs=cfg["epochs"],
        stage2_epochs=cfg["stage2_epochs"],
        batch_size=cfg["batch_size"],
        val_batch_size=cfg["val_batch_size"],
        workers=cfg["workers"],
        val_interval=cfg["val_interval"],
        base_lr=cfg["base_lr"],
        device=cfg["device"],
        seed=cfg["seed"],
        project_dir=cfg["project_dir"],
        package_dir=cfg["package_dir"],
        mmdet_root=cfg["mmdet_root"],
        mmdeploy_root=cfg["mmdeploy_root"],
        pretrained_checkpoint=cfg["pretrained_checkpoint"],
        class_names=cfg["class_names"],
        nc=cfg["nc"],
        normalize_names=cfg["normalize_names"],
        convert_segments_to_boxes=cfg["convert_segments_to_boxes"],
        stop_on_validation_errors=cfg["stop_on_validation_errors"],
        allow_missing_labels=cfg["allow_missing_labels"],
        prepare_only=cfg["prepare_only"],
        run_training=cfg["run_training"],
        run_evaluation=cfg["run_evaluation"],
        run_export=False,
        run_packaging=cfg["run_packaging"],
        amp=cfg["amp"],
        resume=cfg["resume"],
        save_onnx_weights=cfg["save_onnx_weights"],
        onnx_score_threshold=cfg["onnx_score_threshold"],
        onnx_iou_threshold=cfg["onnx_iou_threshold"],
        onnx_keep_top_k=cfg["onnx_keep_top_k"],
        early_stopping=cfg["early_stopping"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
    )

    run_rtmdet_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
