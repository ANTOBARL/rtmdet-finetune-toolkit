"""
Optional preprocessing tool — Convert a raw YOLO dataset to RTMDet/MMDetection format.

This creates a <dataset_name>_rtmdet/ copy with:
  - images and labels preserved
  - COCO JSON annotations generated (annotations/instances_train.json, etc.)
  - classes.txt and data.yaml

Run this ONCE before normalize_dataset.py if your dataset is not yet
in the _rtmdet format. Otherwise, the pipeline in finetune_rtmdet.py
handles COCO conversion automatically.

Usage: edit the variables below, then run this file.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml

from train_rtmdet.pipeline import (
    RTMDetPipelineConfig,
    collect_image_files,
    convert_yolo_dataset_to_coco,
    detect_split_dirs,
    parse_names_from_yaml,
    resolve_dataset_root,
    validate_yolo_dataset,
)


# ============================================================
# Edit these variables, then run this file.
# ============================================================

dataset_path = Path(r"C:\path\to\your\YOLO_dataset")

# If None, creates a sibling folder:  Fire-Smoke → Fire-Smoke_rtmdet
output_path = None

# Set True only if you want to overwrite an existing _rtmdet folder.
overwrite_output = False

# Keep original .txt YOLO labels in train/labels, valid/labels, test/labels.
copy_yolo_labels = True

# Convert YOLO segmentation rows to bounding boxes automatically.
convert_segments_to_boxes = True

# Accept images without a .txt label as negatives.
allow_missing_labels = True


# ============================================================
# Implementation — no need to edit below this line.
# ============================================================

def default_output_root(source_root: Path) -> Path:
    return source_root.parent / f"{source_root.name}_rtmdet"


def prepare_output_root(target_root: Path, overwrite: bool) -> None:
    if target_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {target_root}\n"
                "Set overwrite_output = True to regenerate it."
            )
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(target_root: Path, class_names: list[str], has_test_split: bool) -> None:
    payload = {
        "path": str(target_root),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(class_names),
        "names": class_names,
    }
    if has_test_split:
        payload["test"] = "test/images"
    with (target_root / "data.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=False)


def copy_split(source_split: Path, target_split: Path, copy_labels: bool) -> None:
    source_images = source_split / "images"
    source_labels = source_split / "labels"
    target_images = target_split / "images"
    target_labels = target_split / "labels"

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    for image_path in collect_image_files(source_images):
        shutil.copy2(image_path, target_images / image_path.name)

    if copy_labels:
        for label_path in sorted(source_labels.glob("*.txt"), key=lambda p: p.name.lower()):
            shutil.copy2(label_path, target_labels / label_path.name)
    else:
        for image_path in collect_image_files(source_images):
            source_label = source_labels / f"{image_path.stem}.txt"
            if source_label.is_file():
                shutil.copy2(source_label, target_labels / source_label.name)


def write_classes_file(target_root: Path, class_names: list[str]) -> None:
    with (target_root / "classes.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(class_names) + "\n")


def write_metadata(
    source_root: Path,
    target_root: Path,
    validation_stats: dict,
    class_names: list[str],
) -> None:
    metadata = {
        "source_dataset": str(source_root),
        "rtmdet_dataset": str(target_root),
        "format": "COCO detection for MMDetection/RTMDet",
        "classes": class_names,
        "stats": validation_stats,
    }
    with (target_root / "rtmdet_dataset_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=True)


def prepare_dataset_for_rtmdet(
    dataset_input: str | Path,
    output_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    source_root, source_yaml = resolve_dataset_root(dataset_input)
    source_splits = detect_split_dirs(source_root)
    class_names = parse_names_from_yaml(source_yaml)
    if not class_names:
        raise ValueError(
            "Cannot read class names from data.yaml. "
            "Add a 'names' field before converting."
        )
    target_root = Path(output_root).resolve() if output_root else default_output_root(source_root)

    prepare_output_root(target_root, overwrite)

    print(f"Source dataset : {source_root}")
    print(f"RTMDet dataset : {target_root}")

    for split_name, source_split in source_splits.items():
        target_split_name = "valid" if split_name == "val" else split_name
        print(f"Copying split {split_name} -> {target_split_name}")
        copy_split(
            source_split=source_split,
            target_split=target_root / target_split_name,
            copy_labels=copy_yolo_labels,
        )

    write_dataset_yaml(
        target_root=target_root,
        class_names=class_names,
        has_test_split="test" in source_splits,
    )

    validation_config = RTMDetPipelineConfig(
        dataset_path=target_root,
        model_name=target_root.name,
        convert_segments_to_boxes=convert_segments_to_boxes,
        allow_missing_labels=allow_missing_labels,
        stop_on_validation_errors=True,
        prepare_only=True,
    )

    validation = validate_yolo_dataset(validation_config)
    if validation.errors:
        for error in validation.errors[:50]:
            print(f"ERROR: {error}")
        raise ValueError("RTMDet dataset is invalid. Fix the errors above.")

    print("Generating COCO annotations...")
    convert_yolo_dataset_to_coco(validation)

    write_classes_file(target_root, validation.class_names)
    write_metadata(
        source_root=source_root,
        target_root=target_root,
        validation_stats={name: asdict(stats) for name, stats in validation.stats.items()},
        class_names=validation.class_names,
    )

    print(f"\nPreparation complete. RTMDet dataset ready at: {target_root}")
    return target_root


def main() -> None:
    prepare_dataset_for_rtmdet(
        dataset_input=dataset_path,
        output_root=output_path,
        overwrite=overwrite_output,
    )


if __name__ == "__main__":
    main()
