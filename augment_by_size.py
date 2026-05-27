"""
Dataset object-size augmentation tool.

Reads dataset_path and augmentation settings from dataset_workflow_config.yaml
and creates a new dataset named
<dataset_name>_dimension_augmented.

The tool preserves the original dataset and adds augmented images that help
move the COCO object-size distribution toward the configured target.

Usage:
    1. Edit dataset_workflow_config.yaml if needed.
    2. python augment_by_size.py
"""

from __future__ import annotations

from pathlib import Path

from train_rtmdet.dataset_workflow_config import (
    find_dataset_workflow_config,
    load_dataset_workflow_dataset_path,
)
from train_rtmdet.dimension_augmenter import (
    generate_augmented_dataset,
    load_dimension_config,
)


def main() -> None:
    dim_cfg_path = find_dataset_workflow_config(Path(__file__).resolve().parent)
    if dim_cfg_path is None:
        raise FileNotFoundError("dataset_workflow_config.yaml not found.")
    dim_cfg = load_dimension_config(dim_cfg_path)

    dataset_path = load_dataset_workflow_dataset_path(dim_cfg_path)
    if dataset_path is None:
        raise ValueError("dataset_path is not set in dataset_workflow_config.yaml.")

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    dst_root = dataset_root.parent / f"{dataset_root.name}{dim_cfg.output_suffix}"

    print(f"Source dataset : {dataset_root}")
    print(f"Output dataset : {dst_root}")
    print(f"Target sizes   : {dim_cfg.target_distribution}")
    print(f"Tolerance      : {dim_cfg.tolerance:.3f}")
    print(f"Preserve orig. : {dim_cfg.preserve_originals}")
    print(f"Apply to       : {dim_cfg.apply_to}")
    print(f"Downscale      : {dim_cfg.allow_downscale}")
    print(f"Upscale        : {dim_cfg.allow_upscale}")
    print(f"Max new images : {dim_cfg.max_new_images}")
    print(f"Max new %      : {dim_cfg.max_new_images_percent}")
    print(f"Balance classes: {dim_cfg.balance_classes}"
          + (f"  (weight={dim_cfg.balance_classes_weight})" if dim_cfg.balance_classes else ""))
    print()

    manifest = generate_augmented_dataset(
        src_root=dataset_root,
        dst_root=dst_root,
        cfg=dim_cfg,
    )

    print("Augmentation completed.\n")
    for split_label, result in manifest.get("applied_splits", {}).items():
        print(f"[{split_label}]")
        print(f"  status          : {result.get('status')}")
        print(f"  effective_max   : {result.get('max_new_images_effective', 0)}")
        print(f"  generated_images: {result.get('generated_images', 0)}")
        print(f"  initial_dist    : {result.get('initial_distribution')}")
        print(f"  final_dist      : {result.get('final_distribution')}")
        if result.get("initial_class_distribution") is not None:
            print(f"  initial_class   : {result.get('initial_class_distribution')}")
            print(f"  final_class     : {result.get('final_class_distribution')}")
        print()

    print(f"Saved to: {dst_root}")
    print(f"Manifest : {dst_root / 'dimension_augmentation_manifest.json'}")


if __name__ == "__main__":
    main()
