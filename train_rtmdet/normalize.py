from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_PREFIXES = {
    "train": "train",
    "test": "test",
    "valid": "val",
    "val": "val",
}


@dataclass
class RenameItem:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename a YOLOv8 dataset so that images and labels have consistent "
            "names across train, test, and valid/val splits."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Root path of the dataset containing train, test, and valid/val folders.",
    )
    parser.add_argument(
        "--label-prefix",
        default="",
        help="Optional prefix added to .txt label files (e.g. 'label_' → label_test_1.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without modifying any files.",
    )
    return parser.parse_args()


def find_split_dirs(dataset_path: Path) -> list[Path]:
    split_dirs: list[Path] = []
    for split_name in ("train", "test", "valid", "val"):
        split_dir = dataset_path / split_name
        if split_dir.is_dir():
            split_dirs.append(split_dir)
    return split_dirs


def validate_split_dir(split_dir: Path) -> tuple[Path, Path]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder missing: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder missing: {labels_dir}")

    return images_dir, labels_dir


def collect_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def build_rename_plan(
    dataset_path: Path,
    label_prefix: str,
) -> tuple[list[RenameItem], list[str]]:
    split_dirs = find_split_dirs(dataset_path)
    if not split_dirs:
        raise FileNotFoundError(
            "No train/test/valid/val folder found in the dataset root."
        )

    rename_plan: list[RenameItem] = []
    warnings: list[str] = []

    for split_dir in split_dirs:
        images_dir, labels_dir = validate_split_dir(split_dir)
        split_key = split_dir.name.lower()
        split_prefix = SPLIT_PREFIXES[split_key]
        image_files = collect_image_files(images_dir)

        if not image_files:
            warnings.append(f"[{split_dir.name}] No images found in {images_dir}")
            continue

        image_stems = {image_path.stem for image_path in image_files}
        label_files = list(labels_dir.glob("*.txt"))
        label_stems = {label_path.stem for label_path in label_files}

        orphan_labels = sorted(label_stems - image_stems)
        unlabeled_images = sorted(image_stems - label_stems)

        for stem in orphan_labels:
            warnings.append(
                f"[{split_dir.name}] Label without matching image: {labels_dir / f'{stem}.txt'}"
            )

        for stem in unlabeled_images:
            warnings.append(
                f"[{split_dir.name}] Image without matching label: {stem}"
            )

        for index, image_path in enumerate(image_files, start=1):
            image_stem = image_path.stem
            label_path = labels_dir / f"{image_stem}.txt"

            new_image_stem = f"{split_prefix}_{index}"
            new_label_stem = f"{label_prefix}{new_image_stem}"

            new_image_path = images_dir / f"{new_image_stem}{image_path.suffix.lower()}"
            rename_plan.append(RenameItem(source=image_path, target=new_image_path))

            if label_path.exists():
                new_label_path = labels_dir / f"{new_label_stem}.txt"
                rename_plan.append(RenameItem(source=label_path, target=new_label_path))

    ensure_unique_targets(rename_plan)
    ensure_no_external_conflicts(rename_plan)

    return rename_plan, warnings


def ensure_unique_targets(rename_plan: list[RenameItem]) -> None:
    seen_targets: dict[Path, Path] = {}
    for item in rename_plan:
        existing_source = seen_targets.get(item.target)
        if existing_source is not None and existing_source != item.source:
            raise RuntimeError(
                f"Target collision: {item.target} would be generated from multiple files."
            )
        seen_targets[item.target] = item.source


def ensure_no_external_conflicts(rename_plan: list[RenameItem]) -> None:
    source_paths = {item.source.resolve() for item in rename_plan}

    for item in rename_plan:
        target_resolved = item.target.resolve()
        if target_resolved == item.source.resolve():
            continue
        if item.target.exists() and target_resolved not in source_paths:
            raise RuntimeError(
                f"Target file already exists and is not part of the rename plan: {item.target}"
            )


def print_plan(rename_plan: list[RenameItem], warnings: list[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")

    for item in rename_plan:
        if item.source.resolve() == item.target.resolve():
            continue
        print(f"{item.source} -> {item.target}")


def _print_progress(done: int, total: int) -> None:
    pct = done / total
    filled = int(30 * pct)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\r  [{bar}] {pct:5.1%}  ({done}/{total})", end="", flush=True)


def execute_rename_plan(rename_plan: list[RenameItem]) -> None:
    temp_map: list[tuple[Path, Path, Path]] = []
    token = uuid.uuid4().hex[:8]
    total = len(rename_plan)

    for i, item in enumerate(rename_plan):
        if item.source.resolve() == item.target.resolve():
            continue
        # Short temp name avoids Windows MAX_PATH (260 chars) on long filenames.
        temp_path = item.source.with_name(f"_t{i}_{token}")
        try:
            item.source.rename(temp_path)
        except Exception:
            for done_temp, _, original_source in reversed(temp_map):
                if done_temp.exists():
                    done_temp.rename(original_source)
            raise
        temp_map.append((temp_path, item.target, item.source))

    n = len(temp_map)
    step = max(1, n // 100)
    try:
        for done, (temp_path, target_path, _original_source) in enumerate(temp_map, 1):
            temp_path.rename(target_path)
            if done % step == 0 or done == n:
                _print_progress(done, n)
    except Exception:
        print()
        for temp_path, target_path, original_source in reversed(temp_map):
            if temp_path.exists():
                temp_path.rename(original_source)
            elif target_path.exists():
                target_path.rename(original_source)
        raise
    print()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()

    if not dataset_path.is_dir():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    try:
        rename_plan, warnings = build_rename_plan(
            dataset_path=dataset_path,
            label_prefix=args.label_prefix,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rename_plan:
        print("No files to rename.")
        return 0

    print_plan(rename_plan, warnings)

    if args.dry_run:
        print("\nDry run complete: no files modified.")
        return 0

    try:
        execute_rename_plan(rename_plan)
    except Exception as exc:
        print(f"Error during rename: {exc}", file=sys.stderr)
        return 1

    print("\nRename completed successfully.")
    if args.label_prefix:
        print(
            "Note: with a label prefix, basenames no longer match image names. "
            "Standard YOLO format requires matching basenames."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
