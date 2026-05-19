"""
Step 1 of 2 — Normalize dataset filenames.

Renames images and labels in all splits (train/valid/test) to a canonical
format:  train_1.jpg / train_1.txt,  val_1.jpg / val_1.txt, etc.

Usage:
    python normalize_dataset.py

dataset_path is read from dataset_workflow_config.yaml.
"""

from __future__ import annotations

from pathlib import Path

from train_rtmdet.dataset_workflow_config import load_dataset_workflow_dataset_path
from train_rtmdet.normalize import build_rename_plan, execute_rename_plan


def main() -> None:
    dataset_path = load_dataset_workflow_dataset_path()
    if dataset_path is None:
        raise ValueError("dataset_path is not set in dataset_workflow_config.yaml.")

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    print(f"Dataset: {dataset_root}")

    rename_plan, warnings = build_rename_plan(dataset_root, label_prefix="")

    for w in warnings:
        print(f"WARNING: {w}")

    if not rename_plan:
        print("All filenames are already normalized — nothing to do.")
        return

    print(f"{len(rename_plan)} files to rename.")
    execute_rename_plan(rename_plan)
    print(f"Done — {len(rename_plan)} files renamed.")


if __name__ == "__main__":
    main()
