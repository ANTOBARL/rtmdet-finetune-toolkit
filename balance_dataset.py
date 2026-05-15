"""
Dataset balancing tool — Step 0.5 (optional, run after analyze_dataset.py).

Pools images from ALL splits, redistributes them proportionally by dominant
class, and saves the result as <dataset_name>_balanced/ next to the original.

  - The original dataset is NEVER modified.
  - No images are deleted — total image count stays the same.
  - Suggested class weights are printed for manual entry in
    hyperparameter_config.yaml under training.class_weights.

Usage:
    1. Edit dataset_balancer.yaml if needed.
    2. python balance_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from train_rtmdet.balancer import (
    BalancerConfig,
    collect_all_images,
    compute_class_weights,
    copy_balanced_dataset,
    global_class_counts,
    imbalance_ratio,
    redistribute,
    reduction_alerts,
)
from train_rtmdet.config_loader import load_pipeline_config

_BALANCER_CFG = Path(__file__).resolve().parent / "dataset_balancer.yaml"
_PIPELINE_CFG = Path(__file__).resolve().parent / "hyperparameter_config.yaml"


def _load_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")
    with data_yaml.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def _load_balancer_config() -> BalancerConfig:
    if not _BALANCER_CFG.is_file():
        raise FileNotFoundError(f"Config not found: {_BALANCER_CFG}")
    with _BALANCER_CFG.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    b = raw.get("balancer", {})
    cfg = BalancerConfig(
        train_ratio=float(b.get("train_ratio", 0.80)),
        val_ratio=float(b.get("val_ratio", 0.15)),
        test_ratio=float(b.get("test_ratio", 0.05)),
        imbalance_tolerance=float(b.get("imbalance_tolerance", 1.5)),
        min_images_per_class=int(b.get("min_images_per_class", 50)),
        reduction_alert_threshold=float(b.get("reduction_alert_threshold", 0.30)),
        seed=int(b.get("seed", 42)),
    )
    if abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    return cfg


def main() -> None:
    pipeline_cfg = load_pipeline_config(_PIPELINE_CFG)
    bal = _load_balancer_config()

    dataset_path = pipeline_cfg.get("dataset_path")
    if dataset_path is None:
        raise ValueError("dataset_path not set in hyperparameter_config.yaml.")

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    class_names = pipeline_cfg.get("class_names") or _load_class_names(dataset_root)
    nc = len(class_names)
    dst_root = dataset_root.parent / (dataset_root.name + "_balanced")

    print(f"Source  : {dataset_root}")
    print(f"Output  : {dst_root}")
    print(f"Classes : {class_names}")
    print()

    # ── Collect ───────────────────────────────────────────────────────────────
    print("Collecting images from all splits...")
    records = collect_all_images(dataset_root, nc)
    print(f"  Total: {len(records):,} images\n")

    # ── Current distribution ──────────────────────────────────────────────────
    counts = global_class_counts(records, nc)
    ratio = imbalance_ratio(counts)

    print("Current annotation counts:")
    for name, count in zip(class_names, counts):
        print(f"  {name:<30} {count:>8,}")
    print(f"\n  Imbalance ratio (max/min): {ratio:.2f}x  "
          f"(tolerance: {bal.imbalance_tolerance:.2f}x)\n")

    # ── Check if redistribution is needed ────────────────────────────────────
    if ratio <= bal.imbalance_tolerance:
        print("Dataset is already within tolerance — no redistribution needed.")
        print("Class weight suggestions are still printed below.\n")
        _print_class_weights(class_names, counts)
        return

    # ── Redistribute ──────────────────────────────────────────────────────────
    train, val, test = redistribute(records, bal)

    # Min images per class warning
    train_per_class = [0] * nc
    for rec in train:
        if rec.primary_class is not None:
            train_per_class[rec.primary_class] += 1

    low_warnings = [
        f"  '{class_names[i]}': {train_per_class[i]} train images "
        f"(below min_images_per_class={bal.min_images_per_class})"
        for i in range(nc)
        if 0 < train_per_class[i] < bal.min_images_per_class
    ]

    # Reduction alerts
    alerts = reduction_alerts(records, train, nc, bal.reduction_alert_threshold)

    if low_warnings:
        print("WARNING — Low image count after redistribution:")
        for w in low_warnings:
            print(w)
        print()

    if alerts:
        orig_train_n = sum(1 for r in records if r.original_split == "train")
        new_train_n = len(train)
        delta_pct = (orig_train_n - new_train_n) / orig_train_n * 100 if orig_train_n else 0

        print("WARNING — Redistribution significantly reduces some train classes:")
        for a in alerts:
            print(f"  {a}")
        print(f"\n  Overall train: {orig_train_n:,} → {new_train_n:,} "
              f"({delta_pct:+.1f}%)")
        print("\n  Proceeding will copy the redistributed dataset to the output folder.")
        answer = input("  Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return
        print()

    print(f"Split result:  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # ── Copy ──────────────────────────────────────────────────────────────────
    if dst_root.exists():
        print(f"\nRemoving existing {dst_root.name}/ ...")
        shutil.rmtree(dst_root)

    print(f"\nCopying to {dst_root} ...")
    copy_balanced_dataset(train, val, test, dataset_root, dst_root)
    print(f"Done — balanced dataset saved to:\n  {dst_root}\n")

    # ── Class weights ─────────────────────────────────────────────────────────
    _print_class_weights(class_names, counts)


def _print_class_weights(class_names: list[str], counts: list[int]) -> None:
    weights = compute_class_weights(counts)
    sep = "=" * 60
    print(sep)
    print("SUGGESTED CLASS WEIGHTS")
    print("Add these to hyperparameter_config.yaml under training:")
    print()
    print("  class_weights:")
    for name, w in zip(class_names, weights):
        print(f"    - {w:<8}  # {name}")
    print()
    print("Formula: w_i = total / (n_classes * count_i), normalized to min=1.0")
    print("Higher weight = model penalized more for missing that class.")
    print(sep)


if __name__ == "__main__":
    main()
