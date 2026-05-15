from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SPLIT_SEARCH_ORDER = ("train", "valid", "val", "test")


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path | None
    original_split: str
    class_counts: list[int] = field(default_factory=list)
    primary_class: int | None = None


@dataclass
class BalancerConfig:
    train_ratio: float = 0.80               # original: 0.80
    val_ratio: float = 0.15                 # original: 0.15
    test_ratio: float = 0.05               # original: 0.05
    imbalance_tolerance: float = 1.5        # original: 1.5
    min_images_per_class: int = 50          # original: 50
    reduction_alert_threshold: float = 0.30 # original: 0.30
    seed: int = 42                          # original: 42


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_all_images(dataset_root: Path, nc: int) -> list[ImageRecord]:
    """Pool images from all splits, deduplicating by stem."""
    records: list[ImageRecord] = []
    seen_stems: set[str] = set()

    for split_name in _SPLIT_SEARCH_ORDER:
        images_dir = dataset_root / split_name / "images"
        labels_dir = dataset_root / split_name / "labels"
        if not images_dir.is_dir():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img_path.stem in seen_stems:
                continue
            seen_stems.add(img_path.stem)
            label_path = labels_dir / (img_path.stem + ".txt")
            lp = label_path if label_path.exists() else None
            counts = _count_classes(lp, nc)
            records.append(ImageRecord(
                image_path=img_path,
                label_path=lp,
                original_split=split_name,
                class_counts=counts,
                primary_class=_primary_class(counts),
            ))

    return records


def _count_classes(label_path: Path | None, nc: int) -> list[int]:
    counts = [0] * nc
    if not label_path:
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cls = int(line.split()[0])
            if 0 <= cls < nc:
                counts[cls] += 1
        except (ValueError, IndexError):
            pass
    return counts


def _primary_class(counts: list[int]) -> int | None:
    """Class with the most instances; None if image has no annotations."""
    if not any(counts):
        return None
    return max(range(len(counts)), key=lambda i: counts[i])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def global_class_counts(records: list[ImageRecord], nc: int) -> list[int]:
    totals = [0] * nc
    for rec in records:
        for i, c in enumerate(rec.class_counts):
            totals[i] += c
    return totals


def imbalance_ratio(counts: list[int]) -> float:
    """max / min annotation count across classes that have at least one sample."""
    nonzero = [c for c in counts if c > 0]
    if len(nonzero) < 2:
        return 1.0
    return max(nonzero) / min(nonzero)


# ---------------------------------------------------------------------------
# Redistribution
# ---------------------------------------------------------------------------

def redistribute(
    records: list[ImageRecord],
    cfg: BalancerConfig,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    """
    Group images by primary class, then split each group proportionally
    into train / val / test.  Images with no annotations go entirely to train.
    """
    rng = random.Random(cfg.seed)

    groups: dict[int | None, list[ImageRecord]] = {}
    for rec in records:
        groups.setdefault(rec.primary_class, []).append(rec)

    train_out: list[ImageRecord] = []
    val_out: list[ImageRecord] = []
    test_out: list[ImageRecord] = []

    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)

        if key is None:
            train_out.extend(group)
            continue

        n_test = max(0, round(n * cfg.test_ratio))
        n_val = max(0, round(n * cfg.val_ratio))
        n_train = n - n_val - n_test

        train_out.extend(group[:n_train])
        val_out.extend(group[n_train:n_train + n_val])
        test_out.extend(group[n_train + n_val:])

    return train_out, val_out, test_out


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def reduction_alerts(
    all_records: list[ImageRecord],
    new_train: list[ImageRecord],
    nc: int,
    threshold: float,
) -> list[str]:
    """
    Warn when a class loses more than `threshold` fraction of its original
    train images after redistribution.
    """
    orig_counts = [0] * nc
    for rec in all_records:
        if rec.original_split == "train" and rec.primary_class is not None:
            orig_counts[rec.primary_class] += 1

    new_counts = [0] * nc
    for rec in new_train:
        if rec.primary_class is not None:
            new_counts[rec.primary_class] += 1

    alerts: list[str] = []
    for i in range(nc):
        if orig_counts[i] == 0:
            continue
        loss = (orig_counts[i] - new_counts[i]) / orig_counts[i]
        if loss > threshold:
            alerts.append(
                f"class {i}: {orig_counts[i]} → {new_counts[i]} train images "
                f"(-{loss:.0%}), exceeds alert threshold of {threshold:.0%}"
            )
    return alerts


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(counts: list[int]) -> list[float]:
    """
    Inverse-frequency weights normalized so the most common class = 1.0.
    Formula: w_i = total / (n_classes * count_i), then divide by min(w).
    Classes with zero annotations get weight 0.0.
    """
    nc = len(counts)
    total = sum(counts)
    raw = [total / (nc * c) if c > 0 else 0.0 for c in counts]
    min_w = min((w for w in raw if w > 0), default=1.0)
    return [round(w / min_w, 4) if w > 0 else 0.0 for w in raw]


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def copy_balanced_dataset(
    train: list[ImageRecord],
    val: list[ImageRecord],
    test: list[ImageRecord],
    src_root: Path,
    dst_root: Path,
) -> None:
    """Copy images and labels to the new balanced folder, with progress."""
    split_map = [("train", train), ("valid", val), ("test", test)]

    for split_name, records in split_map:
        if not records:
            continue
        (dst_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    total = sum(len(r) for _, r in split_map)
    step = max(1, total // 100)
    done = 0

    for split_name, records in split_map:
        for rec in records:
            shutil.copy2(rec.image_path, dst_root / split_name / "images" / rec.image_path.name)
            if rec.label_path:
                shutil.copy2(rec.label_path, dst_root / split_name / "labels" / rec.label_path.name)
            done += 1
            if done % step == 0 or done == total:
                print(f"\r  {done:,}/{total:,}", end="", flush=True)
    print()

    src_yaml = src_root / "data.yaml"
    if src_yaml.is_file():
        shutil.copy2(src_yaml, dst_root / "data.yaml")
