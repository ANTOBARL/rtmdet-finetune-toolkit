"""
Dataset class distribution analysis.

Reads dataset_path from hyperparameter_config.yaml, counts annotations per
class in each split (train / val / test), and saves a grouped bar chart as
class_distribution.png inside the dataset folder.

Usage:
    python analyze_dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

from train_rtmdet.config_loader import load_pipeline_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLITS = [
    ("train",  "train",       "#4C72B0"),
    ("valid",  "val", "#DD8452"),
    ("val",    "val", "#DD8452"),
    ("test",   "test",        "#55A868"),
]


@dataclass
class SplitStats:
    annotation_counts: list[int]
    total_images: int
    annotated_images: int

    @property
    def background_images(self) -> int:
        return self.total_images - self.annotated_images


def load_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")
    with data_yaml.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def analyze_split(split_dir: Path, nc: int, split_label: str) -> SplitStats:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Total images
    total_images = 0
    if images_dir.is_dir():
        total_images = sum(
            1 for f in images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

    counts = [0] * nc
    annotated_images = 0

    label_files = list(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    n_files = len(label_files)
    step = max(1, n_files // 100)
    print(f"  [{split_label}] {total_images:,} images, {n_files:,} label files")

    for done, txt in enumerate(label_files, 1):
        has_annotation = False
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls = int(line.split()[0])
                if 0 <= cls < nc:
                    counts[cls] += 1
                    has_annotation = True
            except (ValueError, IndexError):
                pass
        if has_annotation:
            annotated_images += 1
        if done % step == 0 or done == n_files:
            print(f"\r  {done:,}/{n_files:,}", end="", flush=True)
    print()

    # Images with no label file at all also count as background
    # annotated_images already excludes them since we only iterated label files
    # total_images - annotated_images covers both missing labels and empty labels

    return SplitStats(
        annotation_counts=counts,
        total_images=total_images,
        annotated_images=annotated_images,
    )


def collect_split_stats(
    dataset_root: Path, class_names: list[str]
) -> dict[str, SplitStats]:
    nc = len(class_names)
    seen_labels: set[str] = set()
    result: dict[str, SplitStats] = {}

    for folder, label, _ in SPLITS:
        if label in seen_labels:
            continue
        split_dir = dataset_root / folder
        if not split_dir.is_dir():
            continue
        seen_labels.add(label)
        result[label] = analyze_split(split_dir, nc, label)

    return result


def plot_distribution(
    class_names: list[str],
    split_stats: dict[str, SplitStats],
    output_path: Path,
) -> None:
    split_colors = {label: color for _, label, color in SPLITS}
    splits = list(split_stats.keys())
    nc = len(class_names)
    n_splits = len(splits)

    x = np.arange(nc)
    width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

    split_pct: dict[str, list[float]] = {}
    for split_label, stats in split_stats.items():
        total = sum(stats.annotation_counts)
        split_pct[split_label] = [
            c / total * 100 if total else 0.0
            for c in stats.annotation_counts
        ]

    fig, (ax, ax_info) = plt.subplots(
        2, 1,
        figsize=(max(10, nc * 0.9), 8),
        gridspec_kw={"height_ratios": [5, 1]},
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    for offset, split_label in zip(offsets, splits):
        pcts = split_pct[split_label]
        color = split_colors.get(split_label, "#888888")
        bars = ax.bar(x + offset, pcts, width, label=split_label, color=color, alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("% of annotations in split")
    ax.set_title(f"Class distribution — {output_path.parent.name}")
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Background summary panel ───────────────────────────────────────────────
    ax_info.axis("off")
    col_w = 1.0 / len(splits)
    for i, split_label in enumerate(splits):
        stats = split_stats[split_label]
        n_tot = stats.total_images
        n_ann = stats.annotated_images
        n_bg = stats.background_images
        pct_ann = n_ann / n_tot * 100 if n_tot else 0.0
        pct_bg = n_bg / n_tot * 100 if n_tot else 0.0
        color = split_colors.get(split_label, "#888888")

        cx = col_w * i + col_w / 2
        ax_info.text(cx, 0.85, split_label, ha="center", va="top",
                     fontsize=11, fontweight="bold", color=color,
                     transform=ax_info.transAxes)
        ax_info.text(cx, 0.55, f"{n_tot:,} images total", ha="center", va="top",
                     fontsize=10, transform=ax_info.transAxes)
        ax_info.text(cx, 0.28,
                     f"annotated: {pct_ann:.1f}%  ({n_ann:,})",
                     ha="center", va="top", fontsize=10,
                     color="#2ca02c", transform=ax_info.transAxes)
        ax_info.text(cx, 0.02,
                     f"background: {pct_bg:.1f}%  ({n_bg:,})",
                     ha="center", va="top", fontsize=10,
                     color="#d62728", transform=ax_info.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config_path = Path(__file__).resolve().parent / "hyperparameter_config.yaml"
    cfg = load_pipeline_config(config_path)

    dataset_path = cfg.get("dataset_path")
    if dataset_path is None:
        raise ValueError(
            "dataset_path is not set in hyperparameter_config.yaml. "
            "Edit the dataset section before running."
        )

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    class_names = cfg.get("class_names") or load_class_names(dataset_root)
    nc = len(class_names)
    print(f"Dataset : {dataset_root}")
    print(f"Classes : {nc} — {class_names}")

    split_stats = collect_split_stats(dataset_root, class_names)
    if not split_stats:
        raise FileNotFoundError("No split folders found in the dataset.")

    for split_label, stats in split_stats.items():
        total_ann = sum(stats.annotation_counts)
        n_tot = stats.total_images
        pct_ann = stats.annotated_images / n_tot * 100 if n_tot else 0.0
        pct_bg = stats.background_images / n_tot * 100 if n_tot else 0.0
        print(f"\n[{split_label}]  {n_tot:,} images total  |  "
              f"annotated: {pct_ann:.1f}% ({stats.annotated_images:,})  |  "
              f"background: {pct_bg:.1f}% ({stats.background_images:,})")
        print(f"  {total_ann:,} total annotations")
        for name, count in zip(class_names, stats.annotation_counts):
            pct = count / total_ann * 100 if total_ann else 0
            print(f"  {name:<30} {count:>8,}  ({pct:.1f}%)")

    output_path = dataset_root / "class_distribution.png"
    plot_distribution(class_names, split_stats, output_path)
    print(f"\nChart saved → {output_path}")


if __name__ == "__main__":
    main()
