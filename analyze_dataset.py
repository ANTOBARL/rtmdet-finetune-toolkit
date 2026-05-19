"""
Dataset class and object-size distribution analysis.

Reads dataset_path from dataset_workflow_config.yaml, counts annotations per
class in each split (train / val / test), analyzes object sizes using the
COCO area bins (small / medium / large), and saves charts inside the dataset
folder.

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
from PIL import Image

from train_rtmdet.dataset_workflow_config import load_dataset_workflow_dataset_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COCO_SIZE_BINS = [
    ("small", 0.0, 32.0 ** 2),
    ("medium", 32.0 ** 2, 96.0 ** 2),
    ("large", 96.0 ** 2, float("inf")),
]

SPLITS = [
    ("train", "train", "#4C72B0"),
    ("valid", "val", "#DD8452"),
    ("val", "val", "#DD8452"),
    ("test", "test", "#55A868"),
]


@dataclass
class SplitStats:
    annotation_counts: list[int]
    size_counts: list[int]
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


def build_image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        return {}
    return {
        image_path.stem: image_path
        for image_path in images_dir.iterdir()
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    }


def parse_object_area(line: str, image_width: int, image_height: int) -> float | None:
    parts = line.split()
    if len(parts) < 5:
        return None

    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(values) == 4:
        _, _, box_w, box_h = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        box_w = max(xs) - min(xs)
        box_h = max(ys) - min(ys)
    else:
        return None

    if box_w <= 0 or box_h <= 0:
        return None

    return box_w * image_width * box_h * image_height


def size_bin_index(area_px2: float) -> int:
    for idx, (_, min_area, max_area) in enumerate(COCO_SIZE_BINS):
        if min_area <= area_px2 < max_area:
            return idx
    return len(COCO_SIZE_BINS) - 1


def analyze_split(split_dir: Path, nc: int, split_label: str) -> SplitStats:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    total_images = 0
    if images_dir.is_dir():
        total_images = sum(
            1 for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )

    counts = [0] * nc
    size_counts = [0] * len(COCO_SIZE_BINS)
    annotated_images = 0
    image_index = build_image_index(images_dir)

    label_files = list(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    n_files = len(label_files)
    step = max(1, n_files // 100)
    print(f"  [{split_label}] {total_images:,} images, {n_files:,} label files")

    for done, txt in enumerate(label_files, 1):
        has_annotation = False
        image_path = image_index.get(txt.stem)
        image_size: tuple[int, int] | None = None
        if image_path is not None:
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except OSError:
                image_size = None

        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                cls = int(line.split()[0])
            except (ValueError, IndexError):
                continue

            if not 0 <= cls < nc:
                continue

            counts[cls] += 1
            has_annotation = True

            if image_size is not None:
                area_px2 = parse_object_area(line, image_size[0], image_size[1])
                if area_px2 is not None:
                    size_counts[size_bin_index(area_px2)] += 1

        if has_annotation:
            annotated_images += 1
        if done % step == 0 or done == n_files:
            print(f"\r  {done:,}/{n_files:,}", end="", flush=True)
    print()

    return SplitStats(
        annotation_counts=counts,
        size_counts=size_counts,
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
            count / total * 100 if total else 0.0 for count in stats.annotation_counts
        ]

    fig, (ax, ax_info) = plt.subplots(
        2,
        1,
        figsize=(max(10, nc * 0.9), 8),
        gridspec_kw={"height_ratios": [5, 1]},
    )

    for offset, split_label in zip(offsets, splits):
        pcts = split_pct[split_label]
        color = split_colors.get(split_label, "#888888")
        bars = ax.bar(
            x + offset, pcts, width, label=split_label, color=color, alpha=0.85
        )
        for bar, pct in zip(bars, pcts):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=45,
                )

    max_pct = max((p for pcts in split_pct.values() for p in pcts), default=0.0)
    ax.set_ylim(0, max(8.0, max_pct + 12))
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("% of annotations in split")
    ax.set_title(f"Class distribution - {output_path.parent.name}")
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

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
        ax_info.text(
            cx,
            0.85,
            split_label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color,
            transform=ax_info.transAxes,
        )
        ax_info.text(
            cx,
            0.55,
            f"{n_tot:,} images total",
            ha="center",
            va="top",
            fontsize=10,
            transform=ax_info.transAxes,
        )
        ax_info.text(
            cx,
            0.28,
            f"annotated: {pct_ann:.1f}%  ({n_ann:,})",
            ha="center",
            va="top",
            fontsize=10,
            color="#2ca02c",
            transform=ax_info.transAxes,
        )
        ax_info.text(
            cx,
            0.02,
            f"background: {pct_bg:.1f}%  ({n_bg:,})",
            ha="center",
            va="top",
            fontsize=10,
            color="#d62728",
            transform=ax_info.transAxes,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_size_distribution(
    split_stats: dict[str, SplitStats],
    output_path: Path,
) -> None:
    split_colors = {label: color for _, label, color in SPLITS}
    splits = list(split_stats.keys())
    size_names = [name for name, _, _ in COCO_SIZE_BINS]
    n_sizes = len(size_names)
    n_splits = len(splits)

    x = np.arange(n_sizes)
    width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

    split_pct: dict[str, list[float]] = {}
    for split_label, stats in split_stats.items():
        total = sum(stats.size_counts)
        split_pct[split_label] = [
            count / total * 100 if total else 0.0 for count in stats.size_counts
        ]

    fig, (ax, ax_info) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [5, 1]},
    )

    for offset, split_label in zip(offsets, splits):
        pcts = split_pct[split_label]
        color = split_colors.get(split_label, "#888888")
        bars = ax.bar(
            x + offset, pcts, width, label=split_label, color=color, alpha=0.85
        )
        for bar, pct in zip(bars, pcts):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    max_pct = max((p for pcts in split_pct.values() for p in pcts), default=0.0)
    ax.set_ylim(0, max(8.0, max_pct + 10))
    ax.set_xticks(x)
    ax.set_xticklabels(size_names, fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("% of objects in split")
    ax.set_title(f"Object size distribution (COCO area bins) - {output_path.parent.name}")
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax_info.axis("off")
    ax_info.text(
        0.5,
        0.65,
        r"small: area < $32^2$ px | medium: $32^2 \leq$ area < $96^2$ px | large: area $\geq 96^2$ px",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_info.transAxes,
    )

    summary_parts = []
    for split_label in splits:
        total = sum(split_stats[split_label].size_counts)
        summary_parts.append(f"{split_label}: {total:,} objects")
    ax_info.text(
        0.5,
        0.15,
        " | ".join(summary_parts),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_info.transAxes,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    dataset_path = load_dataset_workflow_dataset_path()
    if dataset_path is None:
        raise ValueError("dataset_path is not set in dataset_workflow_config.yaml.")

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    class_names = load_class_names(dataset_root)
    nc = len(class_names)
    print(f"Dataset : {dataset_root}")
    print(f"Classes : {nc} - {class_names}")

    split_stats = collect_split_stats(dataset_root, class_names)
    if not split_stats:
        raise FileNotFoundError("No split folders found in the dataset.")

    for split_label, stats in split_stats.items():
        total_ann = sum(stats.annotation_counts)
        total_sizes = sum(stats.size_counts)
        n_tot = stats.total_images
        pct_ann = stats.annotated_images / n_tot * 100 if n_tot else 0.0
        pct_bg = stats.background_images / n_tot * 100 if n_tot else 0.0
        print(
            f"\n[{split_label}]  {n_tot:,} images total  |  "
            f"annotated: {pct_ann:.1f}% ({stats.annotated_images:,})  |  "
            f"background: {pct_bg:.1f}% ({stats.background_images:,})"
        )
        print(f"  {total_ann:,} total annotations")
        for name, count in zip(class_names, stats.annotation_counts):
            pct = count / total_ann * 100 if total_ann else 0.0
            print(f"  {name:<30} {count:>8,}  ({pct:.1f}%)")

        print("  Object sizes (COCO area bins):")
        for (size_name, _, _), count in zip(COCO_SIZE_BINS, stats.size_counts):
            pct = count / total_sizes * 100 if total_sizes else 0.0
            print(f"  {size_name:<30} {count:>8,}  ({pct:.1f}%)")

    class_output_path = dataset_root / "class_distribution.png"
    size_output_path = dataset_root / "object_size_distribution.png"
    plot_distribution(class_names, split_stats, class_output_path)
    plot_size_distribution(split_stats, size_output_path)
    print(f"\nChart saved -> {class_output_path}")
    print(f"Chart saved -> {size_output_path}")


if __name__ == "__main__":
    main()
