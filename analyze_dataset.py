"""
Dataset class distribution analysis.

Reads dataset_path from hyperparameter_config.yaml, counts annotations per
class in each split (train / val / test), and saves a grouped bar chart as
class_distribution.png inside the dataset folder.

Usage:
    python analyze_dataset.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

from train_rtmdet.config_loader import load_pipeline_config

SPLITS = [
    ("train",  "train",       "#4C72B0"),
    ("valid",  "val / valid", "#DD8452"),
    ("val",    "val / valid", "#DD8452"),
    ("test",   "test",        "#55A868"),
]


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


def count_annotations(labels_dir: Path, nc: int, split_label: str) -> list[int]:
    counts = [0] * nc
    files = list(labels_dir.glob("*.txt"))
    total = len(files)
    step = max(1, total // 100)
    print(f"  [{split_label}] {total:,} label files")
    for done, txt in enumerate(files, 1):
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls = int(line.split()[0])
                if 0 <= cls < nc:
                    counts[cls] += 1
            except (ValueError, IndexError):
                pass
        if done % step == 0 or done == total:
            print(f"\r  {done:,}/{total:,}", end="", flush=True)
    print()
    return counts


def collect_split_counts(
    dataset_root: Path, class_names: list[str]
) -> dict[str, list[int]]:
    nc = len(class_names)
    seen_labels: set[str] = set()
    result: dict[str, list[int]] = {}

    for folder, label, _ in SPLITS:
        if label in seen_labels:
            continue
        split_dir = dataset_root / folder
        labels_dir = split_dir / "labels"
        if not labels_dir.is_dir():
            continue
        seen_labels.add(label)
        result[label] = count_annotations(labels_dir, nc, label)

    return result


def plot_distribution(
    class_names: list[str],
    split_counts: dict[str, list[int]],
    output_path: Path,
) -> None:
    split_colors = {label: color for _, label, color in SPLITS}
    splits = list(split_counts.keys())
    nc = len(class_names)
    n_splits = len(splits)

    x = np.arange(nc)
    width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * width

    # Convert counts to percentages within each split
    split_pct: dict[str, list[float]] = {}
    for split_label, counts in split_counts.items():
        total = sum(counts)
        split_pct[split_label] = [c / total * 100 if total else 0.0 for c in counts]

    fig, ax = plt.subplots(figsize=(max(10, nc * 0.9), 6))

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

    split_counts = collect_split_counts(dataset_root, class_names)
    if not split_counts:
        raise FileNotFoundError("No split folders with labels found in the dataset.")

    for split_label, counts in split_counts.items():
        total = sum(counts)
        print(f"\n[{split_label}]  {total:,} total annotations")
        for name, count in zip(class_names, counts):
            pct = count / total * 100 if total else 0
            print(f"  {name:<30} {count:>8,}  ({pct:.1f}%)")

    output_path = dataset_root / "class_distribution.png"
    plot_distribution(class_names, split_counts, output_path)
    print(f"\nChart saved → {output_path}")


if __name__ == "__main__":
    main()
