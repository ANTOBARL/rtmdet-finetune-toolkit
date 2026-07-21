from __future__ import annotations

import heapq
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COCO_SIZE_BINS = (
    ("small", 0.0, 32.0 ** 2),
    ("medium", 32.0 ** 2, 96.0 ** 2),
    ("large", 96.0 ** 2, float("inf")),
)
SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


@dataclass
class SizeAugmenterConfig:
    dataset_path: Path | None
    target_distribution: dict[str, float]
    tolerance: float = 0.02
    seed: int = 1
    output_suffix: str = "_dimension_augmented"
    preserve_originals: bool = True
    apply_to: dict[str, bool] = field(
        default_factory=lambda: {"train": True, "val": False, "test": False}
    )
    allow_downscale: bool = True
    allow_upscale: bool = False
    max_new_images: int = 3000
    max_new_images_percent: float | None = None
    max_copies_per_source_image: int = 3
    scale_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "to_small": (0.35, 0.65),
            "to_medium": (0.65, 0.90),
            "to_large": (1.10, 1.35),
        }
    )
    padding_mode: str = "constant"
    padding_color: tuple[int, int, int] = (114, 114, 114)
    min_bbox_size_px: float = 4.0
    min_bbox_area_px2: float = 16.0
    skip_images_without_labels: bool = True
    stop_when_within_tolerance: bool = True
    # class balancing
    balance_classes: bool = True
    balance_classes_weight: float = 2.0


@dataclass
class BBoxAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class SourceImageRecord:
    image_path: Path
    label_path: Path | None
    split_label: str
    width: int
    height: int
    annotations: list[BBoxAnnotation]
    size_counts: list[int]


@dataclass
class GeneratedSample:
    source: SourceImageRecord
    scale: float
    output_image_name: str
    output_label_name: str
    annotations: list[BBoxAnnotation]
    size_counts: list[int]


def load_dimension_config(config_path: Path) -> SizeAugmenterConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    section = raw.get("dimension_augmenter", {})
    dataset_path_raw = section.get("dataset_path")
    target = section.get("target_distribution", {})
    target_distribution = {
        "small": float(target.get("small", 0.35)),
        "medium": float(target.get("medium", 0.40)),
        "large": float(target.get("large", 0.25)),
    }

    target_sum = sum(target_distribution.values())
    if not math.isclose(target_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("target_distribution must sum to 1.0.")

    apply_to_raw = section.get("apply_to", {})
    apply_to = {
        "train": bool(apply_to_raw.get("train", True)),
        "val": bool(apply_to_raw.get("val", False)),
        "test": bool(apply_to_raw.get("test", False)),
    }

    scale_ranges_raw = section.get("scale_ranges", {})
    scale_ranges = {
        "to_small": _read_scale_range(scale_ranges_raw.get("to_small"), (0.35, 0.65)),
        "to_medium": _read_scale_range(scale_ranges_raw.get("to_medium"), (0.65, 0.90)),
        "to_large": _read_scale_range(scale_ranges_raw.get("to_large"), (1.10, 1.35)),
    }

    padding = section.get("padding", {})
    color_raw = padding.get("color", [114, 114, 114])
    if not isinstance(color_raw, list) or len(color_raw) != 3:
        raise ValueError("padding.color must be a list of 3 integers.")
    padding_color = tuple(int(max(0, min(255, v))) for v in color_raw)

    filters = section.get("filters", {})

    cfg = SizeAugmenterConfig(
        dataset_path=Path(str(dataset_path_raw)).expanduser() if dataset_path_raw else None,
        target_distribution=target_distribution,
        tolerance=float(section.get("tolerance", 0.02)),
        seed=int(section.get("seed", 1)),
        output_suffix=str(section.get("output_suffix", "_dimension_augmented")).strip(),
        preserve_originals=bool(section.get("preserve_originals", True)),
        apply_to=apply_to,
        allow_downscale=bool(section.get("allow_downscale", True)),
        allow_upscale=bool(section.get("allow_upscale", False)),
        max_new_images=int(section.get("max_new_images", 3000)),
        max_new_images_percent=_read_optional_percent(
            section.get("max_new_images_percent")
        ),
        max_copies_per_source_image=int(section.get("max_copies_per_source_image", 3)),
        scale_ranges=scale_ranges,
        padding_mode=str(padding.get("mode", "constant")).strip().lower(),
        padding_color=padding_color,
        min_bbox_size_px=float(filters.get("min_bbox_size_px", 4.0)),
        min_bbox_area_px2=float(filters.get("min_bbox_area_px2", 16.0)),
        skip_images_without_labels=bool(filters.get("skip_images_without_labels", True)),
        stop_when_within_tolerance=bool(section.get("stop_when_within_tolerance", True)),
        balance_classes=bool(section.get("balance_classes", True)),
        balance_classes_weight=float(section.get("balance_classes_weight", 2.0)),
    )

    if cfg.padding_mode != "constant":
        raise ValueError("Only padding.mode='constant' is currently supported.")
    if cfg.max_new_images < 0:
        raise ValueError("max_new_images must be >= 0.")
    if cfg.max_new_images_percent is not None and cfg.max_new_images_percent < 0:
        raise ValueError("max_new_images_percent must be >= 0.")
    if cfg.max_copies_per_source_image < 1:
        raise ValueError("max_copies_per_source_image must be >= 1.")
    if cfg.tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    if not cfg.allow_downscale and not cfg.allow_upscale:
        raise ValueError("At least one of allow_downscale / allow_upscale must be true.")

    return cfg


def _read_scale_range(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        return default
    lo = float(value[0])
    hi = float(value[1])
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"Invalid scale range: {value}")
    return lo, hi


def _read_optional_percent(value: object) -> float | None:
    if value is None:
        return None
    percent = float(value)
    return percent


def collect_split_records(dataset_root: Path, split_label: str) -> list[SourceImageRecord]:
    folder_name = _split_folder_name(dataset_root, split_label)
    if folder_name is None:
        return []

    split_dir = dataset_root / folder_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        return []

    image_paths = [
        p
        for p in sorted(images_dir.iterdir(), key=lambda p: p.name.lower())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    n_total = len(image_paths)
    step = max(1, n_total // 100)
    print(f"  [{split_label}] scanning {n_total:,} images")

    records: list[SourceImageRecord] = []
    for done, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        lp = label_path if label_path.is_file() else None
        width, height = _load_image_size(image_path)
        annotations = _load_annotations(lp, width, height)
        size_counts = count_size_bins(annotations, width, height)
        records.append(
            SourceImageRecord(
                image_path=image_path,
                label_path=lp,
                split_label=split_label,
                width=width,
                height=height,
                annotations=annotations,
                size_counts=size_counts,
            )
        )
        if done % step == 0 or done == n_total:
            print(f"\r  {done:,}/{n_total:,}", end="", flush=True)
    if n_total:
        print()
    return records


def _split_folder_name(dataset_root: Path, split_label: str) -> str | None:
    for folder_name, label in SPLIT_ALIASES.items():
        if label == split_label and (dataset_root / folder_name).is_dir():
            return folder_name
    return None


def _load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def _load_annotations(
    label_path: Path | None,
    image_width: int,
    image_height: int,
) -> list[BBoxAnnotation]:
    if label_path is None or not label_path.is_file():
        return []

    annotations: list[BBoxAnnotation] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ann = _parse_yolo_line(line, image_width, image_height)
        if ann is not None:
            annotations.append(ann)
    return annotations


def _parse_yolo_line(
    line: str,
    image_width: int,
    image_height: int,
) -> BBoxAnnotation | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        x_center, y_center, width, height = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
    else:
        return None

    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))

    if width <= 0 or height <= 0:
        return None
    if width * image_width <= 0 or height * image_height <= 0:
        return None

    return BBoxAnnotation(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def count_size_bins(
    annotations: list[BBoxAnnotation],
    image_width: int,
    image_height: int,
) -> list[int]:
    counts = [0] * len(COCO_SIZE_BINS)
    for ann in annotations:
        area = ann.width * image_width * ann.height * image_height
        counts[size_bin_index(area)] += 1
    return counts


def size_bin_index(area_px2: float) -> int:
    for idx, (_name, min_area, max_area) in enumerate(COCO_SIZE_BINS):
        if min_area <= area_px2 < max_area:
            return idx
    return len(COCO_SIZE_BINS) - 1


def distribution_from_counts(counts: list[int]) -> dict[str, float]:
    total = sum(counts)
    names = [name for name, _, _ in COCO_SIZE_BINS]
    if total == 0:
        return {name: 0.0 for name in names}
    return {name: counts[i] / total for i, name in enumerate(names)}


def within_tolerance(
    counts: list[int],
    target_distribution: dict[str, float],
    tolerance: float,
) -> bool:
    current = distribution_from_counts(counts)
    for name, target in target_distribution.items():
        if abs(current[name] - target) > tolerance:
            return False
    return True


def select_target_bin(
    counts: list[int],
    target_distribution: dict[str, float],
) -> str | None:
    current = distribution_from_counts(counts)
    deficits = {
        name: target_distribution[name] - current[name]
        for name, _, _ in COCO_SIZE_BINS
    }
    positive = {name: delta for name, delta in deficits.items() if delta > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def build_bin_candidates(
    records: list[SourceImageRecord],
    cfg: SizeAugmenterConfig,
) -> dict[str, list[tuple[SourceImageRecord, tuple[float, float], int]]]:
    """Precompute, once per split, the static (non-usage-dependent) part of
    candidate eligibility for each target size bin: whether a record can be
    scaled toward that bin at all, its allowed scale range, and how many
    "useful" source objects it contributes. This avoids recomputing these
    values for every one of the up to `max_new_images` selection rounds.
    """
    bin_names = [name for name, _, _ in COCO_SIZE_BINS]
    result: dict[str, list[tuple[SourceImageRecord, tuple[float, float], int]]] = {
        name: [] for name in bin_names
    }
    for rec in records:
        if cfg.skip_images_without_labels and not rec.annotations:
            continue
        for target_bin in bin_names:
            scale_range = _candidate_scale_range(rec, target_bin, cfg)
            if scale_range is None:
                continue
            useful_objects = _useful_source_objects(rec, target_bin)
            if useful_objects <= 0:
                continue
            result[target_bin].append((rec, scale_range, useful_objects))
    return result


def compact_bin_candidates(
    bin_candidates: dict[str, list[tuple[SourceImageRecord, tuple[float, float], int]]],
    usage_counts: dict[Path, int],
    max_copies_per_source_image: int,
) -> None:
    """Drop candidates that have already reached their copy limit, in place.

    Called periodically (not every round) so the per-round scan shrinks as
    source images get exhausted, without paying rebuild cost every time.
    """
    for target_bin, items in bin_candidates.items():
        bin_candidates[target_bin] = [
            item
            for item in items
            if usage_counts.get(item[0].image_path, 0) < max_copies_per_source_image
        ]


def choose_candidate(
    candidates: list[tuple[SourceImageRecord, tuple[float, float], int]],
    cfg: SizeAugmenterConfig,
    usage_counts: dict[Path, int],
    rng: random.Random,
    class_counts: dict[int, int] | None = None,
) -> tuple[SourceImageRecord, float] | None:
    scored: list[tuple[float, SourceImageRecord, tuple[float, float]]] = []

    for rec, scale_range, useful_objects in candidates:
        if usage_counts.get(rec.image_path, 0) >= cfg.max_copies_per_source_image:
            continue

        score = float(useful_objects)
        if cfg.balance_classes and class_counts:
            score += _class_balance_score(rec, class_counts, cfg.balance_classes_weight)
        score += rng.random() * 0.05
        scored.append((score, rec, scale_range))

    if not scored:
        return None

    top_k = min(25, len(scored))
    top = heapq.nlargest(top_k, scored, key=lambda item: item[0])
    _score, rec, scale_range = rng.choice(top)
    scale = rng.uniform(scale_range[0], scale_range[1])
    return rec, scale


def _useful_source_objects(rec: SourceImageRecord, target_bin: str) -> int:
    small_idx = 0
    medium_idx = 1
    large_idx = 2
    if target_bin == "small":
        return rec.size_counts[medium_idx] + rec.size_counts[large_idx]
    if target_bin == "medium":
        return rec.size_counts[large_idx] + rec.size_counts[small_idx]
    return rec.size_counts[medium_idx] + rec.size_counts[small_idx]


def _count_class_objects(records: list[SourceImageRecord]) -> dict[int, int]:
    """Count total annotations per class_id across all records."""
    counts: dict[int, int] = {}
    for rec in records:
        for ann in rec.annotations:
            counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
    return counts


def _class_distribution_from_counts(class_counts: dict[int, int]) -> dict[str, float]:
    """Return fractional distribution {str(class_id): fraction} sorted by class_id."""
    total = sum(class_counts.values())
    if total == 0:
        return {}
    return {
        str(cls_id): round(count / total, 4)
        for cls_id, count in sorted(class_counts.items())
    }


def _class_balance_score(
    rec: SourceImageRecord,
    class_counts: dict[int, int],
    weight: float,
) -> float:
    """Extra score for images that contain objects of under-represented classes.

    For each annotation in *rec* we compute the normalised deficit of that
    class relative to the ideal equal split, then multiply by *weight*.
    Images rich in rare classes therefore rank higher in candidate selection.
    """
    if not rec.annotations or not class_counts:
        return 0.0
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    if num_classes <= 1 or total == 0:
        return 0.0
    ideal = total / num_classes
    bonus = 0.0
    for ann in rec.annotations:
        count = class_counts.get(ann.class_id, 0)
        deficit = max(0.0, ideal - count) / ideal   # 0 when at/above ideal, 1 when count=0
        bonus += deficit * weight
    return bonus


def _candidate_scale_range(
    rec: SourceImageRecord,
    target_bin: str,
    cfg: SizeAugmenterConfig,
) -> tuple[float, float] | None:
    if target_bin == "small" and cfg.allow_downscale:
        return cfg.scale_ranges["to_small"]
    if target_bin == "medium" and cfg.allow_downscale and rec.size_counts[2] > 0:
        return cfg.scale_ranges["to_medium"]
    if target_bin == "medium" and cfg.allow_upscale and rec.size_counts[0] > 0:
        return (1.05, max(1.05, cfg.scale_ranges["to_large"][0]))
    if target_bin == "large" and cfg.allow_upscale:
        return cfg.scale_ranges["to_large"]
    return None


def simulate_annotations(
    annotations: list[BBoxAnnotation],
    scale: float,
    image_width: int,
    image_height: int,
    min_bbox_size_px: float,
    min_bbox_area_px2: float,
) -> list[BBoxAnnotation]:
    transformed: list[BBoxAnnotation] = []
    for ann in annotations:
        x1 = ann.x_center - ann.width / 2
        y1 = ann.y_center - ann.height / 2
        x2 = ann.x_center + ann.width / 2
        y2 = ann.y_center + ann.height / 2

        new_x1 = (x1 - 0.5) * scale + 0.5
        new_y1 = (y1 - 0.5) * scale + 0.5
        new_x2 = (x2 - 0.5) * scale + 0.5
        new_y2 = (y2 - 0.5) * scale + 0.5

        clip_x1 = max(0.0, min(1.0, new_x1))
        clip_y1 = max(0.0, min(1.0, new_y1))
        clip_x2 = max(0.0, min(1.0, new_x2))
        clip_y2 = max(0.0, min(1.0, new_y2))

        new_w = clip_x2 - clip_x1
        new_h = clip_y2 - clip_y1
        if new_w <= 0 or new_h <= 0:
            continue

        bbox_w_px = new_w * image_width
        bbox_h_px = new_h * image_height
        area_px2 = bbox_w_px * bbox_h_px
        if bbox_w_px < min_bbox_size_px or bbox_h_px < min_bbox_size_px:
            continue
        if area_px2 < min_bbox_area_px2:
            continue

        transformed.append(
            BBoxAnnotation(
                class_id=ann.class_id,
                x_center=(clip_x1 + clip_x2) / 2,
                y_center=(clip_y1 + clip_y2) / 2,
                width=new_w,
                height=new_h,
            )
        )
    return transformed


def evaluate_candidate_gain(
    current_counts: list[int],
    candidate_counts: list[int],
    target_distribution: dict[str, float],
) -> float:
    before = _distribution_distance(current_counts, target_distribution)
    after = _distribution_distance(
        [a + b for a, b in zip(current_counts, candidate_counts)],
        target_distribution,
    )
    return before - after


def _distribution_distance(
    counts: list[int],
    target_distribution: dict[str, float],
) -> float:
    current = distribution_from_counts(counts)
    return sum(
        abs(current[name] - target_distribution[name]) for name, _, _ in COCO_SIZE_BINS
    )


def generate_augmented_dataset(
    src_root: Path,
    dst_root: Path,
    cfg: SizeAugmenterConfig,
) -> dict[str, object]:
    rng = random.Random(cfg.seed)
    source_records = {
        split: collect_split_records(src_root, split)
        for split in ("train", "val", "test")
    }

    if dst_root.exists():
        shutil.rmtree(dst_root)
    _copytree_with_progress(src_root, dst_root)

    manifest: dict[str, object] = {
        "source_dataset": str(src_root),
        "output_dataset": str(dst_root),
        "target_distribution": cfg.target_distribution,
        "tolerance": cfg.tolerance,
        "preserve_originals": cfg.preserve_originals,
        "allow_downscale": cfg.allow_downscale,
        "allow_upscale": cfg.allow_upscale,
        "applied_splits": {},
    }

    for split_label, enabled in cfg.apply_to.items():
        if not enabled:
            continue
        split_result = augment_split(
            src_root=src_root,
            dst_root=dst_root,
            split_label=split_label,
            records=source_records[split_label],
            cfg=cfg,
            rng=rng,
        )
        manifest["applied_splits"][split_label] = split_result

    manifest_path = dst_root / "dimension_augmentation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def augment_split(
    src_root: Path,
    dst_root: Path,
    split_label: str,
    records: list[SourceImageRecord],
    cfg: SizeAugmenterConfig,
    rng: random.Random,
) -> dict[str, object]:
    folder_name = _split_folder_name(dst_root, split_label)
    if folder_name is None:
        return {"status": "missing_split"}

    split_dir = dst_root / folder_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.preserve_originals:
        _clear_split_contents(images_dir, labels_dir)

    initial_counts = (
        aggregate_split_size_counts(records) if cfg.preserve_originals else [0, 0, 0]
    )
    current_counts = initial_counts[:]
    initial_distribution = distribution_from_counts(initial_counts)
    max_new_images = resolve_max_new_images(records, cfg)

    # ── class-balance tracking ────────────────────────────────────────────────
    initial_class_counts: dict[int, int] = {}
    if cfg.balance_classes:
        raw_class_counts = _count_class_objects(records)
        if cfg.preserve_originals:
            initial_class_counts = raw_class_counts
        # even when not preserving originals we still use the source distribution
        # as the starting reference for scoring, but report 0 in the manifest
        class_counts: dict[int, int] = dict(initial_class_counts) if cfg.preserve_originals else dict.fromkeys(raw_class_counts, 0)
    else:
        class_counts = {}

    generated = 0
    usage_counts: dict[Path, int] = {}
    accepted_samples: list[GeneratedSample] = []
    bin_candidates = build_bin_candidates(records, cfg)
    compact_every = 500

    last_progress_pct = -1
    if max_new_images > 0:
        last_progress_pct = _print_progress(0, max_new_images, split_label, last_progress_pct)

    while generated < max_new_images:
        if cfg.stop_when_within_tolerance and within_tolerance(
            current_counts, cfg.target_distribution, cfg.tolerance
        ):
            break

        target_bin = select_target_bin(current_counts, cfg.target_distribution)
        if target_bin is None:
            break

        choice = choose_candidate(
            bin_candidates[target_bin], cfg, usage_counts, rng,
            class_counts=class_counts if cfg.balance_classes else None,
        )
        if choice is None:
            break
        source, scale = choice

        transformed = simulate_annotations(
            annotations=source.annotations,
            scale=scale,
            image_width=source.width,
            image_height=source.height,
            min_bbox_size_px=cfg.min_bbox_size_px,
            min_bbox_area_px2=cfg.min_bbox_area_px2,
        )
        if not transformed:
            usage_counts[source.image_path] = usage_counts.get(source.image_path, 0) + 1
            continue

        size_counts = count_size_bins(transformed, source.width, source.height)
        gain = evaluate_candidate_gain(current_counts, size_counts, cfg.target_distribution)
        usage_counts[source.image_path] = usage_counts.get(source.image_path, 0) + 1
        if gain <= 0:
            continue

        copy_idx = usage_counts[source.image_path]
        image_name = f"{source.image_path.stem}_dimaug_{copy_idx:02d}{source.image_path.suffix.lower()}"
        label_name = f"{source.image_path.stem}_dimaug_{copy_idx:02d}.txt"
        sample = GeneratedSample(
            source=source,
            scale=scale,
            output_image_name=image_name,
            output_label_name=label_name,
            annotations=transformed,
            size_counts=size_counts,
        )
        _write_generated_sample(sample, images_dir, labels_dir, cfg)
        accepted_samples.append(sample)
        current_counts = [a + b for a, b in zip(current_counts, size_counts)]

        # update per-class counters for the next candidate selection
        if cfg.balance_classes:
            for ann in sample.annotations:
                class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1

        generated += 1
        if generated % compact_every == 0:
            compact_bin_candidates(bin_candidates, usage_counts, cfg.max_copies_per_source_image)
        last_progress_pct = _print_progress(
            generated, max_new_images, split_label, last_progress_pct
        )

    if max_new_images > 0:
        print()

    final_distribution = distribution_from_counts(current_counts)
    result: dict[str, object] = {
        "status": "ok",
        "source_split_path": str(src_root / (_split_folder_name(src_root, split_label) or split_label)),
        "initial_counts": dict(zip(size_bin_names(), initial_counts)),
        "initial_distribution": initial_distribution,
        "max_new_images_effective": max_new_images,
        "final_counts": dict(zip(size_bin_names(), current_counts)),
        "final_distribution": final_distribution,
        "generated_images": generated,
        "samples": [
            {
                "source_image": str(sample.source.image_path),
                "scale": round(sample.scale, 4),
                "output_image": sample.output_image_name,
                "output_label": sample.output_label_name,
                "size_counts": dict(zip(size_bin_names(), sample.size_counts)),
            }
            for sample in accepted_samples
        ],
    }

    if cfg.balance_classes:
        result["initial_class_distribution"] = _class_distribution_from_counts(initial_class_counts)
        result["final_class_distribution"] = _class_distribution_from_counts(class_counts)

    return result


def _copytree_with_progress(src_root: Path, dst_root: Path) -> None:
    files = [p for p in src_root.rglob("*") if p.is_file()]
    n_total = len(files)
    step = max(1, n_total // 100)
    print(f"  copying {n_total:,} files from {src_root.name} to {dst_root.name}")

    for done, src_path in enumerate(files, 1):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        if done % step == 0 or done == n_total:
            print(f"\r  {done:,}/{n_total:,}", end="", flush=True)
    if n_total:
        print()


def _clear_split_contents(images_dir: Path, labels_dir: Path) -> None:
    for path in images_dir.iterdir():
        if path.is_file():
            path.unlink()
    for path in labels_dir.iterdir():
        if path.is_file():
            path.unlink()


def aggregate_split_size_counts(records: list[SourceImageRecord]) -> list[int]:
    counts = [0] * len(COCO_SIZE_BINS)
    for rec in records:
        for idx, value in enumerate(rec.size_counts):
            counts[idx] += value
    return counts


def resolve_max_new_images(
    records: list[SourceImageRecord],
    cfg: SizeAugmenterConfig,
) -> int:
    caps = [cfg.max_new_images]
    if cfg.max_new_images_percent is not None:
        base_count = len(records)
        percent_cap = math.ceil(base_count * cfg.max_new_images_percent / 100.0)
        caps.append(percent_cap)
    return max(0, min(caps))


def size_bin_names() -> list[str]:
    return [name for name, _, _ in COCO_SIZE_BINS]


def _write_generated_sample(
    sample: GeneratedSample,
    images_dir: Path,
    labels_dir: Path,
    cfg: SizeAugmenterConfig,
) -> None:
    output_image_path = images_dir / sample.output_image_name
    output_label_path = labels_dir / sample.output_label_name

    with Image.open(sample.source.image_path) as image:
        image = image.convert("RGB")
        transformed = _transform_image(image, sample.scale, cfg.padding_color)
        transformed.save(output_image_path)

    label_lines = [
        f"{ann.class_id} {ann.x_center:.6f} {ann.y_center:.6f} {ann.width:.6f} {ann.height:.6f}"
        for ann in sample.annotations
    ]
    output_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def _transform_image(
    image: Image.Image,
    scale: float,
    padding_color: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    scaled_width = max(1, round(width * scale))
    scaled_height = max(1, round(height * scale))
    resized = image.resize((scaled_width, scaled_height), resample=_resample_for_scale(scale))

    if scale <= 1.0:
        canvas = Image.new("RGB", (width, height), color=padding_color)
        offset_x = (width - scaled_width) // 2
        offset_y = (height - scaled_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    left = max(0, (scaled_width - width) // 2)
    top = max(0, (scaled_height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _resample_for_scale(scale: float) -> int:
    return Image.Resampling.LANCZOS if scale > 1.0 else Image.Resampling.BOX


def _print_progress(
    done: int,
    total: int,
    split_label: str,
    last_progress_pct: int,
) -> int:
    if total <= 0:
        return last_progress_pct
    progress_pct = min(100, int((done * 100) / total))
    if progress_pct == last_progress_pct and done not in (0, total):
        return last_progress_pct
    width = 28
    pct = done / total
    filled = min(width, int(width * pct))
    bar = "#" * filled + "." * (width - filled)
    print(
        f"\r[{split_label}] generating {done}/{total} [{bar}] {pct:5.1%}",
        end="",
        flush=True,
    )
    return progress_pct
