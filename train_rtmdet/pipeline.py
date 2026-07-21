from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import yaml

from .normalize import build_rename_plan, execute_rename_plan


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROJECT_DIR = Path.cwd() / "runs" / "rtmdet"
DEFAULT_PACKAGE_DIR = Path.cwd() / "models" / "rtmdet"

RTMDET_MODELS = {
    "tiny": {
        "base_config": "rtmdet_tiny_8xb32-300e_coco.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_tiny_8xb32-300e_coco/"
            "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
        ),
    },
    "s": {
        "base_config": "rtmdet_s_8xb32-300e_coco.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_s_8xb32-300e_coco/"
            "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
        ),
    },
    "m": {
        "base_config": "rtmdet_m_8xb32-300e_coco.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_m_8xb32-300e_coco/"
            "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
        ),
    },
    "l": {
        "base_config": "rtmdet_l_8xb32-300e_coco.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_l_8xb32-300e_coco/"
            "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
        ),
    },
    "x": {
        "base_config": "rtmdet_x_8xb32-300e_coco.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_x_8xb32-300e_coco/"
            "rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
        ),
    },
}


@dataclass
class SplitStats:
    images: int = 0
    label_files: int = 0
    annotations: int = 0
    segment_annotations: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0


@dataclass
class DatasetValidationResult:
    dataset_root: Path
    class_names: list[str]
    split_dirs: dict[str, Path]
    stats: dict[str, SplitStats]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class RTMDetPipelineConfig:
    dataset_path: str | Path
    model_name: str
    variant: str = "s"
    imgsz: int = 640
    logger_interval: int = 10
    epochs: int = 100
    stage2_epochs: int = 35
    batch_size: int = 16
    workers: int = 8
    val_batch_size: int = 8
    val_interval: int = 5
    base_lr: float = 0.001
    device: str = "cuda"
    num_gpus: int = 1
    seed: int = 1
    project_dir: str | Path = field(default_factory=lambda: Path.cwd() / "runs" / "rtmdet")
    package_dir: str | Path = field(default_factory=lambda: Path.cwd() / "models" / "rtmdet")
    mmdet_root: str | Path | None = None
    mmdeploy_root: str | Path | None = None
    pretrained_checkpoint: str | Path | None = None
    class_names: list[str] | None = None
    nc: int | None = None
    normalize_names: bool = False
    convert_segments_to_boxes: bool = True
    stop_on_validation_errors: bool = True
    allow_missing_labels: bool = True
    prepare_only: bool = True
    run_training: bool = True
    run_evaluation: bool = True
    run_export: bool = False
    run_packaging: bool = True
    amp: bool = True
    resume: bool = False
    python_executable: str = sys.executable
    deploy_config: str | Path | None = None
    sample_image: str | Path | None = None
    checkpoint_for_export: str | Path | None = None
    trtexec_path: str | Path = "trtexec"
    benchmark_iterations: int = 500
    # ONNX export (triggered when save_onnx_weights=True after training)
    save_onnx_weights: bool = False
    onnx_score_threshold: float = 0.05
    onnx_iou_threshold: float = 0.5
    onnx_keep_top_k: int = 300
    early_stopping: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    generate_plots: bool = True
    eval_conf_threshold: float = 0.25
    eval_iou_threshold: float = 0.50


def is_mmdet_root(path: str | Path | None) -> bool:
    if not path:
        return False
    root = Path(path).expanduser().resolve()
    return (
        root.is_dir()
        and (root / "tools" / "train.py").is_file()
        and (root / "tools" / "test.py").is_file()
        and (root / "configs" / "rtmdet").is_dir()
    )


def discover_mmdet_root(config: RTMDetPipelineConfig) -> Path | None:
    candidates: list[Path] = []

    if config.mmdet_root:
        candidate = Path(config.mmdet_root).expanduser()
        candidates.append(candidate)

    for env_name in ("MMDET_ROOT", "MMDETECTION_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    workspace_root = Path.cwd()
    home = Path.home()
    candidates.extend(
        [
            workspace_root / "mmdetection",
            workspace_root.parent / "mmdetection",
            workspace_root / "MMDetection",
            workspace_root.parent / "MMDetection",
            home / "mmdetection",
            home / "MMDetection",
            home / "Desktop" / "mmdetection",
            home / "Desktop" / "MMDetection",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_mmdet_root(resolved):
            config.mmdet_root = resolved
            return resolved

    return None


def resolve_dataset_root(dataset_input: str | Path) -> tuple[Path, Path | None]:
    dataset_path = Path(dataset_input).resolve()

    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            raise FileNotFoundError(f"Unsupported dataset file: {dataset_path}")
        return dataset_path.parent, dataset_path

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    for candidate in (dataset_path / "data.yaml", dataset_path / "data.yml"):
        if candidate.is_file():
            return dataset_path, candidate

    extra_yaml = sorted(dataset_path.glob("*.yaml")) + sorted(dataset_path.glob("*.yml"))
    return dataset_path, extra_yaml[0] if extra_yaml else None


def detect_split_dirs(dataset_root: Path) -> dict[str, Path]:
    split_dirs: dict[str, Path] = {}

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    split_dirs["train"] = train_dir

    if valid_dir.is_dir():
        split_dirs["val"] = valid_dir
    elif val_dir.is_dir():
        split_dirs["val"] = val_dir
    else:
        raise FileNotFoundError(f"Missing valid or val folder inside {dataset_root}")

    if test_dir.is_dir():
        split_dirs["test"] = test_dir

    for split_name, split_dir in split_dirs.items():
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing images folder for {split_name}: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels folder for {split_name}: {labels_dir}")

    return split_dirs


def parse_names_from_yaml(yaml_path: Path | None) -> list[str] | None:
    if yaml_path is None or not yaml_path.is_file():
        return None

    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    names = data.get("names")
    if names is None:
        return None

    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names, key=lambda item: int(item))
        except Exception:
            ordered_keys = sorted(names)
        return [str(names[key]) for key in ordered_keys]

    return None


def resolve_class_names(config: RTMDetPipelineConfig, yaml_path: Path | None) -> list[str]:
    if config.class_names:
        return [str(name) for name in config.class_names]

    yaml_names = parse_names_from_yaml(yaml_path)
    if yaml_names:
        return yaml_names

    if config.nc is not None and config.nc > 0:
        return [f"class_{index}" for index in range(config.nc)]

    raise ValueError(
        "Cannot determine class names. Add a data.yaml with 'names', "
        "or set class_names / nc in hyperparameter_config.yaml."
    )


def collect_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    size = read_image_size_stdlib(image_path)
    if size is not None:
        return size

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        return width, height
    except ModuleNotFoundError:
        pass

    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        return width, height
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow or OpenCV is required to read image dimensions. "
            "Install with: pip install pillow"
        ) from exc


def read_image_size_stdlib(image_path: Path) -> tuple[int, int] | None:
    suffix = image_path.suffix.lower()

    with image_path.open("rb") as stream:
        header = stream.read(32)

        if suffix == ".png" and header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)

        if suffix == ".bmp" and header.startswith(b"BM"):
            width, height = struct.unpack("<ii", header[18:26])
            return abs(int(width)), abs(int(height))

        if suffix in {".jpg", ".jpeg"} and header.startswith(b"\xff\xd8"):
            stream.seek(2)
            while True:
                marker_prefix = stream.read(1)
                if not marker_prefix:
                    return None
                if marker_prefix != b"\xff":
                    continue

                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue

                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    return None
                segment_length = struct.unpack(">H", length_bytes)[0]
                if segment_length < 2:
                    return None

                marker_code = marker[0]
                if marker_code in {
                    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
                }:
                    data = stream.read(5)
                    if len(data) != 5:
                        return None
                    height, width = struct.unpack(">HH", data[1:5])
                    return int(width), int(height)

                stream.seek(segment_length - 2, 1)

    return None


def parse_yolo_label_file(
    label_path: Path,
    image_path: Path,
    class_count: int,
    convert_segments_to_boxes: bool = True,
) -> tuple[list[dict[str, float | int]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, float | int]] = []

    if not label_path.is_file():
        return rows, errors

    with label_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                class_id_float = float(parts[0])
                class_id = int(class_id_float)
            except ValueError:
                errors.append(f"{label_path}:{line_number} contains non-numeric values.")
                continue

            is_segment = False
            if len(parts) == 5:
                try:
                    xc, yc, width, height = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number} contains non-numeric bbox values.")
                    continue
            elif convert_segments_to_boxes and len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                try:
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number} contains non-numeric segment values.")
                    continue
                xs = coords[0::2]
                ys = coords[1::2]
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                xc = (x_min + x_max) / 2.0
                yc = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min
                is_segment = True
            else:
                errors.append(
                    f"{label_path}:{line_number} must have 5 YOLO bbox fields "
                    "or segment format class_id x1 y1 x2 y2 ..."
                )
                continue

            if class_id_float != class_id:
                errors.append(f"{label_path}:{line_number} non-integer class_id: {parts[0]}")
            if class_id < 0 or class_id >= class_count:
                errors.append(
                    f"{label_path}:{line_number} class_id {class_id} out of range "
                    f"0..{class_count - 1}."
                )

            # Zero-area boxes (degenerate segments where all points coincide) are
            # silently dropped — they carry no training signal and cannot be fixed.
            if width <= 0 or height <= 0:
                continue

            # Clamp tiny floating-point overruns (e.g. 1.0000000001 → 1.0) that
            # arise from segment-to-bbox conversion.  Truly out-of-range values
            # (further than 1e-4 from the boundary) are still reported as errors.
            _tol = 1e-4
            if not (-_tol <= xc <= 1 + _tol and -_tol <= yc <= 1 + _tol
                    and -_tol < width <= 1 + _tol and -_tol < height <= 1 + _tol):
                errors.append(f"{label_path}:{line_number} coordinates out of range: "
                               f"xc={xc:.4f} yc={yc:.4f} w={width:.4f} h={height:.4f}.")
                continue
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            rows.append(
                {
                    "class_id": class_id,
                    "x_center": xc,
                    "y_center": yc,
                    "width": width,
                    "height": height,
                    "is_segment": int(is_segment),
                }
            )

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        errors.append(f"Unsupported image extension: {image_path}")

    return rows, errors


def validate_yolo_dataset(config: RTMDetPipelineConfig) -> DatasetValidationResult:
    dataset_root, yaml_path = resolve_dataset_root(config.dataset_path)
    split_dirs = detect_split_dirs(dataset_root)
    class_names = resolve_class_names(config, yaml_path)

    stats: dict[str, SplitStats] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for split_name, split_dir in split_dirs.items():
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        image_files = collect_image_files(images_dir)
        image_stems = {image_path.stem for image_path in image_files}
        label_files = sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower())
        label_stems = {label_path.stem for label_path in label_files}

        split_stats = SplitStats(images=len(image_files), label_files=len(label_files))
        stats[split_name] = split_stats

        if not image_files:
            errors.append(f"[{split_name}] No images found in {images_dir}")

        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        split_stats.missing_labels = len(missing_labels)
        split_stats.orphan_labels = len(orphan_labels)

        if missing_labels:
            message = f"[{split_name}] {len(missing_labels)} images without label."
            if config.allow_missing_labels:
                warnings.append(message)
            else:
                errors.append(message)

        if orphan_labels:
            warnings.append(f"[{split_name}] {len(orphan_labels)} labels without image.")

        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            rows, row_errors = parse_yolo_label_file(
                label_path,
                image_path,
                len(class_names),
                convert_segments_to_boxes=config.convert_segments_to_boxes,
            )
            errors.extend(row_errors)
            split_stats.annotations += len(rows)
            split_stats.segment_annotations += sum(int(row.get("is_segment", 0)) for row in rows)
            if label_path.is_file() and not rows:
                split_stats.empty_labels += 1

    return DatasetValidationResult(
        dataset_root=dataset_root,
        class_names=class_names,
        split_dirs=split_dirs,
        stats=stats,
        warnings=warnings,
        errors=errors,
    )


def maybe_normalize_dataset(dataset_root: Path) -> None:
    rename_plan, warnings = build_rename_plan(dataset_root, label_prefix="")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if not rename_plan:
        print("Dataset already normalized.")
        return

    print("Normalizing filenames...")
    execute_rename_plan(rename_plan)
    print("Normalization complete.")


def fix_yolo_labels(dataset_root: Path) -> None:
    """Fix degenerate YOLO annotations in-place across all splits.

    Two corrections are applied:
    - Zero-area boxes (w=0 or h=0): the annotation line is removed entirely.
      These arise from segment polygons where all points coincide; they carry
      no training signal and cannot be repaired.
    - Coordinates with a tiny floating-point overrun outside [0, 1] (tolerance
      1e-4): clamped to the valid range.  Coordinates further out of range are
      left untouched so the validator can report them.

    Only files that actually change are rewritten.
    """
    _tol = 1e-4

    total_removed = 0
    total_clamped = 0
    total_files = 0

    for split_dir in sorted(dataset_root.iterdir()):
        if not split_dir.is_dir():
            continue
        labels_dir = split_dir / "labels"
        if not labels_dir.is_dir():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            original = label_file.read_text(encoding="utf-8").splitlines()
            new_lines: list[str] = []
            removed = 0
            clamped = 0

            for raw in original:
                line = raw.strip()
                if not line:
                    continue

                parts = line.split()

                # ── Detect format ────────────────────────────────────────────
                is_bbox = len(parts) == 5
                is_seg = len(parts) >= 7 and (len(parts) - 1) % 2 == 0

                if not (is_bbox or is_seg):
                    new_lines.append(line)
                    continue

                try:
                    cls_id = parts[0]
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    new_lines.append(line)
                    continue

                if is_bbox:
                    xc, yc, w, h = coords
                else:
                    # Segment → derive bbox to check area
                    xs = coords[0::2]
                    ys = coords[1::2]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    xc = yc = 0.0  # not used further for segments

                # ── Drop zero-area annotations ───────────────────────────────
                if w <= 0 or h <= 0:
                    removed += 1
                    continue

                # ── Clamp tiny float overruns (bbox only) ────────────────────
                if is_bbox:
                    xc, yc, w, h = coords
                    needs_clamp = (
                        not (0 <= xc <= 1 and 0 <= yc <= 1
                             and 0 < w <= 1 and 0 < h <= 1)
                        and (-_tol <= xc <= 1 + _tol and -_tol <= yc <= 1 + _tol
                             and -_tol < w <= 1 + _tol and -_tol < h <= 1 + _tol)
                    )
                    if needs_clamp:
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        w  = max(1e-6, min(1.0, w))
                        h  = max(1e-6, min(1.0, h))
                        new_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        clamped += 1
                        continue

                new_lines.append(line)

            if removed or clamped:
                text = "\n".join(new_lines)
                label_file.write_text(text + "\n" if text else "", encoding="utf-8")
                total_files += 1
                total_removed += removed
                total_clamped += clamped

    if total_removed or total_clamped:
        print(
            f"Annotation fix: {total_removed} zero-area box(es) removed, "
            f"{total_clamped} coordinate(s) clamped — {total_files} file(s) rewritten."
        )
    else:
        print("Annotation fix: no issues found.")


def yolo_box_to_coco_bbox(
    row: dict[str, float | int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    xc = float(row["x_center"])
    yc = float(row["y_center"])
    width = float(row["width"])
    height = float(row["height"])

    x_min = (xc - width / 2.0) * image_width
    y_min = (yc - height / 2.0) * image_height
    box_width = width * image_width
    box_height = height * image_height

    x_min = max(0.0, min(x_min, float(image_width)))
    y_min = max(0.0, min(y_min, float(image_height)))
    box_width = max(0.0, min(box_width, float(image_width) - x_min))
    box_height = max(0.0, min(box_height, float(image_height) - y_min))

    return x_min, y_min, box_width, box_height


def convert_split_to_coco(
    split_name: str,
    split_dir: Path,
    class_names: list[str],
    output_json: Path,
) -> Path:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_files = collect_image_files(images_dir)

    categories = [
        {"id": class_id + 1, "name": class_name, "supercategory": "object"}
        for class_id, class_name in enumerate(class_names)
    ]

    coco: dict[str, Any] = {
        "info": {
            "description": f"{split_name} converted from YOLO normalized txt",
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    for image_id, image_path in enumerate(image_files, start=1):
        width, height = read_image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        rows, row_errors = parse_yolo_label_file(label_path, image_path, len(class_names))
        if row_errors:
            raise ValueError("\n".join(row_errors[:20]))

        for row in rows:
            x_min, y_min, box_width, box_height = yolo_box_to_coco_bbox(row, width, height)
            if box_width <= 0 or box_height <= 0:
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(row["class_id"]) + 1,
                    "bbox": [
                        round(x_min, 4),
                        round(y_min, 4),
                        round(box_width, 4),
                        round(box_height, 4),
                    ],
                    "area": round(box_width * box_height, 4),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(coco, stream, indent=2, ensure_ascii=True)

    return output_json


def convert_yolo_dataset_to_coco(validation: DatasetValidationResult) -> dict[str, Path]:
    annotation_dir = validation.dataset_root / "annotations"
    outputs: dict[str, Path] = {}

    for split_name, split_dir in validation.split_dirs.items():
        output_json = annotation_dir / f"instances_{split_name}.json"
        outputs[split_name] = convert_split_to_coco(
            split_name=split_name,
            split_dir=split_dir,
            class_names=validation.class_names,
            output_json=output_json,
        )
        print(f"COCO {split_name}: {outputs[split_name]}")

    return outputs


def get_variant_info(variant: str) -> dict[str, str]:
    key = variant.lower().strip()
    if key not in RTMDET_MODELS:
        raise ValueError(
            f"Unsupported RTMDet variant: '{variant}'. "
            f"Valid options: {sorted(RTMDET_MODELS)}"
        )
    return RTMDET_MODELS[key]


def resolve_base_config(config: RTMDetPipelineConfig) -> str:
    variant_info = get_variant_info(config.variant)
    base_config_name = variant_info["base_config"]
    # Use the portable mmdet:: notation so the generated config works on any
    # machine regardless of where mmdetection is cloned or installed.
    return f"mmdet::rtmdet/{base_config_name}"


def resolve_checkpoint(config: RTMDetPipelineConfig) -> str:
    if config.pretrained_checkpoint:
        return str(config.pretrained_checkpoint)
    return get_variant_info(config.variant)["checkpoint"]


def build_pipeline_block(imgsz: int) -> str:
    return f"""
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{{{_base_.backend_args}}}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=({imgsz}, {imgsz}), pad_val=114.0),
    dict(type='RandomResize', scale=({imgsz * 2}, {imgsz * 2}), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=({imgsz}, {imgsz})),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=({imgsz}, {imgsz}), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=({imgsz}, {imgsz}), ratio_range=(1.0, 1.0), max_cached_images=20, pad_val=(114, 114, 114)),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{{{_base_.backend_args}}}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=({imgsz}, {imgsz}), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=({imgsz}, {imgsz})),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=({imgsz}, {imgsz}), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{{{_base_.backend_args}}}}),
    dict(type='Resize', scale=({imgsz}, {imgsz}), keep_ratio=True),
    dict(type='Pad', size=({imgsz}, {imgsz}), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
"""


def _build_early_stopping_hook(config: RTMDetPipelineConfig) -> str:
    if not config.early_stopping:
        return ""
    return (
        f"    dict(\n"
        f"        type='EarlyStoppingHook',\n"
        f"        monitor='coco/bbox_mAP',\n"
        f"        patience={config.early_stopping_patience},\n"
        f"        min_delta={config.early_stopping_min_delta},\n"
        f"        rule='greater'),\n"
    )


_TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def tidy_checkpoints_dir(run_dir: Path) -> None:
    """Move MMEngine's own housekeeping out of checkpoints/ into logs/.

    MMEngine is given checkpoints/ as its --work-dir, so alongside the
    actual epoch_*.pth / best_*.pth files it also drops one timestamped
    folder per launch (vis_data/scalars.json, a dumped copy of the config,
    log.json, ...) plus a loose copy of the config .py — that's what turns
    checkpoints/ into a mix of run metadata and checkpoint files. This
    relocates the metadata to run_dir/logs/, leaving checkpoints/ with only
    .pth files and the last_checkpoint marker. Safe to call repeatedly —
    already-moved runs are simply skipped.
    """
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return
    logs_dir = run_dir / "logs"

    for entry in checkpoints_dir.iterdir():
        if entry.is_dir() and _TIMESTAMP_DIR_RE.match(entry.name):
            logs_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(logs_dir / entry.name))
        elif entry.is_file() and entry.suffix == ".py":
            logs_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(logs_dir / entry.name))


def generate_mmdet_config(
    config: RTMDetPipelineConfig,
    validation: DatasetValidationResult,
    coco_annotations: dict[str, Path],
) -> Path:
    project_dir = Path(config.project_dir).resolve()
    config_dir = project_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_config = config_dir / f"{config.model_name}_rtmdet_{config.variant}_{timestamp}.py"

    dataset_root = validation.dataset_root
    train_ann = coco_annotations["train"].relative_to(dataset_root)
    val_ann = coco_annotations["val"].relative_to(dataset_root)
    test_ann = coco_annotations.get("test")
    test_ann_rel = test_ann.relative_to(dataset_root) if test_ann else val_ann
    test_prefix = (
        validation.split_dirs["test"].name
        if "test" in validation.split_dirs
        else validation.split_dirs["val"].name
    )

    class_tuple = tuple(validation.class_names)
    metainfo = {"classes": class_tuple}
    max_epochs = int(config.epochs)
    stage2_epochs = max(1, min(int(config.stage2_epochs), max_epochs))
    val_interval = max(1, int(config.val_interval))
    switch_epoch = max(1, max_epochs - stage2_epochs)
    # keep every stage-2 checkpoint so the SWA script has enough to average
    swa_keep = max(3, -(-stage2_epochs // val_interval) + 1)

    config_text = f"""# Auto-generated RTMDet fine-tuning config.
# Source dataset: {dataset_root}

_base_ = {resolve_base_config(config)!r}

data_root = {str(dataset_root).replace(chr(92), '/') + '/'!r}
metainfo = {pformat(metainfo, width=100)}
num_classes = {len(validation.class_names)}
max_epochs = {max_epochs}
stage2_num_epochs = {stage2_epochs}
base_lr = {float(config.base_lr)!r}
interval = {val_interval}

load_from = {resolve_checkpoint(config)!r}
resume = {bool(config.resume)!r}
randomness = dict(seed={int(config.seed)}, deterministic=False)

model = dict(
    bbox_head=dict(num_classes=num_classes),
)
{build_pipeline_block(int(config.imgsz))}
train_dataloader = dict(
    batch_size={int(config.batch_size)},
    num_workers={int(config.workers)},
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={str(train_ann).replace(chr(92), '/')!r},
        data_prefix=dict(img='{validation.split_dirs["train"].name}/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size={int(config.val_batch_size)},
    num_workers={int(config.workers)},
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={str(val_ann).replace(chr(92), '/')!r},
        data_prefix=dict(img='{validation.split_dirs["val"].name}/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size={int(config.val_batch_size)},
    num_workers={int(config.workers)},
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={str(test_ann_rel).replace(chr(92), '/')!r},
        data_prefix=dict(img='{test_prefix}/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + {str(val_ann).replace(chr(92), '/')!r},
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 1, 10),
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + {str(test_ann_rel).replace(chr(92), '/')!r},
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 1, 10),
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[({switch_epoch}, 1)],
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts={swa_keep}, save_best='coco/bbox_mAP'),
    logger=dict(interval={int(config.logger_interval)}, type='LoggerHook'),
    timer=dict(type='IterTimerHook'),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch={switch_epoch},
        switch_pipeline=train_pipeline_stage2),
{_build_early_stopping_hook(config)}]

work_dir ={str((project_dir / config.model_name / "checkpoints").resolve()).replace(chr(92), '/')!r}
"""

    with output_config.open("w", encoding="utf-8") as stream:
        stream.write(config_text)

    return output_config


def print_validation_summary(validation: DatasetValidationResult) -> None:
    print(f"Dataset: {validation.dataset_root}")
    print(f"Classes ({len(validation.class_names)}): {', '.join(validation.class_names)}")

    for split_name, split_stats in validation.stats.items():
        print(
            f"  {split_name}: {split_stats.images} images, "
            f"{split_stats.label_files} labels, {split_stats.annotations} boxes"
        )
        if split_stats.segment_annotations:
            print(f"    Segments converted to bbox: {split_stats.segment_annotations}")

    for warning in validation.warnings:
        print(f"WARNING: {warning}")
    for error in validation.errors[:50]:
        print(f"ERROR: {error}")
    if len(validation.errors) > 50:
        print(f"ERROR: {len(validation.errors) - 50} additional errors not shown.")


def find_mmdet_tool(config: RTMDetPipelineConfig, tool_name: str) -> Path:
    mmdet_root = discover_mmdet_root(config)
    if not mmdet_root:
        raise ValueError(
            "MMDetection is not configured. "
            "Set mmdet_root in hyperparameter_config.yaml to the path of your "
            "MMDetection clone (the folder containing tools/train.py). "
            "Example: mmdet_root = C:\\Users\\name\\Desktop\\mmdetection"
        )

    tool_path = mmdet_root / "tools" / tool_name
    if not tool_path.is_file():
        raise FileNotFoundError(f"MMDetection tool not found: {tool_path}")
    return tool_path


_SUPPRESS_INLINE = (
    " paramwise_options --",
    '"FileClient" will be deprecated',
    '"HardDiskBackend" is the alias',
    " is duplicate. It is skipped",
)

_BLOCK_STARTS = (
    " - INFO - Config:",
    " - INFO - Hooks will be executed in the following order:",
)

_BLOCK_ENDS = (
    "- INFO - Distributed training",
    "loading annotations into memory",
)


def run_command(command: list, cwd=None) -> None:
    print("\nRunning command:")
    print(" ".join(str(part) for part in command))
    if cwd:
        print(f"Working directory: {cwd}")

    env = os.environ.copy()
    if cwd:
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(cwd) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    in_block = False
    for line in proc.stdout:
        text = line.rstrip("\n")
        if in_block:
            if any(p in text for p in _BLOCK_ENDS):
                in_block = False
                sys.stdout.write(line)
                sys.stdout.flush()
            continue
        if any(p in text for p in _BLOCK_STARTS):
            in_block = True
            continue
        if any(p in text for p in _SUPPRESS_INLINE):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def verify_training_environment(config: RTMDetPipelineConfig) -> None:
    modules_to_check = ["numpy", "mmengine", "mmcv", "mmdet"]
    if config.run_export:
        modules_to_check.extend(
            ["onnx", "onnxruntime", "onnxsim", "mmdeploy", "aenum",
             "grpc", "multiprocess", "prettytable", "google.protobuf"]
        )
    check_script = (
        "import importlib.util; "
        "mods=" + repr(tuple(modules_to_check)) + "; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('\\n'.join(missing))"
    )
    result = subprocess.run(
        [str(config.python_executable), "-c", check_script],
        capture_output=True,
        text=True,
        check=True,
    )
    missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not missing:
        return

    raise ValueError(
        "The selected Python environment is missing required modules. "
        f"Missing in {config.python_executable}: {', '.join(missing)}. "
        "Install them in your Conda/venv environment and retry."
    )


def verify_tensorrt_backend(config: RTMDetPipelineConfig) -> None:
    if not config.run_export:
        return

    if not config.mmdeploy_root:
        raise ValueError("Set mmdeploy_root in hyperparameter_config.yaml for TensorRT export.")

    check_script = """
import json, os, shutil
from mmdeploy.backend.tensorrt import TensorRTManager
from mmdeploy.backend.tensorrt.init_plugins import get_ops_path
payload = {
    "tensorrt_python": TensorRTManager.is_available(),
    "tensorrt_custom_ops": TensorRTManager.is_available(with_custom_ops=True),
    "ops_path": get_ops_path(),
    "TENSORRT_DIR": os.environ.get("TENSORRT_DIR"),
    "CUDNN_DIR": os.environ.get("CUDNN_DIR"),
    "trtexec": shutil.which("trtexec"),
    "cmake": shutil.which("cmake"),
}
print(json.dumps(payload))
""".strip()

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(Path(config.mmdeploy_root).resolve()) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )

    result = subprocess.run(
        [str(config.python_executable), "-c", check_script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    info = json.loads(result.stdout.strip())

    if info["tensorrt_python"] and info["tensorrt_custom_ops"]:
        return

    problems: list[str] = []
    if not info["tensorrt_python"]:
        problems.append("Python package `tensorrt` is not importable")
    if not info["tensorrt_custom_ops"]:
        problems.append("MMDeploy TensorRT custom ops are not built/loaded")
    if not info.get("TENSORRT_DIR"):
        problems.append("`TENSORRT_DIR` environment variable is not set")
    if not info.get("CUDNN_DIR"):
        problems.append("`CUDNN_DIR` environment variable is not set")
    if not info.get("trtexec"):
        problems.append("`trtexec` is not in PATH")
    if not info.get("cmake"):
        problems.append("`cmake` is not in PATH")

    raise ValueError(
        "TensorRT export cannot start: "
        + "; ".join(problems)
        + f". Expected plugin: {info['ops_path']}."
    )


def build_train_command(config: RTMDetPipelineConfig, mmdet_config: Path) -> list[str]:
    train_script = find_mmdet_tool(config, "train.py")
    work_dir = str(Path(config.project_dir).resolve() / config.model_name / "checkpoints")

    if config.num_gpus > 1:
        command = [
            str(config.python_executable),
            "-m", "torch.distributed.run",
            f"--nproc_per_node={config.num_gpus}",
            str(train_script),
            str(mmdet_config),
            "--work-dir", work_dir,
            "--launcher", "pytorch",
        ]
    else:
        command = [
            str(config.python_executable),
            str(train_script),
            str(mmdet_config),
            "--work-dir", work_dir,
        ]

    if config.amp:
        command.append("--amp")
    if config.resume:
        command.append("--resume")
    return command


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    # Checkpoints live in run_dir/checkpoints/ for new runs;
    # fall back to run_dir/ root for runs created before the layout change.
    search_dirs = [run_dir / "checkpoints", run_dir]

    for search in search_dirs:
        best = sorted(search.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
        if best:
            return best[-1]
        latest = search / "latest.pth"
        if latest.is_file():
            return latest
        epochs = sorted(search.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        if epochs:
            return epochs[-1]

    return None


# Keys shown in the primary summary table, in display order.
_COCO_SUMMARY_KEYS: list[tuple[str, str]] = [
    ("coco/bbox_mAP_50",  "mAP50"),
    ("coco/bbox_mAP",     "mAP50-95"),
    ("coco/bbox_mAP_75",  "mAP75"),
    ("coco/bbox_mAP_s",   "mAP-s"),
    ("coco/bbox_mAP_m",   "mAP-m"),
    ("coco/bbox_mAP_l",   "mAP-l"),
]


def parse_best_metrics_from_json_log(run_dir: Path) -> dict[str, Any] | None:
    """Return the validation record with the highest coco/bbox_mAP from the MMEngine JSON log.

    MMEngine writes scalars to {timestamp}/vis_data/scalars.json under whichever directory it
    was given as --work-dir. tidy_checkpoints_dir() relocates those timestamped folders to
    {run_dir}/logs/ after training, so that's checked first; the two older layouts
    ({run_dir}/checkpoints/{timestamp}/... and {run_dir}/{timestamp}/...) are kept as fallbacks
    for runs that predate that move. The most recently modified file across all wins.
    """
    log_files = sorted(
        list(run_dir.glob("logs/*/vis_data/scalars.json"))
        or list(run_dir.glob("checkpoints/*/vis_data/scalars.json"))
        or list(run_dir.glob("*/vis_data/scalars.json")),
        key=lambda p: p.stat().st_mtime,
    )
    if not log_files:
        return None

    best_record: dict[str, Any] | None = None
    best_map: float = -1.0

    with log_files[-1].open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = record.get("coco/bbox_mAP")
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > best_map:
                best_map = val
                best_record = record

    return best_record


def write_best_metrics_log(
    out_dir: Path,
    metrics: dict[str, Any],
    checkpoint: Path | None,
    class_names: list[str],
) -> Path:
    """Write a YOLO-style best-epoch metrics summary to {out_dir}/best_metrics.txt."""
    shown_keys: set[str] = {k for k, _ in _COCO_SUMMARY_KEYS}
    col_w = 11

    def fmt(key: str) -> str:
        v = metrics.get(key)
        if v is None:
            return "-"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "-"
        return "-" if f < 0 else f"{f:.4f}"

    epoch = metrics.get("step", metrics.get("epoch", "?"))
    header = f"{'':>24}" + "".join(f"{lbl:>{col_w}}" for _, lbl in _COCO_SUMMARY_KEYS)
    overall = f"{'all':>24}" + "".join(f"{fmt(k):>{col_w}}" for k, _ in _COCO_SUMMARY_KEYS)

    lines: list[str] = [
        f"Results saved to {out_dir.parent}",
        "",
        f"  Best epoch : {epoch}",
        f"  Checkpoint : {checkpoint.name if checkpoint else 'N/A'}",
        f"  Timestamp  : {datetime.now().isoformat(timespec='seconds')}",
        "",
        header,
        overall,
    ]

    # Per-class AP rows (present when classwise=True in CocoMetric).
    # MMDetection 3.x logs per-category AP50-95 as "coco/{cls}_precision"
    # (misleading name from the COCO API — it is NOT true Precision).
    # We probe that key first, then fall back to alternative naming conventions.
    for cls_name in class_names:
        ap50_candidates = [
            f"coco/bbox_mAP_50/{cls_name}",
            f"coco/{cls_name}_ap50",
            f"coco/bbox_{cls_name}_ap50",
        ]
        # "coco/{cls}_precision" is the actual MMDet 3.x classwise AP50-95 key
        ap_candidates = [
            f"coco/{cls_name}_precision",
            f"coco/bbox_mAP/{cls_name}",
            f"coco/{cls_name}_ap",
            f"coco/bbox_{cls_name}_ap",
        ]
        val_50 = next((fmt(k) for k in ap50_candidates if k in metrics), None)
        val_ap = next((fmt(k) for k in ap_candidates if k in metrics), None)
        if val_50 is not None or val_ap is not None:
            cls_vals = [val_50 or "-", val_ap or "-"] + ["-"] * (len(_COCO_SUMMARY_KEYS) - 2)
            lines.append(f"{cls_name:>24}" + "".join(f"{v:>{col_w}}" for v in cls_vals))
            shown_keys.update(ap50_candidates + ap_candidates)

    # All remaining coco/* numeric metrics not already displayed.
    extra = {
        k: v for k, v in sorted(metrics.items())
        if k.startswith("coco/") and k not in shown_keys and isinstance(v, (int, float)) and float(v) >= 0
    }
    if extra:
        lines += ["", "Additional metrics:"]
        for k, v in extra.items():
            lines.append(f"  {k}: {float(v):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_metrics.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def build_test_command(
    config: RTMDetPipelineConfig,
    mmdet_config: Path,
    checkpoint: Path,
    split: str,
) -> list[str]:
    test_script = find_mmdet_tool(config, "test.py")
    eval_dir = Path(config.project_dir).resolve() / config.model_name / "eval" / split
    out_file = eval_dir / f"{split}_predictions.pkl"
    return [
        str(config.python_executable),
        str(test_script),
        str(mmdet_config),
        str(checkpoint),
        "--work-dir",
        str(eval_dir),
        "--out",
        str(out_file),
    ]


def first_sample_image(validation: DatasetValidationResult) -> Path:
    image_files = collect_image_files(validation.split_dirs["val"] / "images")
    if not image_files:
        image_files = collect_image_files(validation.split_dirs["train"] / "images")
    if not image_files:
        raise FileNotFoundError("No sample image available for export.")
    return image_files[0]


def resolve_deploy_config(config: RTMDetPipelineConfig) -> Path:
    if config.deploy_config:
        deploy_config = Path(config.deploy_config)
        if deploy_config.is_file():
            return deploy_config.resolve()

    if not config.mmdeploy_root:
        raise ValueError("Set mmdeploy_root or deploy_config for TensorRT export.")

    deploy_name = f"detection_tensorrt-fp16_static-{config.imgsz}x{config.imgsz}.py"
    deploy_path = (
        Path(config.mmdeploy_root).resolve()
        / "configs" / "mmdet" / "detection" / deploy_name
    )
    if deploy_path.is_file():
        return deploy_path

    fallback = (
        Path(config.mmdeploy_root).resolve()
        / "configs" / "mmdet" / "detection"
        / f"detection_tensorrt_static-{config.imgsz}x{config.imgsz}.py"
    )
    if fallback.is_file():
        print(f"WARNING: FP16 config not found, using fallback: {fallback}")
        return fallback

    raise FileNotFoundError(f"MMDeploy TensorRT config not found: {deploy_path}")


def build_export_command(
    config: RTMDetPipelineConfig,
    validation: DatasetValidationResult,
    mmdet_config: Path,
    checkpoint: Path,
) -> list[str]:
    if not config.mmdeploy_root:
        raise ValueError("Set config.mmdeploy_root to use MMDeploy.")

    deploy_script = Path(config.mmdeploy_root).resolve() / "tools" / "deploy.py"
    if not deploy_script.is_file():
        raise FileNotFoundError(f"MMDeploy tool not found: {deploy_script}")

    deploy_config = resolve_deploy_config(config)
    sample_image = (
        Path(config.sample_image).resolve() if config.sample_image
        else first_sample_image(validation)
    )
    export_dir = Path(config.project_dir).resolve() / config.model_name / "export"

    return [
        str(config.python_executable),
        str(deploy_script),
        str(deploy_config),
        str(mmdet_config),
        str(checkpoint),
        str(sample_image),
        "--work-dir", str(export_dir),
        "--device", config.device,
        "--dump-info",
    ]


def build_trtexec_benchmark_command(
    config: RTMDetPipelineConfig,
    engine_path: str | Path,
) -> list[str]:
    return [
        str(config.trtexec_path),
        f"--loadEngine={engine_path}",
        "--fp16",
        f"--iterations={int(config.benchmark_iterations)}",
        "--useCudaGraph",
        "--noDataTransfers",
    ]


def package_model_artifacts(
    config: RTMDetPipelineConfig,
    validation: DatasetValidationResult,
    mmdet_config: Path,
    checkpoint: Path | None,
    weights_only_checkpoint: Path | None = None,
) -> Path:
    package_root = Path(config.package_dir).resolve() / config.model_name
    package_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(mmdet_config, package_root / mmdet_config.name)

    classes_path = package_root / "classes.txt"
    with classes_path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(validation.class_names) + "\n")

    # Prefer the weights-only checkpoint: it's what a next fine-tune or a
    # deployment would load, and unlike the full training-state checkpoint it
    # doesn't hit PyTorch >= 2.6's weights_only=True default on load.
    checkpoint_to_package = weights_only_checkpoint or checkpoint
    if checkpoint_to_package and checkpoint_to_package.is_file():
        shutil.copy2(checkpoint_to_package, package_root / checkpoint_to_package.name)

    export_dir = Path(config.project_dir).resolve() / config.model_name / "export"
    for artifact_name in ("end2end.onnx", "end2end.engine", "pipeline.json", "deploy.json"):
        artifact = export_dir / artifact_name
        if artifact.is_file():
            shutil.copy2(artifact, package_root / artifact.name)

    metadata = {
        "model_name": config.model_name,
        "family": "RTMDet",
        "variant": config.variant,
        "image_size": config.imgsz,
        "precision_target": "TensorRT FP16",
        "classes": validation.class_names,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(validation.dataset_root),
        "mmdet_config": mmdet_config.name,
        "checkpoint": checkpoint_to_package.name if checkpoint_to_package else None,
    }
    with (package_root / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=True)

    return package_root


def write_run_manifest(
    config: RTMDetPipelineConfig,
    validation: DatasetValidationResult,
    coco_annotations: dict[str, Path],
    mmdet_config: Path,
) -> Path:
    run_dir = Path(config.project_dir).resolve() / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "dataset_root": str(validation.dataset_root),
        "classes": validation.class_names,
        "stats": {key: asdict(value) for key, value in validation.stats.items()},
        "warnings": validation.warnings,
        "coco_annotations": {key: str(value) for key, value in coco_annotations.items()},
        "mmdet_config": str(mmdet_config),
        "train_command": (
            build_train_command(config, mmdet_config)
            if discover_mmdet_root(config) else None
        ),
    }

    if config.mmdeploy_root and config.save_onnx_weights:
        manifest["deploy_config_hint"] = str(resolve_deploy_config(config))
    manifest["jetson_benchmark_hint"] = build_trtexec_benchmark_command(
        config, "end2end.engine"
    )

    manifest_path = run_dir / "rtmdet_pipeline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=True)
    return manifest_path


def run_rtmdet_pipeline(config: RTMDetPipelineConfig) -> dict[str, Path | None]:
    if config.variant.lower().strip() not in RTMDET_MODELS:
        raise ValueError(f"variant must be one of {sorted(RTMDET_MODELS)}.")
    if config.imgsz <= 0:
        raise ValueError("imgsz must be > 0.")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0.")

    dataset_root, _yaml_path = resolve_dataset_root(config.dataset_path)
    if config.normalize_names:
        maybe_normalize_dataset(dataset_root)

    print("\nFixing annotations...")
    fix_yolo_labels(dataset_root)

    validation = validate_yolo_dataset(config)
    print_validation_summary(validation)
    if validation.errors and config.stop_on_validation_errors:
        raise ValueError("Invalid dataset. Fix the errors above before training.")

    print("\nConverting YOLO TXT -> COCO JSON...")
    coco_annotations = convert_yolo_dataset_to_coco(validation)

    print("\nGenerating MMDetection config...")
    mmdet_config = generate_mmdet_config(config, validation, coco_annotations)
    print(f"RTMDet config: {mmdet_config}")

    manifest_path = write_run_manifest(config, validation, coco_annotations, mmdet_config)
    print(f"Run manifest: {manifest_path}")

    run_dir = Path(config.project_dir).resolve() / config.model_name
    checkpoint: Path | None = None
    package_root: Path | None = None

    if config.prepare_only:
        print("Preparation complete. Training/export not started.")
        if config.run_packaging:
            package_root = package_model_artifacts(config, validation, mmdet_config, checkpoint)
            print(f"Initial package: {package_root}")
        return {
            "dataset_root": validation.dataset_root,
            "mmdet_config": mmdet_config,
            "manifest": manifest_path,
            "package_root": package_root,
        }

    weights_only_checkpoint: Path | None = None
    if config.run_training:
        verify_training_environment(config)
        train_command = build_train_command(config, mmdet_config)
        run_command(
            train_command,
            cwd=Path(config.mmdet_root).resolve() if config.mmdet_root else None,
        )
        tidy_checkpoints_dir(run_dir)
        checkpoint = find_latest_checkpoint(run_dir)
        if checkpoint:
            print(f"Selected checkpoint: {checkpoint}")
            from .checkpoint_tools import extract_weights_only
            weights_only_checkpoint = extract_weights_only(checkpoint)
            print(f"Weights-only checkpoint: {weights_only_checkpoint}")
        else:
            print("WARNING: no checkpoint found after training.")

        best_metrics = parse_best_metrics_from_json_log(run_dir)
        if best_metrics:
            metrics_log = write_best_metrics_log(run_dir / "metrics", best_metrics, checkpoint, validation.class_names)
            print(f"Best metrics log: {metrics_log}")
        else:
            print("WARNING: could not parse best metrics from training log.")

    if config.checkpoint_for_export:
        checkpoint = Path(config.checkpoint_for_export).resolve()

    if config.run_evaluation and checkpoint:
        run_command(build_test_command(config, mmdet_config, checkpoint, "val"))
        if "test" in validation.split_dirs:
            run_command(build_test_command(config, mmdet_config, checkpoint, "test"))

    onnx_file: Path | None = None
    if config.save_onnx_weights and checkpoint:
        if not config.mmdeploy_root:
            print("WARNING: save_onnx_weights=True but mmdeploy_root is not set — skipping ONNX export.")
        else:
            from .export.onnx import run_onnx_export, validate_onnx as _validate_onnx

            sample_img = (
                Path(config.sample_image).resolve()
                if config.sample_image
                else first_sample_image(validation)
            )
            onnx_dir = Path(config.project_dir).resolve() / config.model_name / "export"
            mmdet_root_path = Path(config.mmdet_root).resolve() if config.mmdet_root else None

            onnx_file = run_onnx_export(
                checkpoint_file=checkpoint,
                mmdet_config=mmdet_config,
                sample_image=sample_img,
                output_dir=onnx_dir,
                imgsz=int(config.imgsz),
                mmdeploy_root=Path(config.mmdeploy_root).resolve(),
                python_exe=str(config.python_executable),
                device=config.device,
                score_threshold=config.onnx_score_threshold,
                iou_threshold=config.onnx_iou_threshold,
                keep_top_k=config.onnx_keep_top_k,
                mmdet_root=mmdet_root_path,
            )
            _validate_onnx(onnx_file, int(config.imgsz))
            print(f"ONNX export: {onnx_file}")

    if config.run_export and checkpoint:
        verify_tensorrt_backend(config)
        export_command = build_export_command(config, validation, mmdet_config, checkpoint)
        run_command(
            export_command,
            cwd=Path(config.mmdeploy_root).resolve() if config.mmdeploy_root else None,
        )

    # Generate plots (training curves, per-class mAP, confusion matrix)
    if config.generate_plots:
        try:
            from .plots import generate_plots
            _stage2 = max(1, min(int(config.stage2_epochs), int(config.epochs)))
            _switch = max(1, int(config.epochs) - _stage2)
            plot_files = generate_plots(
                run_dir=run_dir,
                class_names=validation.class_names,
                coco_val_ann=coco_annotations.get("val"),
                model_name=config.model_name,
                switch_epoch=_switch,
                conf_thr=config.eval_conf_threshold,
                iou_thr=config.eval_iou_threshold,
            )
            for p in plot_files:
                print(f"Plot: {p}")
        except Exception as exc:
            print(f"WARNING: plot generation failed: {exc}")

    if config.run_packaging:
        package_root = package_model_artifacts(
            config, validation, mmdet_config, checkpoint, weights_only_checkpoint
        )
        print(f"Model package: {package_root}")

    return {
        "dataset_root": validation.dataset_root,
        "mmdet_config": mmdet_config,
        "manifest": manifest_path,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "weights_only_checkpoint": weights_only_checkpoint,
        "onnx_file": onnx_file,
        "package_root": package_root,
    }
