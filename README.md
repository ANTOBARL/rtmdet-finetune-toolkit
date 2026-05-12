# train-rtmdet

Fine-tuning and export pipeline for **RTMDet** object detection models on custom YOLO datasets.

RTMDet is a real-time single-stage detector from OpenMMLab. This pipeline covers the full workflow from a raw YOLO dataset to a trained model exported as ONNX, ready for deployment.

---

## Repository structure

```
train_RTM_det/
│
├── train_rtmdet/                       ← pip-installable Python package
│   ├── __init__.py
│   ├── pipeline.py                     ← core training pipeline logic
│   ├── normalize.py                    ← dataset filename normalization
│   ├── config_loader.py                ← reads iperparameter_config.txt
│   └── export/
│       ├── __init__.py
│       ├── onnx.py                     ← ONNX export functions
│       └── tensorrt.py                 ← IDLE (TensorRT, future work)
│
├── normalize_dataset.py                ← Step 1: normalize dataset filenames
├── finetune_rtmdet.py                  ← Step 2: train + optional ONNX export
│
├── tools/
│   ├── dataset/
│   │   └── prepare_dataset.py          ← optional: convert raw YOLO → RTMDet format
│   ├── export_onnx/
│   │   ├── export_config.txt           ← standalone ONNX export config (edit this)
│   │   └── export_onnx.py              ← standalone ONNX export (post-training)
│   └── export_tensorrt/
│       ├── export_config.txt           ← IDLE (TensorRT export config)
│       ├── export_tensorrt.py          ← IDLE (TensorRT export, future work)
│       └── benchmark_trt.py            ← IDLE (TensorRT benchmark, future work)
│
├── iperparameter_config.txt            ← all hyperparameters and paths (edit this)
├── pyproject.toml
└── README.md
```

---

## Installation

### Prerequisites — clone the OpenMMLab repos

These two repositories must be cloned manually. They are **not** on PyPI.

```bash
# from the repo root
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmdeploy.git   # only needed for ONNX export
```

After cloning, set the corresponding paths in `iperparameter_config.txt`:

```ini
[paths]
mmdet_root    = C:\path\to\train_RTM_det\mmdetection
mmdeploy_root = C:\path\to\train_RTM_det\mmdeploy
```

### Python environment

**Tested configuration:** Python 3.11 · CUDA 11.8 · PyTorch 2.1 · Windows (RTX 4090)

> **Why CUDA 11.8 and not 12.x?**
> OpenMMLab does not publish mmcv binary wheels for Windows on CUDA 12.x, so pip
> would fall back to building from source (requires a full MSVC + CUDA toolchain).
> CUDA 11.8 has pre-built Windows wheels and the RTX 4090 runs it without any
> performance penalty.

> **Why setuptools 68.2.2?**
> PyTorch 2.1 imports `pkg_resources` from setuptools internally. That module was
> removed in setuptools 70+, so anything newer breaks the mmdetection install.
> `requirements.txt` pins this version automatically.

**Step 1 — install all pip dependencies** (torch GPU + mmcv + ONNX tools + rest):

```bash
pip install -r requirements.txt
```

`requirements.txt` handles everything: PyTorch CUDA index, mmcv Windows wheel from
the OpenMMLab CDN, and the setuptools pin — no manual steps required.

**Step 2 — install MMDetection** from the local clone:

```bash
pip install -e mmdetection/ --no-build-isolation
```

`--no-build-isolation` is required because mmdetection's `setup.py` imports `torch`
at build time, which is only available in the current environment, not in pip's
default isolated build sandbox.

**Step 3 — install this package** in editable mode:

```bash
pip install -e .
```

---

## Workflow

The normal workflow has three steps.

### Step 0 — Configure `iperparameter_config.txt`

Open `iperparameter_config.txt` and fill in at minimum:

```ini
[dataset]
dataset_path = C:\path\to\your\dataset

[paths]
mmdet_root = C:\path\to\mmdetection
```

Everything else has a working default. See the [Configuration reference](#configuration-reference) section for a full description of all options.

---

### Step 1 — Normalize dataset filenames

```bash
python normalize_dataset.py
```

Renames all images and labels across train/valid/test splits to a canonical format:

```
train_1.jpg / train_1.txt
val_1.jpg   / val_1.txt
...
```

This ensures MMDetection can reliably pair images with their labels. The script reads `dataset_path` from `iperparameter_config.txt` and is safe to run multiple times — files that are already normalized are skipped.

---

### Step 2 — Train (and optionally export to ONNX)

```bash
python finetune_rtmdet.py
```

Internally the pipeline runs these steps in order:

1. Validate the dataset structure and all annotations.
2. Convert YOLO `.txt` labels to COCO JSON format (`annotations/instances_train.json`, etc.).
3. Generate a MMDetection training config (saved to `runs/rtmdet/_configs/`).
4. Launch `mmdetection/tools/train.py` with the generated config.
5. If `save_onnx_weights = True`: export the best checkpoint to `end2end.onnx` via MMDeploy, then validate the ONNX graph with ONNXRuntime.
6. Package the best checkpoint, config, `classes.txt`, and ONNX file (if produced) into `models/rtmdet/<model_name>/`.

> **Quick test run** — set `prepare_only = True` to stop after step 3. This validates the dataset and generates the config without starting training, useful to check that everything is set up correctly before committing to a full run.

---

## Output structure

After a successful run:

```
runs/rtmdet/<model_name>/
├── _configs/
│   └── <model_name>_rtmdet_s_<timestamp>.py   ← generated MMDetection config
├── best_coco_bbox_mAP_epoch_N.pth              ← best checkpoint
├── latest.pth
├── export_onnx/                                ← present if save_onnx_weights=True
│   ├── end2end.onnx
│   ├── pipeline.json
│   └── deploy.json
└── rtmdet_pipeline_manifest.json               ← full run metadata (paths, commands, stats)

models/rtmdet/<model_name>/                     ← packaged artifacts
├── <config>.py
├── best_coco_bbox_mAP_epoch_N.pth
├── classes.txt
├── end2end.onnx                                ← present if save_onnx_weights=True
└── metadata.json
```

---

## Dataset format

The pipeline expects a YOLO-format dataset:

```
dataset_root/
├── train/
│   ├── images/   ← .jpg / .png / .bmp / .tif / .webp
│   └── labels/   ← .txt  one file per image, YOLO normalized format
├── valid/        ← or val/
│   ├── images/
│   └── labels/
├── test/         ← optional
│   ├── images/
│   └── labels/
└── data.yaml     ← must contain a `names` list
```

**YOLO bounding box format** — each `.txt` line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to `[0, 1]` relative to image dimensions.

**Segmentation annotations** — if your dataset uses YOLO-seg format (`class_id x1 y1 x2 y2 ...`), set `convert_segments_to_boxes = True` (default) to convert them to bounding boxes automatically.

**Images without labels** are treated as hard negatives when `allow_missing_labels = True`.

### Optional: convert a raw YOLO dataset to RTMDet format

If your dataset needs to be restructured (e.g. it uses `val/` instead of `valid/`, or is missing COCO JSON files), run the optional preprocessing tool first:

```bash
# edit dataset_path and output_path at the top of the file, then:
python tools/dataset/prepare_dataset.py
```

This creates a `<dataset_name>_rtmdet/` copy with images, labels, COCO JSON annotations, and `classes.txt` already in place. Point `dataset_path` in `iperparameter_config.txt` to the new folder.

---

## Configuration reference

All options live in `iperparameter_config.txt`. Inline comments explain each parameter; the table below is a quick reference.

### `[dataset]`

| Key | Default | Description |
|---|---|---|
| `dataset_path` | — | Absolute path to the dataset root. **Required.** |
| `class_names` | *(from data.yaml)* | Comma-separated class names override. |
| `nc` | *(from data.yaml)* | Number of classes override. |

### `[model]`

| Key | Default | Description |
|---|---|---|
| `model_name` | `my_model_rtmdet_s_640` | Name used for the run folder and output package. |
| `variant` | `s` | `s` (small, faster) or `m` (medium, more accurate). |
| `imgsz` | `640` | Input size in pixels. Must be a multiple of 32. |
| `pretrained_checkpoint` | *(COCO official)* | Path to a custom `.pth` to start from instead of COCO weights. |

### `[training]`

| Key | Default | Description |
|---|---|---|
| `epochs` | `200` | Total training epochs. |
| `stage2_epochs` | `20` | Final epochs without mosaic/mixup (fine-tuning phase). |
| `batch_size` | `32` | Training batch size. |
| `val_batch_size` | `8` | Validation batch size. |
| `workers` | `8` | DataLoader worker processes. |
| `val_interval` | `5` | Run validation every N epochs. |
| `base_lr` | `0.001` | Peak learning rate (AdamW). |
| `device` | `cuda:0` | Training device (`cuda:0`, `cuda:1`, `cpu`). |
| `amp` | `True` | Automatic Mixed Precision. Recommended on modern GPUs. |
| `resume` | `False` | Resume from the latest checkpoint in the run folder. |
| `logger_interval` | `100` | Print training log every N iterations. |

### `[pipeline]`

| Key | Default | Description |
|---|---|---|
| `prepare_only` | `False` | Stop after config generation — do not start training. |
| `run_training` | `True` | Run the training step. |
| `run_evaluation` | `False` | Run evaluation on val/test after training. |
| `save_onnx_weights` | `False` | Export best checkpoint to ONNX after training. Requires `mmdeploy_root`. |
| `run_packaging` | `True` | Copy artifacts to `models/rtmdet/<model_name>/`. |

### `[onnx_export]`

These parameters define the post-processing baked permanently into the ONNX graph. They cannot be changed at inference without re-exporting.

| Key | Default | Description |
|---|---|---|
| `score_threshold` | `0.05` | Detections below this confidence are discarded before NMS. |
| `iou_threshold` | `0.5` | IoU threshold for Non-Maximum Suppression. |
| `keep_top_k` | `300` | Maximum detections per image (output tensor size). |

### `[paths]`

| Key | Default | Description |
|---|---|---|
| `project_dir` | `<cwd>/runs/rtmdet` | Where training runs are saved. |
| `package_dir` | `<cwd>/models/rtmdet` | Where model packages are saved. |
| `mmdet_root` | — | Path to the MMDetection clone. **Required for training.** |
| `mmdeploy_root` | — | Path to the MMDeploy clone. Required when `save_onnx_weights = True`. |

### `[preprocessing]`

| Key | Default | Description |
|---|---|---|
| `convert_segments_to_boxes` | `True` | Convert YOLO-seg polygon rows to bounding boxes. |
| `allow_missing_labels` | `True` | Accept images without a `.txt` label (treated as negatives). |
| `stop_on_validation_errors` | `True` | Abort if the dataset has annotation errors. |

---

## Standalone ONNX export

To export a checkpoint without re-running training (e.g. to change thresholds):

```bash
# 1. Edit tools/export_onnx/export_config.txt
#    Set project_dir + model_name (or checkpoint_path directly).
# 2. Run:
python tools/export_onnx/export_onnx.py
```

The script auto-detects the MMDetection config and input size from the run manifest (`rtmdet_pipeline_manifest.json`) in the checkpoint folder. All settings (score threshold, IoU threshold, output size, device, paths) are configured exclusively in `tools/export_onnx/export_config.txt` — the main `iperparameter_config.txt` is not read.

---

## TensorRT export (future work)

TensorRT export is currently **IDLE**. The scripts in `tools/export/export_tensorrt/` contain the implementation but are not active yet, because they require hardware-specific setup (TensorRT SDK version, CUDA/cuDNN versions, custom ops build for MMDeploy). They will be activated once the target hardware is defined.

When ready:
1. Edit `tools/export_tensorrt/export_config.txt`.
2. Run `python tools/export_tensorrt/export_tensorrt.py`.
3. Benchmark on the target device: `python tools/export_tensorrt/benchmark_trt.py`.
