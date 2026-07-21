# train-rtmdet

Fine-tuning and export pipeline for **RTMDet** object detection models on custom YOLO datasets.

RTMDet is a real-time single-stage detector from OpenMMLab. This pipeline covers the full workflow from a raw YOLO dataset to a trained model exported as ONNX, ready for deployment.

---

## RTMDet model variants

All variants use the **CSPNeXt backbone**, are pretrained on COCO, and are released under **Apache 2.0 (commercial use allowed)**.

| Variant | `variant` key | COCO box AP | Params | FLOPs | Latency RTX 3090 ¹ | Latency T4 ¹ |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| RTMDet-tiny | `tiny` | 41.1 | 4.8 M | 8.1 G | 0.98 ms | 2.34 ms |
| RTMDet-s    | `s`    | 44.6 | 8.9 M | 14.8 G | 1.22 ms | 2.96 ms |
| RTMDet-m    | `m`    | 49.4 | 24.7 M | 39.3 G | 1.62 ms | 6.41 ms |
| RTMDet-l    | `l`    | 51.5 | 52.3 M | 80.2 G | 2.44 ms | 10.32 ms |
| RTMDet-x    | `x`    | 52.8 | 94.9 M | 141.7 G | 3.10 ms | 18.80 ms |

¹ TensorRT 8.4.3 · cuDNN 8.2.0 · FP16 · batch size 1 · **NMS escluso** · input 640×640.  
Fonte: [OpenMMLab MMDetection — RTMDet model zoo](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)

To select a variant set `variant` in `hyperparameter_config.yaml` under `model`.

---

## Repository structure

```
train_RTM_det/
│
├── train_rtmdet/                       ← pip-installable Python package
│   ├── __init__.py
│   ├── pipeline.py                     ← core training pipeline logic
│   ├── normalize.py                    ← dataset filename normalization
│   ├── config_loader.py                ← reads hyperparameter_config.yaml
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
│   │   ├── export_config.yaml          ← standalone ONNX export config (edit this)
│   │   └── export_onnx.py              ← standalone ONNX export (post-training, no retraining)
│   └── export_tensorrt/
│       ├── export_config.yaml          ← IDLE (TensorRT export config)
│       ├── export_tensorrt.py          ← IDLE (TensorRT export, future work)
│       └── benchmark_trt.py            ← IDLE (TensorRT benchmark, future work)
│
├── hyperparameter_config.yaml          ← your local config (git-ignored, not committed)
├── hyperparameter_config.template.yaml ← template to copy and fill in
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

After cloning, set the corresponding paths in `hyperparameter_config.yaml`:

```yaml
paths:
  mmdet_root: 'C:\path\to\train_RTM_det\mmdetection'
  mmdeploy_root: 'C:\path\to\train_RTM_det\mmdeploy'
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

### Step 0 — Configure `hyperparameter_config.yaml`

Copy the template and fill in your paths:

```bash
cp hyperparameter_config.template.yaml hyperparameter_config.yaml
```

Open `hyperparameter_config.yaml` and fill in at minimum:

```yaml
dataset:
  dataset_path: 'C:\path\to\your\dataset'

paths:
  mmdet_root: 'C:\path\to\mmdetection'
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

This ensures MMDetection can reliably pair images with their labels. The script reads `dataset_path` from `dataset_workflow_config.yaml` and is safe to run multiple times — files that are already normalized are skipped.

---

### Step 1.5 — Analyze dataset before training

```bash
python analyze_dataset.py
```

This optional check helps you understand the dataset before training by reporting:

- class distribution for each split (`train` / `val` / `test`)
- annotated vs background image ratio
- object size distribution using COCO area bins: `small`, `medium`, `large`

The script saves two charts in the dataset folder:

- `class_distribution.png`
- `object_size_distribution.png`

COCO size bins are computed from the real bbox area in pixels:

- `small`: area `< $32^2$`
- `medium`: `$32^2 \leq$ area < $96^2$`
- `large`: area `$\geq 96^2$`

This is useful before training because class imbalance and object-scale imbalance often affect model selection, input resolution, augmentation strategy, and expected detection quality on small objects.

---

### Step 1.6 — Augment object-size distribution

```bash
python augment_by_size.py
```

This optional step reads `dataset_workflow_config.yaml` and creates a new dataset:

- `<dataset_name>_dimension_augmented`

You set `dataset_path` directly in `dataset_workflow_config.yaml`.

The tool works toward a target COCO size distribution (`small`, `medium`, `large`) by generating resized copies of source images and updating YOLO labels automatically.

Default target distribution:

- `small`: `35%`
- `medium`: `40%`
- `large`: `25%`

Behavior:

- by default it augments only `train`
- originals are preserved unless `preserve_originals: false`
- downscale uses resize + padding back to the original image size
- upscale, if enabled, uses resize + center-crop back to the original image size
- you can cap generated samples with both `max_new_images` and `max_new_images_percent`
- output is saved next to the input dataset, never in place

Recommended workflow:

1. Run `python analyze_dataset.py`
2. Run `python balance_dataset.py` if split balancing is needed
3. Run `python normalize_dataset.py`
4. Run `python analyze_dataset.py` again
5. Tune `dataset_workflow_config.yaml`
6. Run `python augment_by_size.py`
7. Run `python analyze_dataset.py` on the augmented dataset

The script also writes `dimension_augmentation_manifest.json` in the output dataset folder with the initial/final size distributions and generated samples.

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
5. If `save_onnx_weights: true`: export the best checkpoint to `end2end.onnx` via MMDeploy, then validate the ONNX graph with ONNXRuntime.
6. Package the best checkpoint, config, `classes.txt`, and ONNX file (if produced) into `models/rtmdet/<model_name>/`.

> **Quick test run** — set `prepare_only: true` to stop after step 3. This validates the dataset and generates the config without starting training, useful to check that everything is set up correctly before committing to a full run.

---

## Output structure

After a successful run:

```
runs/rtmdet/_configs/
└── <model_name>_rtmdet_s_<timestamp>.py        ← generated MMDetection config (one per launch)

runs/rtmdet/<model_name>/
├── checkpoints/                                ← .pth files only, nothing else
│   ├── best_coco_bbox_mAP_epoch_N.pth          ← lightweight: weights + meta only
│   ├── best_coco_bbox_mAP_epoch_N_weights_only.pth  ← auto-generated, see "Checkpoint formats" below
│   ├── epoch_N.pth, epoch_N-1.pth, ...          ← full training state (weights+optimizer+scheduler)
│   └── last_checkpoint                          ← marker file MMEngine uses for --resume
├── logs/<timestamp>/                            ← MMEngine's own housekeeping, one folder per launch
│   ├── vis_data/scalars.json                    ← raw metrics log (used by plots + best-metrics parsing)
│   └── <config>.py                              ← dumped copy of the config used for that launch
├── metrics/
│   ├── best_metrics.txt                         ← YOLO-style summary of the best epoch
│   └── plots/                                   ← present if generate_plots: true
│       ├── training_curves.png
│       ├── per_class_map50.png
│       └── confusion_matrix.png                 ← present if run_evaluation: true
├── export/                                      ← present if save_onnx_weights: true
│   ├── end2end.onnx
│   ├── pipeline.json
│   └── deploy.json
└── rtmdet_pipeline_manifest.json                ← full run metadata (paths, commands, stats)

models/rtmdet/<model_name>/                      ← packaged artifacts
├── <config>.py
├── best_coco_bbox_mAP_epoch_N_weights_only.pth  ← packaged checkpoint is always the weights-only one
├── classes.txt
├── end2end.onnx                                 ← present if save_onnx_weights: true
└── metadata.json
```

`checkpoints/` and `logs/` used to be the same folder (MMEngine's `--work-dir`), which mixed per-launch
metadata in with the actual `.pth` files. The pipeline now moves MMEngine's housekeeping into `logs/`
after training (`tidy_checkpoints_dir` in `train_rtmdet/pipeline.py`) so `checkpoints/` only ever holds
checkpoint files. Runs from before this change keep their old flat layout; nothing needs migrating.

---

## Checkpoint formats

MMEngine checkpoints come in two shapes, and the pipeline produces a third:

| File | Contents | Size | Use for |
|---|---|---|---|
| `epoch_N.pth` | weights + optimizer state + LR scheduler + message_hub + meta | full (largest) | genuine `resume: true` — continuing an interrupted run with optimizer state intact |
| `best_<metric>_epoch_N.pth` | weights + message_hub + meta (no optimizer/scheduler) | ~half of `epoch_N.pth` | evaluation, manual inspection |
| `best_<metric>_epoch_N_weights_only.pth` | weights only | smallest | `model.pretrained_checkpoint` for a new fine-tune, packaging, deployment |

The pipeline auto-generates the third file for the best checkpoint at the end of every training run
(`train_rtmdet/checkpoint_tools.py::extract_weights_only`) and packages that one, not the raw
`best_*.pth`. To do it manually for any other checkpoint (e.g. a specific `epoch_N.pth`):

```bash
python tools/extract_weights.py runs/rtmdet/<model_name>/checkpoints/epoch_120.pth
```

**Why this matters:** since PyTorch 2.6, `torch.load()` defaults to `weights_only=True`. MMEngine's
`message_hub` — present in both `epoch_*.pth` and `best_*.pth` — carries a `HistoryBuffer` object that
isn't on PyTorch's default safe-unpickling allowlist, so loading either of them as
`model.pretrained_checkpoint` fails with:

```
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL mmengine.logging.history_buffer.HistoryBuffer ...
```

The `*_weights_only.pth` file sidesteps this entirely — it only contains tensors, so it loads fine
under the new default. Extracting it requires `torch.load(..., weights_only=False)` once, which is safe
only because it's your own trusted checkpoint (never do this for a `.pth` from an untrusted source).

This only affects `load_from` / `pretrained_checkpoint` (weights-only init). A genuine `resume: true`
still needs the full `epoch_N.pth` with optimizer state — MMEngine's own `--resume` loading goes through
the same `weights_only=True` default internally, so on PyTorch >= 2.6 it can hit the same error. There's
no clean workaround for that case yet short of an MMEngine-side fix or explicitly allowlisting
`HistoryBuffer` via `torch.serialization.add_safe_globals(...)` before training starts.

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

**Segmentation annotations** — if your dataset uses YOLO-seg format (`class_id x1 y1 x2 y2 ...`), set `convert_segments_to_boxes: true` (default) to convert them to bounding boxes automatically.

**Images without labels** are treated as hard negatives when `allow_missing_labels: true`.

### Optional: convert a raw YOLO dataset to RTMDet format

If your dataset needs to be restructured (e.g. it uses `val/` instead of `valid/`, or is missing COCO JSON files), run the optional preprocessing tool first:

```bash
# edit dataset_path and output_path at the top of the file, then:
python tools/dataset/prepare_dataset.py
```

This creates a `<dataset_name>_rtmdet/` copy with images, labels, COCO JSON annotations, and `classes.txt` already in place. Point `dataset_path` in `hyperparameter_config.yaml` to the new folder.

---

## Configuration reference

All options live in `hyperparameter_config.yaml`. Inline comments explain each parameter; the table below is a quick reference.

### `dataset`

| Key | Default | Description |
|---|---|---|
| `dataset_path` | — | Absolute path to the dataset root. **Required.** |
| `class_names` | *(from data.yaml)* | List of class names override. |
| `nc` | *(from data.yaml)* | Number of classes override. |

### `model`

| Key | Default | Description |
|---|---|---|
| `model_name` | `my_model_rtmdet_s_640` | Name used for the run folder and output package. |
| `variant` | `s` | One of `tiny`, `s`, `m`, `l`, `x`. See model variants table above. |
| `imgsz` | `640` | Input size in pixels. Must be a multiple of 32. |
| `pretrained_checkpoint` | *(COCO official)* | Path to a custom `.pth` to start from instead of COCO weights. Use a `*_weights_only.pth` file if it comes from a prior run of this pipeline — see "Checkpoint formats" above. |

### `training`

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
| `amp` | `true` | Automatic Mixed Precision. Recommended on modern GPUs. |
| `resume` | `false` | Resume from the latest checkpoint in the run folder. |
| `logger_interval` | `100` | Print training log every N iterations. |

### `pipeline`

| Key | Default | Description |
|---|---|---|
| `prepare_only` | `false` | Stop after config generation — do not start training. |
| `run_training` | `true` | Run the training step. |
| `run_evaluation` | `false` | Run evaluation on val/test after training. |
| `save_onnx_weights` | `false` | Export best checkpoint to ONNX after training. Requires `mmdeploy_root`. |
| `run_packaging` | `true` | Copy artifacts to `models/rtmdet/<model_name>/`. |

### `onnx_export`

These parameters define the post-processing baked permanently into the ONNX graph. They cannot be changed at inference without re-exporting.

| Key | Default | Description |
|---|---|---|
| `score_threshold` | `0.05` | Detections below this confidence are discarded before NMS. |
| `iou_threshold` | `0.5` | IoU threshold for Non-Maximum Suppression. |
| `keep_top_k` | `300` | Maximum detections per image (output tensor size). |

### `paths`

| Key | Default | Description |
|---|---|---|
| `project_dir` | `<cwd>/runs/rtmdet` | Where training runs are saved. |
| `package_dir` | `<cwd>/models/rtmdet` | Where model packages are saved. |
| `mmdet_root` | — | Path to the MMDetection clone. **Required for training.** |
| `mmdeploy_root` | — | Path to the MMDeploy clone. Required when `save_onnx_weights: true`. |

### `preprocessing`

| Key | Default | Description |
|---|---|---|
| `convert_segments_to_boxes` | `true` | Convert YOLO-seg polygon rows to bounding boxes. |
| `allow_missing_labels` | `true` | Accept images without a `.txt` label (treated as negatives). |
| `stop_on_validation_errors` | `true` | Abort if the dataset has annotation errors. |

---

## Standalone ONNX export

Use `tools/export_onnx/export_onnx.py` when you want to export an already-trained
checkpoint to ONNX **without re-running the training pipeline** — for example to
change NMS thresholds, try a different `keep_top_k`, or export a checkpoint produced
by a previous run.

> The main `hyperparameter_config.yaml` is **not** read by this tool. All settings
> are configured exclusively in `tools/export_onnx/export_config.yaml`.

```bash
# 1. Edit tools/export_onnx/export_config.yaml
#    Set project_dir + model_name (or checkpoint_path directly).
# 2. Run:
python tools/export_onnx/export_onnx.py
```

The script auto-detects the MMDetection config and input size from the run manifest
(`rtmdet_pipeline_manifest.json`) in the checkpoint folder.

---

## TODO / Future work

### TensorRT export

TensorRT export is currently **IDLE**. The scripts in `tools/export_tensorrt/` contain
the implementation but are not active yet, because they require hardware-specific setup
(TensorRT SDK version, CUDA/cuDNN versions, custom ops build for MMDeploy). They will
be activated once the target hardware is defined.

When ready:
1. Edit `tools/export_tensorrt/export_config.yaml`.
2. Run `python tools/export_tensorrt/export_tensorrt.py`.
3. Benchmark on the target device: `python tools/export_tensorrt/benchmark_trt.py`.

### Class-weight support in the training loss

RTMDet uses `QualityFocalLoss` as its classification loss.
The current MMDetection implementation does **not** expose a `class_weight` parameter,
so per-class loss weighting is not yet supported in this pipeline.

The recommended workaround is dataset redistribution via `balance_dataset.py`, which
rebalances the train/val/test splits without discarding any images. This alone is
often sufficient for moderate imbalance.

Once a future MMDetection release adds `class_weight` support to `QualityFocalLoss`
(or the custom config generator is extended to inject it manually), the
`training.class_weights` key should be re-added to `hyperparameter_config.yaml` and
wired into the loss config generated by `train_rtmdet/pipeline.py`.

The inverse-frequency weight formula used by `balance_dataset.py` is already
implemented in `train_rtmdet/balancer.py::compute_class_weights()` and can be reused:

```
w_i = total / (n_classes × count_i),  normalized so min(w) = 1.0
```
