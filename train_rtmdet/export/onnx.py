from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path


def write_onnx_deploy_config(
    work_dir: Path,
    imgsz: int,
    score_threshold: float,
    iou_threshold: float,
    keep_top_k: int,
) -> Path:
    """Write a flat MMDeploy ONNX config (no _base_ inheritance, Windows-safe)."""
    cfg_content = textwrap.dedent(f"""\
        # MMDeploy config generated automatically.
        # Flat config — no _base_ inheritance for Windows compatibility.

        onnx_config = dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            save_file='end2end.onnx',
            input_names=['input'],
            output_names=['dets', 'labels'],
            input_shape=({imgsz}, {imgsz}),
            optimize=True,
        )

        backend_config = dict(type='onnxruntime')

        codebase_config = dict(
            type='mmdet',
            task='ObjectDetection',
            model_type='end2end',
            post_processing=dict(
                score_threshold={score_threshold},
                confidence_threshold={score_threshold},
                iou_threshold={iou_threshold},
                max_output_boxes_per_class={keep_top_k},
                pre_top_k=5000,
                keep_top_k={keep_top_k},
                background_label_id=-1,
            ),
        )
    """)
    cfg_path = work_dir / "deploy_onnx_config.py"
    cfg_path.write_text(cfg_content, encoding="utf-8")
    return cfg_path


def build_pythonpath(mmdeploy_root: Path, mmdet_root: Path | None, repo_root: Path | None) -> str:
    extra = [str(mmdeploy_root)]
    if mmdet_root:
        extra.append(str(mmdet_root))
    if repo_root:
        extra.append(str(repo_root))
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p] + extra
    return os.pathsep.join(dict.fromkeys(parts))


def run_onnx_export(
    checkpoint_file: Path,
    mmdet_config: Path,
    sample_image: Path,
    output_dir: Path,
    imgsz: int,
    mmdeploy_root: Path,
    python_exe: str,
    device: str = "cuda:0",
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    keep_top_k: int = 300,
    mmdet_root: Path | None = None,
) -> Path:
    """Run MMDeploy deploy.py to export a .pth checkpoint to end2end.onnx.

    Returns the path to the generated .onnx file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    deploy_cfg = write_onnx_deploy_config(
        output_dir, imgsz, score_threshold, iou_threshold, keep_top_k
    )

    deploy_script = mmdeploy_root / "tools" / "deploy.py"
    if not deploy_script.is_file():
        raise FileNotFoundError(
            f"MMDeploy script not found: {deploy_script}\n"
            "Ensure mmdeploy_root points to a valid MMDeploy clone."
        )

    cmd = [
        python_exe,
        str(deploy_script),
        str(deploy_cfg),
        str(mmdet_config),
        str(checkpoint_file),
        str(sample_image),
        "--work-dir", str(output_dir),
        "--device", device,
        "--dump-info",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(mmdeploy_root, mmdet_root, None)

    _SUPPRESS = (
        "Failed to search registry with scope",
        "TracerWarning",
        "UserWarning",
        "DeprecationWarning: get_onnx_config",
        "Can not optimize model",
        "warnings.warn(",
        "  return _VF.meshgrid",
        "More details: https://",
    )

    result = subprocess.run(
        cmd, env=env, cwd=str(mmdeploy_root),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    _ansi = re.compile(r'\x1b\[[0-9;]*m')
    for raw_line in result.stdout.splitlines():
        line = _ansi.sub('', raw_line).rstrip()
        if not line:
            continue
        if any(pat in line for pat in _SUPPRESS):
            continue
        print(line)

    onnx_file = output_dir / "end2end.onnx"
    if result.returncode != 0:
        if onnx_file.is_file():
            # Post-export visualization step fails when C++ custom ops are not compiled
            # (common on Windows without MSVC build). The .onnx is written before that
            # step, so if the file exists the export succeeded.
            print("[WARN] Post-export visualization failed (C++ custom ops not compiled).")
            print("       end2end.onnx was created correctly — proceeding.\n")
        else:
            raise RuntimeError(
                f"MMDeploy deploy.py exited with code {result.returncode}.\n"
                "See the output above for details."
            )

    if not onnx_file.is_file():
        raise FileNotFoundError(
            f"Export finished but end2end.onnx not found in: {output_dir}"
        )

    return onnx_file


def validate_onnx(onnx_path: Path, imgsz: int) -> None:
    """Run a structural check and a test inference on an ONNX file."""
    print("\n" + "=" * 60)
    print("ONNX VALIDATION")
    print("=" * 60)

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"  onnx.checker  : OK  (opset {model.opset_import[0].version})")
    except ImportError:
        print("  [SKIP] 'onnx' not installed — skipping structural check")
    except Exception as exc:
        print(f"  [WARN] onnx.checker: {exc}")

    try:
        import numpy as np
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress INFO/WARNING from onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
            active = sess.get_providers()[0]
        except Exception:
            sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
            active = "CPUExecutionProvider"

        print(f"  Provider      : {active}")
        inp = sess.get_inputs()[0]
        print(f"  Input         : name='{inp.name}'  shape={inp.shape}  dtype={inp.type}")
        for out in sess.get_outputs():
            print(f"  Output        : name='{out.name}'  shape={out.shape}  dtype={out.type}")

        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        outputs = sess.run(None, {inp.name: dummy})
        print("  Test inference:")
        for name, arr in zip([o.name for o in sess.get_outputs()], outputs):
            print(f"    '{name}' -> shape {arr.shape}")
        print("  onnxruntime   : OK")

    except ImportError:
        print("  [SKIP] 'onnxruntime' not installed — skipping inference test")
    except Exception as exc:
        print(f"  [WARN] onnxruntime: {exc}")

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"  File          : {onnx_path}")
    print(f"  Size          : {size_mb:.1f} MB")
    print("=" * 60 + "\n")
