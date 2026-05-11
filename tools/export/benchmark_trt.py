# =============================================================================
# IDLE — TensorRT benchmark requires a compiled .engine file.
#
# Run this script on the target device (e.g. Jetson) after copying the
# model package produced by finetune_rtmdet.py + export_tensorrt.py.
#
# Requirements: TensorRT SDK installed, trtexec in PATH.
# =============================================================================

from __future__ import annotations

import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATION — edit these variables before running.
# ============================================================

engine_path = Path("end2end.engine")
trtexec_path = "trtexec"

iterations = 500
warmup_ms = 1000
duration_seconds = 60
use_cuda_graph = True
measure_gpu_only = True


# ============================================================
# Implementation
# ============================================================

def build_command() -> list[str]:
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    command = [
        trtexec_path,
        f"--loadEngine={engine_path}",
        "--fp16",
        f"--iterations={iterations}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_seconds}",
    ]

    if use_cuda_graph:
        command.append("--useCudaGraph")
    if measure_gpu_only:
        command.append("--noDataTransfers")

    return command


def main() -> None:
    command = build_command()
    print("Running TensorRT benchmark:")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
