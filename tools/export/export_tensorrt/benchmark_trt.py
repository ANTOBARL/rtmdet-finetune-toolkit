"""
TensorRT benchmark — reads settings from export_config.txt [benchmark] section.

Requirements: TensorRT SDK installed, trtexec in PATH (or set trtexec_path).

Usage:
  1. Edit export_config.txt — set engine_path and optionally trtexec_path.
  2. python benchmark_trt.py
"""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "export_config.txt"


def _load_config() -> configparser.ConfigParser:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read(str(CONFIG_FILE), encoding="utf-8")
    return parser


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def build_command(
    engine_path: Path,
    trtexec_path: str,
    iterations: int,
    warmup_ms: int,
    duration_seconds: int,
    use_cuda_graph: bool,
    measure_gpu_only: bool,
) -> list[str]:
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
    cfg = _load_config()

    engine_raw = cfg.get("benchmark", "engine_path", fallback="end2end.engine").strip()
    engine_path = Path(engine_raw) if engine_raw else Path("end2end.engine")
    if not engine_path.is_absolute():
        engine_path = (Path(__file__).resolve().parent / engine_path).resolve()

    trtexec_path = cfg.get("benchmark", "trtexec_path", fallback="trtexec").strip() or "trtexec"
    iterations = int(cfg.get("benchmark", "iterations", fallback="500"))
    warmup_ms = int(cfg.get("benchmark", "warmup_ms", fallback="1000"))
    duration_seconds = int(cfg.get("benchmark", "duration_seconds", fallback="60"))
    use_cuda_graph = _to_bool(cfg.get("benchmark", "use_cuda_graph", fallback="true"))
    measure_gpu_only = _to_bool(cfg.get("benchmark", "measure_gpu_only", fallback="true"))

    command = build_command(
        engine_path=engine_path,
        trtexec_path=trtexec_path,
        iterations=iterations,
        warmup_ms=warmup_ms,
        duration_seconds=duration_seconds,
        use_cuda_graph=use_cuda_graph,
        measure_gpu_only=measure_gpu_only,
    )

    print("Running TensorRT benchmark:")
    print(" ".join(str(p) for p in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
