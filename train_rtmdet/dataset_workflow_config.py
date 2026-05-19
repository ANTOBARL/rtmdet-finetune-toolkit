from __future__ import annotations

from pathlib import Path

import yaml


PREFERRED_CONFIG_NAME = "dataset_workflow_config.yaml"


def find_dataset_workflow_config(start_dir: Path | None = None) -> Path | None:
    root = start_dir or Path(__file__).resolve().parent.parent
    preferred = root / PREFERRED_CONFIG_NAME
    if preferred.is_file():
        return preferred
    return None


def load_dataset_workflow_raw(config_path: Path | None = None) -> dict:
    path = config_path or find_dataset_workflow_config()
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Config file '{PREFERRED_CONFIG_NAME}' not found in the project root."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_workflow_dataset_path(config_path: Path | None = None) -> Path | None:
    raw = load_dataset_workflow_raw(config_path)
    section = raw.get("dimension_augmenter", {})
    dataset_path = section.get("dataset_path")
    if dataset_path is None or str(dataset_path).strip() == "":
        return None
    return Path(str(dataset_path)).expanduser()
