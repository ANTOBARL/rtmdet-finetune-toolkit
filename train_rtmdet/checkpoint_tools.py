"""
Checkpoint post-processing utilities.

MMEngine saves full training-state checkpoints (state_dict + optimizer +
param_schedulers + message_hub + meta) so a run can be resumed exactly where
it left off. For anything else — fine-tuning from that checkpoint,
packaging, deployment — only the model weights (state_dict) are needed.

This matters beyond file size: PyTorch >= 2.6 defaults `torch.load` to
`weights_only=True`, and MMEngine's `message_hub` carries a
`HistoryBuffer` object that isn't on PyTorch's default safe-unpickling
allowlist, so loading a full checkpoint as `pretrained_checkpoint` /
`load_from` fails with `_pickle.UnpicklingError` unless it's stripped down
first. See README.md, section "Checkpoint formats", for the full story.
"""

from __future__ import annotations

from pathlib import Path

import torch


def extract_weights_only(checkpoint_path: Path | str, output_path: Path | str | None = None) -> Path:
    """Strip a full MMEngine checkpoint down to just its state_dict.

    Reading the source checkpoint requires `weights_only=False` — safe here
    because this is meant to run on a checkpoint you trust (your own
    training output), which is exactly the case PyTorch's weights_only
    warning describes as fine to override.
    """
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_weights_only{checkpoint_path.suffix}"
        )
    output_path = Path(output_path)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, str(output_path))
    return output_path
