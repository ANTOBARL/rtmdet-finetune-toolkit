"""
Extract a weights-only checkpoint from a full MMEngine training checkpoint.

MMEngine checkpoints (epoch_*.pth, best_*.pth) bundle the model weights
together with optimizer state, LR scheduler state, and a message_hub used
for logging. That's needed to resume an interrupted run, but it's dead
weight — and a loading hazard — for anything else: using the checkpoint as
`pretrained_checkpoint` for a new fine-tune, packaging it, or deploying it.

PyTorch >= 2.6 defaults `torch.load` to `weights_only=True`, and the
message_hub inside a full MMEngine checkpoint carries a HistoryBuffer object
that isn't on PyTorch's default safe-unpickling allowlist — so loading a
full checkpoint anywhere other than a genuine `--resume` fails with
`_pickle.UnpicklingError`. This script strips it down to just the weights.

The training pipeline (finetune_rtmdet.py) already does this automatically
for the best checkpoint at the end of every run — use this script when you
want to do it manually for a specific file (e.g. a mid-run epoch_*.pth).

Usage:
    python tools/extract_weights.py <checkpoint.pth> [--out <output.pth>]

Example:
    python tools/extract_weights.py runs/rtmdet/my_run/checkpoints/epoch_120.pth
    # -> runs/rtmdet/my_run/checkpoints/epoch_120_weights_only.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train_rtmdet.checkpoint_tools import extract_weights_only


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip a full MMEngine checkpoint down to just its state_dict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a full MMEngine checkpoint (.pth)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: <checkpoint>_weights_only.pth next to the input)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    before_mb = args.checkpoint.stat().st_size / (1024 ** 2)
    out_path = extract_weights_only(args.checkpoint, args.out)
    after_mb = out_path.stat().st_size / (1024 ** 2)

    print(f"Input : {args.checkpoint}  ({before_mb:.1f} MB)")
    print(f"Output: {out_path}  ({after_mb:.1f} MB)")


if __name__ == "__main__":
    main()
