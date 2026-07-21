"""Sample text from a trained checkpoint.

    python scripts/generate.py --run experiments/runs/<run_id> --prompt "ROMEO:"
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from nanoscale.config import Config
from nanoscale.eval import sample_text
from nanoscale.model import GPT
from nanoscale.tokenizer import Tokenizer


def latest_run_with_checkpoint(base="experiments/runs") -> str:
    runs = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime, reverse=True)
    for r in runs:
        if os.path.exists(os.path.join(r, "checkpoint.pt")):
            return r
    raise FileNotFoundError("no run directory with a checkpoint.pt found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: latest with a checkpoint)")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--tokenizer", default=None, help="tokenizer json (default: bytes)")
    args = ap.parse_args()

    run_dir = args.run or latest_run_with_checkpoint()
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.load(args.tokenizer) if args.tokenizer else Tokenizer.bytes_tokenizer()

    text = sample_text(model, tok, args.prompt, args.tokens, "cpu")
    print(f"# sample from {run_dir}\n")
    print(text)


if __name__ == "__main__":
    main()
