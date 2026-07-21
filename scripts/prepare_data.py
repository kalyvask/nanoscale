"""Prepare a corpus into memmapped token bins.

Examples
--------
Smoke corpus (downloads TinyShakespeare, bytes tokenizer)::

    python scripts/prepare_data.py --dataset tinyshakespeare

Local corpus with a trained BPE tokenizer::

    python scripts/prepare_data.py --input data/corpus.jsonl --text-field text \
        --train-tokenizer --vocab-size 16384 --tokenizer-sample-mb 50

TinyShakespeare numbers are plumbing checks, never research findings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanoscale.data import prepare
from nanoscale.tokenizer import Tokenizer

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download_tinyshakespeare(dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    resp = requests.get(TINYSHAKESPEARE_URL, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return dest


def build_tokenizer(args, text: str) -> Tokenizer:
    if args.tokenizer_path:
        return Tokenizer.load(args.tokenizer_path)
    if args.train_tokenizer:
        max_bytes = int(args.tokenizer_sample_mb * 1_000_000) if args.tokenizer_sample_mb else None
        tok = Tokenizer.train(text, vocab_size=args.vocab_size, max_bytes=max_bytes)
        if args.save_tokenizer:
            tok.save(args.save_tokenizer)
            print(f"saved tokenizer -> {args.save_tokenizer} (vocab {tok.vocab_size})")
        return tok
    return Tokenizer.bytes_tokenizer()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="'tinyshakespeare' for the smoke corpus")
    ap.add_argument("--input", default=None, help="path to a local .txt or .jsonl corpus")
    ap.add_argument("--out", default=None, help="output dir (default: data/<name>)")
    ap.add_argument("--text-field", default="text", help="JSONL field to read")
    ap.add_argument("--delimiter", default="\n\n", help="TXT document delimiter")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-tokenizer", action="store_true")
    ap.add_argument("--tokenizer-path", default=None, help="load a frozen tokenizer")
    ap.add_argument("--save-tokenizer", default=None)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=50.0,
                    help="cap the tokenizer-training sample (0 = whole corpus)")
    args = ap.parse_args()

    if args.dataset == "tinyshakespeare":
        input_path = download_tinyshakespeare(Path("data/tinyshakespeare/input.txt"))
        name = "tinyshakespeare"
        # TinyShakespeare has one blank-line-separated stream; split on newlines-pairs
    elif args.input:
        input_path = Path(args.input)
        name = input_path.stem
    else:
        ap.error("provide --dataset tinyshakespeare or --input <path>")
        return

    text = Path(input_path).read_text(encoding="utf-8")
    tokenizer = build_tokenizer(args, text)
    out_dir = Path(args.out) if args.out else Path("data") / name

    meta = prepare(
        input_path,
        tokenizer,
        out_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        text_field=args.text_field,
        delimiter=args.delimiter,
        dataset_name=name,
    )
    print(f"prepared {name} -> {out_dir}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
