"""M1.5 Tokenizer study.

Compare a bytes-only tokenizer against byte-level BPE at several vocabulary sizes on
one corpus, and pick the tokenizer to freeze for the architecture experiments.

Reported per tokenizer:
  * compression      bytes / token   (higher = denser)
  * fertility        tokens / word   (lower = denser)
  * utilization      fraction of the vocabulary used on held-out text
  * train_time       seconds to learn the merges (bytes tokenizer: 0)
  * encode_speed     bytes / second when encoding held-out text
  * qualitative      how sample strings segment into tokens
  * bits_per_byte    equal-budget model quality (with --with-model)

bits-per-byte is the deciding metric: it is tokenizer-independent, so models trained on
different tokenizers compare fairly. "Equal budget" here means the same model
architecture trained for the same number of tokens; note that a higher-compression
tokenizer therefore sees more underlying text per token (a real advantage, not a bug).

On TinyShakespeare/CPU this is plumbing. The real decision is made on FineWeb-Edu.

    python scripts/tokenizer_study.py --dataset tinyshakespeare --with-model --model-steps 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root; make the
# `scripts` package importable either way.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanoscale.config import Config
from nanoscale.data import prepare, read_documents
from nanoscale.tokenizer import Tokenizer
from nanoscale.train import train

SAMPLE_STRINGS = [
    "To be, or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "internationalization and tokenization",
]


# ---------------------------------------------------------------------- #
# building tokenizers
# ---------------------------------------------------------------------- #
def build_tokenizers(train_text: str, vocab_sizes: list[int], sample_bytes: int | None):
    """Return an ordered dict label -> (tokenizer, train_time_seconds)."""
    out: dict[str, tuple[Tokenizer, float]] = {"bytes": (Tokenizer.bytes_tokenizer(), 0.0)}
    for v in vocab_sizes:
        t0 = time.perf_counter()
        tok = Tokenizer.train(train_text, vocab_size=v, max_bytes=sample_bytes)
        out[f"bpe_{v}"] = (tok, time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------- #
# intrinsic metrics
# ---------------------------------------------------------------------- #
def intrinsic_metrics(tok: Tokenizer, eval_text: str, reps: int = 1) -> dict:
    st = tok.stats(eval_text)
    n_bytes = st["n_bytes"]
    t0 = time.perf_counter()
    for _ in range(reps):
        tok.encode_ordinary(eval_text)
    encode_time = (time.perf_counter() - t0) / reps
    return {
        "vocab_size": tok.vocab_size,
        "compression": st["compression"],
        "fertility": st["fertility"],
        "utilization": tok.utilization(eval_text),
        "encode_bytes_per_sec": n_bytes / max(encode_time, 1e-9),
        "n_tokens_eval": st["n_tokens"],
    }


def segmentations(tok: Tokenizer, samples: list[str]) -> dict[str, str]:
    return {s: "|".join(tok.segment(s)) for s in samples}


# ---------------------------------------------------------------------- #
# equal-budget model quality
# ---------------------------------------------------------------------- #
def model_bits_per_byte(tok: Tokenizer, docs_path: Path, tmp_dir: Path,
                        steps: int, base_dir: Path) -> dict:
    data_dir = tmp_dir / f"data_v{tok.vocab_size}"
    prepare(docs_path, tok, data_dir, val_frac=0.1, seed=1)
    cfg = Config(
        name=f"tokstudy_v{tok.vocab_size}", group="tokenizer_study",
        vocab_size=tok.vocab_size, block_size=128, n_layer=3, n_head=4, n_embd=128,
        batch_size=16, max_steps=steps, eval_interval=max(1, steps // 3),
        eval_iters=20, warmup_frac=0.1, lr=1e-3, z_loss=1e-4,
        attention_backend="sdpa", dataset="tokstudy", device="cpu",
    )
    summary = train(cfg, base_dir=base_dir, data_dir=data_dir)
    return {
        "bits_per_byte": summary["bits_per_byte"],
        "final_val_loss": summary["final_val_loss"],
        "tokens_per_sec": summary["tokens_per_sec"],
    }


# ---------------------------------------------------------------------- #
# corpus loading
# ---------------------------------------------------------------------- #
def load_corpus_path(args) -> tuple[Path, str]:
    if args.dataset == "tinyshakespeare":
        from scripts.prepare_data import download_tinyshakespeare

        path = download_tinyshakespeare(Path("data/tinyshakespeare/input.txt"))
        return path, "tinyshakespeare"
    if args.input:
        return Path(args.input), Path(args.input).stem
    raise SystemExit("provide --dataset tinyshakespeare or --input <path>")


def split_train_eval(text: str, sample_bytes: int, eval_bytes: int) -> tuple[str, str]:
    """Disjoint tokenizer-training sample (head) and held-out eval slice (tail)."""
    b = text.encode("utf-8")
    train_b = b[:sample_bytes]
    eval_b = b[-eval_bytes:] if len(b) > sample_bytes + eval_bytes else b[sample_bytes:]
    return train_b.decode("utf-8", errors="ignore"), eval_b.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------- #
# report
# ---------------------------------------------------------------------- #
def render_report(rows: list[dict], segs: dict[str, dict], is_smoke: bool,
                  with_model: bool) -> str:
    lines = []
    bar = "=" * 92
    lines.append(bar)
    title = "TOKENIZER STUDY"
    if is_smoke:
        title += " -- SMOKE-TEST PLUMBING (TinyShakespeare, CPU), NOT A FINDING"
    lines.append(title)
    lines.append(bar)
    hdr = f"{'tokenizer':10} {'vocab':>7} {'compress':>9} {'fertility':>10} {'util%':>7} {'train_s':>8} {'enc_MB/s':>9}"
    if with_model:
        hdr += f" {'bits/byte':>10} {'tok/s':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        line = (f"{r['label']:10} {r['vocab_size']:>7,} {r['compression']:>9.3f} "
                f"{r['fertility']:>10.3f} {r['utilization'] * 100:>6.1f} "
                f"{r['train_time']:>8.2f} {r['encode_bytes_per_sec'] / 1e6:>9.2f}")
        if with_model:
            bpb = r.get("bits_per_byte")
            tps = r.get("tokens_per_sec")
            line += f" {bpb:>10.4f}" if bpb is not None else f" {'n/a':>10}"
            line += f" {tps:>8,.0f}" if tps is not None else f" {'n/a':>8}"
        lines.append(line)
    lines.append(bar)
    lines.append("Qualitative segmentation (piece boundaries shown with '|'):")
    for label, seg_map in segs.items():
        lines.append(f"\n[{label}]")
        for s, seg in seg_map.items():
            lines.append(f"  {s!r}")
            lines.append(f"    -> {seg}")
    lines.append(bar)
    if with_model:
        lines.append("Decision metric: lowest equal-budget bits/byte. "
                     "Compression/fertility/util are informative, not decisive.")
    else:
        lines.append("Run with --with-model to add the deciding equal-budget bits/byte.")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="M1.5 tokenizer study.")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--vocab-sizes", default="1024,4096,8192,16384")
    ap.add_argument("--sample-mb", type=float, default=5.0,
                    help="cap the tokenizer-training sample (MB)")
    ap.add_argument("--eval-mb", type=float, default=1.0, help="held-out eval slice (MB)")
    ap.add_argument("--with-model", action="store_true",
                    help="also train an equal-budget model per tokenizer (bits/byte)")
    ap.add_argument("--model-steps", type=int, default=60)
    ap.add_argument("--out", default="analysis/tokenizer_study")
    ap.add_argument("--freeze", default=None,
                    help="save the winning tokenizer to this path (e.g. data/tokenizer.json)")
    ap.add_argument("--base-dir", default="experiments")
    args = ap.parse_args(argv)

    corpus_path, name = load_corpus_path(args)
    text = corpus_path.read_text(encoding="utf-8")
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",") if v]
    sample_bytes = int(args.sample_mb * 1_000_000) if args.sample_mb else None
    eval_bytes = int(args.eval_mb * 1_000_000)
    train_text, eval_text = split_train_eval(text, sample_bytes or len(text.encode()), eval_bytes)

    print(f"Corpus '{name}': {len(text.encode()):,} bytes; "
          f"tokenizer sample {len(train_text.encode()):,} B, eval {len(eval_text.encode()):,} B")

    tokenizers = build_tokenizers(train_text, vocab_sizes, sample_bytes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmpdata"
    rows, segs = [], {}
    for label, (tok, train_time) in tokenizers.items():
        m = intrinsic_metrics(tok, eval_text)
        row = {"label": label, "train_time": train_time, **m}
        if args.with_model:
            print(f"  training equal-budget model for {label} (vocab {tok.vocab_size}) ...")
            row.update(model_bits_per_byte(tok, corpus_path, tmp_dir,
                                           args.model_steps, Path(args.base_dir)))
        rows.append(row)
        segs[label] = segmentations(tok, SAMPLE_STRINGS)

    is_smoke = name == "tinyshakespeare"
    report = render_report(rows, segs, is_smoke, args.with_model)
    print("\n" + report)

    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {out_dir / 'report.txt'} and results.json")

    if args.with_model:
        winner = min(rows, key=lambda r: r["bits_per_byte"])
        print(f"Winner by bits/byte: {winner['label']} ({winner['bits_per_byte']:.4f})")
        if args.freeze:
            tokenizers[winner["label"]][0].save(args.freeze)
            print(f"Froze {winner['label']} -> {args.freeze}")
    elif args.freeze:
        print("Refusing to freeze without --with-model (no deciding metric).")


if __name__ == "__main__":
    main()
