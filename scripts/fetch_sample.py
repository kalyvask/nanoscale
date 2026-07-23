"""Stream a bounded sample from a pinned corpus to a local JSONL file.

Used for the tokenizer study and for smoke-testing ingestion without pulling the whole
corpus. Writes outside the repo by default: prepared data is large and must not land in
a synced folder or in git.

    python scripts/fetch_sample.py --corpus configs/corpora/fineweb_edu.yaml \
        --max-mb 40 --out C:/Users/alexa/nanoscale-data/fineweb_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanoscale.corpora import CorpusSpec, iter_corpus, write_shard_manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="configs/corpora/fineweb_edu.yaml")
    ap.add_argument("--max-mb", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-every-mb", type=float, default=5.0)
    args = ap.parse_args(argv)

    spec = CorpusSpec.from_yaml(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    budget = int(args.max_mb * 1_000_000)
    report_every = int(args.report_every_mb * 1_000_000)

    print(f"corpus '{spec.name}' revision {spec.revision}")
    print(f"shards: {len(spec.ordered_shards)}  ->  writing up to {args.max_mb} MB to {out}")

    total = docs = 0
    next_report = report_every
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for doc in iter_corpus(spec):
            f.write(json.dumps({"text": doc}) + "\n")
            total += len(doc.encode("utf-8"))
            docs += 1
            if total >= next_report:
                rate = total / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {total/1e6:.1f} MB, {docs:,} docs, {rate:.2f} MB/s", flush=True)
                next_report += report_every
            if total >= budget:
                break

    write_shard_manifest(spec, out.parent)
    print(f"done: {docs:,} docs, {total/1e6:.1f} MB in {time.time()-t0:.0f}s")
    print(f"corpus manifest -> {out.parent / 'corpus_manifest.json'}")


if __name__ == "__main__":
    main()
