"""
Train TOAM's 32k byte-level BPE tokenizer.

Design choices:
  - sampled in the intended pretraining proportions, not uniformly
  - the agent prompt markers are added as single special tokens, so the prompt
    structure costs ~8 tokens per example instead of ~40
  - digits are split individually: IDs, amounts and timestamps are arguments
    the model must copy exactly, and merged number tokens make that harder

The released tokenizer was trained this way on ~2 GB of text from the
pretraining corpus plus real tool schemas, with documents mentioning held-out
evaluation tools removed before sampling. That filter and the schema files are
not part of this repository, so a re-run produces a similar tokenizer, not a
byte-identical one. Use the released tokenizer (in the Hugging Face model repo)
to work with the released weights.

    python pretrain/train_tokenizer.py --corpus corpus/pretrain_raw --out tokenizer_32k
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

# Slice -> share of sampled characters. Mirrors the pretraining mixture.
MIX = {
    "web_text": 0.42,
    "code": 0.22,
    "edu_prose": 0.16,
    "schema_text": 0.12,
    "encyclopedic": 0.05,
    "docstring_pairs": 0.03,
}

SPECIALS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    # agent prompt structure (see toam/inference.py)
    "<|system|>", "<|tools|>", "<|policy|>", "<|goal|>",
    "<|history|>", "<|state|>", "<|observation|>", "<|action|>",
]

def iter_texts(path: Path):
    """Yield document text from a parquet or json.gz shard."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        cols = set(pf.schema_arrow.names)
        col = next((c for c in ("text", "content", "code", "whole_func_string", "markdown") if c in cols), None)
        if col is None:
            return
        for batch in pf.iter_batches(batch_size=1000, columns=[col]):
            for value in batch.column(0).to_pylist():
                if value:
                    yield str(value)
    elif path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                value = row.get("content") or row.get("text")
                if value:
                    yield str(value)


def shards_for(root: Path, slice_name: str) -> list[Path]:
    base = root / slice_name
    if not base.exists():
        return []
    out = [p for p in base.rglob("*.parquet")]
    out += [p for p in base.rglob("*.json.gz")]
    return sorted(out)


def sample_texts(root: Path, budget_chars: int, seed: int, extra_jsonl: list[Path]):
    """Yield documents in the intended mixture."""
    rng = random.Random(seed)
    stats = {"kept": 0, "chars": 0}

    for slice_name, share in MIX.items():
        shards = shards_for(root, slice_name)
        if not shards:
            print(f"  [{slice_name}] no shards found, skipping", flush=True)
            continue
        rng.shuffle(shards)
        target = int(budget_chars * share)
        got = 0
        for shard in shards:
            if got >= target:
                break
            for text in iter_texts(shard):
                if got >= target:
                    break
                if len(text) < 200:
                    continue
                # Long documents would dominate the merges; take a window.
                if len(text) > 20000:
                    start = rng.randrange(0, len(text) - 20000)
                    text = text[start:start + 20000]
                got += len(text)
                stats["kept"] += 1
                stats["chars"] += len(text)
                yield text
        print(f"  [{slice_name}] {got/1e6:.0f}M chars from {len(shards)} shards", flush=True)

    # Optional JSON-lines files of tool schemas: small in volume, decisive in shape.
    for path in extra_jsonl:
        n = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                blob = json.dumps(row, ensure_ascii=False)
                n += 1
                stats["kept"] += 1
                stats["chars"] += len(blob)
                yield blob
        print(f"  [{path.name}] {n:,} records", flush=True)

    print(f"  sampled {stats['chars']/1e9:.2f}B chars, {stats['kept']:,} docs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/pretrain_raw",
                    help="folder with one sub-folder per slice in MIX (*.parquet or *.json.gz)")
    ap.add_argument("--out", default="tokenizer_32k")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--budget-gb", type=float, default=2.0,
                    help="characters of training text to sample, in GB")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--extra-jsonl", nargs="*", default=[],
                    help="optional JSON-lines files (e.g. tool schemas) added after the mixture")
    args = ap.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # Split numbers into digits so IDs/amounts stay copyable.
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Sampling ~{args.budget_gb:.1f}GB of text in mixture {MIX}", flush=True)
    started = time.time()
    tokenizer.train_from_iterator(
        sample_texts(Path(args.corpus), int(args.budget_gb * 1e9), args.seed,
                     [Path(p) for p in args.extra_jsonl]),
        trainer=trainer,
    )
    train_min = (time.time() - started) / 60

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 4096,
    }, indent=2), encoding="utf-8")
    (out / "training_meta.json").write_text(json.dumps({
        "vocab_size": tokenizer.get_vocab_size(),
        "mixture": MIX,
        "budget_gb": args.budget_gb,
        "specials": SPECIALS,
        "minutes": round(train_min, 1),
        "pre_tokenizer": "Digits(individual)+ByteLevel",
    }, indent=2), encoding="utf-8")

    print(f"\nTrained in {train_min:.1f} min. vocab={tokenizer.get_vocab_size():,}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
