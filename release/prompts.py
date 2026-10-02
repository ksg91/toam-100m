"""
Prompts for the release equivalence checks (export_hf, make_half, verify_gguf).

A fixed random sample of BFCL prompts rendered in TOAM's format, if
eval/bfcl/prepare.py has been run; otherwise the hand-written examples in
toam/examples.py. The checks compare two versions of the same model on these
prompts, so any realistic tool prompts will do.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toam.examples import REQUESTS, TOOLS  # noqa: E402
from toam.inference import format_prompt  # noqa: E402

BFCL_PREPARED = ROOT / "eval" / "bfcl" / "prepared"


def sample_prompts(n: int, seed: int = 0) -> list[str]:
    files = sorted(BFCL_PREPARED.glob("*.jsonl"))
    if not files:
        print("note: eval/bfcl/prepared not found, using the hand-written examples", flush=True)
        return [format_prompt(TOOLS, r, policy=p, state=s) for r, p, s in REQUESTS][:n]
    entries = []
    for p in files:
        entries += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = random.Random(seed).sample(entries, min(n, len(entries)))
    return [format_prompt(e["function"], "\n".join(m["content"] for m in e["messages"]
                                                   if m["role"] in ("system", "user") and m.get("content")))
            for e in entries]
