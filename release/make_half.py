"""
Cast an exported TOAM HF folder (fp32) to fp16 or bf16 and check that it decodes the same.

Verification: greedy decodes of the half-precision model vs the fp32 model on a fixed random
sample of tool prompts (release/prompts.py, same sample as export_hf.py), on CPU so the GPU
stays free. The folder is only written with --out; without it this is a dry check.

    python release/make_half.py --src release/toam-hf --dtype float16 --n 100 --out release/toam-hf-fp16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts import sample_prompts  # noqa: E402

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@torch.no_grad()
def greedy(model, tok, ids, max_new=160):
    x = torch.tensor(ids)[None]
    out = model.generate(x, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.pad_token_id)[0, len(ids):].tolist()
    return out[:-1] if out and out[-1] == tok.eos_token_id else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dtype", choices=list(DTYPES), default="float16")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.src)
    ref = LlamaForCausalLM.from_pretrained(args.src, dtype=torch.float32).eval()
    half = LlamaForCausalLM.from_pretrained(args.src, dtype=DTYPES[args.dtype]).eval()

    lim = torch.finfo(DTYPES[args.dtype]).max
    wmax = max(p.abs().max().item() for p in ref.parameters())
    print(f"max |weight| {wmax:.3f} ({args.dtype} max {lim:.0f})", flush=True)

    rows = sample_prompts(args.n)
    same, checked, diffs, t0 = 0, 0, [], time.time()
    for i, prompt in enumerate(rows):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > 1800:
            continue
        a, b = greedy(ref, tok, ids), greedy(half, tok, ids)
        checked += 1
        if a == b:
            same += 1
        else:
            diffs.append({"i": i, "fp32": tok.decode(a), args.dtype: tok.decode(b)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}  identical {same}/{checked}  {time.time()-t0:.0f}s", flush=True)
    print(f"verify {args.dtype}: identical greedy decodes {same}/{checked}", flush=True)
    for d in diffs:
        print(json.dumps(d, ensure_ascii=False), flush=True)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        half.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        for f in ("generation_config.json",):
            if (Path(args.src) / f).exists():
                shutil.copy2(Path(args.src) / f, out / f)
        (out / f"verify_{args.dtype}.json").write_text(json.dumps(
            {"src": args.src, "dtype": args.dtype, "device": "cpu", "prompts": checked,
             "identical_greedy_vs_fp32": same, "diffs": diffs}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
