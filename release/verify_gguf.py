"""
Check a TOAM GGUF served by llama.cpp's llama-server against the HF export:
  1. tokenization: /tokenize (special tokens parsed) vs the HF tokenizer, on tool prompts;
  2. greedy decodes: /completion (temperature 0) vs transformers generate, same prompts;
  3. CPU latency per request as a by-product.

    llama-server -m release/toam-100m-q8_0.gguf -c 2048 --port 8087      (in another shell)
    python release/verify_gguf.py --hf ksg91/toam-100m-v0.2 --port 8087 --n 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from prompts import sample_prompts  # noqa: E402


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32).eval()

    rows = sample_prompts(args.n)
    # plus unusual text for the tokenizer: digits, unicode, emoji, code
    extra = ["Pay 1234.50 EUR to acct 000123 on 2026-10-01 at 9:05pm 🚀 naïve café — ok?",
             "def f(x):\n    return x**2  # 3 33 333\t tabs\n\n", "我想在apple工作1314151天 нещо на Български"]

    tok_ok = tok_n = 0
    for t in rows + extra:
        a = tok(t, add_special_tokens=False)["input_ids"]
        b = post(args.port, "/tokenize", {"content": t, "add_special": False, "parse_special": True})["tokens"]
        tok_n += 1
        tok_ok += a == b
        if a != b and tok_n - tok_ok <= 3:
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            print(f"  tokenize mismatch at {i}: hf {a[i:i+6]} gguf {b[i:i+6]} | {tok.decode(a[max(0,i-3):i+6])!r}")
    print(f"tokenization identical: {tok_ok}/{tok_n}", flush=True)

    same, lat, results = 0, [], []
    for i, prompt in enumerate(rows):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > 1800:
            continue
        with torch.no_grad():
            g = model.generate(torch.tensor([ids]), max_new_tokens=160, do_sample=False,
                               eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)[0, len(ids):].tolist()
        hf_text = tok.decode(g, skip_special_tokens=True).strip()
        t0 = time.time()
        res = post(args.port, "/completion", {"prompt": ids, "n_predict": 160, "temperature": 0, "top_k": 1,
                                              "cache_prompt": False})
        lat.append(time.time() - t0)
        gg_text = res["content"].strip()
        ok = hf_text == gg_text
        same += ok
        results.append({"i": i, "hf": hf_text, "gguf": gg_text, "same": ok})
        if not ok and len(results) - same <= 3:
            print(f"  decode differs:\n    hf   {hf_text[:160]}\n    gguf {gg_text[:160]}")
    n = len(results)
    print(f"greedy decodes identical: {same}/{n}; CPU latency per request p50 {statistics.median(lat)*1000:.0f} ms, "
          f"max {max(lat)*1000:.0f} ms", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"tokenization": [tok_ok, tok_n], "decodes": [same, n],
                                              "latency_p50_ms": statistics.median(lat) * 1000,
                                              "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
