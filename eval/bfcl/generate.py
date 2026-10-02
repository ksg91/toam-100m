"""
Stage 2 of the BFCL harness (run in the main environment): generate raw outputs.

Two model kinds:

  toam   A TOAM model from Hugging Face (or a local folder). Each BFCL entry is
         rendered with `toam.inference.format_prompt`, the prompt the model was
         fine-tuned on: tools from the entry, the system/user messages as the
         goal, no policy, first-step state. Greedy decoding.
  hf     Any Hugging Face causal LM with a BFCL handler, here FunctionGemma. Uses
         the prompt that BFCL's own FunctionGemmaHandler rendered in prepare.py.
         Greedy.

TOAM is decoded the way the published numbers were produced: FP32 weights,
BF16 autocast on CUDA, one full forward pass per generated token (no KV cache).
On CPU it runs in FP32.

Also records per-call latency and token counts, and per-run peak VRAM and peak
process RAM.

    python eval/bfcl/generate.py --kind toam --model ksg91/toam-100m-v0.2 --name toam_100m_v0.2
    python eval/bfcl/generate.py --kind hf --hf-model google/functiongemma-270m-it \\
        --name functiongemma_270m --max-new 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from toam.inference import format_prompt  # noqa: E402

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "prepared"
RESULTS = HERE / "results"


def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MB (Windows working set / Unix maxrss)."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        # Explicit argtypes/restype: without them the 64-bit pseudo-handle is
        # truncated and the call silently fails.
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
        fn.restype = ctypes.wintypes.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return c.PeakWorkingSetSize / 2**20
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KB on Linux


def goal_of(entry: dict) -> str:
    return "\n".join(m["content"] for m in entry["messages"]
                     if m["role"] in ("system", "user") and m.get("content"))


class ToamRunner:
    def __init__(self, model_id: str, device, max_new: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        self.max_seq_len = self.model.config.max_position_embeddings
        self.device, self.max_new = device, max_new
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, entry: dict) -> dict:
        prompt = format_prompt(entry["function"], goal_of(entry))
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        n_in = len(ids)
        if n_in + 16 > self.max_seq_len:
            # Cannot even start an answer inside the context window.
            return {"raw": "", "overlength": True, "input_tokens": n_in, "output_tokens": 0}
        out = list(ids)
        eos = self.tok.eos_token_id
        for _ in range(self.max_new):
            if len(out) >= self.max_seq_len:
                break
            x = torch.tensor(out, dtype=torch.long, device=self.device)[None, :]
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                nxt = int(self.model(input_ids=x, use_cache=False).logits[0, -1].argmax().item())
            if nxt == eos:
                break
            out.append(nxt)
        return {"raw": self.tok.decode(out[n_in:], skip_special_tokens=True).strip(),
                "overlength": False, "input_tokens": n_in, "output_tokens": len(out) - n_in}


class HFRunner:
    def __init__(self, model_id: str, device, max_new: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.device, self.max_new = device, max_new
        self.params = sum(p.numel() for p in self.model.parameters())
        # Stop exactly where vLLM (BFCL's backend) stops: the model's own
        # generation_config eos list. For FunctionGemma that is
        # [<eos>, <start_function_response>, <end_of_turn>]; the model emits
        # <start_function_response> when it hands control back for tool output.
        eos = self.model.generation_config.eos_token_id
        self.stop = list(eos) if isinstance(eos, (list, tuple)) else [eos]

    @torch.inference_mode()
    def __call__(self, entry: dict) -> dict:
        # The vLLM completions endpoint BFCL uses adds BOS; add_special_tokens does the same.
        enc = self.tok(entry["functiongemma_prompt"], return_tensors="pt").to(self.device)
        n_in = enc["input_ids"].shape[1]
        out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                  top_k=None, top_p=None,
                                  eos_token_id=self.stop, pad_token_id=self.tok.pad_token_id)
        gen = out[0, n_in:]
        # Keep FunctionGemma's control tokens: its BFCL parser needs <start_function_call>.
        text = self.tok.decode(gen, skip_special_tokens=False)
        for t in ("<end_of_turn>", "<start_function_response>", "<eos>", "<pad>"):
            text = text.replace(t, "")
        return {"raw": text.strip(), "overlength": False, "input_tokens": n_in,
                "output_tokens": int(gen.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["toam", "hf"], required=True)
    ap.add_argument("--name", required=True, help="results/<name>/")
    ap.add_argument("--model", default="ksg91/toam-100m-v0.2", help="TOAM model id or local folder")
    ap.add_argument("--hf-model", default="google/functiongemma-270m-it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--limit", type=int, default=0, help="per-category cap, for smoke tests only")
    args = ap.parse_args()

    if not PREPARED.exists():
        sys.exit(f"{PREPARED} not found: run eval/bfcl/prepare.py first (in the BFCL environment).")
    torch.manual_seed(0)
    device = torch.device(args.device)
    runner = (ToamRunner(args.model, device, args.max_new) if args.kind == "toam"
              else HFRunner(args.hf_model, device, args.max_new))
    out_dir = RESULTS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(p.stem for p in PREPARED.glob("*.jsonl")) if args.categories == "all" \
        else args.categories.split(",")

    for cat in cats:
        dst = out_dir / f"{cat}.jsonl"
        if dst.exists() and not args.limit:
            print(f"{cat:24} exists, skipping", flush=True)
            continue
        entries = [json.loads(l) for l in (PREPARED / f"{cat}.jsonl").read_text(encoding="utf-8").splitlines()]
        if args.limit:
            entries = entries[: args.limit]  # smoke only; never used for reported numbers
        tmp = dst.with_suffix(".partial")
        t0 = time.time()
        with tmp.open("w", encoding="utf-8") as f:
            for e in entries:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                s = time.perf_counter()
                r = runner(e)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                r["latency_s"] = time.perf_counter() - s
                f.write(json.dumps({"id": e["id"], **r}, ensure_ascii=False) + "\n")
        tmp.replace(dst)
        print(f"{cat:24} n={len(entries):5}  {time.time()-t0:7.1f}s", flush=True)

    meta = {
        "name": args.name, "kind": args.kind,
        "model": args.model if args.kind == "toam" else None,
        "hf_model": args.hf_model if args.kind == "hf" else None,
        "parameters": runner.params, "device": str(device),
        "decoding": "greedy", "max_new_tokens": args.max_new,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "peak_rss_mb": peak_rss_mb(),
        "kv_cache": args.kind == "hf",
        "limit": args.limit,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
