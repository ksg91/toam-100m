"""
Language-model pretraining on the packed, decontaminated corpus.

Built to run in sessions across several days:

  --max-hours N   soft limit. Checked only between optimizer steps: the current
                  step finishes, validation runs, a checkpoint is written, and
                  the process exits. Overshoot is one step plus the save.
  Ctrl+C          first press = the same graceful stop; second press = abort.
  STOP file       creating <out>/STOP requests a graceful stop from anywhere.
  --resume        continues from <out>/last.pt. Refuses to continue if the
                  training settings differ from the ones the run started with.
  --decay-now N   ends the run early: starts the final LR decay from the
                  current step, lasting N steps (WSD makes this legitimate).

Checkpoints are written atomically (temp file + rename), so a crash during a
save cannot corrupt the only copy.

Data: corpus/pretrain_tokens/<slice>/*.bin — flat arrays of uint16 token ids
(tokenized with the TOAM tokenizer, documents separated by <|eos|>), one folder
per slice named in MIXTURE. The last shard of every slice is held out for
validation and never trained on. The corpus download, decontamination and
packing scripts are not part of this repository; any corpus written in this
format works.

TOAM-100M v0.2 was trained with (run in ~7-hour sessions on one RTX 5080):

    python pretrain/pretrain.py --config wide --out runs/pretrain_wide_100m --tokens 10e9 \\
        --micro-batch 8 --grad-accum 16 --compile --schedule wsd --warmup-frac 0.02 \\
        --decay-frac 0.2 --lr 1e-3 --val-every 500 --val-windows 48 --save-every 500 \\
        --milestones 1e9,2e9,5e9,10e9 --max-hours 7

and the same command plus --resume for every later session. (Early sessions ran
uncompiled as --micro-batch 4 --grad-accum 32: the same 128 sequences per step
in the same order.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toam.model import AgentLM

MIXTURE = {
    "web_text": 0.42,
    "code": 0.22,
    "edu_prose": 0.16,
    "schema_text": 0.12,
    "encyclopedic": 0.05,
    "docstring_pairs": 0.03,
}

CONFIGS = {
    "deep": dict(d_model=576, n_layers=24, n_heads=9, n_kv_heads=3, mlp_mult=2.7),
    "wide": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, mlp_mult=2.7),
}

# Settings that define the training trajectory. Changing any of them between
# sessions would silently train a different run, so resume refuses.
# micro_batch and grad_accum are not listed: only their product (sequences per
# optimizer step) defines the trajectory. The sampler draws windows one at a
# time, so 4x32 and 8x16 see the same windows in the same order; the split is a
# memory/speed knob and may change between sessions (checked via BATCH_SEQS).
TRAJECTORY_ARGS = [
    "config", "tokens", "seq_len", "lr",
    "min_lr_ratio", "schedule", "warmup", "warmup_frac", "decay_frac",
    "weight_decay", "seed", "tokenizer_vocab",
]


def batch_seqs(a: dict) -> int:
    return a["micro_batch"] * a["grad_accum"]


class ShardSampler:
    """Random windows across memory-mapped shards, weighted by mixture."""

    def __init__(self, root: Path, seq_len: int, seed: int, split: str):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.slices: dict[str, list[tuple[np.memmap, int]]] = {}
        for name in MIXTURE:
            files = sorted((root / name).glob("*.bin"))
            if len(files) < 2:
                continue
            chosen = files[:-1] if split == "train" else files[-1:]
            maps = []
            for f in chosen:
                mm = np.memmap(f, dtype=np.uint16, mode="r")
                if mm.size > seq_len + 1:
                    maps.append((mm, mm.size))
            if maps:
                self.slices[name] = maps
        self.names = list(self.slices)
        weights = np.array([MIXTURE[n] for n in self.names], dtype=np.float64)
        self.weights = weights / weights.sum()

    def window(self, slice_name: str | None = None) -> np.ndarray:
        name = slice_name or self.names[self.rng.choice(len(self.names), p=self.weights)]
        maps = self.slices[name]
        sizes = np.array([s for _, s in maps], dtype=np.float64)
        mm, size = maps[self.rng.choice(len(maps), p=sizes / sizes.sum())]
        start = int(self.rng.integers(0, size - self.seq_len - 1))
        return np.asarray(mm[start:start + self.seq_len + 1], dtype=np.int64)

    def batch(self, n: int) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.window() for _ in range(n)]))


def lr_cosine(step, total, peak, warmup, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(progress, 1.0)))


def lr_wsd(step, total, peak, warmup, floor, decay_start):
    """Warmup -> constant -> linear decay. Depends only on the step number."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    progress = (step - decay_start) / max(1, total - decay_start)
    return peak - (peak - floor) * min(progress, 1.0)


@torch.no_grad()
def validate(model, val_sets, device, batch):
    model.eval()
    out = {}
    for name, windows in val_sets.items():
        losses = []
        for i in range(0, windows.size(0), batch):
            x = windows[i:i + batch].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   x[:, 1:].reshape(-1))
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    present = [k for k in out if k in MIXTURE]
    out["mixture"] = float(sum(MIXTURE[k] * out[k] for k in present)
                           / sum(MIXTURE[k] for k in present))
    model.train()
    return out


@torch.no_grad()
def benchmark_latency(model, device, vocab, prompt_len=1222):
    """Prefill latency for a typical agent prompt (1,222 tokens)."""
    model.eval()
    x = torch.randint(0, vocab, (1, prompt_len), device=device)
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    res = {"gpu_prefill_ms": round((time.time() - t) / 20 * 1000, 2)}
    cpu_model = AgentLM(**model.config).eval()
    cpu_model.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
    xc = x.cpu()
    cpu_model(xc[:, :64])
    t = time.time()
    for _ in range(3):
        cpu_model(xc)
    res["cpu_prefill_ms"] = round((time.time() - t) / 3 * 1000, 1)
    res["cpu_threads"] = torch.get_num_threads()
    res["prompt_tokens"] = prompt_len
    model.train()
    return res


class StopRequest:
    """Graceful-stop flag set by Ctrl+C, a STOP file, or the time limit."""

    def __init__(self, stop_file: Path):
        self.stop_file = stop_file
        self.reason: str | None = None
        self._presses = 0
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        self._presses += 1
        if self._presses >= 2:
            print("\n[abort] second Ctrl+C: exiting WITHOUT saving", flush=True)
            raise KeyboardInterrupt
        self.reason = "ctrl-c"
        print("\n[stop requested] finishing current step, then saving "
              "(press Ctrl+C again to abort without saving)", flush=True)

    def check(self, started: float, max_hours: float | None) -> bool:
        if self.reason:
            return True
        if self.stop_file.exists():
            self.reason = "stop-file"
            return True
        if max_hours and (time.time() - started) / 3600 >= max_hours:
            self.reason = f"max-hours ({max_hours:g}h)"
            return True
        return False


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--data", default="corpus/pretrain_tokens")
    ap.add_argument("--tokenizer-vocab", type=int, default=32768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=200e6, help="total training tokens")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="if >0, warmup = frac * total steps (overrides --warmup)")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="WSD: final decay share")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-windows", type=int, default=48, help="per slice")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--milestones", default="", help="token counts, e.g. 2e9,5e9,10e9")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0.0, help="soft session limit")
    ap.add_argument("--decay-now", type=int, default=0,
                    help="WSD: start the final decay now, lasting N steps")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (needs triton)")
    ap.add_argument("--vram-cap-gb", type=float, default=12.0,
                    help="hard cap on PyTorch's VRAM; exceeding it raises OOM instead "
                         "of Windows silently paging to system RAM (0 = off)")
    ap.add_argument("--bench-steps", type=int, default=0,
                    help="benchmark only: run N steps, report speed/VRAM, save nothing")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    if args.vram_cap_gb > 0:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total_gb), 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    if stop_file.exists():
        stop_file.unlink()          # a stale STOP from last session must not end this one

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = int(args.tokens // tokens_per_step)
    warmup = int(total_steps * args.warmup_frac) if args.warmup_frac > 0 else args.warmup
    decay_start = int(total_steps * (1 - args.decay_frac))
    floor = args.lr * args.min_lr_ratio
    milestones = sorted(float(x) for x in args.milestones.split(",") if x.strip())

    cfg = dict(CONFIGS[args.config], vocab_size=args.tokenizer_vocab,
               max_seq_len=args.seq_len, scaled_residual_init=True)
    model = AgentLM(**cfg).to(device)
    n_params = model.num_parameters

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    step = 0
    history: list[dict] = []
    sessions: list[dict] = []
    milestones_done: list[float] = []
    ckpt_path = out / "last.pt"

    if args.resume:
        if not ckpt_path.exists():
            sys.exit(f"--resume given but no checkpoint at {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck["args"]
        diffs = [f"{k}: checkpoint={saved.get(k)!r} now={getattr(args, k)!r}"
                 for k in TRAJECTORY_ARGS if saved.get(k) != getattr(args, k)]
        if batch_seqs(saved) != batch_seqs(vars(args)):
            diffs.append(f"micro_batch*grad_accum: checkpoint={batch_seqs(saved)} "
                         f"now={batch_seqs(vars(args))}")
        if diffs:
            sys.exit("Refusing to resume: training settings differ from the run's start.\n  "
                     + "\n  ".join(diffs))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        history = ck.get("history", [])
        sessions = ck.get("sessions", [])
        milestones_done = ck.get("milestones_done", [])
        # An early-decay decision made in an earlier session persists.
        total_steps = ck.get("total_steps", total_steps)
        decay_start = ck.get("decay_start", decay_start)
        print(f"Resumed at step {step:,}/{total_steps:,} "
              f"({step*tokens_per_step/1e9:.2f}B tokens)", flush=True)
    elif ckpt_path.exists() and not args.bench_steps:
        sys.exit(f"{ckpt_path} exists. Use --resume to continue it, or choose a new --out.")

    if args.decay_now:
        if args.schedule != "wsd":
            sys.exit("--decay-now requires --schedule wsd")
        decay_start = step
        total_steps = step + args.decay_now
        print(f"Early decay: linear to floor over steps {decay_start:,}->{total_steps:,}",
              flush=True)

    root = Path(args.data)
    # Seeded by step so a resumed session does not replay the same windows.
    train = ShardSampler(root, args.seq_len, args.seed + step, "train")
    val = ShardSampler(root, args.seq_len, args.seed + 10_000_019, "val")
    val_sets = {
        name: torch.from_numpy(np.stack([val.window(name) for _ in range(args.val_windows)]))
        for name in val.names
    }

    def lr_at(s):
        if args.schedule == "wsd":
            return lr_wsd(s, total_steps, args.lr, warmup, floor, decay_start)
        return lr_cosine(s, total_steps, args.lr, warmup, floor)

    def checkpoint(path: Path):
        atomic_save({
            "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
            "config": model.config, "args": vars(args), "history": history,
            "sessions": sessions, "milestones_done": milestones_done,
            "total_steps": total_steps, "decay_start": decay_start,
        }, path)

    print(f"config={args.config} params={n_params/1e6:.1f}M tokens/step={tokens_per_step:,} "
          f"steps={total_steps:,} ({total_steps*tokens_per_step/1e9:.2f}B tokens) "
          f"schedule={args.schedule} warmup={warmup} decay_start={decay_start:,} "
          f"peak_lr={args.lr:g}", flush=True)
    if args.max_hours:
        print(f"Session limit: {args.max_hours:g}h (soft - finishes the step, "
              f"validates and saves)", flush=True)

    def train_loss(x):
        # Loss inside the (optionally compiled) function so the cross-entropy
        # over the 32k vocab is fused too. Validation keeps the eager model.
        # The loss is taken in fp32: on bf16 logits it was bf16-rounded (before
        # 2026-10-01, step 17,229). Under compile the upcast fuses into the loss
        # kernel; eager it costs one fp32 logits tensor.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               x[:, 1:].reshape(-1))

    if args.compile:
        train_loss = torch.compile(train_loss)
    print(f"micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
          f"compile={args.compile} vram_cap={args.vram_cap_gb:g}GB", flush=True)

    stopper = StopRequest(stop_file)
    session_start = time.time()
    session_start_step = step
    last_log = session_start
    model.train()

    if args.bench_steps:
        # Warm-up steps (compilation, allocator) are excluded from the timing.
        # Weights are updated in memory only; nothing is written to disk.
        warm = 3 if args.compile else 1
        for i in range(warm + args.bench_steps):
            if i == warm:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            for _ in range(args.grad_accum):
                x = train.batch(args.micro_batch).to(device, non_blocking=True)
                (train_loss(x) / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"BENCH micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
              f"compile={args.compile}: {args.bench_steps * tokens_per_step / dt / 1e3:.1f}k tok/s, "
              f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB allocated / "
              f"{torch.cuda.max_memory_reserved()/1e9:.2f}GB reserved", flush=True)
        return

    try:
        while step < total_steps:
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr

            # Accumulated on the GPU: a .item() per micro-step would make the
            # CPU wait for the GPU grad_accum times per step.
            loss_accum = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x = train.batch(args.micro_batch).to(device, non_blocking=True)
                loss = train_loss(x)
                (loss / args.grad_accum).backward()
                loss_accum += loss.detach() / args.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 25 == 0 or step == session_start_step + 1:
                loss_accum, grad_norm = loss_accum.item(), grad_norm.item()
                now = time.time()
                n = 25 if step % 25 == 0 else 1
                tps = tokens_per_step * n / max(now - last_log, 1e-6)
                last_log = now
                eta_h = (total_steps - step) * tokens_per_step / max(tps, 1) / 3600
                print(f"step {step:,}/{total_steps:,} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                      f"gnorm {grad_norm:.2f} | {tps/1e3:.0f}k tok/s | "
                      f"{step*tokens_per_step/1e9:.2f}B tok | "
                      f"VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB | "
                      f"run eta {eta_h:.1f}h | session {(now-session_start)/3600:.2f}h",
                      flush=True)

            if step % args.val_every == 0:
                v = validate(model, val_sets, device, args.micro_batch)
                v.update(step=step, tokens=step * tokens_per_step)
                history.append(v)
                per = " ".join(f"{k}={v[k]:.3f}" for k in MIXTURE if k in v)
                print(f"VAL @ {step:,}: mixture={v['mixture']:.4f} | {per}", flush=True)

            tokens_done = step * tokens_per_step
            for m in milestones:
                if tokens_done >= m and m not in milestones_done:
                    milestones_done.append(m)
                    checkpoint(out / f"milestone_{m/1e9:g}B.pt")
                    print(f"[milestone] saved {m/1e9:g}B-token checkpoint", flush=True)

            if step % args.save_every == 0:
                checkpoint(ckpt_path)

            # Soft stop: only ever checked here, between optimizer steps.
            if stopper.check(session_start, args.max_hours):
                break

    except KeyboardInterrupt:
        sys.exit(1)

    finished = step >= total_steps
    if not history or history[-1]["step"] != step:
        v = validate(model, val_sets, device, args.micro_batch)
        v.update(step=step, tokens=step * tokens_per_step)
        history.append(v)
        print(f"VAL @ {step:,}: mixture={v['mixture']:.4f} | "
              + " ".join(f"{k}={v[k]:.3f}" for k in MIXTURE if k in v), flush=True)

    sessions.append({
        "start_step": session_start_step, "end_step": step,
        "hours": round((time.time() - session_start) / 3600, 2),
        "stop_reason": "complete" if finished else (stopper.reason or "unknown"),
        "ended_at": time.strftime("%Y-%m-%d %H:%M"),
    })
    checkpoint(ckpt_path)
    if stop_file.exists():
        stop_file.unlink()

    result = {
        "config": args.config, "model_config": model.config,
        "parameters_m": round(n_params / 1e6, 2),
        "tokens_trained": step * tokens_per_step, "steps": step,
        "total_steps": total_steps, "finished": finished,
        "sessions": sessions, "val_history": history,
        "final_val": history[-1] if history else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    if finished and not args.skip_latency:
        result["latency"] = benchmark_latency(model, device, cfg["vocab_size"])
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    done_b = step * tokens_per_step / 1e9
    if finished:
        print(f"\nRUN COMPLETE: {done_b:.2f}B tokens. Final mixture val "
              f"{history[-1]['mixture']:.4f}", flush=True)
    else:
        print(f"\nSESSION ENDED ({sessions[-1]['stop_reason']}): step {step:,}/{total_steps:,}, "
              f"{done_b:.2f}B tokens, checkpoint saved to {ckpt_path}", flush=True)
        print("Resume with the same command plus --resume.", flush=True)


if __name__ == "__main__":
    main()
