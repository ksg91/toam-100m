"""
Export an AgentLM checkpoint to a standard Hugging Face LlamaForCausalLM folder
(safetensors + tokenizer), then verify it against the original model.

AgentLM is Llama-shaped (RMSNorm, SwiGLU, RoPE, GQA, no biases, tied embeddings).
Two differences are handled here:
  * RoPE pairing: AgentLM rotates interleaved pairs (0,1),(2,3)...; HF Llama rotates
    halves (j, j+D/2). Permuting the rows of q_proj and k_proj per head maps one to
    the other; attention scores are unchanged.
  * RMSNorm eps: torch.nn.RMSNorm(eps=None) uses float32 machine eps.

    python release/export_hf.py --checkpoint runs/finetuned.pt --out release/toam-hf

The checkpoint is an AgentLM state dict as written by pretrain/pretrain.py
({"model": ..., "config": ...}).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toam.model import AgentLM  # noqa: E402
from prompts import sample_prompts  # noqa: E402

FP32_EPS = torch.finfo(torch.float32).eps


def convert(sd: dict, cfg: dict) -> tuple[LlamaConfig, dict]:
    d, nh, nkv = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"]
    hd = d // nh
    hidden = sd["blocks.0.mlp.down.weight"].shape[1]
    perm = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])

    def rope_rows(w, heads):
        return w.view(heads, hd, -1)[:, perm, :].reshape(heads * hd, -1).contiguous()

    lc = LlamaConfig(
        vocab_size=cfg["vocab_size"], hidden_size=d, intermediate_size=hidden,
        num_hidden_layers=cfg["n_layers"], num_attention_heads=nh, num_key_value_heads=nkv,
        head_dim=hd, max_position_embeddings=cfg["max_seq_len"], rms_norm_eps=FP32_EPS,
        rope_theta=10000.0, hidden_act="silu", attention_bias=False, mlp_bias=False,
        tie_word_embeddings=True,
    )
    out = {"model.embed_tokens.weight": sd["tok_emb.weight"], "model.norm.weight": sd["norm.weight"]}
    for i in range(cfg["n_layers"]):
        b, o = f"blocks.{i}.", f"model.layers.{i}."
        kv = sd[b + "attn.kv.weight"]
        out[o + "input_layernorm.weight"] = sd[b + "norm1.weight"]
        out[o + "post_attention_layernorm.weight"] = sd[b + "norm2.weight"]
        out[o + "self_attn.q_proj.weight"] = rope_rows(sd[b + "attn.q.weight"], nh)
        out[o + "self_attn.k_proj.weight"] = rope_rows(kv[: nkv * hd], nkv)
        out[o + "self_attn.v_proj.weight"] = kv[nkv * hd:].contiguous()
        out[o + "self_attn.o_proj.weight"] = sd[b + "attn.out.weight"]
        gu = sd[b + "mlp.gate_up.weight"]
        out[o + "mlp.gate_proj.weight"] = gu[:hidden].contiguous()
        out[o + "mlp.up_proj.weight"] = gu[hidden:].contiguous()
        out[o + "mlp.down_proj.weight"] = sd[b + "mlp.down.weight"]
    return lc, out


@torch.no_grad()
def greedy_ref(model, ids, eos, max_new=160):
    out = list(ids)
    for _ in range(max_new):
        nxt = int(model(torch.tensor(out)[None])[0, -1].argmax())
        if nxt == eos:
            break
        out.append(nxt)
    return out[len(ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="ksg91/toam-100m-v0.2", help="TOAM tokenizer (HF id or folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-n", type=int, default=40)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or ck["model_config"]
    ref = AgentLM(**cfg)
    ref.load_state_dict(ck["model"])
    ref.eval()

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    lc, sd = convert(ck["model"], cfg)
    lc.bos_token_id, lc.eos_token_id, lc.pad_token_id = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    hf = LlamaForCausalLM(lc)
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    assert not missing and not unexpected, (missing, unexpected)
    hf.tie_weights()
    assert torch.equal(hf.lm_head.weight, hf.model.embed_tokens.weight)
    hf.eval()

    # --- verify: logits (fp32) and greedy decodes on a fixed sample of tool prompts
    rows = sample_prompts(args.verify_n)
    max_diff, same = 0.0, 0
    for prompt in rows:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > 1800:
            continue
        x = torch.tensor(ids)[None]
        a = ref(x)[0].float()
        b = hf(input_ids=x).logits[0].float()
        max_diff = max(max_diff, (a - b).abs().max().item())
        ga = greedy_ref(ref, ids, tok.eos_token_id)
        gb = hf.generate(x, max_new_tokens=160, do_sample=False, eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.pad_token_id)[0, len(ids):].tolist()
        if gb and gb[-1] == tok.eos_token_id:
            gb = gb[:-1]
        same += int(ga == gb)
    print(f"verify: {len(rows)} prompts, max |logit diff| {max_diff:.2e}, identical greedy decodes {same}/{len(rows)}",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    hf.generation_config.save_pretrained(out)
    meta = {"source_checkpoint": args.checkpoint, "agentlm_config": cfg, "prompt_version": "v1",
            "verify": {"prompts": len(rows), "max_abs_logit_diff": max_diff, "identical_greedy": same}}
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
