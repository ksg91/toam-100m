"""
TOAM's native PyTorch model: a Llama-style decoder (RMSNorm, SwiGLU, RoPE,
grouped-query attention, tied embeddings). Used for pretraining; the released
weights are converted to a standard Hugging Face LlamaForCausalLM by
release/export_hf.py.

TOAM-100M config: vocab 32,768 · d_model 768 · 12 layers · 12 heads / 4 KV ·
mlp_mult 2.7 · context 2,048 · scaled residual init → 102.5M parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k):
        # q,k: B,H,T,D
        T = q.shape[-2]
        cos = self.cos_cached[:T].to(dtype=q.dtype, device=q.device)[None,None,:,:]
        sin = self.sin_cached[:T].to(dtype=q.dtype, device=q.device)[None,None,:,:]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, n_kv_heads=None):
        super().__init__()
        assert d_model % n_heads == 0
        n_kv_heads = n_kv_heads or n_heads
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        # Grouped-query attention: fewer K/V heads shrink the KV cache, which
        # is what makes a deep, thin model affordable on-device. Plain MHA
        # keeps the fused qkv projection so existing checkpoints still load.
        self.gqa = n_kv_heads != n_heads
        if self.gqa:
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.kv = nn.Linear(d_model, 2*n_kv_heads*self.head_dim, bias=False)
        else:
            self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B,T,C = x.shape
        if self.gqa:
            q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
            kv = self.kv(x).view(B,T,2,self.n_kv_heads,self.head_dim)
            k,v = kv.unbind(dim=2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
        else:
            qkv = self.qkv(x).view(B,T,3,self.n_heads,self.head_dim)
            q,k,v = qkv.unbind(dim=2)
            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
        q,k = self.rope(q,k)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=self.gqa)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.out(y)

class SwiGLU(nn.Module):
    def __init__(self, d_model, mlp_mult=2.7):
        super().__init__()
        hidden = int(d_model * mlp_mult)
        # Round for nicer kernels.
        hidden = ((hidden + 63)//64)*64
        self.gate_up = nn.Linear(d_model, 2*hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, mlp_mult, n_kv_heads=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model,n_heads,max_seq_len,n_kv_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model,mlp_mult)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AgentLM(nn.Module):
    def __init__(
        self,
        vocab_size=16384,
        d_model=512,
        n_layers=12,
        n_heads=8,
        max_seq_len=1024,
        mlp_mult=2.7,
        n_kv_heads=None,
        scaled_residual_init=False,
    ):
        super().__init__()
        self.config = {
            "vocab_size":vocab_size,
            "d_model":d_model,
            "n_layers":n_layers,
            "n_heads":n_heads,
            "max_seq_len":max_seq_len,
            "mlp_mult":mlp_mult,
            "n_kv_heads":n_kv_heads or n_heads,
            "scaled_residual_init":scaled_residual_init,
        }
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size,d_model)
        self.blocks = nn.ModuleList([
            Block(d_model,n_heads,max_seq_len,mlp_mult,n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        if scaled_residual_init:
            # Projections that write into the residual stream are scaled by
            # 1/sqrt(2 * n_layers), so the stream's variance stays bounded as
            # depth grows. Matters for deep, thin models; opt-in so existing
            # training scripts initialise exactly as before.
            std = 0.02 / math.sqrt(2 * n_layers)
            for block in self.blocks:
                nn.init.normal_(block.attn.out.weight, mean=0.0, std=std)
                nn.init.normal_(block.mlp.down.weight, mean=0.0, std=std)

    def _init_weights(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)

    def forward(self,input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
