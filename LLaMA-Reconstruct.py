import torch
import torch.nn as nn
from einops import rearrange


class LLaMAConfig:
    block_size: int = 2048
    vocabsize_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

import torch
import torch.nn as nn
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.qkv_proj = nn.Linear(self.n_embd, self.n_embd * 3)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    @staticmethod
    def rope(pos, dim, theta=10000):
        assert dim % 2 == 0, "Dimension must be even for RoPE."
        half_dim = dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))

        out = torch.einsum("...n,d->...nd", pos, inv_freq)
        sin, cos = out.sin(), out.cos()
        out = torch.stack((cos, -sin, sin, cos), dim=-1)
        out = rearrange(out, '... n d (i j) -> ... n d i j', i=2, j=2)

        return out.float()

    def forward(self, x, rope):
        batch_size, seq_length, n_embd = x.size()

        # Project inputs to query, key, and value
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)

        if rope is not None:
            q = q.reshape(*q.shape[:-1], -1, 1, 2)
            k = k.reshape(*k.shape[:-1], -1, 1, 2)
            q = rope[..., 0] * q[..., 0] + rope[..., 1] * q[..., 1]
            k = rope[..., 0] * k[..., 0] + rope[..., 1] * k[..., 1]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = attn.matmul(v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embd)
        out = self.out_proj(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.hidden_dim = int(2/3 * 4 * config.n_embd)
        self.w1 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)  # gate (W_a)
        self.w3 = nn.Linear(config.n_embd, self.hidden_dim, bias=False)  # content (W_b)
        self.w2 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)  # down-projection
    def forward(self, x):
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)
        content = self.w3(x)
        x = gate * content
        x = self.w2(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.ff(self.ln2(x))
        return x
    
class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabsize_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocabsize_size, bias=False)

        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.head_dim = config.n_embd // config.n_head
        positions = torch.arange(config.block_size)
        rope = SelfAttention.rope(positions, head_dim=self.head_dim)
        self.register_buffer("rope", rope)

    def forward(self, x):
        batch_size, seq_length = x.size()
        x = self.token_embedding(x)
        rope_slice = self.rope[:seq_length, :seq_length]

        for block in self.blocks:
            x = block(x, rope_slice)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


