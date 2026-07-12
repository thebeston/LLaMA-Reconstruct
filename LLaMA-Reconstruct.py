import torch
import torch.nn as nn


class LLaMAConfig:
    block_size: int = 2048
    vocabsize_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

class SelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.qkv_proj = nn.Linear(self.n_embd, self.n_embd * 3)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        self.register_buffer(
            "tril", torch.tril(torch.ones(self.n_embd, self.n_embd))
        )

    def forward(self, x):
        batch_size, seq_length, vocab_size = x.size()

        # Project inputs to query, key, and value
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(self.tril == 0, float = '-inf')
        attn = torch.softmax(attn, dim=-1)

        out = attn.matmul(v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, vocab_size)
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
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
