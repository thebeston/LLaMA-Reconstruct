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
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x
