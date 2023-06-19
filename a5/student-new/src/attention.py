import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

"""
Write your SynthesizerAttention below.
Hint: paste over the CausalSelfAttention above and modify it minimally.
"""

class SynthesizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # NEW learnable weights
        
        # 明明是多头注意力，为什么没有体现出来呢？
        # 因为 X @ w1 其实就是利用分块矩阵的思想实现了多头注意力
        # 如果为每一个头都准备一个矩阵的话，速度可能会慢，而且还会损失一部分 X 的信息
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        
        # 之所以是 config.block_size-1 ，是因为 torch 的矩阵是从 "第 0 维" 开始的，而不是像
        # Linear layer 那样从 "n 维" 映射为 "m 维"
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        # TODO [part g]: Write your SynthesizerAttention below.
        #   Do not modify __init__().
        # Hints:
        #   - Paste over the CausalSelfAttention above and modify it minimally.
        #   - Consider especially the parameters self.w1, self.w2 and self.b2.
        #       How do these map to the matrices in the handout?

        B, T, C = x.size()
        
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = F.relu(self.w1(x)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 广播机制的核心就是
        # 如果是一个向量，那么通过堆叠就可以形成一个矩阵
        # 如果是一个矩阵，那么通过这个矩阵的堆叠可以形成任意高维度的矩阵
        att = att @ self.w2[:, :T] + self.b2[:T]
        
        # a[][] 这种写法只会对同一维度进行多次切片，而 a[ , ] 这种可以对多维度进行切片
        
        # decoder 要在 softmax 之前进行掩码
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)
        att = F.softmax(att, dim = -1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
