import torch
from torch import nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_q, num_kv):
        super().__init__()
        assert embed_dim % num_q == 0
        assert num_q % num_kv == 0

        
        self.embed_dim = embed_dim
        self.num_q = num_q
        self.num_kv = num_kv
        self.head_dim = embed_dim//num_q
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim*num_kv)
        self.v_proj = nn.Linear(embed_dim, self.head_dim*num_kv)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q, -1).transpose(1,2) # (bs, heads, seqlen, head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv, -1).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv, -1).transpose(1,2)

        repeat_factor = self.num_q // self.num_kv
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1) # (bs, heads, seqlen, head_dim)
            v = v.repeat_interleave(repeat_factor, dim=1) 

        scores = torch.matmul(q,k.transpose(-2,-1))
        scale = math.sqrt(self.head_dim)
        scores = scores / scale
        if mask is not None:
            scores.masked_fill(mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v).transpose(1,2).contiguous().view(batch_size,seq_len,embed_dim)
        attention = self.o_proj(attention)
        return attention, weights




