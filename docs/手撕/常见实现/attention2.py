import torch
from torch import nn
import math
import torch.nn.functional as F

"""
写一个标准版的Transformer attention 函数，输入是一个随机的embedding，shape: batch=2, length=8, dim=32，输出attention 的结果
"""

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x) # (bs, seqlen, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        score = torch.matmul(q,k.transpose(1,2)) # bs, seqlen, seqlen
        scale = math.sqrt(self.embed_dim)
        score = score / scale
        if mask is not None:
            score.masked_fill(mask, float('-inf'))

        weights = F.softmax(score, dim=-1)

        attention = torch.matmul(weights, v)
        return attention, weights

mySelfAttention = SelfAttention()
x = torch.zeros((2,8,32))

print(mySelfAttention(x))