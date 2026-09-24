"""
交叉注意力机制

1. __init__部分:
只需要传入d_model, 持久化QKV的权重矩阵

2. forward部分:
有x和context两个序列入参, 以及附加的mask
注意Q是靠x算, KV是靠context算
注意力矩阵不是正方形而是长方形

"""

from torch import nn
import torch
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, context, mask=None):
        q = self.W_q(x) # (batch_size, decoder_seq_len, d_model)
        k = self.W_k(context) # (batch_size, encoder_seq_len, d_model)
        v = self.W_v(context)

        atten_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_model)
        if mask is not None:
            atten_scores.masked_fill(mask==0,float('-inf'))
        atten_weights = torch.softmax(atten_scores,dim=-1)

        outputs = torch.matmul(atten_weights, v)
        return outputs, atten_weights


# ================= 测试脚本 =================
def test_cross_attention():
    # 定义超参数
    batch_size = 2
    d_model = 16
    
    # 核心：目标序列和上下文序列的长度完全可以不同
    tgt_len = 3   # 比如当前正在生成的 3 个词
    src_len = 10  # 比如检索到的 10 个词的外部知识库片段
    
    # 实例化模型
    attention = CrossAttention(d_model=d_model)
    
    # 模拟输入
    x = torch.randn(batch_size, tgt_len, d_model)
    context = torch.randn(batch_size, src_len, d_model)
    
    print("=== 测试：单头 Cross Attention 前向传播 ===")
    outputs, weights = attention(x, context)
    
    print(f"Target 输入 (x) 形状: {x.shape}")
    print(f"Source 输入 (context) 形状: {context.shape}")
    print(f"注意力权重矩阵形状: {weights.shape}  <-- 注意这里是 (tgt_len, src_len) 的长方形")
    print(f"最终输出形状: {outputs.shape}  <-- 长度对齐回了 Target 序列")
    
    # 验证在 src_len 维度上的概率和是否为 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, tgt_len))
    print("✅ 形状验证通过，每一行 (tgt) 对所有 context (src) 的注意力概率和为 1。")

if __name__ == "__main__":
    test_cross_attention()
        