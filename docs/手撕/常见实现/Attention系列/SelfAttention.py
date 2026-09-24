import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        atten_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_model)

        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask==0,float('-inf'))

        atten_weights = torch.softmax(atten_scores,dim=-1)
        outputs = torch.matmul(atten_weights,v)
        return outputs, atten_weights

# ================= 测试脚本 =================
def test_self_attention():
    # 1. 定义超参数
    batch_size = 2
    seq_len = 4
    d_model = 16
    
    # 2. 实例化模型
    attention = SelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("=== 测试 1：基础前向传播 (无 Mask) ===")
    outputs, weights = attention(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {outputs.shape}")
    print(f"注意力权重形状: {weights.shape}")
    # 验证权重在最后一个维度上求和为 1 (Softmax 的性质)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len))
    print("✅ 基础形状验证通过，Softmax 概率和为 1。\n")

    print("=== 测试 2：因果掩码 (Causal Mask) 效果验证 ===")
    # 生成下三角掩码矩阵：对角线及以下为 1，右上角为 0
    # 形状: (seq_len, seq_len) -> broadcast 到 (batch_size, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print("使用的 Mask 矩阵:\n", causal_mask)
    
    outputs_masked, weights_masked = attention(x, mask=causal_mask)
    
    # 打印第一个 batch 的注意力权重矩阵
    print("\n施加 Mask 后的注意力权重 (weights_masked[0]):")
    # 为了方便查看，四舍五入保留三位小数
    print(torch.round(weights_masked[0] * 1000) / 1000)
    
    # 验证右上角（未来的信息）的权重是否严格为 0
    upper_triangle = torch.triu(weights_masked[0], diagonal=1)
    assert torch.all(upper_triangle == 0), "Mask 失败，右上角不为 0！"
    print("✅ Mask 验证通过，当前 Token 没有看到未来的 Token。")

if __name__ == "__main__":
    test_self_attention()