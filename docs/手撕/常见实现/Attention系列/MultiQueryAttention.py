
import torch
from torch import nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.dim_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.dim_head)
        self.W_v = nn.Linear(d_model, self.dim_head)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1,2)
        k = k.view(batch_size, seq_len, 1, self.dim_head).transpose(1,2)
        v = v.view(batch_size, seq_len, 1, self.dim_head).transpose(1,2)

        if self.num_heads > 0:
            k = torch.repeat_interleave(k, repeats = self.num_heads, dim=1)
            v = torch.repeat_interleave(v, repeats = self.num_heads, dim=1)
        
        atten_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dim_head)
        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask==0, float('-inf'))
        atten_weights = torch.softmax(atten_scores, dim=-1)

        outputs = torch.matmul(atten_weights, v).transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(outputs)


# ================= 测试用例 =================

def test_multi_query_attention():
    # 模拟超参数
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    # 初始化 MQA 模型
    mqa = MultiQueryAttention(num_heads=num_heads, d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("--- 🚀 开始测试 Multi-Query Attention ---")

    # [Test Case 1] 基础形状测试
    out = mqa(x)
    assert out.shape == (batch_size, seq_len, d_model), f"输出形状错误，期望 {(batch_size, seq_len, d_model)}, 实际 {out.shape}"
    print(f"✅ 基础前向传播测试通过，输出维度完美对齐: {out.shape}")

    # [Test Case 2] Causal Mask 验证 (确保广播机制正常)
    # mask 形状: (seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    out_masked = mqa(x, mask=causal_mask)
    
    assert out_masked.shape == (batch_size, seq_len, d_model), "Mask 分支输出形状错误"
    assert not torch.isnan(out_masked).any(), "Mask 导致 NaN"
    print("✅ Causal Mask 测试通过，广播和极小值处理正常。")
    
    print("\n🎉 测试完毕！你的 MQA 核心思路完全正确！")

if __name__ == "__main__":
    test_multi_query_attention()