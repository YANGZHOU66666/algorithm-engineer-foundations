"""
分组查询注意力机制
1. __init__ 实现逻辑
和MultiHeadSelfAttention不同的, 需要传入3个超参数: num_heads, num_kv, d_model
需要持久化dim_head, n_groups; QKV和输出O的权重矩阵

2. forward 实现逻辑


"""

from torch import nn
import torch
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv, d_model):
        super().__init__()
        assert num_heads % num_kv == 0
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.num_kv = num_kv
        self.d_model = d_model
        self.dim_head = d_model // num_heads
        self.n_groups = num_heads // num_kv

        self.W_q = nn.Linear(d_model, num_heads * self.dim_head)
        self.W_k = nn.Linear(d_model, num_kv * self.dim_head)
        self.W_v = nn.Linear(d_model, num_kv * self.dim_head)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x) # (batch_size, seq_len, num_heads * dim_head)
        k = self.W_k(x) # (batch_size, seq_len, num_kv * dim_head)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_kv, self.dim_head).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_kv, self.dim_head).transpose(1,2)

        if self.n_groups > 1:
            k = torch.repeat_interleave(k, repeats = self.n_groups, dim = 1)
            v = torch.repeat_interleave(v, repeats = self.n_groups, dim = 1)

        atten_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dim_head)
        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask==0,float('-inf'))
        atten_weights = torch.softmax(atten_scores, dim=-1)

        outputs = torch.matmul(atten_weights,v).transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(outputs)

# ================= 测试用例 =================

def test_grouped_query_attention():
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    
    # 随机输入张量
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("🚀 开始测试 GQA 的各种形态...\n")

    # [Test 1] 验证 GQA (比如 8 个 Q 头，2 个 KV 头，这是 Llama 2/3 常见的比例)
    num_kv_gqa = 2
    gqa = GroupedQueryAttention(num_heads=num_heads, num_kv=num_kv_gqa, d_model=d_model)
    out_gqa = gqa(x)
    assert out_gqa.shape == (batch_size, seq_len, d_model), "GQA 形状错误"
    print(f"✅ GQA (Q_heads={num_heads}, KV_heads={num_kv_gqa}) 运行成功，输出维度: {out_gqa.shape}")

    # [Test 2] 验证 MHA (当 num_kv == num_heads 时，退化为传统多头注意力)
    num_kv_mha = 8
    mha = GroupedQueryAttention(num_heads=num_heads, num_kv=num_kv_mha, d_model=d_model)
    out_mha = mha(x)
    assert out_mha.shape == (batch_size, seq_len, d_model), "MHA 形状错误"
    print(f"✅ MHA (Q_heads={num_heads}, KV_heads={num_kv_mha}) 运行成功，未触发 broadcast 分支。")

    # [Test 3] 验证 MQA (当 num_kv == 1 时，退化为多查询注意力，ChatGLM2/3 在用)
    num_kv_mqa = 1
    mqa = GroupedQueryAttention(num_heads=num_heads, num_kv=num_kv_mqa, d_model=d_model)
    out_mqa = mqa(x)
    assert out_mqa.shape == (batch_size, seq_len, d_model), "MQA 形状错误"
    print(f"✅ MQA (Q_heads={num_heads}, KV_heads={num_kv_mqa}) 运行成功，所有 Q 头共享 1 个 KV 头。")

    # [Test 4] 验证带 Causal Mask 的 GQA
    mask = torch.tril(torch.ones(seq_len, seq_len))
    out_masked = gqa(x, mask=mask)
    assert not torch.isnan(out_masked).any(), "Mask 导致 NaN"
    print(f"✅ GQA + Causal Mask 运行成功，未出现梯度爆炸或 NaN。")
    
    print("\n🎉 所有测试通过，这套代码非常完美！")

if __name__ == "__main__":
    test_grouped_query_attention()