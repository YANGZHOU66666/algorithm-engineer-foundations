"""
多头自注意力机制

1. __init__部分：
- 要记得语法__init__(self, xxx); super().__init__()
= 入参为num_heads和d_model，需要持久化头数、隐藏层维度、头维度、QKV权重矩阵、【合流权重矩阵】!!

2. forward()部分：
- 先转成大块的q、k、v向量
- 拆分多头：view和transpose，注意换成最后两维是seq_len和head_dim的格式，便于后续在单个头上算整个注意力; 注意view的时候还是先seq_len再num_heads然后再transpose!
- 每个头上分别计算注意力。注意除以根号dk、mask机制和softmax
- 拼接多头
- 最终再过一个W_o得到输出

3. 需要注意的语法点
- Tensor.view(): 变换维度, 拆多头和合并多头时用
- Tensor.transpose(): 某两个维度之间交换
- Tensor.contiguous(): transpose之后, view之前, 需要让内存连续
- 几个数学公式: torch.sqrt(), torch.softmax(Tensor,dim=n)

"""
from torch import nn
import torch
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x) # (batch_size, seq_len, d_model)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2) # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,float('-inf'))

        attention_weights = torch.softmax(attention_scores,dim=-1) # (batch_size, num_heads, seq_len, seq_len)

        outputs = torch.matmul(attention_weights,v) # (batch_size, num_heads, seq_len, head_dim)

        outputs = outputs.transpose(1,2).contiguous() # (batch_size, seq_len, num_heads, head_dim)

        outputs = outputs.view(batch_size, seq_len, self.d_model)

        return self.W_o(outputs)

def test_multi_head_attention():
    # 模拟超参数
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    # 初始化模型
    mha = MultiHeadSelfAttention(num_heads=num_heads, d_model=d_model)
    
    print("--- 开始测试 ---")

    # [Test Case 1] 基础形状测试 (无 Mask)
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x)
    assert output.shape == (batch_size, seq_len, d_model), f"Shape错误: 期望 {(batch_size, seq_len, d_model)}, 实际 {output.shape}"
    print("✅ Test 1: 基础前向传播 (无 Mask) 测试通过，输出维度正确。")

    # [Test Case 2] Causal Mask (下三角掩码，模拟生成式任务如 GPT)
    # 形状通常为 (batch_size, 1, seq_len, seq_len) 以便利用广播机制
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_causal = mha(x, mask=causal_mask)
    assert output_causal.shape == (batch_size, seq_len, d_model), "Causal Mask 输出 Shape 错误"
    
    # 验证输出是否有 NaN (防止 -inf 处理不当引发计算溢出)
    assert not torch.isnan(output_causal).any(), "Causal Mask 前向计算出现 NaN"
    print("✅ Test 2: Causal Mask (下三角掩码) 测试通过，前向传播未报错且维度正确。")

    # [Test Case 3] Padding Mask (模拟文本长度不一致)
    # 假设 batch 中第一个序列全有效，第二个序列只有前 3 个 token 有效
    padding_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ]).unsqueeze(1).unsqueeze(2) # 调整形状为 (batch, 1, 1, seq_len) 触发广播
    
    output_padding = mha(x, mask=padding_mask)
    assert output_padding.shape == (batch_size, seq_len, d_model), "Padding Mask 输出 Shape 错误"
    assert not torch.isnan(output_padding).any(), "Padding Mask 前向计算出现 NaN"
    print("✅ Test 3: Padding Mask 测试通过，广播机制工作正常。")

if __name__ == "__main__":
    test_multi_head_attention()