# SFT

预训练 - 大模型获得所有的能力

指令微调 - 如何更好的回答问题



指令微调和预训练没有本质上的不同

网络结构完全相同，Loss基本相同，只有数据不同



## Chat Template

微调需要和原厂的对话模板保持一致

有时需要加上指定的标识符：

```
Hugging Face格式的数据：
[{"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "天空为什么是蓝色的？"},
 {"role": "assistant", "content": "这是由于光的散射引起的。"}]

实际用来训练大模型的数据格式：
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
天空为什么是蓝色的？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
这是由于光的散射引起的。<|eot_id|>
```

每个大模型和自己的Tokenizer唯一对应，和Chat Template唯一对应

示例：Llama3.1版本的tokenizer_config

```
{
    "add_prefix_space": false,
    "added_tokens_decoder": {
        "151643": {
            "content": "<|endoftext|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "151644": {
            "content": "<|im_start|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "151645": {
            "content": "<|im_end|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        }
    },
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    "bos_token": null,
    "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}[{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }]{% endif %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
    "clean_up_tokenization_spaces": false,
    "eos_token": "<|im_end|>",
    "errors": "replace",
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "split_token": "<|endoftext|>",
    "spilt_special_tokens": false,
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": null
}
```

将结构化的语料转化为大模型需要的格式：

```python
dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"},
          {"role": "assistant", "content": "这是由于光的散射引起的。"}]

msg = tokenizer.apply_chat_template(dialog, tokenize=False)
```

推理时，使用add_generation_prompt=True，来给prompt后面加上额外的标记，让大模型知道要生成回答

```python
dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"}]

msg = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
```

## Completion Only

对于每个问答对，由于前面的系统提示部分相同，我们也更需要关注模型回答问题的能力，可以选择只对回答部分计算Loss

解决方案：使用loss mask，对预测系统回答部分设为1，前面的部分设为0。用原本的每个位置预测的损失乘以loss mask得到最终需要的损失

示例代码：

```python
# 定义分隔符并进行分词
end_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"

inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True) # 注意这里使用padding和truncation把所有序列做成同一长度了，因此下面可以使用len(input_ids[0])算样本长度
input_ids = inputs["input_ids"]
input_len = len(input_ids[0])

end_ids = tokenizer(end_str)['input_ids']
end_id_len = len(end_ids)
# 核心逻辑：创建损失掩码 (Loss Mask)
loss_mask = []
for input_id in input_ids:
    for i in range(len(input_id) - end_id_len, -1, -1):
        if input_id[i : i + end_id_len] == end_ids:
            mask = [1] * (input_len - 1)
            mask[:i + end_id_len - 1] = [0] * (i + end_id_len - 1)
            loss_mask.append(mask)
            break
    if i == 0: # 所有回答部分都被截断
        loss_mask.append([0] * (input_len - 1))
# 转换为Tensor并计算最终损失
inputs = {k: torch.tensor(v) for k, v in inputs.items()}
loss_mask = torch.tensor(loss_mask)

# --- 下面的代码在模型前向传播之后执行 ---
loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
loss = loss * loss_mask
loss = torch.mean(loss)
```



## NEFTune - Noisy Embeddings Finetuning

背景：计算机视觉领域往往要对图片数据进行增强，以获得更多数据

Word Embedding：将token映射到某一空间，词义相似的词距离更近

NEFTune：embedding时随机添加噪声，增加样本的丰富度。下面的实现中向原向量中添加的是$[-\frac{alpha}{\sqrt{seq\_len*embedding\_size}},\frac{alpha}{\sqrt{seq\_len*embedding\_size}}]$的一个均匀分布

除以$\sqrt{dims}$的意义：保证在所有情况下（即无论序列长度），添加噪声前的样本和添加噪声后的样本的欧氏距离是相同的

示例代码：

```python
neftune_noise_alpha = 10
for i in range(epoch):
    for inputs, loss_mask in data_loader:
        input_ids = inputs.pop("input_ids")
        input_embeddings = model.base_model.model.model.embed_tokens(input_ids)
        dims = torch.tensor(input_embeddings.size(1) * input_embeddings.size(2))
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        input_embeddings = input_embeddings + torch.zeros_like(input_embeddings).uniform_(-mag_norm, mag_norm)
        inputs["inputs_embeds"] = input_embeddings
```

