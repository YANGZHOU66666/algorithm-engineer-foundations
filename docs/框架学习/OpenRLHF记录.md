## 2025.8.15 安装

CUDA Version: 12.8，服务器AutoDL 3090*1

按照pytorch官网：

```
pip3 install torch torchvision
```

克隆OpenRLHF官方GitHub：

```
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
```

尝试直接安装：

```
pip install .
```

失败！一直卡在编译Flash Attention: 

> Building wheel for flash-attn (setup.py) ... |



Ctrl+C后尝试使用网上的办法：

[flash-attention安装速度慢的解决方法_flash-attn安装卡住-CSDN博客](https://blog.csdn.net/dongbidsaxue/article/details/146322800)

在[Releases · mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)中找到合适的版本的whl文件，直接安装

```
pip install flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl
```

再切入OpenRLHF的文件夹下`pip install .`

应该成功了（？



## 2025.8.16 跑通demo

从hf-mirror上下载facebook/opt-1.3b，上传到AutoDL平台，在OpenRLHF文件夹下使用如下指令：

```
deepspeed --module openrlhf.cli.train_sft \
    --pretrain "../facebook-opt-1.3b" \
    --dataset "./my_sft_data" \
    --input_key "question" \
    --output_key "response" \
    --save_path "./sft_checkpoint_opt1.3b" \
    --max_epochs 2 \
    --learning_rate 5e-6 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --zero_stage 2 \
    --bf16 \
    --train_batch_size 2 \
    --micro_train_batch_size 2 \
    --gradient_checkpointing
```

得到微调后的权重。

合并Lora权重至主模型：

```
python -m openrlhf.cli.lora_combiner \
    --model_path "../facebook-opt-1.3b" \
    --lora_path "./sft_checkpoint_opt1.3b" \
    --output_path "./sft_merged_model_opt1.3b" \
    --bf16
```

使用脚本测试微调后的模型效果：

```python
# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 直接加载合并后的完整模型
merged_model_path = "./sft_merged_model_opt1.3b"
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # 自动将模型加载到 GPU
)
model.eval()

# 2. 进行对话
prompt = "写一首关于夏天的诗。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("模型生成中...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    repetition_penalty=1.1
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("="*20)
print(f"提示 (Prompt): {prompt}")
# 注意：模型可能会重复你的输入，这是正常现象
print(f"模型回答 (Response): {response}")
print("="*20)
```

运行脚本：

```
python inference.py
```

