# Qwen-Image-2

## Abstract & Intro

**历史薄弱问题：**

超长文本渲染；多语言排版；高分辨率下照片逼真度下降；复杂指令遵循都解决的不好

此外，历史工作没有把所有能力用单一模型解决。要么逼真性高要么文本渲染准确，要么文生图要么图片编辑。

**数据：**

数据pipeline：逐步引入经过过滤的语料库、编辑数据对、合成数据、精选的高分辨率样本

自动化数据飞轮：评估信号、用户反馈，用于识别失败的模式、驱动迭代

**模型结构：**

Qwen3-VL encoder理解图和指令

high-compression VAE（16倍压缩）：用了residual autoencoding、enlarged latent channels、semantic alignment loss

MMDiT用于生成：用了MARoPE、RMSNorm QK normalization、bias-free modulation、SwiGLU

**训练：**

预训练、继续预训练、SFT、RLHF

偏好对齐上，主要对齐审美、文本和图的一致性、人像质量、指令遵循、视觉一致性（多张图角色是否统一、画面元素前后矛盾等）

再用GRPO优化生成模型

## 数据——筛选、合成、训练数据策略（2 Data）

### 数据筛选

文生图（Text-to-Image，T2I）：(图片, 文本指令)对，真实的摄影图（人像、风景、静物等，也有长尾类别）、风格丰富且排版敏感的内容（幻灯片、海报、漫画、信息图等）

图像编辑（image editing，TI2I）：(原图, 指令->修改后图片)对，分单图编辑（属性修改如换衣服颜色、背景替换、风格迁移、文字编辑、修复、结构化调整等）和多图编辑（基于参考图的生成、保持主体一致性、跨图风格迁移、组合合成等）

### 数据注释

通用描述（General Captions）：给不同分辨率的图像，添加全面、详细的自然语言描述，全面说明图里面有什么。包括物体、背景细节、空间关系、文字内容等。->帮助模型学会“图里有什么”“文字怎么说”之间的对应关系

文字描述（Text Captions）：针对含密集文字的图像上进行标注。如幻灯片、海报、漫画对话框、产品说明书、路牌招牌等。具体来说：转录图片上所有文字、描述文字的排版布局、描述文字的视觉样式。解决文字渲染乱码、长尾字符稀缺、复杂排版理解问题等

知识描述（Knowledge Captions）：补充图中物品的背景知识和上下文。注入世界知识，让生成内容更符合现实逻辑。

结构描述（Structured Captions）：处理逻辑关系复杂的图像，如流程图、组织架构图、关系图、UI界面、数据图表等。标注不再采用自然语言，而是采用结构化数据格式来明确表达，实体节点、属性信息、拓扑关系等。更好的描述关系、拓扑依赖、视觉元素中的语义联系等。

### 多阶段训练数据策略

每个阶段会有一些Filters，用于控制数据质量

**Stage 1：256P T2I pre-training**

Stage 1用到：Broken Files Filter（过滤坏文件）、Resolution Filter（过滤分辨率不合适、短边<256P的文件）、Deduplication Filter（去重过滤）、NSFW Filter（不适宜内容过滤）、Rotation Filter（把图片转成正向，或者扔掉方向混乱的）、Entropy Filter（过滤掉熵过高-可能噪点、熵过低-可能纯色图）、CLIP Filter（过滤图片和文本相似程度太低的）、Token Length Filter（过滤描述太长/太短的）

**Stage 2：256P T2I & TI2I pre-training**

Stage 2相较于Stage 1引入了TI2I数据

**Stage 3：512P T2I & TI2I pre-training**

Stage 3额外引入了合成数据

**Stage 4：512P/1024P T2I & TI2I pre-training**

Stage 4用到：Resolution Filter、Image Quality Filter（过滤掉低质量的，比如欠曝、清晰度低的？）、Image Aesthetic Filter（过滤不符合人审美的数据？）、Compression Quality Filter，进一步提高数据质量

**Stage 5：Multi-Resolution T2I & TI2I pre-training**

多分辨率。512P、1024P、2048P都会覆盖。

**Stage 6：SFT**

更好的将模型和高质量人类偏好对齐。用了Distribution Filter，相较于前面的Stage，用更严格标准的移除低质量数据

### 数据飞轮

多源信号收集（网页应用/内部评测等bad case）-> case路由至不同解决方法 -> 模型训练。新训出的模型回到阶段一。

1. 多源信号收集（Multi-source signal collection）

   自动收集反馈信号，标准化的模型评测、靶向的bad-case挖掘、多源用户反馈
2. 案例路由&有针对的优化（Case routing\&targeted optimization）

   三条路线：

   RL track——由于未充分RL/对齐的case->自动化奖励调整

   Pre-training track——未覆盖的知识的case，即训练时特定数据缺失->系统自动发生成prompt和图像编辑指令，人工介入

   Prompt Engineering track——模型有能力回答好，但用户给的prompt不够好->用prompt优化器来完成
3. 模型更新&闭环

   获得了新的数据集和参数，模型训练，获得的checkpoints重新回到Stage 1用于评测，继续飞轮循环。

## 模型架构（3 Architecture）

总览：

一个MLLM（Qwen3-VL）用于语义+图像理解，一个VAE Encoder/Decoder用于图像编解码，一个MMDiT用于根据VAE Encode的latent向量+MLLM给的语义向量，给出去噪后的图像latent向量。

另有一个Prompt Enhancer用于用户prompt改写

### VAE

VAE有个三点的权衡：压缩率、重建忠实度（即能不能重建的好）、可扩散性（即是否容易被扩散模型画出来）

压缩率过高，重建忠实度会下降；而若用添加通道数的方法增多保留信息，会更难扩散出来

TODO

### Multi-modal Diffusion Transformer（MMDiT）

<br />

### Prompt Enhancer（PE）

基于Qwen3.5-9B进行后训练

<br />

## 训练（4 Training）
