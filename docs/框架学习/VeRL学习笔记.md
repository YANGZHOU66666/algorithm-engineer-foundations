# VeRL

## 数据准备 Data Preparation

source: [Prepare Data for Post-Training — verl documentation](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)

核心思路：将数据集做成有如下几个字段的列表：

```
1. data_source
作用: 数据集的唯一标识符，用于后续匹配对应的奖励函数。
格式: 字符串 (String)。
示例: 'openai'

2. prompt
作用: 将要输入给大模型的实际内容。
格式: Hugging Face chat_template 格式（一个字典列表）。
角色 (role):
"user": 代表用户的直接输入。
"system": (可选) 用于提供高层次的系统级指令或设定模型的“人设”，通常放在列表开头。
"assistant": (可选) 用于多轮对话中模型的回复。

示例:
[
    { "role": "system", "content": "你是一位专业的代码助手。" },
    { "role": "user", "content": "用 Python 写一个快速排序算法。" }
]

3. ability
作用: 定义任务所属的类别。
格式: 字符串 (String)。
示例: 'math', 'coding', 'translation', 'summarization'。

4. reward_model
作用: 存放用于计算奖励 (Reward) 的信息。
格式: 字典 (Dictionary)。
内部字段:
style: 奖励计算的方式。示例值为 "rule"，表示基于规则匹配。
ground_truth: 标准答案。其数据类型灵活，取决于你的任务。你提供的 ground_truth 必须能被你的奖励函数正确解析。

示例:
{
    "style": "rule",
    "ground_truth": "15"
}

5. extra_info
作用: 记录与训练无关的额外信息，主要用于调试和数据溯源。
格式: 字典 (Dictionary)。
示例: {'split': 'train', 'index': 42}。
```

然后转化为**.parquet**格式即可



## 自定义奖励函数

source: [Implement Reward Function for Dataset — verl documentation](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)

### 运行命令中自定义

- 函数签名要求：

```python
def your_function_name(data_source, solution_str, ground_truth, extra_info=None):
    pass
```

必须包含data_source, solution_str, ground_truth, extra_info四个字段，与数据部分的含义相同



- 命令行参数设置：

```
python -m verl.trainer.main_ppo \
  ... \
  custom_reward_function.path=my_rewards.py \
  custom_reward_function.name=my_amazing_reward_fn \
  ...
```

需要指定py文件的路径和函数名

### 框架内置

`/verl/utils/reward_score`里实现了很多demo的奖励函数，如`gsm8k.py`就实现了QuickStart中的demo的奖励函数：

```python
import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
```

再通过`/verl/utils/reward_score/__init__.py`来注册：注意是根据data_source字段值的不同来判断的

```python
def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])
```

如果要自定义奖励函数，也可以考虑框架在`/verl/utils/reward_score`下开一个py文件，然后在`__init__.py`中的if-else逻辑中指定对应数据集的data_source，判断对应奖励函数

**（来自gemini）总结一下**：除非你有特殊理由，否则**强烈推荐你使用 `custom_reward_function.path` 和 `custom_reward_function.name` 这两个运行时参数**。这是最干净、最方便的做法。

### 补充信息

1. 一个便捷方式

如果你在自定义的 `.py` 文件中，将你的奖励函数**直接命名为 `compute_score`**，那么你在运行命令时就**不需要设置 `custom_reward_function.name` 这个参数了**。

**示例：** 如果你的 `my_rewards.py` 文件内容是：

```python
def compute_score(solution_str, ground_truth, ...):
    # 你的逻辑
    return score
```

那么你的启动命令就可以简化为：

```bash
python3 -m verl.trainer.main_ppo \
  ... \
  custom_reward_function.path=my_rewards.py \
  ...
```

2. 便于实验、A/B测试的最佳实践

你可以在**同一个 `.py` 文件里定义多个不同的奖励函数**。

这样一来，当你想切换不同的打分逻辑进行实验时，你只需要修改启动命令中的 `custom_reward_function.name` 这一个参数，而不需要更改文件路径，非常方便。

**示例：** 你的 `my_rewards.py` 文件可以这样写：

```python
def reward_logic_v1(solution_str, ...):
    # 版本1的打分逻辑
    return score

def reward_logic_v2(solution_str, ...):
    # 版本2的打分逻辑
    return score
```

然后你可以轻松地在两次不同的实验中切换：

- **实验 A**: `... custom_reward_function.name=reward_logic_v1 ...`
- **实验 B**: `... custom_reward_function.name=reward_logic_v2 ...`

3. 奖励函数的不同类型

最后，手册也明确了奖励函数的来源可以是多种多样的，不仅仅是基于规则的代码：

- **规则型**: 就像 GSM8k 的例子，通过字符串匹配和正则表达式来打分。
- **模型型**: 对于 RLHF 数据集，明确提到会使用一个**奖励模型 (Reward Model)** 来打分。
- **沙箱型**: 对于代码生成任务，会使用 **SandBox (沙箱)** 来实际执行代码并根据测试用例的通过情况来打分。

这为你实现自己的 `compute_score` 函数提供了更广阔的思路：你的函数内部不仅可以是简单的 `if/else`，还可以是调用一个外部 API、一个预训练模型，或者一个代码执行环境。



## 自定义配置 Configurations

source: [Config Explanation — verl documentation](https://verl.readthedocs.io/en/latest/examples/config.html)

### 文件结构

强化学习的中央配置枢纽位于verl/verl/trainer/config/ppo_trainer.yaml，其内部将各个模块的细分配置放到了各个文件夹下：

```yaml
defaults:

  # <folder_name>@<field_name>.<field_name>: <yaml_file_name>
  # actor_rollout_ref.actor: trainer/config/actor/dp_actor.yaml
  - actor@actor_rollout_ref.actor: dp_actor

  # data: trainer/config/data/legacy_data.yaml
  - data@data: legacy_data

  # Reference model config.
  # Reference model will be enabled when actor.use_kl_loss or/and algorithm.use_kl_in_reward is/are True.
  - ref@actor_rollout_ref.ref: dp_ref

  # Rollout model config.
  - rollout@actor_rollout_ref.rollout: rollout

  # Critic model config.
  - critic@critic: dp_critic

  # Reward model config.
  - reward_model@reward_model: dp_reward_model
```

如actor相关的配置被放到了config/actor/dp_actor.yaml

实际加载时，优先级：命令行里设置的参数 > ppo_trainer.yaml下面设置的具体参数 > 几个分散的配置文件定义的参数

### 框架内置的配置-加载原理

配置的yaml文件都位于/verl/verl/trainer/config

脚本文件位于/verl/verl/trainer，通过Hydra引入yaml文件到脚本里

以`ppo_main.py`为例：

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # ...
```

通过@hydra引入/config/ppo_trainer.yaml。在实际运行脚本时，默认加载此脚本，也可以设置`--config-dir $CONFIG_DIR`来自定义yaml配置文件，会覆盖@hydra定义的配置文件：

```bash
#!/bin/bash

# 设置你想使用的配置文件夹路径
# 你可以轻松地把这行改成 "experiment_configs_B"
CONFIG_DIR="experiment_configs_A"

echo "--- 正在使用配置文件夹: $CONFIG_DIR ---"

# 在 python 命令中使用 --config-dir 参数
# 它会告诉 Hydra 去 $CONFIG_DIR 文件夹里寻找 ppo_trainer.yaml
python3 -m verl.trainer.main_ppo \
  --config-dir $CONFIG_DIR \
  data.train_batch_size=256 # 你仍然可以继续使用其他参数来覆盖
```

### 配置详解（ppo_trainer.yaml）

由于参数过于多，这里只整理个人当前认为相对重要的部分，本部分个人认为最重要的是知道大概分为哪几类参数，具体细节查文档即可

主要参数类别：**data**-和数据集有关的参数，包括数据集路径、对应数据列名、prompt与response长度限制、最外层的batch size、对数据进行怎么样的中间处理等；**actor_rollout_ref**-策略、采样、参考模型有关参数，包括通用模型路径、actor模型的配置如单次前向传播的batch size大小和actor模型的优化器等、rollout模型的配置、refrence模型的配置；**critic**-状态价值模型相关配置；**reward_model**-奖励函数相关配置；**custom_reward_function**-自定义奖励函数配置；**algorithm**-具体采用的算法相关的配置

#### data: 数据相关配置

**核心文件路径：**

**`data.train_files`**：指定**训练集**的 `.parquet` 文件路径。可以是一个单独的文件路径字符串，也可以是一个包含多个路径的列表。路径可以是本地路径或 HDFS 路径。

**`data.val_files`**：指定**验证集**的 `.parquet` 文件路径。格式与 `train_files` 相同。

**数据格式与长度：**

**`data.prompt_key`**：指定在 Parquet 文件中，哪一列（哪个字段）是**输入提示 (prompt)**。**默认值**: `'prompt'`。

**`data.max_prompt_length`**：设置输入提示的最大 token 长度。所有提示都会被**向左填充 (left-padded)** 到这个长度。如果原始提示超过此长度，会根据 `truncation` 参数进行处理。

**`data.max_response_length`**：设置模型在 RL 训练的 rollout 阶段，**生成回答的最大 token 长度**。

`data.truncation`：定义当输入提示超过 `max_prompt_length` 时的处理策略。**可选值**：`'error'`: (默认) 直接报错；`'left'`: 截断左边（开头）；`'right'`: 截断右边（结尾）；`'middle'`: 从中间截断，保留开头和结尾部分。

**批处理与加载行为：**

**`data.train_batch_size`**：在一次完整的训练迭代中，从数据集中采样的**总样本数量**。这个总批次后续会被 PPO 算法拆分成更小的 `mini_batch` 和 `micro_batch`。

`data.shuffle`：是否在每个 epoch 开始时**打乱**数据加载的顺序。设置为`True` 或 `False`。

**特殊数据返回格式 (重要)**

**`data.return_raw_chat`**：设置为 `True` 时，数据加载器将返回**最原始的、未经处理的聊天数据**（即 `[{'role': ..., 'content': ...}]` 这样的 Python 列表），而不会应用任何聊天模板。

**`data.return_raw_input_ids`**： 一个**至关重要**的开关，用于处理**策略模型**和**奖励模型**聊天模板不一致的情况。当设为 `True` 时，框架会先用策略模型的分词器将 `input_ids` **解码**回文本，然后再用奖励模型的分词器**重新编码**，确保奖励模型收到它能正确理解的输入。

`data.return_full_prompt`：设置为 `True` 时，将返回应用了聊天模板之后的完整提示符字符串。

**性能与过滤**

`data.filter_overlong_prompts`：是否要过滤掉那些长度超过 `max_prompt_length` 的提示。**默认值**为`False` (不过滤)。

`data.filter_overlong_prompts_workers`：当 `filter_overlong_prompts` 开启时，可以使用多个进程来加速过滤过程，特别适用于超大规模数据集。

#### actor_rollout_ref: 关于actor、rollout、refrence三个模型相关配置

1. **`model` (共享模型配置)**

这部分定义了三个角色共同继承的基础模型信息。

- **`path`**: 指定基础模型的路径（本地或 Hugging Face Hub）。
- `external_libs`: 注册自定义模型或分词器时需要导入的额外 Python 库。
- `override_config`: 用于覆盖模型加载时的默认配置，比如 `dropout`。
- `enable_gradient_checkpointing`: (FSDP 专属) 梯度检查点，一种用计算换显存的技术，可以显著降低训练时的显存占用。
- `enable_activation_offload`: 将模型的激活值卸载到 CPU，进一步节省显存。
- `trust_remote_code`: 是否允许加载需要执行远程代码的模型。
- `use_remove_padding`: 一项性能优化，通过移除输入中的填充部分来提升训练效率。

2. **`actor` (演员/策略模型配置) **

这部分专门定义了我们**正在训练的 Actor 模型**的行为。

- **`strategy`**: 指定分布式训练策略，如 `fsdp` 或 `megatron`。
- **`ppo_mini_batch_size`**: PPO 算法进行一次参数更新时使用的样本数量（全局批次大小）。
- **`ppo_micro_batch_size_per_gpu`**: **(核心)** 在单张 GPU 上进行一次前向/后向传播的样本数量，用于梯度累积，是控制显存占用的关键参数。
- **`grad_clip`**: 梯度裁剪的阈值，防止梯度爆炸。
- **`clip_ratio`**: PPO 算法中用于限制策略更新幅度的裁剪比率。
- **`entropy_coeff`**: 熵损失的系数，用于鼓励策略探索。
- **`use_kl_loss`**: 是否在 Actor 的损失函数中直接计算 KL 散度损失（GRPO 等算法需要）。
- **`ppo_epochs`**: 使用同一批采样数据，对模型进行多少轮次的更新。
- **`optim`**: 定义 Actor 模型的优化器 (详见下方 `optim` 配置详解)。
- **`fsdp_config`**: FSDP 策略的详细配置，如是否开启参数卸载 (`param_offload`)、优化器状态卸载 (`optimizer_offload`) 等。
- **`checkpoint`**: 配置模型检查点需要保存哪些内容（如模型权重、优化器状态等）。
- **`tis_imp_ratio_cap`**: (高级) 用于截断重要性采样（Truncated Importance Sampling）的比率上限，以稳定训练。

3. **`rollout` (采样/推理模型配置) **

这部分定义了在**生成样本 (Rollout) 阶段**的行为，本质上是 Actor 模型在推理模式下的配置。

- `name`: 指定使用的推理引擎，如 `vllm` (高性能) 或 `hf` (Hugging Face 默认)。
- `temperaure`, `top_k`, `top_p`: 控制生成文本多样性的采样参数。
- `dtype`: 推理时使用的数据类型，如 `bfloat16`，应与 FSDP 的设置保持一致。
- **`gpu_memory_utilization`**: 控制 vLLM 引擎可以使用的 GPU **总显存**的比例。
- **`tensor_model_parallel_size`**: vLLM 使用的张量并行大小，用于将推理任务分布到多张 GPU 上。
- **`n`**: 每个 prompt 生成多少个不同的回答。当使用 GRPO、RLOO 等算法时，需要设置为大于 1 的值。
- `load_format`: 如何将 Actor 模型的权重加载到 vLLM 引擎中，`dummy_dtensor` 是 FSDP 环境下的常用选项。
- `calculate_log_probs`: (高级) 是否在 rollout 阶段就计算好 `log_probs`，某些高级算法（如 TIS）需要。
- `val_kwargs`: 在**验证阶段**专用的采样参数，通常会使用更确定的采样策略（如 `temperature: 0`）以获得可复现的评估结果。

4. **`ref` (参考模型配置) **

这部分定义了**固定的、不参与训练的 Reference 模型**的行为，它的主要作用是作为计算 KL 散度的基准。

- **`fsdp_config`**: 通常与 Actor 的配置类似，但**强烈建议开启 `param_offload: True`**，因为参考模型只进行前向传播，将其参数卸载到 CPU 可以为 Actor 和 Critic 节省大量宝贵的 GPU 显存。
- **`log_prob_micro_batch_size_per_gpu`**: 在计算 `log_prob` 时，单张 GPU 的微批次大小。

5. **`optim` (优化器配置) **

这部分详细定义了**模型训练时使用的优化器及其学习率策略**。它通常在 `actor` 和 `critic` 的配置块内部定义，因为它们各自需要独立的优化策略。

- **`lr`**: 学习率 (Learning Rate)，决定了每次参数更新的步长。
- `lr_warmup_steps`: 在训练开始阶段，学习率从 0 线性增长到设定 `lr` 值所需的步数。这有助于训练初期的稳定。
- `lr_warmup_steps_ratio`: 另一种设置预热步数的方式，按总训练步数的比例来计算。
- `warmup_style`: 预热阶段结束后的学习率变化策略。
  - `constant`: 学习率在预热后保持为 `lr` 不变。
  - `cosine`: 学习率按余弦曲线从 `lr` 衰减到最低值。
- `min_lr_ratio`: 当使用 `cosine` 衰减时，学习率可以衰减到的最低值占初始 `lr` 的比例。

#### critic: 状态价值函数相关配置

由官方文档，与actor模型的配置参数十分接近，由于GRPO不要critic模型，这里暂时不讨论。

#### reward_model: 奖励模型相关配置

这部分用于配置一个**基于神经网络模型的奖励函数**，它会学习并预测人类的偏好。

- **`enable`**: 一个布尔值 (`True` 或 `False`)。设为 `True` 时，框架才会加载并使用一个模型来打分。如果为 `False`，则完全依赖自定义函数或内置规则。
- **`model`**: 这是一个嵌套配置，定义了奖励模型本身。
  - **`path`**: 指定你训练好的奖励模型（必须是 `AutoModelForSequenceClassification` 类型）的路径。
  - **`input_tokenizer`**: 指定奖励模型的分词器路径。当奖励模型和策略模型的分词器/聊天模板不一致时，这是一个**必须设置**的关键参数。
  - **`trust_remote_code`**: 是否允许加载需要执行远程代码的模型。
- `micro_batch_size_per_gpu`: 在进行模型打分时，每张 GPU 处理的微批次大小。
- `max_length`: 奖励模型处理序列的最大长度。
- `reward_manager`: 定义奖励处理机制的管理器。`naive` 是默认选项，`prime` 则用于支持并行安全的验证函数。

#### custom_reward_function: 自定义奖励函数相关配置

这部分用于配置一个**基于代码规则的奖励函数**，让你用自己的 Python 逻辑来打分。

- **`path`**: 指向一个包含你的打分逻辑的 Python 文件（例如 `my_rewards.py`）。如果留空 (`null`)，框架会尝试使用内置的、基于 `data_source` 的自动匹配函数。
- **`name`**: 指定 `path` 文件中具体是哪个函数。如果你的函数名正好是 `compute_score`，则可以省略这个参数。

#### algorithm: 算法

这部分控制 PPO 算法在数学层面的核心行为。

- **`gamma`**: **折扣因子**。控制对未来奖励的重视程度，`1.0` 代表最有远见。
- **`lam`**: **GAE 参数**。在优势估计的偏差和方差之间进行权衡。
- **`adv_estimator`**: **优势估计器**。选择计算优势函数的具体算法，`gae` 是最常用的。
- **`use_kl_in_reward`**: **是否启用奖励内 KL 惩罚**。一个开关，决定 KL 惩罚是直接从奖励中扣除，还是作为损失函数的一部分。
- **`kl_penalty`**: **KL 散度计算方式**。选择计算 Actor 和 Reference 策略差异的具体数学方法。
- **`kl_ctrl`**: **KL 惩罚控制器**。
  - **`type`**: 控制器类型，`fixed` (固定系数) 或 `adaptive` (自适应调整)。
  - **`kl_coef`**: 惩罚系数，值越大，对模型“跑偏”的惩罚越重。
  - **`horizon` & `target_kl`**: `adaptive` 模式下的参数，分别代表调整时参考的步数和期望的 KL 目标值。

#### trainer: 训练器

这部分管理整个训练流程的宏观控制。

- **`total_epochs`**: **总训练轮次**。
- **`project_name`**: **项目名称**。用于在 WandB 等工具中对实验进行顶层分组。
- **`experiment_name`**: **实验名称**。为本次具体的运行提供一个独特的描述性名字。
- **`logger`**: **日志记录器**。选择记录实验日志的工具，如 `console` (控制台) 和 `wandb`。
- **`nnodes`**: **节点数**。分布式训练中使用的机器数量。
- **`n_gpus_per_node`**: **每节点 GPU 数**。
- **`save_freq`**: **保存频率**。每隔多少次迭代保存一次模型检查点。
- **`val_before_train`**: **训练前验证**。是否在训练开始前先进行一次验证。
- **`test_freq`**: **验证频率**。每隔多少次迭代进行一次验证。
- **`critic_warmup`**: **评论家预热**。在正式开始策略学习前，单独训练 Critic 模型的迭代次数。
- **`resume_mode`**: **断点续训模式**。
  - `auto`: 自动从最新的检查点恢复。
  - `disable`: 禁用，从头开始训练。
  - `resume_path`: 从指定的路径恢复。
- **`resume_from_path`**: 当 `resume_mode` 为 `resume_path` 时，指定检查点的具体路径。
