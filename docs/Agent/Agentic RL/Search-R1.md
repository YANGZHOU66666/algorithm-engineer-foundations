# Search-R1

核心思路：用不同的特殊token标记思考（<think></think>）、搜索词（<search></search>）、搜索结果（<information></information>）、最终答案（<answer></answer>）。每次输出完<search></search>之后停下来调用搜索API，直到返回搜索结果后填入<information>中，继续输出后续内容。

最多搜索4轮，超出的时候再search（貌似）会回填一句话让LLM停止这么干。如果搜索不到4轮直接输出<answer>，提前停止。

奖励：直接基于规则看最终的<answer></answer>里有没有规则到最终ground truth的内容

额外引入一个验码算子，对非LLM推理得到的token不计算损失：

$$I(y_t) = \begin{cases} 1, & \text{如果 } y_t \text{ 是大模型生成的 Token（如 think, search, answer 里的内容）} \\ 0, & \text{如果 } y_t \text{ 是搜索引擎返回的 Retrieved Token（information 里的内容）} \end{cases}$$



prompt：

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. 
You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. 
For example, <answer> xxx </answer>. Question: {question}
```

