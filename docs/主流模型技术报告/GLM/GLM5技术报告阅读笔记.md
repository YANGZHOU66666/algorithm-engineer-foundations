# GLM5 Technical Report

## Abstract & Intro

训练pipeline：27T tokens Pre-training，有Code和Reasoning数据；Mid-training，将上下文长度从4K逐步添加到200K，关注长上下文的Agentic数据；Post-Training，SFT -> Reasoning RL -> Agentic RL -> General RL

1. 用了DSA，减少训练和推理开销，支持长上下文
2. 新的异步强化学习infra：解耦rollout和train，最大化GPU使用
3. 多样、长周期的交互中，通过RL学习Agent能力
4. 中国产GPU适配

