# RL知识地图

## 基石：马尔可夫决策过程

马尔可夫决策过程（MDP，Markov Decision Process）

RL 统一将世界抽象为一个由五个核心要素 $(S, A, P, R, \gamma)$ 组成的物理模型。

$S$：state状态，当前状态点

$A$：action动作，每个状态都可以执行一些动作，执行动作可以变化到下一个状态，并获得即时奖励

$P$：probability概率/策略，在某个状态下，当前策略选择每个动作都有一个概率

$R$：reward奖励，从某状态执行某动作会获得一个奖励。也可以叫$\pi$

$\gamma$：gamma衰减系数，描述较远的奖励信号对当前状态价值评估的加权权重。取值(0,1)。越接近0越注重当下，越接近1越注重远期



后面随着RL的演化，在不同地方引入了一些额外的概念，统一整理一下：

$return$回报：从某状态开始的一条执行路径上，能够获得所有奖励的加权和。从状态$S_1$开始，假设从状态$S_i$执行的动作为$a_i$，获得的奖励为$R_i$，$return = R_1+\gamma R_2+\gamma^2 R_3+\dots+\gamma^{n-1}R_n$

$\lambda$，lambda：$\lambda$-回报调节系数。取值(0,1)。越接近1，越看重长期奖励，方差变大；越接近0，越看重近期奖励，方差也变小

$Q(s,a)$：描述在s状态下执行a动作（如果后续按照某策略执行）能够获得的期望return。或者说，描述了当前策略下，从s出发且第一步选a动作，后面采样足够多路径，获得的return的平均值。$Q(s,a)=R(s,a)+\gamma V(s')$，其中$s'$是从$s$执行动作$a$到达的状态

$V(s)$状态价值：描述状态s下（如果按照某策略执行）能够获得的期望return。或者说，描述了在当前策略下，从s出发采样足够多路径，获得的return的平均值。$V(s)=\sum_{s下能执行的所有a} p(s,a)\times Q(s,a)$。

$A(s,a)$优势：描述在状态s下执行动作a相较于按策略执行动作能够获得的return优势。$A(s,a)=Q(s,a)-V(s)$

$\delta(s,a)$：delta，TD残差（Temporal Difference Residual）。在当前时间步，智能体迈出一步后所看到的“新现实”与“旧预估”之间的差额。本质上，它衡量的是大模型智能体执行某个动作或工具调用（Tool Call）后，环境的反馈和新处境是否超出了原先的预期（即带来了多少“惊喜”或“失望”）。$\delta(s_t,a_t) = R_t+\gamma V(s_{t+1}) - V(s_t)$

$A^{GAE}$：GAE优势。描述了如果按照某策略采样，从某状态开始，能够收获回报的期望。相较于直接算return，引入了V(s)参与计算，更好的控制算长程总回报的方差。这是因为，万一某个动作凑巧没做好，后面一串可能都reward很差，这样总return偏离很多；但如果引入V(s)，就算一个动作很差，也能在后面救回来。$A^{GAE}= \delta_0 + (\gamma\lambda)\delta_1 + (\gamma\lambda)^2\delta_2 + \dots + \mathbf{(\gamma\lambda)^{15}\delta_{15}} + \dots$



## 按照维护的变量类型分类

### Value-based RL

核心在于维护了一系列状态-动作对的价值（即$Q(s,a)$），但不维护每个状态下每个动作的策略概率。

实际推理时，通过$\epsilon-greedy$等策略，根据各个动作Q值大小现场算出每个动作的执行概率

代表方法：Q-Learning（状态动作都可枚举，可以把Q值维护成Q表）、DQN（状态不可枚举，动作可枚举）：

价值估计方法（不是具体的RL算法，而是根据采样结果估计状态/动作的价值，反哺更新Q表）：时序差分TD，Monte Carlo，GAE优势估计等

### Policy-based RL

核心在于维护了一系列状态-动作的概率，不维护状态和动作的价值。

代表方法：Reinforce，GRPO

### Actor-Critic

一个Actor，作为操作者，维护所有的策略，即状态-动作的概率

一个Critic，作为评估者，维护所有状态的价值

二者互相更新、户向较准

代表方法：PPO

### RL for LLM

Transformer可以视为一个策略模型，根据当前已有的token序列（状态），输出下一个token（动作）。因此一定是有policy的，且最终要优化的就是这个Policy

主流方法包括Actor-Critic的PPO、Policy-based的GRPO

## 按照On/Off-Policy分类

On-Policy：正在更新的策略是采样策略

Off-Policy：正在更新的策略不是采样策略

