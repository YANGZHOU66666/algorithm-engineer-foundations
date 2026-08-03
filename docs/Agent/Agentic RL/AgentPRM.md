# AgentPRM

解决Agent多轮工具调用，只能给一个最终奖励，没法给中间步骤奖励的问题。做了一个模型，给每次LLM工具调用打分。



建模：每次调用LLM生成的内容（think+tool）作为action，历史session作为state

目标：训练一个模型预测每个(state, action)的Q值，用于指导Agent训练时给奖励



## 具体做法

每一个iteration，做如下两大块操作：

一、训PRM模型。该模型对$(s_t, a_t)$的输出称为$\mathcal{M}_\phi(s_t, a_t)$，作为对该状态-动作的Q值预测

1. Agent模型采样一系列s-a-s-a...轨迹。每个轨迹看最终任务完成情况由奖励模型给个最终的reward分
2. PRM模型给每个Action打分，得到若干$\mathcal{M}_\phi(s_0, a_0),\mathcal{M}_\phi(s_1, a_1)\dots$
3. 算时序差分：对于中间步骤（t<T），有$$\delta_t = 0 + \gamma \mathcal{M}_\phi(s_{t+1}, a_{t+1}) - \mathcal{M}_\phi(s_t, a_t)$$；对于最后一步t=T，有$$\delta_T = r(\tau) - \mathcal{M}_\phi(s_T, a_T)$$
4. 算GAE优势：$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots$$，可以算出每个step的A。得到了$[\hat{A}_0, \hat{A}_1, \dots, \hat{A}_T]$ 
5. 根据定义，有：$$\hat{Q}_t = \hat{A}_t + \mathcal{M}_\phi(s_{t-1}, a_{t-1})$$，可以算出每个step的Q。得到了$[\hat{Q}_0, \hat{Q}_1, \dots, \hat{Q}_T]$
6. 损失函数包含两部分，一部分尽可能拟合所有A：$$\mathcal{L}_{A}(\phi) = \mathbb{E}_{s_t, a_t \sim \mathcal{D}_Q} \left[ \Big( \mathcal{A}_{\phi}(s_t, a_t) - \hat{A}_t \Big)^2 \right]$$，
   注意这里$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = Q(s_t, a_t) - Q(s_{t-1}, a_{t-1})$$。
   另一部分尽可能拟合所有Q：$$\mathcal{L}_{Q}(\phi) = \mathbb{E}_{s_t, a_t \sim \mathcal{D}_Q} \left[ \frac{1}{2} \Big( \mathcal{M}_{\phi}(s_t, a_t) - \hat{Q}_t \Big)^2 \right] \quad \text{}$$。
   总损失：$$\mathcal{L}_{\text{AgentPRM}}(\phi) = \underbrace{\mathcal{L}_Q(\phi)}_{\text{逼模型输出去贴近 }\hat{Q}} + \beta \times \underbrace{\mathcal{L}_A(\phi)}_{\text{逼模型的前后步差额去贴近 }\hat{A}}$$。

二、根据PRM模型指导Agent模型的RL训练

奖励信号还是只给最后一轮交互，但是用PRM给。相较于之前按照任务最终完成情况给reward，如果前面大多数轨迹都对但最后错，PRM给的奖励还是不低，而之前给的reward为0；如果只是瞎猜了正确回复，PRM给的奖励还是不高，而之前给的reward为0



以一、二两个步骤，重复多个iteration。