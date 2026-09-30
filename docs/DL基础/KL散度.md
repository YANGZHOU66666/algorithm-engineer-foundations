# KL散度

理解：用分布Q【我们的估计】来表示分布P【真实分布】时，所带来的信息损失。衡量两个分布之间的差异

- 非对称
- 一定>0

离散形式：$$D_{KL}(P \vert{}\vert{} Q) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

连续形式：$$D_{KL}(P \vert{}\vert{} Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{p(x)}{q(x)}\right) dx$$



## 前向和反向KL

对于真实分布P(x)和我们的估计分布Q(x)，

Forward KL：$$D_{KL}(P \vert{}\vert{} Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

Reverse KL：$$D_{KL}(Q \vert{}\vert{} P) = \sum Q(x) \log\left(\frac{Q(x)}{P(x)}\right)$$

优化（最小化）Forward KL时，Q(x)会倾向覆盖所有P(x)概率较大的区间，因为分母趋于0时整个式子损失太大

优化（最小化）Reverse KL时，Q(x)会倾向覆盖P(x)尖峰，因为P(x)趋于0的地方无论Q(x)怎么调整，$\log\left(\frac{Q(x)}{P(x)}\right)$都会很大，所以直接系数为0。

## KL散度与交叉熵

$-logP(x)$表示事件x发生带来的“惊讶”程度。概率越低，惊讶程度越大

$信息熵 =\sum -P(x)\log{P(x)}$ 是系统在客观上所固有的最低“平均惊讶量”（或最低编码成本的底线）

$交叉熵 =\sum -P(x)\log{Q(x)}$ 是用我们的估计分布Q(x)来表示P(x)时，带来的总“平均惊讶量”

$KL散度 = \sum P(x)log{\frac{P(x)}{Q(x)}}=交叉熵 - 信息熵$，表示用我们的估计分布Q(x)表示P(x)相较于用P(x)本身，亏了多少“平均惊讶量”

