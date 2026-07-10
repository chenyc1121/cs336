# Lecture 9: Scaling Laws Basics

> 课程来源：`context/09 - Lecture 9  Scaling Laws 重制版.json`
>
> 本讲介绍 scaling laws 的基本思想：用小规模实验建立模型性能随参数量、数据量和计算量变化的规律，从而指导大规模训练。

## 0. 本讲学习目标

- 理解 scaling law 为什么是语言模型工程工具。
- 区分 model size、dataset size、compute budget。
- 理解 loss 随 scale 呈幂律下降的经验规律。
- 理解 compute-optimal training 和 Chinchilla 结论。
- 理解 isoFLOP curves 的实验设计。

## 1. Scaling law 的问题定义

大模型训练成本太高，不能像小实验一样随意调参。Scaling law 试图回答：

- 给定 compute budget，模型应该多大？
- 应该训练多少 tokens？
- 更大模型和更多数据之间如何权衡？
- 小规模实验能否预测大规模 loss？

形式上，validation loss 可被看作以下变量的函数：

```text
L = f(N, D, C)
```

其中：

- `N`: 参数量；
- `D`: 训练 tokens 数；
- `C`: compute budget。

## 2. 幂律关系

许多语言模型实验观察到，loss 随规模增加近似满足 power law：

```text
L(x) = L_inf + A * x^(-alpha)
```

这里：

- `L_inf`: 不可约误差或极限 loss；
- `A`: 常数；
- `alpha`: scaling exponent；
- `x`: 可以是参数量、数据量或计算量。

幂律的工程意义是：在 log-log 图上关系近似线性，便于外推。

## 3. Model-limited 与 data-limited

如果模型太小，即使给很多数据也无法充分吸收信息，这是 model-limited。

如果模型很大但训练 tokens 太少，模型未被充分训练，这是 data-limited 或 undertrained。

训练设计要在两者之间平衡：

```text
too small model + too much data -> capacity bottleneck
too large model + too little data -> undertrained model
```

## 4. Compute-optimal training

给定训练 FLOPs `C`，我们希望选择 `N` 和 `D` 使 loss 最小。

粗略地，Transformer 训练 FLOPs 与参数量和 token 数成正比：

```text
C ≈ k * N * D
```

问题变为：

```text
minimize L(N, D)
subject to N * D = fixed
```

Compute-optimal 不是最大模型，也不是最多数据，而是二者的最优组合。

## 5. Kaplan scaling 与 Chinchilla 修正

早期 scaling laws 倾向于训练更大的模型、相对较少 tokens。后来 Chinchilla 工作显示，许多大模型其实 undertrained：在相同 compute 下，较小模型配更多 tokens 可能更优。

Chinchilla 风格的核心直觉：

```text
compute-optimal scaling requires growing model size and training data together
```

这改变了现代 LLM 训练策略：不只是追求更大参数量，也重视训练 token 数。

## 6. IsoFLOP curves

IsoFLOP curve 是在固定 compute 下训练多个不同大小模型，观察哪个模型 loss 最低。

实验设计：

```text
fixed compute C
train models with different N
adjust D so N*D roughly fixed
measure validation loss
find best N for that C
```

对多个 compute budgets 重复，就能拟合 compute-optimal frontier。

## 7. Batch size 与 learning rate

Scaling law 实验必须控制训练超参数，否则观测到的 loss 可能不是规模规律，而是优化失败。

重要因素：

- batch size；
- learning rate；
- warmup；
- weight decay；
- training duration；
- optimizer；
- initialization。

Batch size 太小会导致训练慢或噪声大；太大可能降低 sample efficiency 或需要调整 learning rate。

## 8. Loss 与能力

Validation loss 是稳定、连续、可预测的指标。但用户关心的是 reasoning、coding、instruction following 等能力。

关系：

- loss 通常与许多下游能力相关；
- 但某些能力可能有 threshold 或非线性表现；
- benchmark 可能受数据污染、prompt 和评估方式影响；
- 因此 scaling law 不能完全替代 evaluation。

## 9. Scaling laws 的用途

实际用途包括：

- 预算规划；
- 选择模型大小；
- 选择训练 tokens；
- 预测最终 loss；
- 判断训练曲线是否异常；
- 比较 architecture 或 dataset 的效率；
- 降低大规模训练失败风险。

## 10. 本讲关键术语

- Scaling law: 性能随规模变化的经验规律。
- Power law: 幂律关系。
- Model size: 参数量。
- Data size: 训练 token 数。
- Compute budget: 可用计算量。
- Compute-optimal: 给定计算预算下 loss 最低的配置。
- Undertrained: 模型参数多但训练 tokens 不足。
- IsoFLOP curve: 固定 FLOPs 下比较不同模型大小的曲线。
- Perplexity: cross-entropy loss 的指数形式。
- Scaling exponent: 幂律下降速度参数。

## 11. 易错点

- 不要把最大模型等同于 compute-optimal 模型。
- 不要忽略数据量。参数大但 tokens 少可能 undertrained。
- 不要把 validation loss 与所有能力完全等同。
- 不要用优化失败的实验拟合 scaling law。
- 不要过度相信远距离外推，外推需要谨慎。

## 12. 自测题

1. Scaling law 解决什么工程问题？
2. Power law 在 log-log 图上有什么特征？
3. Compute-optimal training 的约束是什么？
4. Undertrained model 是什么？
5. Chinchilla 结论改变了什么直觉？
6. IsoFLOP curve 如何构造？
7. 为什么 batch size 会影响 scaling 实验？
8. Loss 和 benchmark 能力是什么关系？
9. 为什么不能只训练最大模型？
10. Scaling law 外推有什么风险？

## 13. 自测题答案

1. 它帮助用小规模实验预测大规模训练表现，从而选择模型大小、数据量和预算分配。
2. 幂律关系在 log-log 图上近似为直线，斜率对应 scaling exponent。
3. 在固定 compute budget 下选择参数量 `N` 和 token 数 `D`，通常近似满足 `N*D` 固定。
4. 参数量很大但训练 tokens 不足，模型没有充分学习，给更多数据可能比增大模型更有效。
5. 它强调许多模型数据训练不足；在相同 compute 下，较小模型加更多 tokens 可能更优。
6. 固定一个 FLOP budget，训练多个不同 `N` 的模型，并相应调整 `D`，比较 validation loss。
7. Batch size 改变优化动态、梯度噪声和硬件效率；设置不当会让实验反映优化问题而非规模规律。
8. Loss 通常与能力相关，但不是一一对应；某些能力可能非线性出现，benchmark 也有噪声和污染问题。
9. 因为最大模型可能训练数据不足，在固定 compute 下 loss 反而不如较小但训练更充分的模型。
10. 远距离外推可能失效，架构、数据、优化器或训练 regime 改变都会破坏已拟合规律。
