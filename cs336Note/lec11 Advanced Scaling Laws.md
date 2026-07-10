# Lecture 11: Advanced Scaling Laws

> 课程来源：`context/11 - Lecture 11   Scaling Laws 重制版.json`
>
> 本讲进入 scaling laws 的实践细节：如何在真实训练中外推超参数、控制 scale drift，并理解 overtraining、MoE scaling 等问题。

## 0. 本讲学习目标

- 理解 scaling law 从论文公式到真实训练之间的落差。
- 解释 hyperparameter transfer 的困难。
- 理解 learning rate、batch size、optimizer 参数随规模变化的漂移。
- 理解 μP / maximal update parameterization 的动机。
- 区分 compute-optimal 训练与 inference-aware overtraining。
- 理解 MoE scaling 中 active parameters 与 total parameters 的差异。

## 1. Scaling law 的实践问题

基础 scaling law 假设可以干净地比较不同规模模型。但真实训练中，改变模型规模会同时改变：

- 最优 learning rate；
- batch size；
- initialization scale；
- optimizer dynamics；
- normalization statistics；
- hardware efficiency；
- data mixture effects。

因此外推不仅是拟合 `loss = f(scale)`，还包括让不同规模实验处在“可比”的训练 regime 中。

## 2. Hyperparameter transfer

Hyperparameter transfer 指从小模型实验中选择超参数，并迁移到大模型。

困难在于：

- 小模型最优 learning rate 未必是大模型最优；
- batch size 和 gradient noise scale 改变；
- depth/width 比例改变；
- activation scale 和 update scale 可能随宽度变化；
- 大模型训练太贵，无法大范围 grid search。

目标是找到一套 parameterization，使超参数在 scale 上更稳定。

## 3. Learning rate scaling

Learning rate 控制参数更新幅度。模型变宽、变深后，同样 learning rate 可能导致：

- update 太大，训练不稳定；
- update 太小，训练不足；
- 不同层更新尺度不一致。

因此 scaling 实验常需要对 learning rate 做 sweep，并观察 loss 曲线是否正常。

## 4. Batch size scaling

Batch size 增大可提高硬件利用率和降低梯度噪声，但也会影响优化。

相关概念：

- gradient noise scale；
- critical batch size；
- sample efficiency；
- hardware throughput。

过大 batch 可能导致每个 token 提供的优化信息利用不足；过小 batch 又可能硬件效率差、训练不稳定。

## 5. μP 的动机

μP / maximal update parameterization 试图让不同宽度模型的训练动态更可比。它关注参数初始化和 learning rate 如何随 width 缩放，使 hidden activations 和 parameter updates 在宽度变化时保持合理尺度。

直觉：

```text
standard parameterization: optimal hyperparameters drift with width
μP: choose scaling rules so hyperparameters transfer better
```

这对大模型很重要，因为你希望在小模型上调好的 learning rate 能更可靠地迁移到大模型。

## 6. Optimizer 与 schedule

现代 LLM 常用 AdamW 或相关变体。关键超参数包括：

- learning rate；
- betas；
- epsilon；
- weight decay；
- warmup steps；
- decay schedule；
- gradient clipping。

Schedule 也影响 scaling。常见策略：

- cosine decay；
- linear decay；
- WSD / warmup-stable-decay；
- constant with decay。

不同 schedule 会影响中途 loss、最终 loss 和 continued training 的可行性。

## 7. Overtraining 与 inference-aware scaling

Compute-optimal 训练只考虑训练 compute。但产品模型还要考虑 inference cost。

如果模型较小，即使训练 tokens 超过 compute-optimal，也可能因为推理更便宜而更适合部署。

这称为 overtraining 或 inference-aware training：

```text
train smaller model on more tokens
pay more training compute
save repeated inference compute
```

当模型要服务大量用户时，推理成本可能主导总成本。

## 8. MoE scaling

MoE 中有两个参数概念：

- total parameters: 所有 experts 加起来的总参数。
- active parameters: 单个 token 实际经过的参数。

Scaling MoE 时不能简单套 dense model 的参数量规律，因为 per-token compute 主要由 active parameters 决定，而容量又受 total parameters 影响。

MoE 的 scaling 优势是增加 total capacity 而不等比例增加 active compute；难点是 routing、load balancing 和 communication。

## 9. 数据与 scaling

Scaling law 依赖数据分布。改变数据质量、去重程度、代码比例、多语言比例，都可能改变 loss 曲线。

因此拟合 scaling law 时应保持：

- tokenizer 一致；
- data mixture 一致；
- preprocessing 一致；
- evaluation set 一致；
- optimization regime 一致。

否则观察到的差异可能来自数据，而不是规模。

## 10. 失败模式

Scaling 外推可能失败的原因：

- 超参数没有正确 transfer；
- 大模型出现训练不稳定；
- 数据分布改变；
- tokenizer 改变；
- hardware bottleneck 改变；
- benchmark 被污染；
- loss 曲线未收敛；
- 外推距离太远。

实践上需要多条曲线、多种预算和异常监控，而不是盲目相信单次拟合。

## 11. 本讲关键术语

- Hyperparameter transfer: 小模型超参数迁移到大模型。
- Scale drift: 最优超参数随规模变化而漂移。
- μP: maximal update parameterization。
- Critical batch size: 继续增大 batch 收益下降的区域。
- Warmup: 训练初期逐渐增大学习率。
- WSD: warmup-stable-decay schedule。
- Overtraining: 相对 compute-optimal 使用更多 tokens 训练模型。
- Inference-aware scaling: 把部署推理成本纳入训练规模选择。
- Active parameters: 每个 token 实际使用的参数。
- Total parameters: 模型全部参数。

## 12. 易错点

- 不要把 scaling law 当成不需要调参的公式。
- 不要把小模型最优 learning rate 直接搬到大模型。
- 不要只按 total parameters 比较 MoE 和 dense model。
- 不要认为 compute-optimal 一定是产品最优。
- 不要忽略数据分布变化对 scaling 曲线的影响。

## 13. 自测题

1. Hyperparameter transfer 为什么困难？
2. Learning rate 为什么会随规模出现 drift？
3. μP 想解决什么问题？
4. Batch size 过大有什么潜在问题？
5. WSD schedule 包含哪几个阶段？
6. 为什么产品模型可能选择 overtraining？
7. MoE 中 active parameters 和 total parameters 有何不同？
8. 数据分布变化为什么会影响 scaling law？
9. Scaling 外推失败的常见原因有哪些？
10. 为什么需要多条 isoFLOP 曲线而不是单次实验？

## 14. 自测题答案

1. 因为模型规模改变会改变更新尺度、激活尺度、梯度噪声、最优 learning rate 和硬件行为，小模型最佳配置未必适合大模型。
2. 宽度、深度和初始化改变后，同样 learning rate 产生的相对参数更新大小不同，可能过大或过小。
3. 它通过特定初始化和 learning-rate scaling 规则，使不同宽度模型的训练动态更可比，从而改善超参数迁移。
4. 可能降低 sample efficiency，使每个 token 的边际优化价值下降，并且需要调整 learning rate。
5. Warmup、stable 和 decay：先升学习率，中段保持，后段衰减。
6. 较小模型推理更便宜；如果服务请求很多，额外训练成本可能被长期推理节省抵消。
7. Total parameters 是全部 experts 和共享层参数；active parameters 是单个 token 实际计算经过的参数。
8. Loss 曲线反映模型对特定数据分布的拟合难度，数据质量和组成改变会改变可预测关系。
9. 包括超参数迁移失败、训练不稳定、数据变化、tokenizer 变化、硬件瓶颈变化和外推距离过远。
10. 多条曲线可估计不同 compute 下的最优规模趋势，单次实验无法区分噪声、优化失败和真实 scaling 规律。
