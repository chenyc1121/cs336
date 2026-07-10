# Lecture 8: Large Scale Training Parallelism

> 课程来源：`context/08 - Lecture 8  Parallelism 重制版.json`
>
> 本讲继续并行主题，重点是大规模训练中如何组合 data、tensor、pipeline、FSDP 和 expert parallelism，并根据硬件拓扑做取舍。

## 0. 本讲学习目标

- 理解为什么真实大模型训练通常混合多种 parallelism。
- 理解 tensor parallel、pipeline parallel、FSDP、expert parallel 的组合逻辑。
- 解释通信/计算重叠与 topology-aware placement。
- 理解大规模集群中高速链路和低速链路的不同角色。

## 1. 并行策略的组合问题

单一并行策略往往不能满足大规模训练：

- DDP 简单但每卡冗余完整状态。
- FSDP 省显存但引入参数 all-gather。
- Tensor parallel 解决单层计算和参数切分，但通信频繁。
- Pipeline parallel 降低每卡层数，但有 bubble。
- Expert parallel 是 MoE 必需，但引入 all-to-all。

因此真实训练常把设备组织成多维网格：

```text
data parallel dimension
tensor parallel dimension
pipeline parallel dimension
expert parallel dimension
```

## 2. Hardware topology

集群不是一个均匀网络。常见层次：

- 单 GPU 内部：HBM、SM、tensor cores。
- 单机多 GPU：NVLink/NVSwitch 或 PCIe。
- 多机之间：InfiniBand 或以太网。
- 更大规模：rack、pod、cluster。

并行策略应尽量把通信频繁的部分放在高速链路内。例如 tensor parallel 通信频繁，适合放在同一节点或 NVLink 域内；data parallel 通信相对粗粒度，可跨节点。

## 3. Tensor parallel in practice

Tensor parallelism 通常在 Transformer 层内切分：

- Attention QKV projection 可按 heads 或 hidden dimension 切。
- MLP up projection 可 column split。
- MLP down projection 可 row split，并需要 reduce。

典型模式：

```text
column parallel -> local compute -> concatenate or keep sharded
row parallel -> local partial result -> all-reduce
```

Tensor parallel 的优点是降低每卡参数和计算；缺点是每层都有通信，因此需要高速互联。

## 4. Pipeline parallel in practice

Pipeline parallelism 按层切分模型。调度常见元素：

- microbatching；
- forward pipeline；
- backward pipeline；
- 1F1B schedule；
- activation transfer between stages。

主要 trade-off：

- stages 多，每卡模型更小；
- stages 多，跨 stage communication 增加；
- microbatches 多，bubble 减少；
- microbatches 多，activation memory 和调度复杂性增加。

## 5. FSDP 与 parameter sharding

FSDP 适合处理训练状态冗余：

- parameters sharded；
- gradients sharded；
- optimizer states sharded。

在大规模训练中，FSDP 常作为 data parallel 维度的增强版本。它让每个 data-parallel rank 不必保存完整模型状态。

关键成本是每层 forward/backward 前后的 all-gather 和 reduce-scatter。是否划算取决于显存压力和通信带宽。

## 6. Expert parallelism for MoE

MoE 模型有许多 experts。Expert parallelism 把 experts 分布到多个 devices。

流程：

```text
token hidden states
-> router chooses experts
-> all-to-all dispatch tokens
-> experts compute
-> all-to-all gather outputs
```

MoE 的优势是 active compute 可控，total parameters 很大；系统难点是 all-to-all 和 load balancing。

## 7. Sequence parallelism 与 activation partitioning

某些操作可以沿 sequence dimension 切分，以减少 activation memory。例如 normalization、dropout 或部分 residual 操作可以在 `[B,T,D]` 的 `T` 维上分片。

Sequence parallelism 常与 tensor parallelism 配合，用来降低每卡 activation 存储压力。

## 8. Communication overlap

高效训练不能只减少通信，还要把通信隐藏在计算后面。

常见 overlap：

- backward 计算某层时，同步后面层的 gradients；
- all-gather 下一层参数，同时计算当前层；
- reduce-scatter gradients 与 backward computation 重叠。

理想状态是 communication time 被 compute time 覆盖。但如果通信太大或计算太小，overlap 不充分，训练会被网络限制。

## 9. 经验性选择规则

实际选择常遵循以下原则：

- 先用 data parallel 扩吞吐。
- 如果模型状态放不下，用 FSDP/ZeRO。
- 如果单层矩阵太大或每卡算力不够，用 tensor parallel。
- 如果层数太多或单卡仍放不下，用 pipeline parallel。
- 如果是 MoE，用 expert parallel，并关注 all-to-all。
- 把频繁通信的 parallel dimension 放在最快链路中。

## 10. 大规模训练的非算法问题

大规模训练还需要处理：

- checkpointing 和恢复；
- straggler 与节点故障；
- deterministic reproducibility；
- dataloader throughput；
- logging 和 monitoring；
- cluster scheduling；
- evaluation during training。

这些问题不改变模型公式，但会决定训练能否稳定完成。

## 11. 本讲关键术语

- Hybrid parallelism: 多种并行方式组合。
- Topology-aware placement: 根据硬件网络拓扑放置并行维度。
- Tensor parallel group: 共同切分单层张量的 ranks。
- Pipeline stage: 负责部分层的设备组。
- Expert parallel group: 持有不同 MoE experts 的 ranks。
- Sequence parallelism: 沿序列维切分 activation。
- Communication overlap: 用计算掩盖通信耗时。
- 1F1B: 一前向一反向的 pipeline 调度。
- Straggler: 拖慢整体同步的慢节点或慢任务。

## 12. 易错点

- 不要认为更多 GPU 自动更快。通信和同步可能抵消收益。
- 不要把所有通信都同等看待。all-to-all、all-reduce、all-gather 的模式不同。
- 不要把 tensor parallel 放在慢网络上，它通常通信频繁。
- 不要忽略 activation memory，参数 sharding 不一定解决所有显存问题。
- 不要只看平均吞吐，长时间训练还要看稳定性和故障恢复。

## 13. 自测题

1. 为什么大模型训练常用 hybrid parallelism？
2. Tensor parallel 为什么通常需要高速互联？
3. Pipeline parallel 的主要效率损失是什么？
4. FSDP 主要解决哪类内存问题？
5. Expert parallelism 的主要通信模式是什么？
6. Communication overlap 的目标是什么？
7. 为什么并行策略要 topology-aware？
8. Sequence parallelism 主要节省什么？
9. Data parallel 和 FSDP 的关系是什么？
10. 大规模训练为什么需要 checkpointing？

## 14. 自测题答案

1. 因为单一策略无法同时解决显存、计算、通信和模型结构问题；组合策略可以分别处理不同瓶颈。
2. 它在每层内部就需要 all-reduce 或 all-gather，通信频率高，慢网络会严重拖慢训练。
3. Bubble，即流水线启动和结束时部分 stage 空闲；跨 stage activation 传输也有开销。
4. 它减少 parameters、gradients 和 optimizer states 在 data-parallel ranks 上的冗余。
5. All-to-all：tokens 根据 router 结果发送到持有对应 experts 的设备，再收集输出。
6. 让通信与计算同时发生，尽量把通信时间隐藏在计算时间内。
7. 因为不同链路带宽和延迟差异巨大；频繁通信必须放在高速链路中，否则成为瓶颈。
8. 主要节省 activation memory，尤其是沿 sequence dimension 可分片的中间张量。
9. FSDP 可以看作 sharded data parallelism：仍有数据并行语义，但训练状态被切分。
10. 大规模训练时间长、故障概率高；checkpointing 允许从中间状态恢复，避免一次故障导致全部计算损失。
