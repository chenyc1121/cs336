# Lecture 18: Inference Systems Guest Lecture Dan Fu

> 课程来源：`context/18 - Lecture 18  Guest Lecture Dan Fu 重制版.json`
>
> 本讲从系统角度看模型部署：训练好模型后，如何把它变成高吞吐、低延迟、成本可控的 inference service。

## 0. 本讲学习目标

- 理解 inference serving stack 的主要组件。
- 理解 prefill/decode 在系统中的不同瓶颈。
- 理解 KV cache memory management 和 PagedAttention 的动机。
- 理解 continuous batching、prefix caching、speculative decoding、custom kernels。
- 理解 latency、throughput、utilization、cost 的工程取舍。

## 1. 从模型到服务

训练完成的 checkpoint 不是产品。Serving system 还需要：

- model loading；
- tokenizer；
- request scheduler；
- batching；
- KV cache manager；
- GPU kernels；
- distributed inference；
- streaming output；
- monitoring；
- fault handling。

最终目标是在约束下服务用户：

```text
low latency + high throughput + low cost + reliable behavior
```

## 2. Serving workload 的特点

请求具有高度不规则性：

- prompt 长度不同；
-输出长度不同；
-到达时间随机；
-用户可能中途取消；
-部分请求共享 prefix；
-不同请求有不同 sampling 参数。

这使 inference 系统比离线 batch evaluation 更复杂。

## 3. Prefill 与 decode 的系统差异

Prefill：

- 处理完整 prompt；
- matmul 较大；
- 更容易 compute-bound；
- TTFT 关键。

Decode：

- 每步生成一个 token；
- 需要读取 KV cache；
- 常 memory-bandwidth-bound；
- ITL 关键；
- 持续时间长，主导服务成本。

系统需要同时优化两者。

## 4. KV cache 管理

KV cache 是 inference 服务的核心状态。问题：

- 长上下文占用大量显存；
- 请求长度不同导致碎片；
- 请求结束后需要回收；
- batching 时每个序列位置不同；
- 多 GPU 时 cache placement 更复杂。

如果 KV cache 管理不好，GPU 显存会被浪费，batch size 受限，吞吐下降。

## 5. PagedAttention 的动机

PagedAttention 借鉴操作系统分页思想，把 KV cache 切成 blocks/pages，而不是为每个请求分配连续大块。

优点：

- 减少内存碎片；
- 支持动态增长；
- 易于回收；
- 便于 continuous batching；
- 支持共享 prefix。

直觉：

```text
logical token positions -> physical KV blocks
```

类似 virtual memory 的地址映射。

## 6. Continuous batching

Serving 中请求不断到达和完成。Continuous batching 每个 decode step 都重组 active batch：

- 新请求可加入；
- 完成请求退出；
- 取消请求释放 cache；
- scheduler 维持 GPU 饱和。

这比静态 batching 更适合在线服务，但要求 KV cache 和 scheduler 协同。

## 7. Prefix caching

许多请求共享相同前缀，例如 system prompt、few-shot examples、长文档开头。

Prefix caching 复用已计算的 prefix KV cache，减少 prefill 成本。

适用场景：

- chat system prompt；
- retrieval-augmented generation 中重复文档；
- agent loop 中固定工具说明；
- batch evaluation 中相同 prompt prefix。

限制：

- cache 命中率决定收益；
- prefix 需要完全一致或可安全共享；
- memory 管理更复杂。

## 8. Custom kernels

Inference 系统常写 custom kernels：

- fused attention；
- decoding attention；
- layernorm/RMSNorm；
- sampling；
- quantized matmul；
- KV cache layout conversion。

原因是通用框架 overhead 在 decode 阶段很明显，小操作很多，memory traffic 关键。

## 9. Quantization 与 serving

Serving 中 quantization 常用于降低成本：

- weight-only quantization；
- activation quantization；
- KV cache quantization；
- FP8 inference。

收益：

- 更大模型放入同样显存；
- 更大 batch；
- 更低 bandwidth；
- 更低成本。

风险：

- 质量下降；
- kernel 不成熟；
- 不同层敏感度不同；
- calibration 复杂。

## 10. Distributed inference

大模型可能单卡放不下，需要 tensor parallel 或 pipeline parallel 推理。

推理并行与训练不同：

- batch 和 sequence 动态变化；
- decode latency 对通信敏感；
- KV cache 需要跨设备管理；
- small batch 下通信开销更明显。

因此 serving parallelism 不只是复用训练并行策略。

## 11. Cost 与工程取舍

Inference service 的关键指标：

- TTFT；
- ITL；
- tokens/s；
- requests/s；
- GPU utilization；
- memory utilization；
- cost per million tokens；
- tail latency。

优化往往有 trade-off：

- 大 batch 提高吞吐但增加 latency；
- quantization 降低成本但可能损失质量；
- prefix cache 节省计算但占显存；
- speculative decoding 多用一个 draft model 但减少 target steps。

## 12. 本讲关键术语

- Serving stack: 模型部署系统全栈。
- Scheduler: 决定请求如何 batch 和执行的组件。
- KV cache manager: 管理 cache 分配、映射和回收。
- PagedAttention: 分页式 KV cache 管理与 attention 执行思想。
- Continuous batching: 动态在线 batching。
- Prefix caching: 复用相同 prompt prefix 的 KV cache。
- Tail latency: 高百分位延迟。
- Weight-only quantization: 只量化权重。
- Distributed inference: 多 GPU 推理。
- Cost per token: 单位 token 服务成本。

## 13. 易错点

- 不要把 checkpoint 当成服务系统。部署还需要 scheduler、cache、kernel 和监控。
- 不要只优化平均 latency。tail latency 对用户体验和 SLA 更关键。
- 不要忽略 KV cache fragmentation。
- 不要认为训练并行策略能直接用于推理。
- 不要认为 GPU utilization 高就一定用户体验好，latency 可能很差。

## 14. 自测题

1. Serving stack 包含哪些组件？
2. 为什么在线请求比离线 batch 更难处理？
3. Decode 阶段为什么常成为服务瓶颈？
4. KV cache fragmentation 是什么问题？
5. PagedAttention 的核心思想是什么？
6. Continuous batching 如何提高吞吐？
7. Prefix caching 适合哪些场景？
8. Inference custom kernels 主要优化什么？
9. Distributed inference 为什么不同于 distributed training？
10. Tail latency 为什么重要？

## 15. 自测题答案

1. 包括 tokenizer、model runtime、scheduler、batching、KV cache manager、GPU kernels、streaming、monitoring 和故障处理。
2. 请求长度、到达时间、输出长度和采样参数都不规则，还可能取消或共享 prefix。
3. 每步只生成一个 token，计算量小但要读取大量历史 KV cache，容易受 memory bandwidth 限制。
4. 不同请求长度导致显存中出现难以复用的小空洞，降低可用 batch size 和显存利用率。
5. 把 KV cache 分成固定大小 blocks/pages，通过映射管理逻辑位置到物理块，减少碎片并支持动态 batching。
6. 每个 decode step 动态加入新请求、移除完成请求，让 GPU 始终有足够 active sequences。
7. system prompt、固定工具说明、重复 few-shot、RAG 中重复文档和共享长前缀的请求。
8. 减少小操作 overhead、降低 HBM 读写、融合操作、支持量化和优化 decode attention。
9. 推理请求动态、batch 小且 latency 敏感，KV cache 状态复杂；训练更规则，主要追求吞吐。
10. 用户和 SLA 关心高百分位请求是否很慢；平均延迟好不代表体验稳定。
