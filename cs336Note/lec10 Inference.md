# Lecture 10: Inference

> 课程来源：`context/10 - Lecture 10  Inference 重制版.json`
>
> 本讲讨论训练完成后的模型如何高效生成文本。核心是 prefill、decode、KV cache、latency、throughput 和 serving。

## 0. 本讲学习目标

- 区分 prefill 和 decode 两个 inference 阶段。
- 理解 KV cache 的结构和成本。
- 解释 latency、throughput、batch size 的取舍。
- 理解 MQA/GQA、quantization、speculative decoding、continuous batching 的作用。
- 理解为什么 Transformer 原始结构并非 inference-optimal。

## 1. Inference 的基本过程

Autoregressive LM 每次生成一个 token：

```text
prompt -> model -> next token
prompt + next token -> model -> next token
...
```

训练时可以并行预测所有位置；推理 decode 时必须串行生成，因为第 `t+1` 个 token 依赖第 `t` 个生成结果。

## 2. Prefill 与 decode

Inference 分两阶段：

- Prefill: 输入完整 prompt，计算所有 prompt tokens 的 hidden states 和 KV cache。
- Decode: 每次输入上一步生成的新 token，读取历史 KV cache，生成下一个 token。

Prefill 特点：

- 序列长度较长；
- 可并行处理 prompt tokens；
- 更像训练前向。

Decode 特点：

- 每步只有一个新 token；
- 必须读取所有历史 KV；
- 常受 memory bandwidth 限制；
- latency 对用户体验关键。

## 3. KV cache

Self-attention 中，每层每个 token 都产生 K/V。推理时历史 K/V 不会改变，可以缓存。

如果没有 KV cache，每生成一个 token 都要重新计算整个上下文的 K/V，成本很高。

KV cache 大小近似：

```text
layers * batch * sequence_length * kv_heads * head_dim * 2
```

其中 `2` 表示 K 和 V。

长上下文、多 batch、多层模型都会显著增加 KV cache memory。

## 4. Latency 与 throughput

- Latency: 单个请求从输入到输出的时间。
- Time to first token / TTFT: prompt 进入后生成第一个 token 的时间，主要受 prefill 影响。
- Inter-token latency / ITL: 相邻输出 tokens 之间的时间，主要受 decode 影响。
- Throughput: 系统单位时间处理 tokens 或 requests 的数量。

大 batch 可以提高 throughput，但可能增加单请求 latency。Serving 系统需要在二者之间平衡。

## 5. Batch size 的作用

Prefill 阶段通常能从 batch 中获益，因为矩阵乘法更大、更能利用 GPU。

Decode 阶段每个请求每步只有一个 token，batching 也重要，但不同请求长度不同，会产生调度问题。

Serving scheduler 的目标：

- 合并多个请求，提高 GPU utilization；
- 避免短请求被长请求阻塞；
- 控制 latency SLA；
- 管理 KV cache memory。

## 6. Continuous batching

传统 batching 等一批请求全部完成再处理下一批，效率低。Continuous batching 允许新请求动态加入，已完成请求动态退出。

流程：

```text
active batch changes every decode step
finished sequences leave
new sequences enter
KV cache manager tracks memory blocks
```

这显著提高 serving throughput，但实现复杂。

## 7. MQA 与 GQA

KV cache 是 decode 瓶颈之一。Multi-query attention / MQA 让所有 query heads 共享 K/V；grouped-query attention / GQA 让一组 query heads 共享 K/V。

效果：

- 减少 K/V heads；
- 降低 KV cache memory；
- 降低 decode 时读取带宽；
- 保留较多 query heads 表达能力。

GQA 是现代 LLM inference-friendly architecture 的常见选择。

## 8. Quantization

Quantization 用更低 bit 表示权重或 activations，例如 int8、int4、FP8。

推理中常见收益：

- 减少模型权重显存；
- 降低 memory bandwidth；
- 提高吞吐；
- 允许更大 batch 或更长上下文。

风险：

- 精度下降；
- calibration 复杂；
- 不同层对量化敏感度不同；
- kernel 支持决定实际速度。

## 9. Speculative decoding

Speculative decoding 使用小模型 draft 多个 tokens，再用大模型并行验证。

直觉：

```text
small model proposes tokens
large model checks them
accept consecutive valid tokens
fallback when mismatch
```

如果 draft model 与 target model 分布接近，大模型一次 forward 可接受多个 tokens，从而减少 decode steps。

它不改变目标模型分布，但依赖 draft quality 和实现效率。

## 10. Inference-friendly architecture

Transformer 最初并不是专为 serving 设计。推理中主要问题：

- decode 串行；
- KV cache 随上下文增长；
- attention 读取历史 KV 带宽高；
- batch 中请求长度不一致；
- long context 占用大量显存。

因此架构和系统都在向 inference-aware 发展，例如 GQA、sliding window attention、state-space alternatives、KV cache compression。

## 11. 本讲关键术语

- Inference: 使用训练好的模型生成输出。
- Prefill: 处理 prompt 并建立 KV cache 的阶段。
- Decode: 自回归逐 token 生成阶段。
- KV cache: 缓存历史 tokens 的 key/value。
- TTFT: time to first token。
- ITL: inter-token latency。
- Throughput: 单位时间服务 tokens/requests 数。
- Continuous batching: 动态维护 active request batch。
- MQA/GQA: 减少 K/V heads 的 attention 变体。
- Quantization: 低精度表示权重或激活。
- Speculative decoding: 小模型草拟、大模型验证的加速方法。

## 12. 易错点

- 不要把训练吞吐和推理吞吐混淆。decode 是串行的。
- 不要忽略 KV cache。长上下文推理常被它限制。
- 不要认为 batch 越大越好。latency 和 memory 会变差。
- 不要认为 quantization 一定加速，kernel 和硬件支持很关键。
- 不要认为 speculative decoding 免费，它需要额外 draft model 和验证逻辑。

## 13. 自测题

1. Prefill 和 decode 的区别是什么？
2. KV cache 缓存的是什么？
3. 为什么 decode 常常 memory-bound？
4. TTFT 主要受哪个阶段影响？
5. Continuous batching 解决什么问题？
6. GQA 为什么减少推理成本？
7. Quantization 的主要收益是什么？
8. Speculative decoding 为什么能加速？
9. Batch size 对 latency 和 throughput 有什么影响？
10. 为什么说 Transformer 不天然 inference-optimal？

## 14. 自测题答案

1. Prefill 并行处理 prompt tokens 并建立 KV cache；decode 每次处理一个新 token，串行生成后续 tokens。
2. 每层 attention 中历史 tokens 的 key 和 value 向量。
3. 每步只生成一个 token，计算量相对小，但需要读取大量历史 K/V，性能受内存带宽限制。
4. TTFT 主要受 prefill 阶段影响，因为第一个输出 token 前必须处理完整 prompt。
5. 它允许请求动态加入和退出 batch，提高 GPU 利用率，避免静态 batch 被长请求拖住。
6. GQA 减少 K/V heads 数，从而减小 KV cache 和 decode 时的 K/V 读取带宽。
7. 降低权重和激活的存储与带宽需求，可能提升吞吐并允许更大 batch。
8. 小模型快速草拟多个 tokens，大模型一次并行验证，若接受多个 tokens，就减少了昂贵的大模型 decode steps。
9. 更大 batch 通常提高 throughput，但会增加排队、显存和单请求 latency。
10. 因为自回归 decode 串行，KV cache 随上下文增长，attention 每步读取历史，服务请求长度还高度不规则。
