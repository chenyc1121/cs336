# Lecture 16: Post-Training and RLVR

> 课程来源：`context/16 - Lecture 16  Post-Training - RLVR 重制版.json`
>
> 本讲讨论 RLVR / reinforcement learning with verifiable rewards，即使用可自动验证的奖励训练 reasoning model。

## 0. 本讲学习目标

- 理解 RLVR 与 RLHF 的差异。
- 理解可验证奖励为何适合数学、代码和形式化任务。
- 理解 rollout、CoT、verifier、policy update 的关系。
- 理解 PPO、GRPO、DPO 在该语境中的位置。
- 理解 reasoning post-training 与 general alignment 的阶段划分。

## 1. RLVR 的基本定义

RLVR 使用可验证奖励训练模型：

```text
prompt -> model rollout -> verifier checks answer -> reward
```

与 RLHF 不同，奖励不是来自人类偏好模型，而是来自自动验证器。例如：

- 数学答案是否正确；
- 代码是否通过 tests；
- 定理证明是否被 checker 接受；
- 格式化任务是否满足规则。

## 2. 为什么可验证奖励重要

Human preference reward 有噪声、成本高且容易被 reward hacking。可验证奖励在某些任务上更客观：

- 正确/错误清晰；
- 可大规模自动采样；
- reward 不依赖主观风格；
- 能支持大量 RL rollouts。

这使 reasoning 能力可以通过更多 test-time-like 采样和强化学习提升。

## 3. Rollout 与 Chain-of-Thought

在 reasoning 任务中，模型会生成完整解题过程和答案：

```text
question -> reasoning trace -> final answer
```

RLVR 通常对整个 rollout 给奖励。即使 verifier 只检查最终答案，训练信号也会影响模型产生中间 reasoning trace 的方式。

Chain-of-thought 不是 reward 本身，而是 policy 生成的中间行为。

## 4. Verifier

Verifier 是奖励来源。它可以是：

- exact answer checker；
- unit test runner；
- symbolic math checker；
- theorem prover；
- rule-based parser；
- another model plus constraints。

Verifier 的质量决定 RLVR 的上限。若 verifier 有漏洞，模型仍可能 reward hack。

## 5. PPO 在 RLVR 中

PPO 可以用于优化语言模型 policy，使其生成高 reward rollouts。目标通常包含：

- outcome reward；
- KL penalty；
- advantage estimate；
- clipping objective。

问题是 PPO 实现复杂，需要 value model、rollout sampling 和稳定性调参。

## 6. GRPO 的直觉

GRPO / group relative policy optimization 类方法通过同一 prompt 的多条 samples 构造相对优势。

直觉：

```text
sample several responses for same prompt
score each with verifier
normalize rewards within group
increase probability of above-average samples
decrease below-average samples
```

这样可以减少对单独 value model 的依赖，并适合可验证奖励场景。

## 7. Reward sparsity

可验证奖励常常稀疏：

```text
correct -> 1
wrong -> 0
```

稀疏奖励的问题：

- 很多 rollouts 得不到正反馈；
- 训练早期探索困难；
- 模型可能学到格式投机；
- verifier 只看最终答案，不知道中间哪步错。

缓解方式包括 curriculum、partial credit、better prompts、process supervision 或更强 base model。

## 8. RLVR 适合和不适合的任务

适合：

- 数学题；
- 代码题；
- 逻辑谜题；
- 形式化证明；
- 有确定答案的 QA；
- 规则明确的工具任务。

不适合或困难：

- 文风偏好；
- 开放式创作；
- 安全边界；
- 含糊的用户满意度；
- 多目标对话质量。

因此 RLVR 通常只解决 reasoning 的一部分，不替代 RLHF/SFT。

## 9. Training pipeline 中的位置

一种常见阶段划分：

1. Pretraining：学习语言和知识。
2. SFT：学习指令和基本格式。
3. RLVR：在可验证任务上强化 reasoning。
4. RLHF/DPO/safety tuning：改善人类偏好、安全、对话行为。

顺序和混合策略仍是研究问题，但 reasoning reward 和 general alignment reward 通常需要区分。

## 10. 本讲关键术语

- RLVR: reinforcement learning with verifiable rewards。
- Verifier: 自动检查答案正确性的系统。
- Rollout: 模型对 prompt 的完整采样输出。
- CoT: chain-of-thought。
- Outcome reward: 基于最终答案的奖励。
- Process reward: 基于中间步骤的奖励。
- PPO: proximal policy optimization。
- GRPO: group relative policy optimization。
- Advantage: 相对基线的奖励优势。
- Reward sparsity: 奖励信号稀疏。

## 11. 易错点

- 不要把 RLVR 当成所有 post-training 的替代品。
- 不要认为 verifier 完全可靠。验证器漏洞会导致 reward hacking。
- 不要把 CoT 等同于正确推理。CoT 可能看似合理但答案错误。
- 不要忽略 KL control。无约束 RL 可能破坏语言质量。
- 不要把 outcome reward 和 process supervision 混淆。

## 12. 自测题

1. RLVR 与 RLHF 的主要区别是什么？
2. 哪些任务适合可验证奖励？
3. Verifier 的作用是什么？
4. Outcome reward 的局限是什么？
5. GRPO 的基本思想是什么？
6. 为什么 RLVR 中 reward 可能稀疏？
7. KL penalty 在 RLVR 中有什么作用？
8. 为什么 RLVR 不适合所有对话质量优化？
9. Process reward 和 outcome reward 有何不同？
10. RLVR 在 post-training pipeline 中通常处于什么位置？

## 13. 自测题答案

1. RLVR 的奖励来自自动验证器；RLHF 的奖励通常来自人类偏好或由偏好训练出的 reward model。
2. 数学、代码、形式化证明、规则明确的问答和可自动判定正确性的任务。
3. 它检查模型 rollout 是否满足正确性标准，并产生 reward。
4. 它只告诉最终结果对错，不直接指出中间哪一步推理错误，训练信号较粗。
5. 对同一 prompt 采样多条回答，用组内相对奖励归一化，提升高于平均的回答概率。
6. 很多任务只有全对才给正分，错误回答都得 0，尤其训练早期正样本少。
7. 限制新 policy 偏离 reference model，防止语言质量退化和利用 verifier 漏洞。
8. 因为开放对话涉及风格、安全、帮助性和主观偏好，难以用单一自动 verifier 判定。
9. Outcome reward 评价最终答案；process reward 评价中间推理步骤。
10. 通常在 SFT 之后用于增强 reasoning，再结合偏好和安全训练塑造最终助手行为。
