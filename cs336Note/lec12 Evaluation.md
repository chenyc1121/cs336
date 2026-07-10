# Lecture 12: Evaluation

> 课程来源：`context/12 - Lecture 12  Evaluation 重制版.json`
>
> 本讲讨论如何评价语言模型。重点是 evaluation 不是单一分数，而是由任务、数据、prompt、metric、污染控制和人类偏好共同构成的测量系统。

## 0. 本讲学习目标

- 区分 language modeling loss、perplexity、benchmark score 和 human preference。
- 理解 benchmark 的组成：task、prompt、metric、aggregation。
- 理解 contamination、benchmark saturation 和 ecological validity。
- 比较 multiple-choice、free-form generation、arena、人类评估和 agentic evaluation。
- 根据评估目标选择合适 evaluation suite。

## 1. 为什么 evaluation 困难

语言模型的目标不是单一任务。一个模型可能同时用于：

- 文本补全；
- 问答；
- 推理；
- 编程；
- 对话；
- 工具调用；
- 长上下文阅读；
- 安全拒答。

因此不存在一个绝对充分的分数。Evaluation 的核心问题是：你要测量什么能力，以及测量结果是否真实反映目标使用场景。

## 2. Language modeling loss

训练中最直接的指标是 next-token cross-entropy loss：

```text
L = - average log p(x_t | x_<t)
```

Perplexity 是 loss 的指数形式：

```text
perplexity = exp(loss)
```

优点：

- 连续、稳定；
- 易于大规模计算；
- 与 scaling law 配合良好；
- 不依赖复杂 prompt。

缺点：

- 不直接等同于用户任务能力；
- 对 instruction following、安全、推理格式不够敏感；
- 受 evaluation data 分布影响很大。

## 3. Benchmark 的基本结构

一个 benchmark 通常由以下部分组成：

- dataset：问题或任务集合；
- prompt format：如何把任务转成模型输入；
- decoding setup：temperature、max tokens、stop tokens；
- metric：正确率、F1、BLEU、pass@k、胜率等；
- aggregation：如何把多个任务分数合成总分。

任何一个环节变化都可能改变结果，因此 benchmark score 必须和 evaluation protocol 一起解释。

## 4. Multiple-choice evaluation

MMLU 等 benchmark 常使用 multiple-choice 格式。

优点：

- 自动评分简单；
- 可覆盖大量知识领域；
- 结果稳定；
- 便于模型比较。

缺点：

- 与真实开放式使用场景不同；
- prompt 格式会显著影响结果；
- 模型可能通过排除法或 test-taking heuristics 得分；
- 数据污染风险较高。

## 5. Free-form generation evaluation

开放生成任务更接近真实使用，但评分更难。可能方法：

- exact match；
- regex / rule-based grading；
- unit tests，例如代码题；
- model-based judge；
- human evaluation。

关键难点：

- 正确答案可能不唯一；
- 表达方式多样；
- 评分器本身可能有偏差；
- 长答案更难稳定评估。

## 6. Coding evaluation

代码 benchmark 常使用 unit tests，如 HumanEval 风格。

常见指标：

- pass@1：一次生成通过测试的比例；
- pass@k：k 次采样中至少一次通过的估计概率。

优势是可自动验证；限制是测试覆盖不完整，模型可能生成通过公开测试但逻辑不稳的代码。

## 7. Human preference 与 arena

对话模型常用人类偏好或 arena 评估。给定两个模型回答，人类或 judge 选择更好者。

优点：

- 更接近用户感受；
- 可评价 helpfulness、style、format、harmlessness；
- 适合开放式回答。

缺点：

- 成本高；
- 噪声大；
- 人类偏好不总是等同事实正确；
- judge model 可能偏向特定风格。

## 8. Contamination

Contamination 指测试数据进入训练数据。后果是 benchmark 分数高估泛化能力。

常见来源：

- benchmark 题目被网页转载；
- GitHub 中含有测试集；
- 模型训练了 benchmark 解答；
- synthetic data 间接复制测试内容。

缓解方法：

- 去重和相似度检测；
- 使用新发布或私有测试集；
- dynamic evaluation；
- 关注真实部署反馈。

## 9. Benchmark saturation

当模型普遍在某 benchmark 上接近满分，该 benchmark 就失去区分度。

解决方式：

- 设计更难任务；
- 使用长上下文、多步骤、工具调用任务；
- 动态生成问题；
- 转向真实用户分布评估；
- 多维度报告而非单一总分。

## 10. Safety evaluation

Safety evaluation 包括：

- harmful request refusal；
- jailbreak robustness；
- bias/toxicity；
- privacy leakage；
- misinformation；
- dangerous capability。

安全评估往往与 helpfulness 存在 trade-off：过度拒答会降低可用性，过度服从会增加风险。

## 11. 本讲关键术语

- Evaluation: 对模型能力和行为的系统测量。
- Cross-entropy loss: next-token 负对数似然。
- Perplexity: `exp(loss)`。
- Benchmark: 标准化任务集合和评分协议。
- Metric: 评分函数。
- Contamination: 测试数据泄漏到训练数据。
- Saturation: benchmark 分数接近上限，失去区分度。
- Human preference: 人类对回答质量的比较判断。
- Arena: 模型匿名对战式评估。
- Ecological validity: 评估是否贴近真实使用场景。

## 12. 易错点

- 不要把 benchmark 排名当作完整模型质量。
- 不要忽略 prompt 和 decoding 设置。
- 不要把 loss 低直接等同于对话好。
- 不要忽略 contamination。
- 不要用单一 benchmark 判断安全、推理、代码和对话能力。

## 13. 自测题

1. Perplexity 和 cross-entropy loss 的关系是什么？
2. Benchmark score 为什么必须和 protocol 一起报告？
3. Multiple-choice evaluation 的优点和局限是什么？
4. 代码评估为什么常用 unit tests？
5. Human preference evaluation 有什么风险？
6. Contamination 为什么会高估模型能力？
7. Benchmark saturation 会带来什么问题？
8. Safety evaluation 为什么不能只看拒答率？
9. Model-based judge 有什么潜在偏差？
10. 如何根据目标选择 evaluation？

## 14. 自测题答案

1. Perplexity 是 cross-entropy loss 的指数形式，即 `perplexity = exp(loss)`。
2. 因为 prompt、decoding、metric 和 aggregation 都会影响分数；没有 protocol，分数不可复现也不可比较。
3. 优点是自动评分简单、稳定、覆盖广；局限是与开放式真实任务不同，容易受 prompt 和污染影响。
4. Unit tests 可自动验证程序行为，比文本相似度更接近代码正确性。
5. 人类偏好有噪声、成本高，且可能偏好流畅或自信的错误回答；judge model 也可能带风格偏差。
6. 如果模型训练过测试题或相似答案，分数反映记忆而非泛化。
7. 高分模型无法被区分，benchmark 失去诊断价值，需要更难或更真实的评估。
8. 因为安全与 helpfulness 有 trade-off；只看拒答率可能奖励无条件拒答的不可用模型。
9. 它可能偏好长答案、特定格式、与自身训练风格相似的回答，或无法可靠识别事实错误。
10. 先定义使用场景和风险，再选择对应任务、metric、污染控制和人工/自动评估组合。
