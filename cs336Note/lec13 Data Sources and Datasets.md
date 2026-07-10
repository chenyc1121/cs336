# Lecture 13: Data Sources and Datasets

> 课程来源：`context/13 - Lecture 13  Data (Sources, Datasets) 重制版.json`
>
> 本讲讨论 pretraining data 从哪里来，以及为什么数据是语言模型训练中最关键也最难标准化的部分。

## 0. 本讲学习目标

- 理解 pretraining data 对模型能力的决定性作用。
- 了解 web、books、code、papers、Wikipedia、forums 等数据来源。
- 理解 Common Crawl、C4、The Pile 等数据集的基本定位。
- 理解版权、许可、terms of service 和 fair use 的工程影响。
- 理解 dataset mixture、token budget、epoch 和 deduplication。

## 1. 数据为什么关键

语言模型通过 next-token prediction 从数据中学习知识、语言模式、推理痕迹、代码结构和对话风格。

架构和优化器决定模型能学什么；数据决定模型实际学到什么。

数据影响：

- 知识覆盖；
- 语言和文化覆盖；
- 代码能力；
- 数学和推理能力；
- 安全风险；
- 偏见和有害内容；
- benchmark contamination；
- 模型风格。

## 2. Web data

Web 是最大规模文本来源。Common Crawl 是常见原始来源之一，它定期抓取网页并提供大规模网页快照。

Web data 的优点：

- 规模巨大；
- 主题多样；
- 覆盖真实语言使用；
- 包含问答、论坛、教程、文档。

缺点：

- 噪声高；
- HTML boilerplate 多；
- spam、SEO、广告多；
- 质量差异极大；
- 版权和隐私问题复杂。

## 3. Books 与长文本

Books 提供高质量长文本：

- 结构完整；
- 语言质量高；
- 长距离叙事和论证；
- 对长上下文建模有价值。

风险：

- 版权争议；
- 获取授权困难；
- 体量相对 web 小；
- 数据来源透明性问题。

## 4. Code data

Code 对现代 LLM 很重要，因为它训练模型学习：

- 精确语法；
- 函数组合；
- API 使用；
- 测试和调试；
- 形式化结构。

主要来源包括 GitHub、package repositories、documentation 和 Q&A。

关键问题：

- license 是否允许训练；
- 代码质量参差不齐；
- secrets、PII 和密钥泄漏；
- benchmark contamination；
- repository-level duplication。

## 5. Scientific papers 与 technical text

论文、教材、技术文档有高信息密度，适合训练模型的专业知识和推理风格。

处理难点：

- PDF extraction；
- 数学公式；
- 表格；
- 引文和页眉页脚；
- copyright。

文本抽取质量会直接影响训练质量。

## 6. Wikipedia 与 curated sources

Wikipedia 等 curated sources 质量较高、结构较清晰，但规模有限。它们适合提供事实性、百科式知识，但不能覆盖所有真实语言分布。

高质量 sources 常用于提高数据 mixture 的质量基线。

## 7. Dataset mixture

训练集通常是多来源混合：

```text
web + books + code + papers + wiki + forums + Q&A + synthetic data
```

Mixture ratio 会影响模型能力。例如：

- code 比例高，编程能力强，但自然语言风格可能变化；
- 多语言比例高，非英语能力提升；
- 数学推理数据多，推理 benchmark 改善；
- 低质量 web 太多，会浪费 compute。

Mixture 是经验性设计，需要通过 evaluation 反馈迭代。

## 8. Token budget 与 epoch

训练通常按 tokens 计量：

```text
training budget = number of tokens seen
```

Epoch 表示完整遍历数据集一次。如果数据集很大，训练可能不到一个 epoch；如果高质量数据有限，模型可能多次看到同一数据。

重复训练的风险：

- memorization；
- overfitting；
- benchmark leakage；
- 多样性下降。

## 9. Deduplication

Web 数据中重复非常常见。Deduplication 的目标是删除重复或近重复内容。

类型：

- exact dedup：完全相同文本；
- near dedup：高度相似文本；
- document-level dedup；
- paragraph/line-level dedup；
- repository-level dedup。

收益：

- 减少 memorization；
- 提高有效 token 多样性；
- 降低 benchmark contamination；
- 避免重复内容浪费 compute。

## 10. 法律与社会问题

数据使用涉及：

- copyright；
- license；
- terms of service；
- fair use；
- privacy；
- PII；
- user consent。

这些问题不是训练后的附加事项，而会影响数据能否收集、能否公开、模型能否部署。

## 11. 本讲关键术语

- Pretraining data: 预训练语料。
- Common Crawl: 大规模网页抓取数据来源。
- Dataset mixture: 多数据源混合比例。
- Token budget: 训练中计划消耗的 token 数。
- Epoch: 遍历数据集一次。
- Deduplication: 去重。
- Near duplicate: 近重复文档。
- PII: personally identifiable information。
- License: 数据或代码使用许可。
- Data contamination: 评估数据进入训练集。

## 12. 易错点

- 不要认为数据越多越好。低质量数据会浪费 compute。
- 不要忽略许可和版权。能抓到不等于能使用。
- 不要把 web data 当作干净文本。它通常是 HTML、模板和噪声。
- 不要只做 exact dedup。近重复同样严重。
- 不要把 mixture ratio 当成固定真理，它需要评估驱动。

## 13. 自测题

1. 为什么数据会决定模型能力？
2. Web data 的优点和缺点是什么？
3. Code data 有哪些特殊风险？
4. PDF 数据为什么难处理？
5. Dataset mixture 影响哪些能力？
6. Token budget 和 epoch 有什么区别？
7. Deduplication 为什么能提高训练效率？
8. Exact dedup 和 near dedup 有何不同？
9. 为什么 license 会影响模型训练？
10. 为什么高质量数据有限时要警惕重复训练？

## 14. 自测题答案

1. 模型通过预测数据中的 token 学习知识、语言、代码和推理模式；没有出现在数据中的能力很难凭空产生。
2. 优点是规模大、主题广、真实；缺点是噪声、重复、spam、HTML 模板、版权和隐私问题多。
3. 包括许可证限制、密钥泄漏、PII、低质量代码、重复仓库和代码 benchmark 污染。
4. PDF 是排版格式而非纯文本，抽取公式、表格、页眉页脚和阅读顺序都容易出错。
5. 影响代码、多语言、数学、专业知识、风格、安全和对话能力。
6. Token budget 是计划训练看到的 token 总量；epoch 是遍历整个数据集一次。
7. 它减少重复 token，让固定 compute 用在更多不同信息上，同时降低记忆和污染风险。
8. Exact dedup 删除完全相同文本；near dedup 删除高度相似但不完全一致的文本。
9. 许可证决定数据是否允许被复制、处理、再分发或用于商业/模型训练。
10. 多次看到同样数据会增加 memorization 和 overfitting，降低有效数据多样性。
