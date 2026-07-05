# MMLU 数据说明

本目录包含多任务测试使用的 dev、val 和 test 数据。

dev 数据集用于 few-shot learning，用来为模型提供上下文示例；test 集是评测问题的来源。

`auxiliary_training` 数据可以用于微调，这对不具备 few-shot 能力的模型尤其重要。该辅助训练数据来自其他 NLP 多选题数据集，例如 MCTest（Richardson et al., 2013）、RACE（Lai et al., 2017）、ARC（Clark et al., 2018, 2016）和 OBQA（Mihaylov et al., 2018）。

除非另有说明，问题所参照的人类知识截止到 2020 年 1 月 1 日。在更久以后的未来使用这些数据时，可以考虑在 prompt 中说明这些问题是为 2020 年的受众编写的。

--

如果这些数据对你的研究有帮助，请考虑引用该测试集以及它所使用的 ETHICS 数据集：

```bibtex
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
