# CS336 2026 春季作业 5：对齐

完整作业说明见讲义：[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)。

关于安全对齐、指令微调和 RLHF 的补充作业是完全可选的，见 [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)。

如果你发现作业讲义或代码有问题，可以提交 GitHub issue，或直接开 pull request 修复。

## 配置

和前几次作业一样，本项目使用 `uv` 管理依赖。

1. 先安装除 `flash-attn` 外的所有包，再安装全部包（`flash-attn` 的安装比较特殊）：

```
uv sync --no-install-package flash-attn
uv sync
```

2. 运行必需的单元测试：

``` sh
uv run pytest tests/test_grpo.py
```

初始状态下，所有测试都应因 `NotImplementedError` 失败。
要把你的实现接入测试，请完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。
