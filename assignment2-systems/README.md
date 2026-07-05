# CS336 2026 春季作业 2：系统

完整作业说明见讲义：[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)。

如果你发现作业讲义或代码有问题，可以提交 GitHub issue，或直接开 pull request 修复。

## 配置

本目录结构如下：

- [`./cs336-basics`](./cs336-basics)：包含 `cs336_basics` 模块及其 `pyproject.toml` 的目录。该模块提供课程组给出的作业 1 语言模型实现。如果你想使用自己的作业 1 实现，可以用自己的实现替换这个目录。
- [`./cs336_systems`](./cs336_systems)：这个目录基本是空的。你将在这里实现优化后的 Transformer 语言模型。可以从作业 1 的 `cs336-basics` 中复制需要的代码作为起点。此外，你还会在这个模块中实现分布式训练和优化。

目录大致应如下所示：

``` sh
.
├── cs336_basics  # 名为 cs336_basics 的 Python 模块
│   ├── __init__.py
│   └── ... 从作业 1 带来的 cs336_basics 模块中的其他文件 ...
├── cs336_systems  # TODO(you): 你为作业 2 编写的代码
│   ├── __init__.py
│   └── ... TODO(you): 作业 2 需要的其他文件或文件夹 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): 作业 2 需要的其他文件或文件夹 ...
```

如果要使用自己的作业 1 实现，可以用自己的实现替换 `cs336-basics` 目录，或修改外层 `pyproject.toml`，让它指向你的实现。

0. 本项目使用 `uv` 管理依赖。可以运行下面的命令，确认 `cs336-basics` 包中的代码可以被访问：

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` 会根据 `pyproject.toml` 自动安装依赖。

## 提交

提交时运行 `./test_and_make_submission.sh`。该脚本会安装代码依赖、运行测试，并创建包含输出的 gzip 压缩 tar 包。课程组应能解压你提交的压缩包，并运行 `./test_and_make_submission.sh` 来验证测试结果。
