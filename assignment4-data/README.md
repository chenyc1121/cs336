# CS336 2026 春季作业 4：数据

完整作业说明见讲义：[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)。

如果你发现作业讲义或代码有问题，可以提交 GitHub issue，或直接开 pull request 修复。

## 配置

本目录结构如下：

- [`./cs336_basics`](./cs336_basics)：该模块包含课程组给出的作业 1 语言模型实现。你会使用这份训练代码在过滤后的数据上训练语言模型。请不要修改训练逻辑，因为排行榜提交必须严格使用它。
- [`./cs336_data`](./cs336_data)：这个目录基本是空的。你将在这里实现数据过滤和处理代码。

目录大致应如下所示：

``` sh
.
├── cs336_basics  # 名为 cs336_basics 的 Python 模块
│   └── ... 优化后的训练实现 ...
├── cs336_data  # TODO(you): 你为作业 4 编写的代码
│   ├── __init__.py
│   └── ... TODO(you): 作业 4 需要的其他文件或文件夹 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): 作业 4 需要的其他文件或文件夹 ...
```

和前几次作业一样，本项目使用 `uv` 管理依赖：

```sh
uv sync
```

## 本地下载数据

原始课程环境中，完整数据位于 Modal 的 `/shared-data`。在本地运行时，代码会使用仓库下的 `local-shared-data` 目录作为共享数据目录，不需要配置 Modal volume。

只下载离线运行作业所需的文件：

```sh
uv run python scripts/download_data.py --offline-only
```

下载完整的本地数据：

```sh
uv run python scripts/download_data.py
```

下载完整非离线数据前，请先实现 [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) 中的 `is_english` 方法。

如果只想下载少于 2500 个 WET 文件，可以修改 [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) 中 `EnglishWetFiles` 的 `n_files`。

默认本地数据目录来自 [`./cs336_data/common.py`](./cs336_data/common.py)：`local-shared-data`。如需放到其他磁盘，可修改该文件中的本地路径。

## 本地训练

[`./scripts/train.py`](./scripts/train.py) 包含完整训练配置。传入 GPT-2 tokenizer 处理后的训练数据路径：

```sh
uv run python scripts/train.py --train-bin local-shared-data/your_data.bin --valid-bin local-shared-data/tokenized_paloma_c4_100_domains_validation.bin
```

该命令在本机单进程运行。没有 CUDA GPU 时训练会很慢；可以先用小数据和较少训练步数调试数据处理流程。

## 提交

提交时运行 `./test_and_make_submission.sh`。该脚本会安装代码依赖、运行测试，并创建包含输出的 zip 文件。课程组应能解压你提交的压缩包，并运行 `./test_and_make_submission.sh` 来验证测试结果。
