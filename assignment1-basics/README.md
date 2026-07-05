# CS336 2025 春季作业 1：基础

完整作业说明见讲义：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)。

如果你发现作业讲义或代码有问题，可以提交 GitHub issue，或直接开 pull request 修复。

## 配置

### 环境

本项目使用 `uv` 管理依赖，以保证环境可复现、可移植且易于使用。
推荐按 [`uv` 官方安装说明](https://github.com/astral-sh/uv#installation)安装，也可以运行 `pip install uv` 或 `brew install uv`。
也建议阅读 [`uv` 项目依赖管理指南](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)。

之后可以用下面的方式运行仓库中的任意 Python 文件：

```sh
uv run <python_file_path>
```

`uv` 会在需要时自动解析依赖并激活环境。

### 运行单元测试

```sh
uv run pytest
```

初始状态下，所有测试都应因 `NotImplementedError` 失败。
要把你的实现接入测试，请完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据

下载 TinyStories 数据和 OpenWebText 的一个子样本：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
