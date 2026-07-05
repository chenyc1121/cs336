# CS336 2026 春季作业 3：Scaling

完整作业说明见讲义：[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf)。

如果你发现作业讲义或代码有问题，可以提交 GitHub issue，或直接开 pull request 修复。

## 本地配置

安装依赖：

```sh
uv sync --extra server
```

如果只需要使用客户端接口，可以运行：

```sh
uv sync
```

本地运行时，将 `A3_API_KEY` 设为你的 8 位学生 ID 或本地测试用 key：

```sh
export A3_API_KEY=06123456
```

原始课程环境使用托管训练 API。若在本地运行，请改用本地 API：

```text
http://127.0.0.1:8000
```

启动本地服务后，文档和仪表盘地址为：

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/dashboard
```

[`./examples/client_example.ipynb`](./examples/client_example.ipynb) 提供了提交和查看训练运行的示例。使用本地 API 时，请将示例中的远程 API 地址改为 `http://127.0.0.1:8000`。

## 本地下载数据

原始下载脚本包含 Modal 入口。若在本机运行，直接执行脚本即可：

```sh
uv run python scripts/1_download_tokenized_data.py
```

该脚本会准备 tokenized DCLM 数据，数据量和计算量都很大。只做本地调试时，建议先在代码中调小 chunk 数量或使用较小的本地样本，确认流程跑通后再扩大规模。

## 本地训练

```sh
uv run python cs336_scaling/training/run.py
```

训练代码默认使用 JAX，并会根据本机可用设备运行。没有 GPU 时可用于功能调试，但大规模训练会非常慢。

## 本地运行 API 和调度器

本地 API 需要数据库连接和内部 API key。准备一个本地 PostgreSQL 数据库，然后设置：

```sh
export DATABASE_URL_DEV="postgresql://USER:PASSWORD@127.0.0.1:5432/cs336_scaling"
export INTERNAL_API_KEY="SOMEKEY"
export API_BASE_URL="http://127.0.0.1:8000"
export LAUNCH_MODE="local"
```

启动 API：

```sh
uv run fastapi run
```

另开一个终端启动调度器：

```sh
uv run dispatcher
```

如果你只是完成作业中的本地实验，通常不需要连接 Stanford 托管服务或 Modal；优先使用上面的本地命令。
