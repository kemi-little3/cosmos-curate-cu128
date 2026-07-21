# 日志 SDK 中文说明

这份文档是对 `logging_sdk_bundle(1)` 中英文说明的中文整理，方便在 `cosmos-curate-cu128` 仓库里直接查阅和接入。

相关原始文档：

- [INSTALL.md](/mlp-01/linyuxi/docs/logging_sdk_bundle(1)/INSTALL.md)
- [DARWINMIND_DATA_ENGINE_LOGGING.md](/mlp-01/linyuxi/docs/logging_sdk_bundle(1)/DARWINMIND_DATA_ENGINE_LOGGING.md)
- [log_sdk/README.md](/mlp-01/linyuxi/docs/logging_sdk_bundle(1)/log_sdk/README.md)

## 这套东西是干什么的

这个 bundle 提供两个本地 SDK：

- `env_sdk`：提供环境信息，当前默认 `get_env()` 返回 `prod`
- `log_sdk`：负责结构化日志输出

日志会同时：

- 打到 stdout
- 可选推送到 Loki
- 最后在 Grafana 里查询

## 你们项目的固定配置

项目日志指南里已经指定了当前项目应使用的标签：

- `app = darwinmind_data_engine`
- `service_name = cosmos_curate`

Loki 和 Grafana 地址是：

- Grafana UI: `http://192.168.9.132:3000`
- Loki push/query API: `http://192.168.9.132:3100`
- Loki push endpoint: `http://192.168.9.132:3100/loki/api/v1/push`

Grafana 登录信息：

- 用户名: `admin`
- 密码: `zjulearning123`

## 安装顺序

先装 `env_sdk`，再装 `log_sdk`。

```bash
pip install /path/to/env_sdk
pip install /path/to/log_sdk
```

开发模式可以用 editable install：

```bash
pip install -e /path/to/env_sdk
pip install -e /path/to/log_sdk
```

## 最小接入代码

这就是文档里的最小用法，适合先做 smoke test：

```python
from log_sdk import LogConfig, get_client

client = get_client(
    LogConfig(
        app="darwinmind_data_engine",
        service_name="cosmos_curate",
        enable_loki=True,
        loki_url="http://192.168.9.132:3100/loki/api/v1/push",
    )
)

client.info("service started", module="bootstrap")
client.warn("cache miss", cache_key="dataset:42")
client.error("curation timeout", task_id="task_1001")
```

## 这些日志该怎么理解

这不是“SDK 自己的日志”，而是你们服务运行时的日志，只是通过这个 SDK 统一格式化并推送到 Loki。

适合记录的内容：

- 服务启动和退出
- 某个 batch 开始和结束
- 跳过原因
- 重试
- 异常和失败
- 关键业务字段，比如 `task_id`、`dataset_id`、`request_id`、`module`

不建议做的事情：

- 把高基数字段做成固定 label
- 每一行都打一个超高频、没信息量的日志
- 随便修改 `app` 和 `service_name`

## 推荐的日志字段

文档建议把业务字段作为关键字参数传入，例如：

```python
client.info(
    "batch finished",
    module="caption_stage",
    task_id="task_123",
    dataset_id="dataset_456",
)
```

固定标签只保留：

- `app`
- `env`
- `service_name`

其中 `env` 会由 `env_sdk.get_env()` 自动加上，当前默认是 `prod`。

## 怎么在 Grafana 查

1. 打开 Grafana: `http://192.168.9.132:3000`
2. 登录 `admin / zjulearning123`
3. 进入 `Explore`
4. 选择数据源 `Loki`
5. 输入查询语句

常用查询：

查这个服务的所有日志：

```logql
{app="darwinmind_data_engine", service_name="cosmos_curate"}
```

只看错误日志：

```logql
{app="darwinmind_data_engine", service_name="cosmos_curate", level="error"}
```

查包含关键字的日志：

```logql
{app="darwinmind_data_engine", service_name="cosmos_curate"} |= "timeout"
```

## 怎么验证接入是否成功

最稳的验证顺序是：

1. 先在 Python 里发一条唯一日志
2. 确认 stdout 能打印 JSON
3. 再在 Loki 里查这条唯一日志

如果你看到 Loki 查到了同一条 `message`，就说明链路已经通了。

## 这份文档建议怎么用

如果你要接入 `cosmos-curate-cu128`：

- 先把这份中文文档当作项目内说明
- 再按原始英文文档核对参数
- 先接入口日志，再考虑 batch 级别的细日志
