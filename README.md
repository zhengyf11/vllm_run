# vllm_run

本项目提供一组 Python 脚本，用于生成和检查 vLLM DeepSeek-V4-Pro on NVIDIA BNT3 的 Docker 与 PD 分离部署命令。当前实现参考 `sglang_run` 的架构：命令生成与执行解耦，Web/API 只返回命令和提示，不直接启动模型服务。

## 项目目标

- 使用指定 Docker 镜像作为 vLLM 服务运行环境。
- 封装 Docker 容器启动参数，支持 dry-run 检查。
- 提供本地 Web/API 命令生成器，覆盖 `host_check`、`docker_run`、`prefill`、`decode`、`pd_proxy` profile。
- 按 Issue #7 体现 DeepSeek-V4-Pro on BNT3 的 vLLM 0.21.0 注意事项。

## Issue #7 注意事项

Issue: <https://github.com/zhengyf11/vllm_run/issues/7>

1. **Mooncake 依赖**：`vllm-0.21.0` 镜像未预置 `mooncake-transfer-engine`，生成的 PD 命令响应会提供安装提示：

   ```bash
   pip install mooncake-transfer-engine-cuda13
   ```

2. **PD 分离前 unset 代理**：现网节点可能配置代理，生成的 `prefill`/`decode` 响应会在 `proxy_unsets` 与 `combined_shell` 中包含 `unset HTTP_PROXY`、`unset HTTPS_PROXY` 等代理清理片段，避免 KV 传输失败。
3. **PD 分离默认不启用 MTP**：当前 vLLM 0.21.0 下 PD 分离部署的 MTP 有精度和可靠性问题，因此 `prefill`/`decode` profile 会强制 `enable_mtp=false`，不会生成 speculative/MTP 参数。

## Docker 容器启动脚本

`scripts/docker_run.py` 使用 `argparse` 生成并执行 Docker 命令。脚本会先处理旧容器：

```bash
docker rm <name>
```

然后启动新容器：

```bash
docker run -itd ... <image>
```

默认参数包括 `--user=0`、`--privileged`、`--ipc=host`、`--network host`、`--runtime=nvidia`、`--gpus all`、`--ulimit memlock=-1:-1`、`-v /sys/fs/cgroup:/sys/fs/cgroup:ro`、`-e NVIDIA_VISIBLE_DEVICES=all`、`--entrypoint /bin/bash`。

### Dry-run 查看命令

测试或检查参数时请使用 `--dry-run`，该模式只打印将执行的命令，不会调用 Docker：

```bash
python scripts/docker_run.py --dry-run \
  --name vllm_dsv4 \
  --image vllm/vllm-openai:v0.21.0-ubuntu2404 \
  --model-path /data/DeepSeek-V4-Pro \
  --host-workdir /home/zhengyingfei \
  --root-path /root \
  --runtime nvidia \
  --gpus all \
  --nvidia-visible-devices all
```

### 真实启动容器

确认 dry-run 输出无误后，去掉 `--dry-run` 即可真实执行 Docker 命令：

```bash
python scripts/docker_run.py \
  --name vllm_dsv4 \
  --image vllm/vllm-openai:v0.21.0-ubuntu2404 \
  --model-path /data/DeepSeek-V4-Pro \
  --host-workdir /home/zhengyingfei \
  --root-path /root \
  --runtime nvidia \
  --gpus all \
  --nvidia-visible-devices all
```

## 本地 vLLM 命令生成 Web/API

启动本地命令生成器：

```bash
python scripts/vllm_command_web.py --host 127.0.0.1 --port 6070
```

浏览器打开：<http://127.0.0.1:6070>

> 安全边界：`scripts/vllm_command_web.py` 只生成命令，不执行 Docker/vLLM 服务；`POST /api/command` 响应始终包含 `executed: false`。

### 支持的 profile

| Profile | 用途 |
| --- | --- |
| `host_check` | 生成 Mooncake 依赖、代理变量、MTP 禁用规则的检查/修复提示。 |
| `docker_run` | 生成 `docker run` argv/shell，用于进入 vLLM 容器环境。 |
| `prefill` | 生成 vLLM PD prefill 节点命令，默认 DeepSeek-V4-Pro、Mooncake KV producer。 |
| `decode` | 生成 vLLM PD decode 节点命令，默认 DeepSeek-V4-Pro、Mooncake KV consumer。 |
| `pd_proxy` | 单独生成 PD 部署前需要 unset 的代理变量片段。 |

### API 示例

获取默认参数：

```bash
curl 'http://127.0.0.1:6070/api/defaults?profile=prefill'
```

生成 prefill 命令：

```bash
curl -sS -X POST 'http://127.0.0.1:6070/api/command' \
  -H 'Content-Type: application/json' \
  -d '{"profile":"prefill"}'
```

响应字段包括：

- `profile`：归一化后的 profile。
- `config`：最终配置。
- `command`：argv list，便于测试和审查。
- `shell_command`：格式化后的单条命令。
- `proxy_unsets`：PD 部署前清理代理的 shell 片段。
- `install_hints`：Mooncake 依赖安装提示。
- `shell_hints`：Issue #7 规则说明。
- `combined_shell`：可复制执行前再人工审查的组合 shell；其中说明性 Issue note 会以 `# ` 注释形式出现，避免复制到 shell 后被当成命令执行。
- `executed`：固定为 `false`。

### 默认 prefill/decode 命令要点

- 入口：`vllm serve /data/DeepSeek-V4-Pro`。
- 模型名：`--served-model-name DeepSeek-V4-Pro`。
- 并行默认：`--tensor-parallel-size 8`、`--pipeline-parallel-size 1`。
- KV 传输：`--kv-transfer-config` 使用 `MooncakeStore`，prefill 为 `kv_producer`，decode 为 `kv_consumer`。
- MTP：默认强制关闭，不生成 speculative/MTP 参数。

## 测试

本仓库测试使用标准库 `unittest`，不需要额外依赖：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

语法编译检查：

```bash
python -m py_compile scripts/docker_run.py scripts/vllm_command_web.py tests/test_docker_run.py tests/test_vllm_command_web.py
```
