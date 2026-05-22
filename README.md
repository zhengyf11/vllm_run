# vllm_run

本项目目标是创建一组 Python 脚本，用于基于指定的 Docker 镜像启动容器，并在容器内执行 vLLM 运行命令，从而拉起模型推理服务。

## 项目目标

- 使用指定的 Docker 镜像作为 vLLM 服务运行环境。
- 通过 Python 脚本封装容器启动、参数传递和 vLLM 服务拉起流程。
- 支持根据模型路径、工作目录、GPU 配置、环境变量和 volume 映射等参数扩展启动逻辑。

## Docker 容器启动脚本

`scripts/docker_run.py` 使用 `argparse` 生成并执行与 `docker_run.sh` 等价的 Docker 命令。脚本会先处理旧容器：

```bash
docker rm <name>
```

然后启动新容器：

```bash
docker run -itd ... <image>
```

默认参数复刻 Issue 示例中的关键配置，包括 `--user=0`、`--privileged`、`--ipc=host`、`--network host`、`--runtime=nvidia`、`--gpus all`、`--ulimit memlock=-1:-1`、`-v /sys/fs/cgroup:/sys/fs/cgroup:ro`、`-e NVIDIA_VISIBLE_DEVICES=all`、`--entrypoint /bin/bash`。

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

### 常用动态参数

- `--name vllm_dsv4`：容器名称，同时用于 `docker rm` 和 `docker run --name`。
- `--image vllm/vllm-openai:v0.21.0-ubuntu2404`：Docker 镜像名。
- `--model-path /data/DeepSeek-V4-Pro`：模型宿主机路径，默认映射到容器内相同路径。
- `--model-target /container/model/path`：可选，指定模型在容器内的目标路径。
- `--host-workdir /home/zhengyingfei`：宿主机工作目录，默认也作为容器工作目录。
- `--container-workdir /workspace`：可选，指定容器内工作目录并作为 `-w` 参数。
- `--root-path /root`：宿主机 root 目录映射，默认映射到容器内相同路径。
- `--root-target /container/root`：可选，指定 root 路径在容器内的目标路径。
- `--runtime nvidia`、`--gpus all`、`--nvidia-visible-devices all`：GPU 相关 Docker 参数。
- `--volume host:container[:mode]`：额外 volume 映射，可重复传入。
- `--env KEY=VALUE`：额外环境变量，可重复传入。
- `--ulimit NAME=SOFT:HARD`：额外 ulimit，可重复传入。

示例：

```bash
python scripts/docker_run.py --dry-run \
  --name custom_vllm \
  --image vllm/vllm-openai:v0.21.0-ubuntu2404 \
  --model-path /models/DeepSeek-V4-Pro \
  --model-target /data/DeepSeek-V4-Pro \
  --host-workdir /home/zhengyingfei \
  --volume /data/cache:/data/cache \
  --env HF_HOME=/data/cache/huggingface \
  --ulimit nofile=1024:2048
```

## 测试

本仓库使用 pytest 风格测试。运行：

```bash
python -m pytest tests/test_docker_run.py -q
```

如果当前环境未安装 pytest，可使用标准库 unittest 的发现模式执行这些测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
