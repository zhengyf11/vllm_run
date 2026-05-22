#!/usr/bin/env python3
"""Build and run the Docker container used for vLLM workloads."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Sequence


DEFAULT_NAME = "vllm_dsv4"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.21.0-ubuntu2404"
DEFAULT_MODEL_PATH = "/data/DeepSeek-V4-Pro"
DEFAULT_HOST_WORKDIR = "/home/zhengyingfei"
DEFAULT_ROOT_PATH = "/root"
DEFAULT_RUNTIME = "nvidia"
DEFAULT_GPUS = "all"
DEFAULT_NVIDIA_VISIBLE_DEVICES = "all"
DEFAULT_ENTRYPOINT = "/bin/bash"
DEFAULT_CGROUP_VOLUME = "/sys/fs/cgroup:/sys/fs/cgroup:ro"
DEFAULT_MEMLOCK_ULIMIT = "memlock=-1:-1"


@dataclass(frozen=True)
class DockerRunConfig:
    """Normalized configuration for the Docker remove/run commands."""

    name: str = DEFAULT_NAME
    image: str = DEFAULT_IMAGE
    model_path: str = DEFAULT_MODEL_PATH
    model_target: str | None = None
    host_workdir: str = DEFAULT_HOST_WORKDIR
    container_workdir: str | None = None
    root_path: str = DEFAULT_ROOT_PATH
    root_target: str | None = None
    runtime: str = DEFAULT_RUNTIME
    gpus: str = DEFAULT_GPUS
    nvidia_visible_devices: str = DEFAULT_NVIDIA_VISIBLE_DEVICES
    entrypoint: str = DEFAULT_ENTRYPOINT
    user: str = "0"
    privileged: bool = True
    ipc: str = "host"
    network: str = "host"
    volumes: tuple[str, ...] = field(default_factory=tuple)
    envs: tuple[str, ...] = field(default_factory=tuple)
    ulimits: tuple[str, ...] = (DEFAULT_MEMLOCK_ULIMIT,)


def _path_mapping(host_path: str, container_path: str | None) -> str:
    target = container_path or host_path
    return f"{host_path}:{target}"


def build_rm_command(config: DockerRunConfig) -> list[str]:
    """Return the docker rm command for the configured container name."""
    return ["docker", "rm", config.name]


def build_run_command(config: DockerRunConfig) -> list[str]:
    """Return the docker run command as an argv list, without using a shell."""
    command = [
        "docker",
        "run",
        "-itd",
        f"--user={config.user}",
        "--name",
        config.name,
    ]

    if config.privileged:
        command.append("--privileged")

    command.extend(
        [
            f"--ipc={config.ipc}",
            "--network",
            config.network,
            f"--runtime={config.runtime}",
            "--gpus",
            config.gpus,
        ]
    )

    for ulimit in config.ulimits:
        command.extend(["--ulimit", ulimit])

    volume_args = [
        DEFAULT_CGROUP_VOLUME,
        _path_mapping(config.model_path, config.model_target),
        _path_mapping(config.root_path, config.root_target),
        _path_mapping(config.host_workdir, config.container_workdir),
        *config.volumes,
    ]
    for volume in volume_args:
        command.extend(["-v", volume])

    env_args = [f"NVIDIA_VISIBLE_DEVICES={config.nvidia_visible_devices}", *config.envs]
    for env in env_args:
        command.extend(["-e", env])

    command.extend(
        [
            "-w",
            config.container_workdir or config.host_workdir,
            "--entrypoint",
            config.entrypoint,
            config.image,
        ]
    )
    return command


def build_commands(config: DockerRunConfig) -> tuple[list[str], list[str]]:
    """Return the docker rm command followed by the docker run command."""
    return build_rm_command(config), build_run_command(config)


def format_command(command: Sequence[str]) -> str:
    """Format an argv list for safe display in dry-run output."""
    return shlex.join(command)


def run_commands(config: DockerRunConfig, *, dry_run: bool = False) -> list[str]:
    """Execute or print the configured Docker commands.

    In dry-run mode, returns the formatted commands and does not call Docker.
    In execution mode, docker rm is allowed to fail so a missing container does
    not prevent the subsequent docker run command.
    """
    commands = build_commands(config)
    formatted = [format_command(command) for command in commands]

    if dry_run:
        return formatted

    rm_command, run_command = commands
    subprocess.run(rm_command, check=False)
    subprocess.run(run_command, check=True)
    return formatted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a vLLM Docker container with configurable Docker run options."
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="Container name.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to run.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Host model path. Defaults to mapping to the same path in the container.",
    )
    parser.add_argument(
        "--model-target",
        default=None,
        help="Optional container target path for --model-path.",
    )
    parser.add_argument(
        "--host-workdir",
        default=DEFAULT_HOST_WORKDIR,
        help="Host working directory to mount and use as the default container workdir.",
    )
    parser.add_argument(
        "--container-workdir",
        default=None,
        help="Optional container workdir target. Defaults to --host-workdir.",
    )
    parser.add_argument("--root-path", default=DEFAULT_ROOT_PATH, help="Host root path to mount.")
    parser.add_argument(
        "--root-target",
        default=None,
        help="Optional container target path for --root-path.",
    )
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME, help="Docker runtime.")
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Docker --gpus value.")
    parser.add_argument(
        "--nvidia-visible-devices",
        default=DEFAULT_NVIDIA_VISIBLE_DEVICES,
        help="NVIDIA_VISIBLE_DEVICES environment value.",
    )
    parser.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT, help="Container entrypoint.")
    parser.add_argument("--user", default="0", help="Container user value for --user.")
    parser.add_argument(
        "--volume",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:MODE]",
        help="Additional volume mapping. Can be repeated.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment variable. Can be repeated.",
    )
    parser.add_argument(
        "--ulimit",
        action="append",
        default=[],
        metavar="NAME=SOFT:HARD",
        help=f"Additional ulimit. Defaults to {DEFAULT_MEMLOCK_ULIMIT} when omitted.",
    )
    parser.add_argument(
        "--no-default-memlock-ulimit",
        action="store_true",
        help="Do not add the default memlock=-1:-1 ulimit.",
    )
    parser.add_argument(
        "--no-privileged",
        action="store_true",
        help="Do not pass --privileged to docker run.",
    )
    parser.add_argument("--ipc", default="host", help="Docker IPC mode.")
    parser.add_argument("--network", default="host", help="Docker network mode.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print docker rm and docker run commands without executing Docker.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DockerRunConfig:
    ulimits = tuple(args.ulimit)
    if not args.no_default_memlock_ulimit:
        ulimits = (DEFAULT_MEMLOCK_ULIMIT, *ulimits)

    return DockerRunConfig(
        name=args.name,
        image=args.image,
        model_path=args.model_path,
        model_target=args.model_target,
        host_workdir=args.host_workdir,
        container_workdir=args.container_workdir,
        root_path=args.root_path,
        root_target=args.root_target,
        runtime=args.runtime,
        gpus=args.gpus,
        nvidia_visible_devices=args.nvidia_visible_devices,
        entrypoint=args.entrypoint,
        user=args.user,
        privileged=not args.no_privileged,
        ipc=args.ipc,
        network=args.network,
        volumes=tuple(args.volume),
        envs=tuple(args.env),
        ulimits=ulimits,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    for command in run_commands(config, dry_run=args.dry_run):
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
