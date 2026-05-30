#!/usr/bin/env python3
"""Serve a local Web/API command generator for vLLM DeepSeek-V4-Pro PD runs.

The module builds argv lists and shell snippets only. It never starts Docker or
vLLM from the Web/API paths; all responses keep ``executed`` set to ``False``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shlex

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from scripts import docker_run

DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 6070
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

HOST_CHECK_PROFILE = "host_check"
DOCKER_RUN_PROFILE = "docker_run"
PREFILL_PROFILE = "prefill"
DECODE_PROFILE = "decode"
PD_PROXY_PROFILE = "pd_proxy"
SUPPORTED_PROFILES = {
    HOST_CHECK_PROFILE,
    DOCKER_RUN_PROFILE,
    PREFILL_PROFILE,
    DECODE_PROFILE,
    PD_PROXY_PROFILE,
}

MODEL_NAME = "DeepSeek-V4-Pro"
MOONCAKE_INSTALL_COMMAND = "pip install mooncake-transfer-engine-cuda13"
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
)
ISSUE_NOTES = (
    "vllm-0.21.0 image does not preinstall mooncake-transfer-engine; install mooncake-transfer-engine-cuda13 in the container.",
    "Unset proxy variables before PD-disaggregated runs to avoid KV transfer failures on production nodes.",
    "Do not enable MTP for vLLM 0.21.0 PD-disaggregated deployment because of precision and reliability issues.",
)

PD_DEFAULTS: dict[str, Any] = {
    "model_path": docker_run.DEFAULT_MODEL_PATH,
    "served_model_name": MODEL_NAME,
    "host": "0.0.0.0",
    "tensor_parallel_size": 8,
    "pipeline_parallel_size": 1,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.9,
    "kv_connector": "MooncakeStore",
    "kv_buffer_device": "cuda",
    "kv_buffer_size": "10e9",
    "kv_host": "127.0.0.1",
    "kv_port": 5000,
    "enable_mtp": False,
    "extra_vllm_args": "",
}
PREFILL_DEFAULTS: dict[str, Any] = {
    **PD_DEFAULTS,
    "port": 8000,
    "kv_role": "kv_producer",
}
DECODE_DEFAULTS: dict[str, Any] = {
    **PD_DEFAULTS,
    "port": 8001,
    "kv_role": "kv_consumer",
}
DOCKER_RUN_DEFAULTS: dict[str, Any] = {
    "name": docker_run.DEFAULT_NAME,
    "image": docker_run.DEFAULT_IMAGE,
    "model_path": docker_run.DEFAULT_MODEL_PATH,
    "model_target": "",
    "host_workdir": docker_run.DEFAULT_HOST_WORKDIR,
    "container_workdir": "",
    "root_path": docker_run.DEFAULT_ROOT_PATH,
    "root_target": "",
    "runtime": docker_run.DEFAULT_RUNTIME,
    "gpus": docker_run.DEFAULT_GPUS,
    "nvidia_visible_devices": docker_run.DEFAULT_NVIDIA_VISIBLE_DEVICES,
    "entrypoint": docker_run.DEFAULT_ENTRYPOINT,
    "user": "0",
    "privileged": True,
    "ipc": "host",
    "network": "host",
    "volumes": "",
    "envs": "",
    "ulimits": docker_run.DEFAULT_MEMLOCK_ULIMIT,
}
HOST_CHECK_ITEMS: tuple[dict[str, str], ...] = (
    {
        "title": "Mooncake transfer engine dependency",
        "check_shell": "pip show mooncake-transfer-engine-cuda13 || true",
        "repair_shell": MOONCAKE_INSTALL_COMMAND,
    },
    {
        "title": "Proxy variables that may break PD KV transfer",
        "check_shell": "env | grep -i '_proxy=' || true",
        "repair_shell": "\n".join(f"unset {key}" for key in PROXY_ENV_VARS),
    },
    {
        "title": "MTP disabled for vLLM 0.21.0 PD deployment",
        "check_shell": "# Review generated command: it must not contain speculative/MTP flags.",
        "repair_shell": "# Keep enable_mtp=false for prefill/decode profiles.",
    },
)
HOST_CHECK_DEFAULTS = {"issue": "https://github.com/zhengyf11/vllm_run/issues/7", "items": HOST_CHECK_ITEMS}
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    PREFILL_PROFILE: PREFILL_DEFAULTS,
    DECODE_PROFILE: DECODE_DEFAULTS,
    DOCKER_RUN_PROFILE: DOCKER_RUN_DEFAULTS,
    HOST_CHECK_PROFILE: HOST_CHECK_DEFAULTS,
    PD_PROXY_PROFILE: {"proxy_unsets": "\n".join(f"unset {key}" for key in PROXY_ENV_VARS)},
}


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_lines(value: Any) -> tuple[str, ...]:
    if not _has_value(value):
        return ()
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(item.strip() for item in value if item.strip())
    if not isinstance(value, str):
        raise ValueError("multi-line fields must be strings or lists of strings")
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def normalize_profile(profile: Any) -> str:
    if not _has_value(profile):
        return PREFILL_PROFILE
    normalized = str(profile).strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported profile: {profile}")
    return normalized


def get_effective_defaults(profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    return dict(PROFILE_DEFAULTS[normalize_profile(profile)])


def normalize_form_payload(payload: Mapping[str, Any] | None, profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    normalized_profile = normalize_profile(profile)
    defaults = get_effective_defaults(normalized_profile)
    raw = {} if payload is None else dict(payload)
    if normalized_profile in {HOST_CHECK_PROFILE, PD_PROXY_PROFILE}:
        return defaults
    config = {key: (raw[key] if _has_value(raw.get(key)) else value) for key, value in defaults.items()}
    if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE}:
        for bool_key in ("trust_remote_code",):
            config[bool_key] = _to_bool(raw.get(bool_key), defaults[bool_key])
        # Issue #7: PD deployment on vLLM 0.21.0 must not enable MTP.
        config["enable_mtp"] = False
    if normalized_profile == DOCKER_RUN_PROFILE:
        config["privileged"] = _to_bool(raw.get("privileged"), defaults["privileged"])
        config["volumes"] = _parse_lines(raw.get("volumes", config.get("volumes")))
        config["envs"] = _parse_lines(raw.get("envs", config.get("envs")))
        config["ulimits"] = _parse_lines(raw.get("ulimits", config.get("ulimits")))
    return config


def _kv_transfer_config(config: Mapping[str, Any]) -> str:
    kv_config = {
        "kv_connector": str(config["kv_connector"]),
        "kv_role": str(config["kv_role"]),
        "kv_buffer_device": str(config["kv_buffer_device"]),
        "kv_buffer_size": str(config["kv_buffer_size"]),
        "kv_host": str(config["kv_host"]),
        "kv_port": int(config["kv_port"]),
    }
    return json.dumps(kv_config, separators=(",", ":"), sort_keys=True)


def parse_extra_vllm_args(value: Any) -> list[str]:
    if not _has_value(value):
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if not isinstance(value, str):
        raise ValueError("extra_vllm_args must be a string or a list of strings")
    return shlex.split(value)


def build_pd_command(config: Mapping[str, Any]) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        str(config["model_path"]),
        "--served-model-name",
        str(config["served_model_name"]),
        "--host",
        str(config["host"]),
        "--port",
        str(config["port"]),
        "--tensor-parallel-size",
        str(config["tensor_parallel_size"]),
        "--pipeline-parallel-size",
        str(config["pipeline_parallel_size"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--kv-transfer-config",
        _kv_transfer_config(config),
    ]
    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    cmd.extend(parse_extra_vllm_args(config.get("extra_vllm_args")))
    return cmd


def build_docker_run_command(config: Mapping[str, Any]) -> list[str]:
    model_path = str(config["model_path"])
    volumes = list(config.get("volumes", ()))
    model_target = str(config.get("model_target") or model_path)
    model_volume = f"{model_path}:{model_target}"
    if model_volume not in volumes:
        volumes = [model_volume, *volumes]
    docker_config = docker_run.DockerRunConfig(
        name=str(config["name"]),
        image=str(config["image"]),
        model_path=model_path,
        model_target=str(config.get("model_target") or "") or None,
        host_workdir=str(config["host_workdir"]),
        container_workdir=str(config.get("container_workdir") or "") or None,
        root_path=str(config["root_path"]),
        root_target=str(config.get("root_target") or "") or None,
        runtime=str(config["runtime"]),
        gpus=str(config["gpus"]),
        nvidia_visible_devices=str(config["nvidia_visible_devices"]),
        entrypoint=str(config["entrypoint"]),
        user=str(config["user"]),
        privileged=bool(config["privileged"]),
        ipc=str(config["ipc"]),
        network=str(config["network"]),
        volumes=tuple(item for item in volumes if item != model_volume),
        envs=tuple(config.get("envs", ())),
        ulimits=tuple(config.get("ulimits", ())),
    )
    return docker_run.build_run_command(docker_config)


def build_profile_command(profile: str, config: Mapping[str, Any]) -> list[str]:
    normalized = normalize_profile(profile)
    if normalized == DOCKER_RUN_PROFILE:
        return build_docker_run_command(config)
    if normalized in {PREFILL_PROFILE, DECODE_PROFILE}:
        return build_pd_command(config)
    return []


def build_shell_command(command: Sequence[str]) -> str:
    if not command:
        return ""
    if len(command) <= 3:
        return shlex.join(command)
    head = shlex.join(command[:2]) if command[:2] == ["vllm", "serve"] else shlex.join(command[:3])
    start = 2 if command[:2] == ["vllm", "serve"] else 3
    groups: list[Sequence[str]] = []
    index = start
    while index < len(command):
        current = command[index]
        if current.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            groups.append(command[index : index + 2])
            index += 2
        elif current in {"-v", "-e", "--ulimit", "--gpus", "--name", "--network", "-w", "--entrypoint"} and index + 1 < len(command):
            groups.append(command[index : index + 2])
            index += 2
        else:
            groups.append(command[index : index + 1])
            index += 1
    lines = [f"{head} \\"]
    for group_index, group in enumerate(groups):
        suffix = " \\" if group_index < len(groups) - 1 else ""
        lines.append(f"  {shlex.join(group)}{suffix}")
    return "\n".join(lines)


def build_proxy_unsets() -> list[str]:
    return [f"unset {key}" for key in PROXY_ENV_VARS]


def build_install_hints() -> list[str]:
    return [MOONCAKE_INSTALL_COMMAND]


def build_host_check_shell() -> str:
    sections = []
    for item in HOST_CHECK_ITEMS:
        sections.append(
            "\n".join(
                [
                    f"# {item['title']}",
                    "# Check",
                    item["check_shell"],
                    "# Repair (review before running manually)",
                    item["repair_shell"],
                ]
            )
        )
    return "\n\n".join(sections)


def build_command_response(payload: Mapping[str, Any] | None, profile: str = PREFILL_PROFILE) -> dict[str, Any]:
    raw = {} if payload is None else dict(payload)
    normalized_profile = normalize_profile(raw.pop("profile", profile))
    if normalized_profile == HOST_CHECK_PROFILE:
        shell = build_host_check_shell()
        return {
            "profile": HOST_CHECK_PROFILE,
            "config": get_effective_defaults(HOST_CHECK_PROFILE),
            "command": [],
            "shell_command": shell,
            "env_exports": [],
            "proxy_unsets": build_proxy_unsets(),
            "install_hints": build_install_hints(),
            "shell_hints": list(ISSUE_NOTES),
            "combined_shell": shell,
            "executed": False,
        }
    if normalized_profile == PD_PROXY_PROFILE:
        proxy_unsets = build_proxy_unsets()
        return {
            "profile": PD_PROXY_PROFILE,
            "config": get_effective_defaults(PD_PROXY_PROFILE),
            "command": [],
            "shell_command": "\n".join(proxy_unsets),
            "env_exports": [],
            "proxy_unsets": proxy_unsets,
            "install_hints": [],
            "shell_hints": [ISSUE_NOTES[1]],
            "combined_shell": "\n".join(proxy_unsets),
            "executed": False,
        }
    config = normalize_form_payload(raw, normalized_profile)
    command = build_profile_command(normalized_profile, config)
    proxy_unsets = build_proxy_unsets() if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE} else []
    install_hints = build_install_hints() if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE} else []
    shell_hints = list(ISSUE_NOTES) if normalized_profile in {PREFILL_PROFILE, DECODE_PROFILE} else []
    commented_shell_hints = [f"# {hint}" for hint in shell_hints]
    shell_command = build_shell_command(command)
    combined_parts = [*install_hints, *proxy_unsets, *commented_shell_hints, shell_command]
    return {
        "profile": normalized_profile,
        "config": config,
        "command": command,
        "shell_command": shell_command,
        "env_exports": [],
        "proxy_unsets": proxy_unsets,
        "install_hints": install_hints,
        "shell_hints": shell_hints,
        "combined_shell": "\n".join(part for part in combined_parts if part),
        "executed": False,
    }


def read_static_asset(path: str) -> tuple[bytes, str]:
    route = "index.html" if path in {"", "/"} else path.lstrip("/")
    if route not in {"index.html", "styles.css", "app.js"}:
        raise FileNotFoundError(path)
    asset_path = WEB_DIR / route
    content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    if asset_path.suffix == ".js":
        content_type = "application/javascript"
    if asset_path.suffix in {".html", ".css", ".js"}:
        content_type = f"{content_type}; charset=utf-8"
    return asset_path.read_bytes(), content_type


class VllmCommandHandler(BaseHTTPRequestHandler):
    server_version = "VllmCommandWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            try:
                query = parse_qs(parsed.query)
                profile = normalize_profile(query.get("profile", [PREFILL_PROFILE])[0])
                self._write_json(HTTPStatus.OK, {"profile": profile, "defaults": get_effective_defaults(profile), "issue_notes": ISSUE_NOTES})
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        try:
            body, content_type = read_static_asset(parsed.path)
        except FileNotFoundError:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._write_bytes(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/command":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            response = build_command_response(payload, payload.get("profile", PREFILL_PROFILE))
        except json.JSONDecodeError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc.msg}"})
            return
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        return

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._write_bytes(status, body, "application/json; charset=utf-8")

    def _write_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local vLLM command-generator Web UI.")
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), VllmCommandHandler)
    print(f"Serving vLLM command generator on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
