from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scripts import vllm_command_web


class CommandGenerationTests(unittest.TestCase):
    def test_prefill_default_command_contains_deepseek_vllm_and_pd_rules(self):
        response = vllm_command_web.build_command_response({}, "prefill")
        cmd = response["command"]

        self.assertEqual(cmd[:2], ["vllm", "serve"])
        self.assertIn("/data/DeepSeek-V4-Pro", cmd)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "DeepSeek-V4-Pro")
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "8")
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8000")
        self.assertIn("--kv-transfer-config", cmd)
        kv_config = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        self.assertEqual(kv_config["kv_role"], "kv_producer")
        self.assertEqual(kv_config["kv_connector"], "MooncakeStore")
        self.assertFalse(response["config"]["enable_mtp"])
        self.assertFalse(response["executed"])
        self.assertNotIn("--speculative-config", cmd)
        self.assertNotIn("--num-lookahead-slots", cmd)
        self.assertIn("pip install mooncake-transfer-engine-cuda13", response["install_hints"])
        self.assertIn("unset HTTP_PROXY", response["proxy_unsets"])
        self.assertIn("unset HTTPS_PROXY", response["combined_shell"])

    def test_decode_default_command_uses_consumer_role_and_disables_mtp(self):
        response = vllm_command_web.build_command_response({}, "decode")
        cmd = response["command"]

        self.assertEqual(cmd[:2], ["vllm", "serve"])
        self.assertIn("/data/DeepSeek-V4-Pro", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "8001")
        kv_config = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        self.assertEqual(kv_config["kv_role"], "kv_consumer")
        self.assertFalse(response["config"]["enable_mtp"])
        self.assertNotIn("--speculative-config", cmd)
        self.assertIn("mooncake-transfer-engine-cuda13", "\n".join(response["install_hints"]))

    def test_payload_defaults_and_overrides_are_normalized_without_shell_execution(self):
        with patch("scripts.docker_run.subprocess.run") as run_mock:
            response = vllm_command_web.build_command_response(
                {
                    "profile": "prefill",
                    "model_path": "/models/custom",
                    "served_model_name": "custom-model",
                    "tensor_parallel_size": 4,
                    "port": 9000,
                    "extra_vllm_args": "--max-model-len 65536",
                    "enable_mtp": True,
                },
                "prefill",
            )

        run_mock.assert_not_called()
        cmd = response["command"]
        self.assertIn("/models/custom", cmd)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "custom-model")
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(cmd[cmd.index("--port") + 1], "9000")
        self.assertIn("--max-model-len", cmd)
        self.assertIn("65536", cmd)
        self.assertFalse(response["config"]["enable_mtp"], "PD profiles force MTP off for vLLM 0.21.0")

    def test_docker_run_profile_reuses_docker_command_builder(self):
        response = vllm_command_web.build_command_response(
            {"name": "custom_vllm", "model_path": "/models/dsv4"}, "docker_run"
        )
        cmd = response["command"]

        self.assertEqual(cmd[:3], ["docker", "run", "-itd"])
        self.assertIn("custom_vllm", cmd)
        self.assertIn("/models/dsv4:/models/dsv4", cmd)
        self.assertFalse(response["executed"])

    def test_combined_shell_comments_issue_notes(self):
        response = vllm_command_web.build_command_response({}, "prefill")
        combined_lines = response["combined_shell"].splitlines()

        for note in response["shell_hints"]:
            self.assertIn(f"# {note}", combined_lines)
            self.assertNotIn(note, [line for line in combined_lines if not line.startswith("# ")])

    def test_pd_proxy_profile_returns_proxy_unsets_only(self):
        response = vllm_command_web.build_command_response({}, "pd_proxy")

        self.assertEqual(response["profile"], "pd_proxy")
        self.assertEqual(response["command"], [])
        self.assertFalse(response["executed"])
        self.assertIn("unset HTTP_PROXY", response["proxy_unsets"])
        self.assertIn("unset http_proxy", response["proxy_unsets"])
        self.assertEqual(response["install_hints"], [])
        self.assertEqual(response["combined_shell"], "\n".join(response["proxy_unsets"]))

    def test_host_check_profile_returns_shell_only(self):
        response = vllm_command_web.build_command_response({}, "host_check")

        self.assertEqual(response["profile"], "host_check")
        self.assertEqual(response["command"], [])
        self.assertIn("pip show mooncake-transfer-engine-cuda13", response["combined_shell"])
        self.assertFalse(response["executed"])

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            vllm_command_web.normalize_profile("bad")


class StaticAssetTests(unittest.TestCase):
    def test_static_assets_exist_and_are_whitelisted(self):
        for route in ("/", "/index.html", "/styles.css", "/app.js"):
            body, content_type = vllm_command_web.read_static_asset(route)
            self.assertGreater(len(body), 0)
            self.assertIn("charset=utf-8", content_type)

        with self.assertRaises(FileNotFoundError):
            vllm_command_web.read_static_asset("/../README.md")

        self.assertTrue((Path("web") / "index.html").exists())
        self.assertTrue((Path("web") / "styles.css").exists())
        self.assertTrue((Path("web") / "app.js").exists())


class ApiHandlerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), vllm_command_web.VllmCommandHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method: str, path: str, body: dict | None = None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, json.loads(data.decode("utf-8"))

    def test_get_defaults_and_post_command(self):
        status, defaults = self.request("GET", "/api/defaults?profile=decode")
        self.assertEqual(status, 200)
        self.assertEqual(defaults["profile"], "decode")
        self.assertEqual(defaults["defaults"]["port"], 8001)
        self.assertFalse(defaults["defaults"]["enable_mtp"])

        status, command = self.request("POST", "/api/command", {"profile": "decode"})
        self.assertEqual(status, 200)
        self.assertEqual(command["profile"], "decode")
        self.assertFalse(command["executed"])
        self.assertIn("unset HTTP_PROXY", command["proxy_unsets"])

    def test_api_rejects_invalid_payload_and_profile(self):
        status, error = self.request("GET", "/api/defaults?profile=bad")
        self.assertEqual(status, 400)
        self.assertIn("unsupported profile", error["error"])

        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        conn.request("POST", "/api/command", body=b"[]", headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 400)
        self.assertIn("JSON body must be an object", data["error"])


if __name__ == "__main__":
    unittest.main()
