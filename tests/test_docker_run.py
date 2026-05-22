import contextlib
import io
import unittest
from unittest.mock import patch

from scripts import docker_run


class DockerRunCommandTests(unittest.TestCase):
    def test_default_commands_match_issue_key_flags(self):
        args = docker_run.parse_args(["--dry-run"])
        config = docker_run.config_from_args(args)

        rm_command, run_command = docker_run.build_commands(config)

        self.assertEqual(rm_command, ["docker", "rm", "vllm_dsv4"])
        self.assertEqual(run_command[:4], ["docker", "run", "-itd", "--user=0"])
        self.assertEqual(["--name", "vllm_dsv4"], run_command[4:6])
        self.assertIn("--privileged", run_command)
        self.assertIn("--ipc=host", run_command)
        self.assertEqual(
            ["--network", "host"],
            run_command[run_command.index("--network") : run_command.index("--network") + 2],
        )
        self.assertIn("--runtime=nvidia", run_command)
        self.assertEqual(
            ["--gpus", "all"],
            run_command[run_command.index("--gpus") : run_command.index("--gpus") + 2],
        )
        self.assertEqual(
            ["--ulimit", "memlock=-1:-1"],
            run_command[run_command.index("--ulimit") : run_command.index("--ulimit") + 2],
        )
        self.assertIn(["-v", "/sys/fs/cgroup:/sys/fs/cgroup:ro"], _pairs(run_command, "-v"))
        self.assertIn(
            ["-v", "/data/DeepSeek-V4-Pro:/data/DeepSeek-V4-Pro"],
            _pairs(run_command, "-v"),
        )
        self.assertIn(["-v", "/root:/root"], _pairs(run_command, "-v"))
        self.assertIn(
            ["-v", "/home/zhengyingfei:/home/zhengyingfei"], _pairs(run_command, "-v")
        )
        self.assertIn(["-e", "NVIDIA_VISIBLE_DEVICES=all"], _pairs(run_command, "-e"))
        self.assertEqual(
            ["-w", "/home/zhengyingfei"],
            run_command[run_command.index("-w") : run_command.index("-w") + 2],
        )
        self.assertEqual(
            ["--entrypoint", "/bin/bash"],
            run_command[
                run_command.index("--entrypoint") : run_command.index("--entrypoint") + 2
            ],
        )
        self.assertEqual(run_command[-1], "vllm/vllm-openai:v0.21.0-ubuntu2404")

    def test_dynamic_name_image_and_paths_change_commands(self):
        args = docker_run.parse_args(
            [
                "--name",
                "custom_container",
                "--image",
                "example/image:tag",
                "--model-path",
                "/models/dsv4",
                "--model-target",
                "/mnt/model",
                "--host-workdir",
                "/workspace/host",
                "--container-workdir",
                "/workspace/container",
                "--root-path",
                "/host-root",
                "--root-target",
                "/container-root",
                "--gpus",
                "device=0",
                "--runtime",
                "custom-runtime",
                "--nvidia-visible-devices",
                "0",
            ]
        )
        rm_command, run_command = docker_run.build_commands(docker_run.config_from_args(args))

        self.assertEqual(rm_command, ["docker", "rm", "custom_container"])
        self.assertEqual(["--name", "custom_container"], run_command[4:6])
        self.assertIn("--runtime=custom-runtime", run_command)
        self.assertEqual(
            ["--gpus", "device=0"],
            run_command[run_command.index("--gpus") : run_command.index("--gpus") + 2],
        )
        self.assertIn(["-v", "/models/dsv4:/mnt/model"], _pairs(run_command, "-v"))
        self.assertIn(["-v", "/host-root:/container-root"], _pairs(run_command, "-v"))
        self.assertIn(
            ["-v", "/workspace/host:/workspace/container"], _pairs(run_command, "-v")
        )
        self.assertIn(["-e", "NVIDIA_VISIBLE_DEVICES=0"], _pairs(run_command, "-e"))
        self.assertEqual(
            ["-w", "/workspace/container"],
            run_command[run_command.index("-w") : run_command.index("-w") + 2],
        )
        self.assertEqual(run_command[-1], "example/image:tag")

    def test_repeated_volume_env_and_ulimit_are_added(self):
        args = docker_run.parse_args(
            [
                "--volume",
                "/host/a:/container/a:ro",
                "--volume",
                "/host/b:/container/b",
                "--env",
                "FOO=bar",
                "--env",
                "BAZ=qux",
                "--ulimit",
                "nofile=1024:2048",
            ]
        )
        run_command = docker_run.build_run_command(docker_run.config_from_args(args))

        self.assertIn(["-v", "/host/a:/container/a:ro"], _pairs(run_command, "-v"))
        self.assertIn(["-v", "/host/b:/container/b"], _pairs(run_command, "-v"))
        self.assertIn(["-e", "FOO=bar"], _pairs(run_command, "-e"))
        self.assertIn(["-e", "BAZ=qux"], _pairs(run_command, "-e"))
        self.assertIn(["--ulimit", "memlock=-1:-1"], _pairs(run_command, "--ulimit"))
        self.assertIn(["--ulimit", "nofile=1024:2048"], _pairs(run_command, "--ulimit"))

    def test_dry_run_returns_commands_without_calling_subprocess(self):
        config = docker_run.config_from_args(docker_run.parse_args(["--name", "safe_name"]))

        with patch.object(docker_run.subprocess, "run") as run_mock:
            formatted_commands = docker_run.run_commands(config, dry_run=True)

        run_mock.assert_not_called()
        self.assertEqual(formatted_commands[0], "docker rm safe_name")
        self.assertTrue(
            formatted_commands[1].startswith("docker run -itd --user=0 --name safe_name")
        )

    def test_execution_uses_subprocess_without_shell_and_allows_rm_failure(self):
        config = docker_run.config_from_args(docker_run.parse_args(["--name", "run_name"]))

        with patch.object(docker_run.subprocess, "run") as run_mock:
            docker_run.run_commands(config, dry_run=False)

        self.assertEqual(run_mock.call_count, 2)
        run_mock.assert_any_call(["docker", "rm", "run_name"], check=False)
        run_mock.assert_any_call(docker_run.build_run_command(config), check=True)

    def test_main_prints_dry_run_commands(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = docker_run.main(["--dry-run", "--name", "print_name"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("docker rm print_name", output)
        self.assertIn("docker run -itd --user=0 --name print_name", output)


def _pairs(command, flag):
    return [[item, command[index + 1]] for index, item in enumerate(command[:-1]) if item == flag]


if __name__ == "__main__":
    unittest.main()
