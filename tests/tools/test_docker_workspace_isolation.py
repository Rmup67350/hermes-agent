"""Strong host-write isolation contracts for Docker-backed code profiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile

import pytest

import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools
import tools.terminal_tool as terminal_tool
from tools.environments import docker as docker_env
from tools.file_operations import ShellFileOperations


def _fake_docker(monkeypatch, *, declared_volumes=None):
    docker_env._cgroup_limits_ok = True
    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(command, **_kwargs):
        command = list(command)
        if command and command[0] == "git":
            return real_run(command, **_kwargs)
        calls.append(command)
        if len(command) > 1 and command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "Docker version", "")
        if len(command) > 3 and command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(declared_volumes),
                "",
            )
        if len(command) > 2 and command[1:3] == ["run", "-d"]:
            return subprocess.CompletedProcess(command, 0, "container-id\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    return calls


def _git_project(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Hermes Test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "hermes@example.invalid"], check=True)
    (path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = path / "sandbox_tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_module.py").write_text(
        "import unittest\n"
        "from module import VALUE\n\n"
        "class ModuleTest(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(VALUE, 1)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(path), "add", "-f", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return path


def _index_transport_bundle(
    gitdir: Path, destination: Path, *, expected_head: str
) -> tuple[Path, str]:
    private_head = subprocess.run(
        ["git", "--git-dir", str(gitdir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "--git-dir", str(gitdir), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    envelope = subprocess.run(
        ["git", "--git-dir", str(gitdir), "commit-tree", tree, "-p", private_head],
        input="Hermes staged index transport\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    transport_ref = "refs/hermes/index-transport"
    subprocess.run(
        ["git", "--git-dir", str(gitdir), "update-ref", transport_ref, envelope],
        check=True,
    )
    try:
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(gitdir),
                "bundle",
                "create",
                str(destination),
                transport_ref,
                f"^{expected_head}",
            ],
            check=True,
        )
    finally:
        subprocess.run(
            ["git", "--git-dir", str(gitdir), "update-ref", "-d", transport_ref],
            check=True,
        )
    return destination, private_head


def _structured_documents(root: Path, marker: str) -> dict[str, Path]:
    notebook = root / "document.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "markdown", "source": marker}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    docx = root / "document.docx"
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            f'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{marker}'
            "</w:t></w:r></w:p></w:body></w:document>",
        )
    xlsx = root / "document.xlsx"
    with zipfile.ZipFile(xlsx, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets><sheet name="Data" '
            'sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" '
            'Type="x"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
            f'2006/main"><sheetData><row r="1"><c r="A1" t="str"><v>{marker}'
            "</v></c></row></sheetData></worksheet>",
        )
    return {path.suffix: path for path in (notebook, docx, xlsx)}


def _workspace_only_environment(project: Path, *, task_id: str):
    return docker_env.DockerEnvironment(
        image=os.environ.get(
            "HERMES_TEST_CODE_IMAGE",
            "nikolaik/python-nodejs:python3.11-nodejs20",
        ),
        cwd="/workspace",
        timeout=120,
        task_id=task_id,
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )


def _configure_active_egress_proxy(monkeypatch, tmp_path):
    from agent.proxy_sources import iron_proxy
    from hermes_cli.config import load_config, save_config

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_config()
    config.setdefault("proxy", {})["enabled"] = True
    config["proxy"]["enforce_on_docker"] = True
    save_config(config)

    proxy_state = iron_proxy._proxy_state_dir()
    proxy_config = proxy_state / "proxy.yaml"
    proxy_config.write_text("sentinel proxy config\n", encoding="utf-8")
    ca_path = proxy_state / "sentinel-ca.crt"
    ca_path.write_text("sentinel ca\n", encoding="utf-8")
    proxy_token = "sentinel-proxy-token"
    iron_proxy.write_mappings(
        [
            iron_proxy.TokenMapping(
                proxy_token=proxy_token,
                real_env_name="SENTINEL_PROVIDER_TOKEN",
                upstream_hosts=("provider.example",),
            )
        ]
    )
    monkeypatch.setattr(
        iron_proxy,
        "get_status",
        lambda: iron_proxy.ProxyStatus(
            enabled=True,
            config_path=proxy_config,
            ca_cert_path=ca_path,
            pid=12345,
            listening=True,
            tunnel_port=19090,
        ),
    )
    return ca_path, proxy_token


def test_workspace_only_ignores_active_egress_proxy_credentials(
    monkeypatch, tmp_path, caplog
):
    project = _git_project(tmp_path / "owned")
    ca_path, proxy_token = _configure_active_egress_proxy(monkeypatch, tmp_path)
    calls = _fake_docker(monkeypatch)
    real_egress_args = docker_env._egress_proxy_args_for_docker
    egress_calls = 0

    def tracked_egress_args():
        nonlocal egress_calls
        egress_calls += 1
        return real_egress_args()

    monkeypatch.setattr(docker_env, "_egress_proxy_args_for_docker", tracked_egress_args)

    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )

    run_command = next(command for command in calls if command[1:3] == ["run", "-d"])
    volumes = [
        run_command[index + 1]
        for index, value in enumerate(run_command[:-1])
        if value == "-v"
    ]
    forbidden = (
        str(ca_path),
        proxy_token,
        "SENTINEL_PROVIDER_TOKEN",
        "--add-host",
        "host.docker.internal:host-gateway",
    )

    assert len(volumes) == 3
    assert "--network=none" in run_command
    assert "--read-only" in run_command
    assert egress_calls == 0
    assert environment._labels["hermes-egress"] == "off"
    assert all(marker not in arg for marker in forbidden for arg in run_command)
    assert all(marker not in caplog.text for marker in forbidden)



def test_terminal_config_reads_workspace_only_toggle(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_WORKSPACE_ONLY", "true")

    config = terminal_tool._get_env_config()

    assert config["docker_workspace_only"] is True


def test_create_environment_passes_workspace_only_toggle(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_environment(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", fake_environment)

    actual = terminal_tool._create_environment(
        env_type="docker",
        image="code-image",
        cwd="/workspace",
        timeout=60,
        host_cwd="/owned",
        container_config={"docker_workspace_only": True},
    )

    assert actual is sentinel
    assert captured["workspace_only"] is True


def test_execute_code_passes_workspace_only_to_shared_environment(monkeypatch):
    captured = {}
    sentinel = object()
    config = {
        "env_type": "docker",
        "docker_image": "code-image",
        "cwd": "/workspace",
        "host_cwd": "/owned",
        "timeout": 60,
        "docker_workspace_only": True,
    }

    def fake_create_environment(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_env_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda task_id: task_id)
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    environment, env_type = code_execution_tool._get_or_create_env("code-a")

    assert environment is sentinel
    assert env_type == "docker"
    assert captured["container_config"]["docker_workspace_only"] is True


class _RecordingEnvironment:
    def __init__(self):
        self.cwd = "/workspace"
        self.commands: list[str] = []

    def execute(self, command, **_kwargs):
        self.commands.append(command)
        return {"output": command, "returncode": 0}


def _dynamic_terminal_config(workspace: Path) -> dict:
    return {
        "env_type": "docker",
        "docker_image": "code-image",
        "cwd": "/root",
        "host_cwd": None,
        "timeout": 60,
        "docker_workspace_only": True,
        "docker_mount_cwd_to_workspace": False,
        "docker_volumes": [],
        "workspace_bootstrap": {"trusted": "spec"},
    }


def _isolate_terminal_environment_state(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_env_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )


def test_dynamic_workspace_bootstrap_runs_once_for_reused_terminal_session(
    monkeypatch, tmp_path
):
    workspace = _git_project(tmp_path / "owned")
    config = _dynamic_terminal_config(workspace)
    environment = _RecordingEnvironment()
    bootstrap_calls = 0
    create_calls = 0

    def fake_prepare(raw_config, *, task_id):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        metadata = workspace.stat(follow_symlinks=False)
        prepared = dict(raw_config)
        prepared.update({
            "host_cwd": str(workspace),
            "cwd": "/workspace",
            "docker_mount_cwd_to_workspace": True,
            "workspace_identity": {
                "path": str(workspace),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            },
        })
        return prepared

    def fake_create(**_kwargs):
        nonlocal create_calls
        create_calls += 1
        return environment

    _isolate_terminal_environment_state(monkeypatch, config)
    monkeypatch.setattr("tools.workspace_bootstrap.prepare_workspace_only_config", fake_prepare)
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create)

    first = json.loads(terminal_tool.terminal_tool("printf first", task_id="session-a"))
    second = json.loads(terminal_tool.terminal_tool("printf second", task_id="session-a"))

    assert first["exit_code"] == second["exit_code"] == 0
    assert environment.commands == ["printf first", "printf second"]
    assert bootstrap_calls == 1
    assert create_calls == 1


def test_concurrent_dynamic_terminal_calls_converge_on_one_bootstrap(
    monkeypatch, tmp_path
):
    workspace = _git_project(tmp_path / "owned")
    config = _dynamic_terminal_config(workspace)
    environment = _RecordingEnvironment()
    bootstrap_calls = 0
    create_calls = 0
    count_lock = threading.Lock()

    def fake_prepare(raw_config, *, task_id):
        nonlocal bootstrap_calls
        with count_lock:
            bootstrap_calls += 1
        metadata = workspace.stat(follow_symlinks=False)
        prepared = dict(raw_config)
        prepared.update({
            "host_cwd": str(workspace),
            "cwd": "/workspace",
            "docker_mount_cwd_to_workspace": True,
            "workspace_identity": {
                "path": str(workspace),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            },
        })
        return prepared

    def fake_create(**_kwargs):
        nonlocal create_calls
        with count_lock:
            create_calls += 1
        return environment

    _isolate_terminal_environment_state(monkeypatch, config)
    monkeypatch.setattr("tools.workspace_bootstrap.prepare_workspace_only_config", fake_prepare)
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda command: json.loads(
                terminal_tool.terminal_tool(command, task_id="session-a")
            ),
            ("printf first", "printf second"),
        ))

    assert all(result["exit_code"] == 0 for result in results)
    assert bootstrap_calls == 1
    assert create_calls == 1


def test_terminal_refuses_workspace_swap_after_bootstrap_before_environment_creation(
    monkeypatch, tmp_path
):
    workspace = _git_project(tmp_path / "owned")
    foreign = _git_project(tmp_path / "foreign")
    parked = tmp_path / "parked"
    stat_result = workspace.stat(follow_symlinks=False)
    config = _dynamic_terminal_config(workspace)
    create_calls = 0

    def prepare_then_swap(raw_config, *, task_id):
        prepared = dict(raw_config)
        prepared.update({
            "host_cwd": str(workspace),
            "cwd": "/workspace",
            "docker_mount_cwd_to_workspace": True,
            "workspace_identity": {
                "path": str(workspace),
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
            },
        })
        workspace.rename(parked)
        workspace.symlink_to(foreign, target_is_directory=True)
        return prepared

    def fake_create(**_kwargs):
        nonlocal create_calls
        create_calls += 1
        return _RecordingEnvironment()

    _isolate_terminal_environment_state(monkeypatch, config)
    monkeypatch.setattr(
        "tools.workspace_bootstrap.prepare_workspace_only_config", prepare_then_swap
    )
    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create)

    result = json.loads(terminal_tool.terminal_tool("printf unsafe", task_id="session-a"))

    assert result["exit_code"] != 0
    assert create_calls == 0


def test_execute_code_treats_dynamic_workspace_bootstrap_as_host_access_before_guard(
    monkeypatch, tmp_path
):
    workspace = _git_project(tmp_path / "owned")
    config = _dynamic_terminal_config(workspace)
    observed = {}

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)

    def deny_guard(_code, _env_type, *, has_host_access):
        observed["has_host_access"] = has_host_access
        return {"approved": False, "message": "approval required"}

    monkeypatch.setattr("tools.approval.check_execute_code_guard", deny_guard)
    monkeypatch.setattr(
        code_execution_tool,
        "_execute_remote",
        lambda *_args, **_kwargs: pytest.fail("environment must not be created before guard"),
    )

    result = json.loads(code_execution_tool.execute_code("print('unsafe')", task_id="session-a"))

    assert observed["has_host_access"] is True
    assert result["status"] == "error"
    assert result["tool_calls_made"] == 0


def test_workspace_only_builds_a_closed_host_mount_set(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    calls = _fake_docker(monkeypatch)

    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )

    run_command = next(command for command in calls if command[1:3] == ["run", "-d"])
    volumes = [
        run_command[index + 1]
        for index, value in enumerate(run_command[:-1])
        if value == "-v"
    ]

    assert "--read-only" in run_command
    assert "--network=none" in run_command
    assert "--user" in run_command
    assert f"{project.resolve()}:/workspace:rw" in volumes
    private_git_mounts = [volume for volume in volumes if volume.endswith(":/hermes-git:rw")]
    assert len(private_git_mounts) == 1
    assert not private_git_mounts[0].startswith(f"{(project / '.git').resolve()}:")
    dotgit_masks = [volume for volume in volumes if volume.endswith(":/workspace/.git:ro")]
    assert len(dotgit_masks) == 1
    assert not dotgit_masks[0].startswith(f"{(project / '.git').resolve()}:")
    assert len(volumes) == 3
    assert "/cache:rw,exec,nosuid,size=2g" in run_command
    assert "HOME=/home" in run_command
    assert "XDG_CACHE_HOME=/cache" in run_command
    assert "TMPDIR=/tmp" in run_command
    assert "GIT_OPTIONAL_LOCKS=0" in run_command
    assert "GIT_DIR=/hermes-git" in run_command
    assert "GIT_WORK_TREE=/workspace" in run_command
    assert environment._labels["hermes-egress"] == "off"


def test_dynamic_workspace_uses_private_named_volumes_and_reaches_docker_run(
    monkeypatch, tmp_path
):
    project = _git_project(tmp_path / "owned")
    metadata = project.stat(follow_symlinks=False)
    identity = {
        "path": str(project),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    calls = _fake_docker(monkeypatch)
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_initialize_private_workspace",
        lambda self: None,
        raising=False,
    )

    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
        workspace_identity=identity,
        workspace_transport="volume",
    )

    run_command = next(command for command in calls if command[1:3] == ["run", "-d"])
    volumes = [
        run_command[index + 1]
        for index, value in enumerate(run_command[:-1])
        if value == "-v"
    ]
    assert len([volume for volume in volumes if volume.endswith(":/workspace:rw")]) == 1
    assert len([volume for volume in volumes if volume.endswith(":/hermes-git:rw")]) == 1
    assert all(str(project) not in volume for volume in volumes)
    assert all(".hermes-git-broker-" not in volume for volume in volumes)
    volume_creates = [command for command in calls if command[1:3] == ["volume", "create"]]
    assert len(volume_creates) == 2
    for command in volume_creates:
        labels = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--label"
        }
        assert "hermes-agent=1" in labels
        assert "hermes-workspace-only=1" in labels
        assert "hermes-task-id=code-a" in labels
        assert any(label.startswith("hermes-profile=") for label in labels)
        assert any(label.startswith("hermes-owner-pid=") for label in labels)
        assert any(label.startswith("hermes-owner-start=") for label in labels)
    assert environment._workspace_volume
    assert environment._git_volume


def test_dynamic_workspace_without_volume_transport_keeps_path_bind_fail_closed(
    monkeypatch, tmp_path
):
    project = _git_project(tmp_path / "owned")
    metadata = project.stat(follow_symlinks=False)
    identity = {
        "path": str(project),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    calls = _fake_docker(monkeypatch)

    with pytest.raises(
        RuntimeError, match="path-based bind mounts cannot guarantee the pinned inode"
    ):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(project),
            auto_mount_cwd=True,
            workspace_only=True,
            workspace_identity=identity,
        )

    assert not any(command[1:3] == ["run", "-d"] for command in calls)


def test_dynamic_workspace_exports_then_publishes_files_and_git_atomically(
    monkeypatch, tmp_path
):
    from tools.workspace_staging import archive_tree

    project = _git_project(tmp_path / "owned")
    metadata = project.stat(follow_symlinks=False)
    identity = {"path": str(project), "device": metadata.st_dev, "inode": metadata.st_ino}
    _fake_docker(monkeypatch)
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_initialize_private_workspace",
        lambda self: None,
        raising=False,
    )
    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
        workspace_identity=identity,
        workspace_transport="volume",
    )

    exported_workspace = tmp_path / "container-workspace"
    shutil.copytree(project, exported_workspace, ignore=shutil.ignore_patterns(".git"))
    (exported_workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (exported_workspace / "created.txt").write_text("created\n", encoding="utf-8")
    exported_git = environment._git_broker_gitdir
    assert exported_git is not None
    docker_env._run_git_checked(
        ["--git-dir", str(exported_git), "--work-tree", str(exported_workspace), "add", "-A"]
    )
    docker_env._run_git_checked(
        [
            "--git-dir",
            str(exported_git),
            "--work-tree",
            str(exported_workspace),
            "commit",
            "-m",
            "container candidate",
        ]
    )
    bundle, private_head = _index_transport_bundle(
        exported_git,
        tmp_path / "container.bundle",
        expected_head=environment._git_broker_head,
    )
    exported_paths = []

    def export_workspace(path):
        exported_paths.append(path)
        assert path == "/workspace"
        return archive_tree(exported_workspace)

    monkeypatch.setattr(environment, "_export_private_tree", export_workspace)
    monkeypatch.setattr(
        environment, "_export_private_bundle", lambda: (bundle.read_bytes(), private_head)
    )
    monkeypatch.setattr(
        environment, "_export_private_index", lambda: (exported_git / "index").read_bytes()
    )
    finalized = []
    monkeypatch.setattr(environment, "_sync_workspace_git_broker_locked", lambda: finalized.append(True))

    environment._sync_private_workspace()

    assert (project / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (project / "created.txt").read_text(encoding="utf-8") == "created\n"
    assert finalized == [True]


def test_dynamic_workspace_restores_staged_new_file_without_commit(monkeypatch, tmp_path):
    from tools.workspace_staging import archive_tree

    project = _git_project(tmp_path / "owned")
    metadata = project.stat(follow_symlinks=False)
    identity = {
        "path": str(project),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    _fake_docker(monkeypatch)
    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
        workspace_identity=identity,
        workspace_transport="volume",
    )

    exported_workspace = tmp_path / "container-workspace"
    shutil.copytree(project, exported_workspace, ignore=shutil.ignore_patterns(".git"))
    (exported_workspace / "staged-only.txt").write_text("staged\n", encoding="utf-8")
    exported_git = environment._git_broker_gitdir
    assert exported_git is not None
    docker_env._run_git_checked(
        [
            "--git-dir",
            str(exported_git),
            "--work-tree",
            str(exported_workspace),
            "add",
            "staged-only.txt",
        ]
    )
    monkeypatch.setattr(
        environment,
        "_export_private_tree",
        lambda path: archive_tree(exported_workspace) if path == "/workspace" else None,
    )
    bundle, private_head = _index_transport_bundle(
        exported_git,
        tmp_path / "staged-index.bundle",
        expected_head=environment._git_broker_head,
    )
    monkeypatch.setattr(
        environment, "_export_private_bundle", lambda: (bundle.read_bytes(), private_head)
    )
    monkeypatch.setattr(
        environment, "_export_private_index", lambda: (exported_git / "index").read_bytes()
    )

    environment._sync_private_workspace()

    assert (project / "staged-only.txt").read_text(encoding="utf-8") == "staged\n"
    staged = docker_env._run_git_checked(
        ["-C", str(project), "diff", "--cached", "--name-only"]
    ).stdout.splitlines()
    assert staged == ["staged-only.txt"]


def test_workspace_publication_journal_recovers_old_or_started_candidate_state(tmp_path):
    project = _git_project(tmp_path / "owned")
    metadata = docker_env._workspace_git_metadata(project.resolve())
    candidate_workspace = tmp_path / "candidate-workspace"
    shutil.copytree(project, candidate_workspace, ignore=shutil.ignore_patterns(".git"))
    (candidate_workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (candidate_workspace / "created.txt").write_text("candidate\n", encoding="utf-8")
    candidate_root, candidate_git = docker_env._prepare_workspace_git_broker(metadata)
    try:
        docker_env._run_git_checked(
            ["--git-dir", str(candidate_git), "--work-tree", str(candidate_workspace), "add", "-A"]
        )
        docker_env._run_git_checked(
            [
                "--git-dir",
                str(candidate_git),
                "--work-tree",
                str(candidate_workspace),
                "commit",
                "-m",
                "candidate",
            ]
        )
        candidate_head = docker_env._run_git_checked(
            ["--git-dir", str(candidate_git), "rev-parse", "HEAD"]
        ).stdout.strip()

        docker_env._begin_workspace_publication_journal(
            metadata,
            candidate_workspace=candidate_workspace,
            candidate_git=candidate_git,
            expected_head=metadata.head,
            candidate_head=candidate_head,
        )
        (project / "module.py").write_text("PARTIAL\n", encoding="utf-8")
        (project / "sandbox_tests" / "test_module.py").unlink()
        assert docker_env._recover_workspace_publication(metadata)
        assert (project / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert (project / "sandbox_tests" / "test_module.py").exists()
        assert not (project / "created.txt").exists()

        docker_env._begin_workspace_publication_journal(
            metadata,
            candidate_workspace=candidate_workspace,
            candidate_git=candidate_git,
            expected_head=metadata.head,
            candidate_head=candidate_head,
        )
        docker_env._run_git_checked(
            [
                "-C",
                str(project),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(candidate_git),
                candidate_head,
            ]
        )
        docker_env._run_git_checked(
            ["-C", str(project), "update-ref", metadata.branch_ref, candidate_head, metadata.head]
        )
        (project / "module.py").write_text("PARTIAL AGAIN\n", encoding="utf-8")
        assert docker_env._recover_workspace_publication(metadata)
        assert (project / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
        assert (project / "created.txt").read_text(encoding="utf-8") == "candidate\n"
        assert docker_env._run_git_checked(
            ["-C", str(project), "rev-parse", "HEAD"]
        ).stdout.strip() == candidate_head
        assert docker_env._run_git_checked(
            ["-C", str(project), "status", "--porcelain"]
        ).stdout == ""

        refreshed = docker_env._workspace_git_metadata(project.resolve())
        docker_env._begin_workspace_publication_journal(
            refreshed,
            candidate_workspace=candidate_workspace,
            candidate_git=candidate_git,
            expected_head=candidate_head,
            candidate_head=candidate_head,
        )
        foreign_lock = refreshed.gitdir / "index.lock"
        foreign_lock.write_bytes(b"foreign lock")
        with pytest.raises(RuntimeError, match="concurrent.*index lock"):
            docker_env._recover_workspace_publication(refreshed)
        assert foreign_lock.read_bytes() == b"foreign lock"
        foreign_lock.unlink()
        journal = docker_env._workspace_publication_journal(refreshed)
        journal_payload = json.loads((journal / "metadata.json").read_text(encoding="utf-8"))
        owned_lock = refreshed.gitdir / f"index.hermes-{journal_payload['index_lock_token']}"
        assert owned_lock.exists()
        os.link(owned_lock, foreign_lock, follow_symlinks=False)
        assert docker_env._recover_workspace_publication(refreshed)
        assert not foreign_lock.exists()
        assert not owned_lock.exists()
    finally:
        shutil.rmtree(candidate_root, ignore_errors=True)
        docker_env._discard_workspace_publication_journal(metadata)


@pytest.mark.parametrize(
    ("journal_state", "expected_value"),
    [
        ("prepared", "VALUE = 1\n"),
        ("publishing", "VALUE = 1\n"),
        ("metadata_published", "VALUE = 2\n"),
        ("committed", "VALUE = 2\n"),
    ],
)
def test_workspace_publication_recovery_uses_durable_state_for_identical_head_and_index(
    tmp_path,
    journal_state,
    expected_value,
):
    project = _git_project(tmp_path / "owned")
    metadata = docker_env._workspace_git_metadata(project.resolve())
    candidate_workspace = tmp_path / "candidate-workspace"
    shutil.copytree(project, candidate_workspace, ignore=shutil.ignore_patterns(".git"))
    (candidate_workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    candidate_root, candidate_git = docker_env._prepare_workspace_git_broker(metadata)
    try:
        candidate_head = docker_env._run_git_checked(
            ["--git-dir", str(candidate_git), "rev-parse", "HEAD"]
        ).stdout.strip()
        assert candidate_head == metadata.head
        (candidate_git / "index").write_bytes((metadata.gitdir / "index").read_bytes())
        assert (candidate_git / "index").read_bytes() == (metadata.gitdir / "index").read_bytes()

        docker_env._begin_workspace_publication_journal(
            metadata,
            candidate_workspace=candidate_workspace,
            candidate_git=candidate_git,
            expected_head=metadata.head,
            candidate_head=candidate_head,
        )
        docker_env._write_journal_state(metadata, journal_state)
        (project / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

        assert docker_env._recover_workspace_publication(metadata)
        assert (project / "module.py").read_text(encoding="utf-8") == expected_value
        assert docker_env._run_git_checked(
            ["-C", str(project), "rev-parse", "HEAD"]
        ).stdout.strip() == metadata.head
    finally:
        docker_env._discard_workspace_publication_journal(metadata)
        shutil.rmtree(candidate_root, ignore_errors=True)


@pytest.mark.parametrize("unsafe_state", ["unknown", "symlink", "hardlink"])
def test_workspace_publication_recovery_rejects_unsafe_or_unknown_state(
    tmp_path,
    unsafe_state,
):
    project = _git_project(tmp_path / "owned")
    metadata = docker_env._workspace_git_metadata(project.resolve())
    candidate_workspace = tmp_path / "candidate-workspace"
    shutil.copytree(project, candidate_workspace, ignore=shutil.ignore_patterns(".git"))
    candidate_root, candidate_git = docker_env._prepare_workspace_git_broker(metadata)
    try:
        docker_env._begin_workspace_publication_journal(
            metadata,
            candidate_workspace=candidate_workspace,
            candidate_git=candidate_git,
            expected_head=metadata.head,
            candidate_head=metadata.head,
        )
        journal = docker_env._workspace_publication_journal(metadata)
        state_path = journal / "state"
        if unsafe_state == "unknown":
            state_path.write_text("unknown\n", encoding="ascii")
        elif unsafe_state == "symlink":
            state_path.unlink()
            state_path.symlink_to("metadata.json")
        else:
            os.link(state_path, journal / "state-link", follow_symlinks=False)

        with pytest.raises(RuntimeError, match="journal state"):
            docker_env._recover_workspace_publication(metadata)
    finally:
        docker_env._discard_workspace_publication_journal(metadata)
        shutil.rmtree(candidate_root, ignore_errors=True)


def test_dynamic_workspace_cleanup_removes_private_named_volumes(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    metadata = project.stat(follow_symlinks=False)
    identity = {"path": str(project), "device": metadata.st_dev, "inode": metadata.st_ino}
    calls = _fake_docker(monkeypatch)
    monkeypatch.setattr(
        docker_env.DockerEnvironment,
        "_initialize_private_workspace",
        lambda self: None,
        raising=False,
    )
    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
        workspace_identity=identity,
        workspace_transport="volume",
    )
    volume_names = {environment._workspace_volume, environment._git_volume}

    environment.cleanup(force_remove=True)
    assert environment.wait_for_cleanup(timeout=5)

    removed = {
        command[-1]
        for command in calls
        if command[1:4] == ["volume", "rm", "-f"]
    }
    assert removed == volume_names


def test_workspace_git_broker_excludes_objects_from_unowned_branches(tmp_path):
    project = _git_project(tmp_path / "owned")
    owned_branch = subprocess.run(
        ["git", "-C", str(project), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(project), "switch", "-qc", "foreign"], check=True)
    secret = project / "foreign-secret.txt"
    secret.write_text("must not enter owned broker\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", secret.name], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-qm", "foreign secret"], check=True)
    secret_oid = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD:foreign-secret.txt"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(project), "switch", "-q", owned_branch], check=True)

    broker_root, private_git = docker_env._prepare_workspace_git_broker(
        docker_env._workspace_git_metadata(project.resolve())
    )
    try:
        leaked = subprocess.run(
            ["git", "--git-dir", str(private_git), "cat-file", "-e", secret_oid],
            capture_output=True,
            text=True,
            check=False,
        )
        assert leaked.returncode != 0
    finally:
        shutil.rmtree(broker_root, ignore_errors=True)


def test_workspace_only_disables_cross_process_reuse(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    calls = _fake_docker(monkeypatch)

    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        persist_across_processes=True,
        workspace_only=True,
    )

    assert environment._persist_across_processes is False
    assert not any(command[1:2] == ["ps"] for command in calls)


@pytest.mark.parametrize("failure_boundary", ["fetch", "update-ref", "copyfileobj", "replace"])
def test_git_broker_sync_failure_does_not_advance_host_ref(
    monkeypatch, tmp_path, failure_boundary
):
    project = _git_project(tmp_path / "owned")
    _fake_docker(monkeypatch)
    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )
    old_head = environment._git_broker_head
    private_git = environment._git_broker_gitdir
    assert private_git is not None
    staged = project / "broker-transaction.txt"
    staged.write_text("transaction\n", encoding="utf-8")
    docker_env._run_git_checked(
        ["--git-dir", str(private_git), "--work-tree", str(project), "add", staged.name]
    )
    docker_env._run_git_checked(
        [
            "--git-dir",
            str(private_git),
            "--work-tree",
            str(project),
            "commit",
            "-m",
            "broker transaction",
        ]
    )

    if failure_boundary in {"fetch", "update-ref"}:
        real_git_checked = docker_env._run_git_checked
        failed = False

        def fail_git_boundary(args, **kwargs):
            nonlocal failed
            command = args[args.index("-C") + 2 :] if "-C" in args else args
            if not failed and command and command[0] == failure_boundary:
                failed = True
                raise OSError(f"{failure_boundary} failed")
            return real_git_checked(args, **kwargs)

        monkeypatch.setattr(docker_env, "_run_git_checked", fail_git_boundary)
    elif failure_boundary == "copyfileobj":
        monkeypatch.setattr(
            docker_env.shutil,
            "copyfileobj",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
        )
    else:
        monkeypatch.setattr(
            docker_env.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
        )

    with pytest.raises(OSError, match="failed"):
        environment._sync_workspace_git_broker()

    host_head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert host_head == old_head


def test_git_broker_sync_holds_canonical_index_lock_while_publishing(
    monkeypatch, tmp_path
):
    project = _git_project(tmp_path / "owned")
    _fake_docker(monkeypatch)
    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )
    private_git = environment._git_broker_gitdir
    assert private_git is not None

    broker_file = project / "broker-staged.txt"
    broker_file.write_text("broker\n", encoding="utf-8")
    docker_env._run_git_checked(
        ["--git-dir", str(private_git), "--work-tree", str(project), "add", broker_file.name]
    )
    concurrent_file = project / "host-concurrent.txt"
    concurrent_file.write_text("host\n", encoding="utf-8")

    real_replace = docker_env.os.replace
    concurrent_add = None

    def add_during_index_publish(source, destination):
        nonlocal concurrent_add
        concurrent_add = subprocess.run(
            ["git", "-C", str(project), "add", concurrent_file.name],
            capture_output=True,
            text=True,
            check=False,
        )
        real_replace(source, destination)

    monkeypatch.setattr(docker_env.os, "replace", add_during_index_publish)

    environment._sync_workspace_git_broker()

    assert concurrent_add is not None
    assert concurrent_add.returncode != 0
    assert "index.lock" in concurrent_add.stderr
    staged = subprocess.run(
        ["git", "-C", str(project), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert staged == [broker_file.name]
    assert not (environment._workspace_git_metadata.gitdir / "index.lock").exists()


def test_workspace_only_does_not_forward_implicit_host_environment(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    _fake_docker(monkeypatch)
    monkeypatch.setenv("WORKSPACE_BOUNDARY_SECRET", "must-not-cross")
    monkeypatch.setattr(
        "tools.env_passthrough.get_all_passthrough",
        lambda: {"WORKSPACE_BOUNDARY_SECRET"},
    )

    environment = docker_env.DockerEnvironment(
        image="code-image",
        cwd="/workspace",
        timeout=60,
        task_id="code-a",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
    )

    assert environment._init_env_args == []


def test_workspace_only_fails_closed_when_docker_is_missing(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    monkeypatch.setattr(docker_env, "find_docker", lambda: None)

    with pytest.raises(RuntimeError, match="Docker executable not found"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(project),
            auto_mount_cwd=True,
            workspace_only=True,
        )


def test_workspace_only_rejects_images_with_implicit_volumes(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    calls = _fake_docker(monkeypatch, declared_volumes={"/unbounded": {}})

    with pytest.raises(ValueError, match="declares implicit writable volumes"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(project),
            auto_mount_cwd=True,
            workspace_only=True,
        )

    assert not any(command[1:3] == ["run", "-d"] for command in calls)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"volumes": ["/foreign:/foreign"]}, "docker_volumes"),
        ({"forward_env": ["TOKEN"]}, "docker_forward_env"),
        ({"env": {"TOKEN": "secret"}}, "docker_env"),
        ({"extra_args": ["--privileged"]}, "docker_extra_args"),
        ({"auto_mount_cwd": False}, "docker_mount_cwd_to_workspace"),
        ({"host_cwd": None}, "terminal.cwd"),
    ],
)
def test_workspace_only_rejects_unsafe_or_incomplete_configuration(
    monkeypatch, tmp_path, overrides, message
):
    project = _git_project(tmp_path / "owned")
    kwargs = {
        "image": "code-image",
        "cwd": "/workspace",
        "timeout": 60,
        "task_id": "code-a",
        "host_cwd": str(project),
        "auto_mount_cwd": True,
        "workspace_only": True,
    }
    kwargs.update(overrides)

    monkeypatch.setattr(
        docker_env,
        "find_docker",
        lambda: pytest.fail("invalid workspace-only config must fail before Docker"),
    )

    with pytest.raises(ValueError, match=message):
        docker_env.DockerEnvironment(**kwargs)


def test_workspace_only_requires_git_metadata(monkeypatch, tmp_path):
    project = tmp_path / "not-a-worktree"
    project.mkdir()
    monkeypatch.setattr(
        docker_env,
        "find_docker",
        lambda: pytest.fail("missing git metadata must fail before Docker"),
    )

    with pytest.raises(ValueError, match="Git worktree"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(project),
            auto_mount_cwd=True,
            workspace_only=True,
        )


def test_workspace_only_rejects_symlinked_dotgit_before_docker(monkeypatch, tmp_path):
    project = _git_project(tmp_path / "owned")
    real_git = tmp_path / "real-git"
    (project / ".git").rename(real_git)
    (project / ".git").symlink_to(real_git, target_is_directory=True)
    monkeypatch.setattr(
        docker_env,
        "find_docker",
        lambda: pytest.fail("symlinked Git metadata must fail before Docker"),
    )

    with pytest.raises(ValueError, match="symlink"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(project),
            auto_mount_cwd=True,
            workspace_only=True,
        )


def test_workspace_only_rejects_symlinked_linked_worktree_metadata(monkeypatch, tmp_path):
    source = _git_project(tmp_path / "source")
    owned = tmp_path / "owned"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-qb", "owned", str(owned)],
        check=True,
    )
    admin = Path(
        subprocess.run(
            ["git", "-C", str(owned), "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    commondir = admin / "commondir"
    real_commondir = admin / "commondir.real"
    commondir.rename(real_commondir)
    commondir.symlink_to(real_commondir.name)
    monkeypatch.setattr(
        docker_env,
        "find_docker",
        lambda: pytest.fail("symlinked Git metadata must fail before Docker"),
    )

    with pytest.raises(ValueError, match="symlink"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(owned),
            auto_mount_cwd=True,
            workspace_only=True,
        )


def test_workspace_only_rejects_intermediate_metadata_symlink(monkeypatch, tmp_path):
    source = _git_project(tmp_path / "source")
    owned = tmp_path / "owned"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-qb", "owned", str(owned)],
        check=True,
    )
    admin = Path(
        subprocess.run(
            ["git", "-C", str(owned), "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    metadata_alias = tmp_path / "metadata-alias"
    metadata_alias.symlink_to(source / ".git", target_is_directory=True)
    aliased_admin = metadata_alias / "worktrees" / admin.name
    assert not aliased_admin.is_symlink()
    (owned / ".git").write_text(f"gitdir: {aliased_admin}\n", encoding="utf-8")
    monkeypatch.setattr(
        docker_env,
        "find_docker",
        lambda: pytest.fail("intermediate metadata symlinks must fail before Docker"),
    )

    with pytest.raises(ValueError, match="symlink"):
        docker_env.DockerEnvironment(
            image="code-image",
            cwd="/workspace",
            timeout=60,
            task_id="code-a",
            host_cwd=str(owned),
            auto_mount_cwd=True,
            workspace_only=True,
        )


def test_workspace_only_config_is_bridged_everywhere():
    from tests.tools.test_terminal_config_env_sync import (
        _cli_env_map_keys,
        _gateway_env_map_keys,
        _save_config_env_sync_keys,
        _terminal_tool_env_var_names,
    )

    assert "docker_workspace_only" in _cli_env_map_keys()
    assert "docker_workspace_only" in _gateway_env_map_keys()
    assert "docker_workspace_only" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_WORKSPACE_ONLY" in _terminal_tool_env_var_names()


@pytest.fixture
def docker_host_tmp():
    """Use a macOS Docker-shared path rather than pytest's /private/tmp."""
    root = Path(__file__).resolve().parents[2] / ".docker-integration-tmp"
    root.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=root) as directory:
            yield Path(directory)
    finally:
        try:
            root.rmdir()
        except OSError:
            pass


@pytest.mark.integration
def test_workspace_only_real_container_blocks_foreign_writes_and_keeps_builds_working(
    docker_host_tmp,
    monkeypatch,
):
    tmp_path = docker_host_tmp
    docker = docker_env.find_docker()
    if not docker:
        pytest.skip("Docker CLI is unavailable")
    available = subprocess.run(
        [docker, "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    source = _git_project(tmp_path / "source")
    owned_a = tmp_path / "owned-a"
    owned_b = tmp_path / "owned-b"
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-qb", "code-a", str(owned_a)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "-qb", "code-b", str(owned_b)],
        check=True,
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    python_escape = foreign / "python-escape"
    symlink_escape = foreign / "symlink-escape"
    traversal_escape = tmp_path / "traversal-escape"
    host_cache_escape = tmp_path / "cache-escape"
    (owned_a / "escape-link").symlink_to(symlink_escape)
    internal_documents = _structured_documents(owned_a, "INTERNAL-DOCUMENT")
    foreign_documents = _structured_documents(foreign, "FOREIGN-DOCUMENT")
    for extension, foreign_document in foreign_documents.items():
        (owned_a / f"foreign-link{extension}").symlink_to(foreign_document)
    monkeypatch.setenv("WORKSPACE_BOUNDARY_SECRET", "must-not-cross")
    monkeypatch.setattr(
        "tools.env_passthrough.get_all_passthrough",
        lambda: {"WORKSPACE_BOUNDARY_SECRET"},
    )

    env_a = _workspace_only_environment(owned_a, task_id=f"code-a-{uuid.uuid4().hex}")
    env_b = _workspace_only_environment(owned_b, task_id=f"code-b-{uuid.uuid4().hex}")
    container_ids = [env_a._container_id, env_b._container_id]
    try:
        build = env_a.execute(
            "python -m compileall -q . && python -m unittest discover -s sandbox_tests -v",
            cwd="/workspace",
            timeout=120,
        )
        assert build["returncode"] == 0, build["output"]

        secret_probe = env_a.execute(
            "test -z \"${WORKSPACE_BOUNDARY_SECRET:-}\"",
            cwd="/workspace",
        )
        assert secret_probe["returncode"] == 0, secret_probe["output"]

        file_ops = ShellFileOperations(env_a)
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: file_ops)
        for extension in internal_documents:
            internal_read = json.loads(
                file_tools.read_file_tool(
                    f"/workspace/document{extension}",
                    task_id=f"structured-{extension}",
                )
            )
            assert internal_read.get("extracted_document") is True, internal_read
            assert "INTERNAL-DOCUMENT" in internal_read["content"]
            for blocked_path in (
                str(foreign_documents[extension]),
                f"/workspace/foreign-link{extension}",
            ):
                blocked_read = json.loads(
                    file_tools.read_file_tool(
                        blocked_path,
                        task_id=f"blocked-{extension}",
                    )
                )
                assert "error" in blocked_read, blocked_read
                assert "FOREIGN-DOCUMENT" not in blocked_read.get("content", "")

        internal_git = env_a.execute(
            "git status --porcelain && mkdir internal-patches && "
            "git format-patch -o /workspace/internal-patches -1 HEAD",
            cwd="/workspace",
        )
        assert internal_git["returncode"] == 0, internal_git["output"]
        assert len(list((owned_a / "internal-patches").glob("*.patch"))) == 1

        host_dotgit = owned_a / ".git"
        host_dotgit_before = host_dotgit.read_bytes()
        dotgit_read = env_a.execute("cat /workspace/.git", cwd="/workspace")
        assert dotgit_read["returncode"] == 0, dotgit_read["output"]
        assert dotgit_read["output"].strip() == "gitdir: /hermes-git"
        assert str(source) not in dotgit_read["output"]
        dotgit_write = env_a.execute(
            "printf PWNED > /workspace/.git",
            cwd="/workspace",
        )
        assert dotgit_write["returncode"] != 0
        assert host_dotgit.read_bytes() == host_dotgit_before

        (owned_a / "committed.txt").write_text("owned commit\n", encoding="utf-8")
        autonomous_commit = env_a.execute(
            "git add committed.txt && git commit -m 'owned container commit'",
            cwd="/workspace",
        )
        assert autonomous_commit["returncode"] == 0, autonomous_commit["output"]
        assert subprocess.run(
            ["git", "-C", str(owned_a), "log", "-1", "--format=%s"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == "owned container commit"

        common_git = Path(
            subprocess.run(
                ["git", "-C", str(owned_a), "rev-parse", "--git-common-dir"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if not common_git.is_absolute():
            common_git = owned_a / common_git
        common_git = common_git.resolve()
        config_before = (common_git / "config").read_bytes()
        hook_sentinel = common_git / "hooks" / "hermes-sentinel"
        hook_sentinel.write_text("HOST-HOOK-SENTINEL\n", encoding="utf-8")
        branch_sentinel = common_git / "refs" / "heads" / "code-b"
        branch_before = branch_sentinel.read_bytes()
        metadata_probe = env_a.execute(
            " && ".join(
                [
                    f"test ! -e {str(common_git / 'config')!r}",
                    f"test ! -e {str(common_git / 'hooks')!r}",
                    f"test ! -e {str(common_git / 'refs')!r}",
                    f"test ! -e {str(common_git / 'worktrees' / owned_b.name)!r}",
                ]
            ),
            cwd="/workspace",
        )
        assert metadata_probe["returncode"] == 0, metadata_probe["output"]
        for metadata_target in (common_git / "config", hook_sentinel, branch_sentinel):
            metadata_write = env_a.execute(
                f"printf PWNED > {str(metadata_target)!r}",
                cwd="/workspace",
            )
            assert metadata_write["returncode"] != 0
        assert (common_git / "config").read_bytes() == config_before
        assert hook_sentinel.read_text(encoding="utf-8") == "HOST-HOOK-SENTINEL\n"
        assert branch_sentinel.read_bytes() == branch_before

        internal = env_a.execute(
            "python -c \"open('/workspace/internal.txt','w').write('ok')\"",
            cwd="/workspace",
        )
        assert internal["returncode"] == 0, internal["output"]
        assert (owned_a / "internal.txt").read_text(encoding="utf-8") == "ok"

        python_result = env_a.execute(
            f"python -c \"open({str(python_escape)!r},'w').write('escaped')\"",
            cwd="/workspace",
        )
        assert python_result["returncode"] != 0
        assert not python_escape.exists()

        patch_result = env_a.execute(
            "git format-patch -o../foreign -1 HEAD",
            cwd="/workspace",
        )
        assert patch_result["returncode"] != 0
        assert list(foreign.glob("*.patch")) == []

        symlink_result = env_a.execute(
            "python -c \"open('/workspace/escape-link','w').write('escaped')\"",
            cwd="/workspace",
        )
        assert symlink_result["returncode"] != 0
        assert not symlink_escape.exists()

        traversal_result = env_a.execute(
            "python -c \"open('../traversal-escape','w').write('escaped')\"",
            cwd="/workspace",
        )
        assert traversal_result["returncode"] != 0
        assert not traversal_escape.exists()

        cache_result = env_a.execute(
            "mkdir -p \"$XDG_CACHE_HOME/python\" && "
            "python -c \"open('/cache/python/canary','w').write('cache')\" && "
            "python -c \"open('/tmp/canary','w').write('tmp')\"",
            cwd="/workspace",
        )
        assert cache_result["returncode"] == 0, cache_result["output"]

        host_cache_result = env_a.execute(
            f"python -c \"open({str(host_cache_escape)!r},'w').write('escaped')\"",
            cwd="/workspace",
        )
        assert host_cache_result["returncode"] != 0
        assert not host_cache_escape.exists()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda pair: pair[0].execute(
                        f"python -c \"open('/workspace/{pair[1]}','w').write('ok')\"",
                        cwd="/workspace",
                    ),
                    [(env_a, "a.txt"), (env_b, "b.txt")],
                )
            )
        assert [result["returncode"] for result in results] == [0, 0]
        assert (owned_a / "a.txt").read_text(encoding="utf-8") == "ok"
        assert (owned_b / "b.txt").read_text(encoding="utf-8") == "ok"
        assert not (owned_a / "b.txt").exists()
        assert not (owned_b / "a.txt").exists()

        (owned_b / "host-staged.txt").write_text("host index owner\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(owned_b), "add", "host-staged.txt"],
            check=True,
        )
        branch_before_conflict = subprocess.run(
            ["git", "-C", str(owned_b), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (owned_b / "broker-commit.txt").write_text("broker\n", encoding="utf-8")
        refused_commit = env_b.execute(
            "git add broker-commit.txt && git commit -m 'must fail closed'",
            cwd="/workspace",
        )
        assert refused_commit["returncode"] == 126, refused_commit["output"]
        assert subprocess.run(
            ["git", "-C", str(owned_b), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() == branch_before_conflict
    finally:
        env_a.cleanup(force_remove=True)
        env_b.cleanup(force_remove=True)
        env_a.wait_for_cleanup(timeout=30)
        env_b.wait_for_cleanup(timeout=30)

    assert not list(tmp_path.glob(".hermes-git-broker-*"))
    for container_id in container_ids:
        inspected = subprocess.run(
            [docker, "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert inspected.returncode != 0


@pytest.mark.integration
def test_dynamic_workspace_reaper_removes_unattached_volume_from_dead_owner(docker_host_tmp):
    del docker_host_tmp
    docker = docker_env.find_docker()
    if not docker:
        pytest.skip("Docker CLI is unavailable")
    available = subprocess.run(
        [docker, "version"], capture_output=True, text=True, timeout=10, check=False
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    volume = f"hermes-workspace-{uuid.uuid4().hex}"
    profile = docker_env._sanitize_label_value(docker_env._get_active_profile_name())
    created = subprocess.run(
        [
            docker,
            "volume",
            "create",
            "--label",
            "hermes-agent=1",
            "--label",
            "hermes-workspace-only=1",
            "--label",
            "hermes-private-kind=workspace",
            "--label",
            "hermes-task-id=dead-owner",
            "--label",
            f"hermes-profile={profile}",
            "--label",
            "hermes-owner-pid=99999999",
            "--label",
            "hermes-owner-start=1",
            volume,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    try:
        docker_env.reap_orphan_containers(
            max_age_seconds=0,
            profile_filter=docker_env._get_active_profile_name(),
            docker_exe=docker,
        )
        assert subprocess.run(
            [docker, "volume", "inspect", volume],
            capture_output=True,
            timeout=10,
            check=False,
        ).returncode != 0
    finally:
        subprocess.run([docker, "volume", "rm", "-f", volume], capture_output=True, check=False)


@pytest.mark.integration
def test_dynamic_workspace_sigkill_is_reaped_with_its_private_volumes(docker_host_tmp):
    tmp_path = docker_host_tmp
    docker = docker_env.find_docker()
    if not docker:
        pytest.skip("Docker CLI is unavailable")
    available = subprocess.run(
        [docker, "version"], capture_output=True, text=True, timeout=10, check=False
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    image = os.environ.get(
        "HERMES_TEST_CODE_IMAGE",
        "nikolaik/python-nodejs:python3.11-nodejs20",
    )
    present = subprocess.run(
        [docker, "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if present.returncode != 0:
        pytest.skip("Docker test image is not present locally; integration test never pulls")

    project = _git_project(tmp_path / "sigkill-owned")
    task_id = f"sigkill-{uuid.uuid4().hex}"
    child_script = (
        "import json, os, signal, sys; "
        "from pathlib import Path; "
        "from tools.environments.docker import DockerEnvironment; "
        "p=Path(sys.argv[1]); st=p.stat(follow_symlinks=False); "
        "e=DockerEnvironment(image=sys.argv[2], cwd='/workspace', timeout=120, "
        "task_id=sys.argv[3], host_cwd=str(p), auto_mount_cwd=True, "
        "workspace_only=True, workspace_identity={'path':str(p),'device':st.st_dev,'inode':st.st_ino}, "
        "workspace_transport='volume'); "
        "print(json.dumps({'container':e._container_id,'volumes':[e._workspace_volume,e._git_volume]}), flush=True); "
        "os.kill(os.getpid(), signal.SIGKILL)"
    )
    child = subprocess.run(
        [sys.executable, "-c", child_script, str(project), image, task_id],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr
    details = json.loads(child.stdout.splitlines()[-1])
    container_id = details["container"]
    volumes = set(details["volumes"])
    try:
        removed = docker_env.reap_orphan_containers(
            max_age_seconds=0,
            profile_filter=docker_env._get_active_profile_name(),
            docker_exe=docker,
        )
        assert removed >= 1
        assert subprocess.run(
            [docker, "inspect", container_id], capture_output=True, timeout=10, check=False
        ).returncode != 0
        remaining = subprocess.run(
            [docker, "volume", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()
        assert not (volumes & set(remaining))
    finally:
        subprocess.run([docker, "rm", "-f", container_id], capture_output=True, check=False)
        for volume in volumes:
            subprocess.run([docker, "volume", "rm", "-f", volume], capture_output=True, check=False)


@pytest.mark.integration
def test_dynamic_workspace_real_container_round_trips_files_and_git_without_host_bind(
    docker_host_tmp,
):
    tmp_path = docker_host_tmp
    docker = docker_env.find_docker()
    if not docker:
        pytest.skip("Docker CLI is unavailable")
    available = subprocess.run(
        [docker, "version"], capture_output=True, text=True, timeout=10, check=False
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    image = os.environ.get(
        "HERMES_TEST_CODE_IMAGE",
        "nikolaik/python-nodejs:python3.11-nodejs20",
    )
    present = subprocess.run(
        [docker, "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if present.returncode != 0:
        pytest.skip("Docker test image is not present locally; integration test never pulls")

    project = _git_project(tmp_path / "dynamic-owned")
    (project / "delete.txt").write_text("delete\n", encoding="utf-8")
    metadata = project.stat(follow_symlinks=False)
    identity = {"path": str(project), "device": metadata.st_dev, "inode": metadata.st_ino}
    environment = docker_env.DockerEnvironment(
        image=image,
        cwd="/workspace",
        timeout=120,
        task_id=f"dynamic-{uuid.uuid4().hex}",
        host_cwd=str(project),
        auto_mount_cwd=True,
        workspace_only=True,
        workspace_identity=identity,
        workspace_transport="volume",
    )
    container_id = environment._container_id
    volume_names = {environment._workspace_volume, environment._git_volume}
    try:
        mutation = environment.execute(
            "printf 'VALUE = 2\\n' > module.py && "
            "python -c \"from pathlib import Path; "
            "p=Path('sandbox_tests/test_module.py'); "
            "p.write_text(p.read_text().replace('VALUE, 1', 'VALUE, 2'))\" && "
            "rm delete.txt && printf 'created\\n' > created.txt && "
            "python -m unittest discover -s sandbox_tests -v && "
            "git add -A && git commit -m 'dynamic staging canary'",
            cwd="/workspace",
            timeout=120,
        )
        assert mutation["returncode"] == 0, mutation["output"]
        assert (project / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
        assert (project / "created.txt").read_text(encoding="utf-8") == "created\n"
        assert not (project / "delete.txt").exists()
        assert subprocess.run(
            ["git", "-C", str(project), "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() == "dynamic staging canary"

        inspect = subprocess.run(
            [docker, "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        details = json.loads(inspect.stdout)[0]
        mounts = details["Mounts"]
        selected = {
            mount["Destination"]: mount
            for mount in mounts
            if mount["Destination"] in {"/workspace", "/hermes-git"}
        }
        assert set(selected) == {"/workspace", "/hermes-git"}
        assert all(mount["Type"] == "volume" for mount in selected.values())
        assert all(str(project) not in json.dumps(mount) for mount in selected.values())
        assert details["HostConfig"]["ReadonlyRootfs"] is True
        assert details["HostConfig"]["NetworkMode"] == "none"
    finally:
        environment.cleanup(force_remove=True)
        assert environment.wait_for_cleanup(timeout=30)

    assert container_id
    assert subprocess.run(
        [docker, "inspect", container_id], capture_output=True, timeout=10, check=False
    ).returncode != 0
    remaining = subprocess.run(
        [docker, "volume", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.splitlines()
    assert not (volume_names & set(remaining))
