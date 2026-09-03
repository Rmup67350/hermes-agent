"""Host-side bootstrap contracts for workspace-only Docker environments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.workspace_bootstrap import (
    WorkspaceBootstrapError,
    prepare_workspace_only_config,
    uses_dynamic_workspace_bootstrap,
)


def _bootstrap_spec(lane: Path) -> dict[str, str]:
    return {
        "registry": "/registry",
        "policy": "/policy.json",
        "profile": "code-a",
        "agent": "code-a",
        "key": "HER-96",
        "lane_sha256": hashlib.sha256(lane.read_bytes()).hexdigest(),
    }


def _config(lane: Path) -> dict:
    return {
        "env_type": "docker",
        "cwd": "/root",
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": False,
        "docker_workspace_only": True,
        "workspace_bootstrap": _bootstrap_spec(lane),
    }


def test_bootstrap_injects_canonical_host_cwd_before_environment_creation(monkeypatch, tmp_path):
    lane = tmp_path / "factory_lane.py"
    lane.write_text("trusted lane\n")
    workspace = tmp_path / "owned"
    workspace.mkdir()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[:2] == [sys.executable, "-c"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"worktree": str(workspace)}), "")
        assert argv[:3] == ["git", "-C", str(workspace)]
        return subprocess.CompletedProcess(argv, 0, str(workspace) + "\n", "")

    monkeypatch.setattr("tools.workspace_bootstrap._lane_script", lambda: lane)
    monkeypatch.setattr("tools.workspace_bootstrap.subprocess.run", fake_run)

    prepared = prepare_workspace_only_config(_config(lane), task_id="session-1")

    assert prepared["host_cwd"] == str(workspace)
    assert prepared["cwd"] == "/workspace"
    assert prepared["workspace_transport"] == "volume"
    assert prepared["docker_mount_cwd_to_workspace"] is True
    assert prepared["workspace_bootstrap"] == _bootstrap_spec(lane)
    assert calls[0][0][:2] == [sys.executable, "-c"]
    assert calls[0][0][3:] == [
        str(lane), "--registry", "/registry", "bootstrap", "HER-96",
        "--policy", "/policy.json", "--profile", "code-a", "--agent", "code-a",
        "--session", "session-1", "--owner-pid", str(os.getpid()),
    ]
    assert calls[0][1]["input"] == b"trusted lane\n"
    assert calls[0][1]["shell"] is False


@pytest.mark.parametrize("returned_path", ["", "not-json", "symlink", "foreign"])
def test_bootstrap_refuses_untrusted_or_noncanonical_result(monkeypatch, tmp_path, returned_path):
    lane = tmp_path / "factory_lane.py"
    lane.write_text("trusted lane\n")
    owned = tmp_path / "owned"
    owned.mkdir()
    target = tmp_path / "target"
    if returned_path == "symlink":
        target.symlink_to(owned, target_is_directory=True)
        payload = {"worktree": str(target)}
    elif returned_path == "foreign":
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        payload = {"worktree": str(foreign)}
    elif returned_path == "not-json":
        payload = None
    else:
        payload = {"worktree": ""}

    def fake_run(argv, **_kwargs):
        if argv[:2] == [sys.executable, "-c"]:
            return subprocess.CompletedProcess(argv, 0, "not json" if payload is None else json.dumps(payload), "")
        return subprocess.CompletedProcess(argv, 1, "", "not a worktree")

    monkeypatch.setattr("tools.workspace_bootstrap._lane_script", lambda: lane)
    monkeypatch.setattr("tools.workspace_bootstrap.subprocess.run", fake_run)

    with pytest.raises(WorkspaceBootstrapError):
        prepare_workspace_only_config(_config(lane), task_id="session-1")


def test_bootstrap_runs_verified_source_with_script_style_argv(monkeypatch, tmp_path):
    lane = tmp_path / "factory_lane.py"
    workspace = tmp_path / "owned"
    workspace.mkdir()
    lane.write_text(
        "import json, sys\n"
        "assert sys.argv[0] == __file__\n"
        f"assert sys.argv[1:] == ['--registry', '/registry', 'bootstrap', 'HER-96', '--policy', '/policy.json', '--profile', 'code-a', '--agent', 'code-a', '--session', 'session-1', '--owner-pid', {str(os.getpid())!r}]\n"
        f"print(json.dumps({{\"worktree\": {str(workspace)!r}}}))\n"
    )

    monkeypatch.setattr("tools.workspace_bootstrap._lane_script", lambda: lane)
    monkeypatch.setattr("tools.workspace_bootstrap._canonical_worktree", lambda value: value)

    prepared = prepare_workspace_only_config(_config(lane), task_id="session-1")

    assert prepared["host_cwd"] == str(workspace)


def test_bootstrap_refuses_hash_drift_before_running_claim(monkeypatch, tmp_path):
    lane = tmp_path / "factory_lane.py"
    lane.write_text("reviewed bytes\n")
    config = _config(lane)
    config["workspace_bootstrap"]["lane_sha256"] = "0" * 64
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("claim must not run")

    monkeypatch.setattr("tools.workspace_bootstrap._lane_script", lambda: lane)
    monkeypatch.setattr("tools.workspace_bootstrap.subprocess.run", fake_run)

    with pytest.raises(WorkspaceBootstrapError, match="hash"):
        prepare_workspace_only_config(config, task_id="session-1")
    assert called is False


def test_bootstrap_executes_verified_bytes_when_lane_path_is_replaced(monkeypatch, tmp_path):
    lane = tmp_path / "factory_lane.py"
    reviewed_bytes = b"reviewed lane\n"
    lane.write_bytes(reviewed_bytes)
    workspace = tmp_path / "owned"
    workspace.mkdir()
    executed_bytes = None

    def fake_run(argv, **_kwargs):
        nonlocal executed_bytes
        if argv[:2] == [sys.executable, "-c"]:
            lane.write_bytes(b"replaced lane\n")
            executed_bytes = _kwargs["input"]
            return subprocess.CompletedProcess(argv, 0, json.dumps({"worktree": str(workspace)}), "")
        return subprocess.CompletedProcess(argv, 0, str(workspace) + "\n", "")

    monkeypatch.setattr("tools.workspace_bootstrap._lane_script", lambda: lane)
    monkeypatch.setattr("tools.workspace_bootstrap.subprocess.run", fake_run)

    prepare_workspace_only_config(_config(lane), task_id="session-1")

    assert executed_bytes == reviewed_bytes


def test_dynamic_bootstrap_requires_a_session_scoped_environment_key(tmp_path):
    lane = tmp_path / "factory_lane.py"
    lane.write_text("trusted lane\n")

    assert uses_dynamic_workspace_bootstrap(_config(lane)) is True
    assert uses_dynamic_workspace_bootstrap({
        "env_type": "docker",
        "docker_workspace_only": True,
        "workspace_bootstrap": {},
    }) is False
