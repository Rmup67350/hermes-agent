"""Minimal fork guarantees retained after externalising Jean-specific features.

This directory is intentionally absent upstream.  It protects only generic
core invariants that remain in the v2026.8.31 port; AI Factory ownership,
business services, and blanket fork-update policy live outside this core.
Missing or renamed guards fail closed instead of skipping.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest


def _guard(module: str, *names: str) -> Any:
    try:
        imported = __import__(module, fromlist=list(names))
    except Exception as exc:  # pragma: no cover - failure text is the contract
        pytest.fail(f"forkguard import failed for {module}: {exc!r}")
    values = []
    for name in names:
        if not hasattr(imported, name):
            pytest.fail(f"forkguard symbol missing: {module}.{name}")
        values.append(getattr(imported, name))
    return values[0] if len(values) == 1 else values


def test_forkguard_requires_posix():
    if os.name != "posix":
        pytest.fail("forkguard cannot prove POSIX filesystem/process invariants")


def test_cron_lock_rejects_symlink_and_accepts_owner_file(tmp_path, monkeypatch):
    jobs = __import__("cron.jobs", fromlist=["_jobs_lock"])
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    victim = tmp_path / "victim"
    victim.write_text("preserve", encoding="utf-8")
    jobs._jobs_lock_file().symlink_to(victim)
    with pytest.raises(RuntimeError, match="cron jobs lock"):
        with jobs._jobs_lock():
            pytest.fail("symlinked cron lock entered the critical section")
    assert victim.read_text(encoding="utf-8") == "preserve"

    jobs._jobs_lock_file().unlink()
    with jobs._jobs_lock():
        lock_stat = os.stat(jobs._jobs_lock_file(), follow_symlinks=False)
    assert stat.S_ISREG(lock_stat.st_mode)
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    assert lock_stat.st_uid == os.geteuid()
    assert lock_stat.st_nlink == 1


@pytest.mark.parametrize("value", [None, "", "banane", "TRUE", "Full-ish", "  "])
def test_mcp_trust_unknown_values_fail_closed(value):
    normalize = _guard("tools.mcp_tool", "_normalize_server_trust")
    assert normalize(value) == "untrusted"


def test_mcp_trust_requires_explicit_full():
    normalize = _guard("tools.mcp_tool", "_normalize_server_trust")
    assert normalize("full") == "full"
    assert normalize("  FULL  ") == "full"
    assert normalize("untrusted") == "untrusted"


@pytest.mark.parametrize(
    "override",
    [
        {"volumes": ["/etc:/etc"]},
        {"env": {"LD_PRELOAD": "/tmp/x"}},
        {"forward_env": ["AWS_SECRET_ACCESS_KEY"]},
        {"extra_args": ["--privileged"]},
        {"cwd": "/"},
        {"auto_mount_cwd": False},
        {"host_cwd": None},
    ],
)
def test_workspace_only_rejects_each_escape(override):
    validate = _guard(
        "tools.environments.docker", "_validate_workspace_only_config"
    )
    config = {
        "cwd": "/workspace",
        "host_cwd": "/tmp/owned",
        "auto_mount_cwd": True,
        "volumes": None,
        "forward_env": None,
        "env": None,
        "extra_args": None,
    }
    config.update(override)
    with pytest.raises(Exception) as exc_info:
        validate(**config)
    assert not isinstance(exc_info.value, TypeError)


def test_worker_identity_requires_matching_pid_and_start_time():
    process_start, identity_state = _guard(
        "hermes_cli.kanban_db",
        "_worker_process_start_time",
        "_worker_identity_state",
    )
    start = process_start(os.getpid())
    assert start
    assert identity_state(os.getpid(), start) == "alive"
    assert identity_state(os.getpid(), "") == "unknown"
    assert identity_state(os.getpid(), "0.000000") == "dead"


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "forkguard",
        "GIT_AUTHOR_EMAIL": "forkguard@example.invalid",
        "GIT_COMMITTER_NAME": "forkguard",
        "GIT_COMMITTER_EMAIL": "forkguard@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_update_ancestry_refuses_local_only_history(tmp_path):
    classify, unsafe = _guard(
        "hermes_cli.update_cmd",
        "_classify_update_ancestry",
        "UPDATE_ANCESTRY_LOCAL_ONLY_OR_DIVERGED",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "local.txt").write_text("local\n", encoding="utf-8")
    _git(repo, "add", "local.txt")
    _git(repo, "commit", "-m", "local")

    state, _detail = classify(["git"], repo, "main")
    assert state == unsafe
