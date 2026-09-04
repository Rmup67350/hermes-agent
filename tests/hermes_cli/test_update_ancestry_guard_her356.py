from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hermes_cli import update_cmd


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "her356-test",
    "GIT_AUTHOR_EMAIL": "her356@example.invalid",
    "GIT_COMMITTER_NAME": "her356-test",
    "GIT_COMMITTER_EMAIL": "her356@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        env={**os.environ, **_GIT_ENV}, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repos(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "main")
    (origin / "file.txt").write_text("base\n", encoding="utf-8")
    _git(origin, "add", "file.txt")
    _git(origin, "commit", "-m", "base")
    local = tmp_path / "local"
    _git(tmp_path, "clone", str(origin), str(local))
    return origin, local


def test_update_ancestry_accepts_equal_or_behind_history(tmp_path):
    origin, local = _repos(tmp_path)
    equal, _ = update_cmd._classify_update_ancestry(["git"], local, "main")
    assert equal == update_cmd.UPDATE_ANCESTRY_FAST_FORWARD_SAFE

    (origin / "file.txt").write_text("remote\n", encoding="utf-8")
    _git(origin, "commit", "-am", "remote")
    _git(local, "fetch", "origin", "main")
    behind, _ = update_cmd._classify_update_ancestry(["git"], local, "main")
    assert behind == update_cmd.UPDATE_ANCESTRY_FAST_FORWARD_SAFE


def test_update_ancestry_rejects_local_only_or_diverged_history(tmp_path):
    origin, local = _repos(tmp_path)
    (local / "local.txt").write_text("local\n", encoding="utf-8")
    _git(local, "add", "local.txt")
    _git(local, "commit", "-m", "local")
    local_only, _ = update_cmd._classify_update_ancestry(["git"], local, "main")
    assert local_only == update_cmd.UPDATE_ANCESTRY_LOCAL_ONLY_OR_DIVERGED

    (origin / "file.txt").write_text("remote\n", encoding="utf-8")
    _git(origin, "commit", "-am", "remote")
    _git(local, "fetch", "origin", "main")
    diverged, _ = update_cmd._classify_update_ancestry(["git"], local, "main")
    assert diverged == update_cmd.UPDATE_ANCESTRY_LOCAL_ONLY_OR_DIVERGED


def test_update_ancestry_fails_closed_when_remote_ref_is_missing(tmp_path):
    _origin, local = _repos(tmp_path)
    classification, detail = update_cmd._classify_update_ancestry(
        ["git"], local, "missing"
    )
    assert classification == update_cmd.UPDATE_ANCESTRY_UNKNOWN
    assert detail
