from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools.workspace_git_transport import (
    WorkspaceGitTransportError,
    validate_workspace_bundle,
    validate_workspace_index,
)


def _git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _repository(path: Path) -> tuple[Path, str, str]:
    path.mkdir()
    _git(["init", "-q", "-b", "owned", str(path)])
    _git(["config", "user.name", "Hermes Test"], cwd=path)
    _git(["config", "user.email", "hermes@example.invalid"], cwd=path)
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(["add", "tracked.txt"], cwd=path)
    _git(["commit", "-qm", "base"], cwd=path)
    base = _git(["rev-parse", "HEAD"], cwd=path)
    (path / "tracked.txt").write_text("candidate\n", encoding="utf-8")
    _git(["commit", "-qam", "candidate"], cwd=path)
    return path, base, _git(["rev-parse", "HEAD"], cwd=path)


def _bundle(repo: Path, destination: Path, *revisions: str) -> Path:
    _git(["bundle", "create", str(destination), *revisions], cwd=repo)
    return destination


def test_bundle_is_seeded_from_trusted_history_then_fscked_and_fast_forwarded(tmp_path):
    repo, base, tip = _repository(tmp_path / "repo")
    bundle = _bundle(repo, tmp_path / "delta.bundle", "owned", f"^{base}")
    validated = validate_workspace_bundle(
        bundle, trusted_repository=repo, branch_ref="refs/heads/owned", expected_head=base
    )
    try:
        assert validated.tip == tip
        assert validated.expected_head == base
        assert validated.branch_ref == "refs/heads/owned"
        assert validated.gitdir.is_dir()
        assert not (validated.gitdir / "hooks" / "post-fetch").exists()
        _git(["--git-dir", str(validated.gitdir), "read-tree", tip])
        validate_workspace_index(
            validated.gitdir, worktree=repo, trusted_repository=repo
        )
        hooks_path = subprocess.run(
            ["git", "--git-dir", str(validated.gitdir), "config", "--get", "core.hooksPath"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert hooks_path.returncode != 0
    finally:
        validated.cleanup()

    with pytest.raises(WorkspaceGitTransportError, match="object limit"):
        validate_workspace_bundle(
            bundle,
            trusted_repository=repo,
            branch_ref="refs/heads/owned",
            expected_head=base,
            max_objects=1,
        )


def test_bundle_rejects_additional_refs_before_fetch(tmp_path):
    repo, base, _tip = _repository(tmp_path / "repo")
    _git(["branch", "foreign", base], cwd=repo)
    bundle = _bundle(repo, tmp_path / "multi.bundle", "owned", "foreign")
    with pytest.raises(WorkspaceGitTransportError, match="exactly the owned branch"):
        validate_workspace_bundle(
            bundle, trusted_repository=repo, branch_ref="refs/heads/owned", expected_head=base
        )


def test_bundle_rejects_non_fast_forward_and_corrupt_or_oversized_input(tmp_path):
    repo, base, original_tip = _repository(tmp_path / "repo")
    _git(["checkout", "-q", "--detach", base], cwd=repo)
    (repo / "diverged.txt").write_text("diverged\n", encoding="utf-8")
    _git(["add", "diverged.txt"], cwd=repo)
    _git(["commit", "-qm", "diverged"], cwd=repo)
    _git(["branch", "-f", "owned", "HEAD"], cwd=repo)
    non_ff = _bundle(repo, tmp_path / "non-ff.bundle", "owned")
    with pytest.raises(WorkspaceGitTransportError, match="fast-forward"):
        validate_workspace_bundle(
            non_ff, trusted_repository=repo, branch_ref="refs/heads/owned", expected_head=original_tip
        )
    corrupt = tmp_path / "corrupt.bundle"
    corrupt.write_bytes(b"not a git bundle")
    with pytest.raises(WorkspaceGitTransportError, match="invalid"):
        validate_workspace_bundle(
            corrupt, trusted_repository=repo, branch_ref="refs/heads/owned", expected_head=base
        )
    with pytest.raises(WorkspaceGitTransportError, match="size limit"):
        validate_workspace_bundle(
            non_ff,
            trusted_repository=repo,
            branch_ref="refs/heads/owned",
            expected_head=base,
            max_bundle_bytes=1,
        )


def test_index_rejects_missing_objects_and_non_regular_modes(tmp_path):
    repo, base, tip = _repository(tmp_path / "repo")
    bundle = _bundle(repo, tmp_path / "delta.bundle", "owned", f"^{base}")
    validated = validate_workspace_bundle(
        bundle, trusted_repository=repo, branch_ref="refs/heads/owned", expected_head=base
    )
    try:
        _git(["--git-dir", str(validated.gitdir), "read-tree", tip])
        poisoned = subprocess.run(
            ["git", "--git-dir", str(validated.gitdir), "update-index", "--index-info"],
            input="100644 1111111111111111111111111111111111111111\tevil.txt\n",
            capture_output=True,
            text=True,
            check=False,
        )
        assert poisoned.returncode == 0, poisoned.stderr
        with pytest.raises(WorkspaceGitTransportError, match="missing index object"):
            validate_workspace_index(
                validated.gitdir, worktree=repo, trusted_repository=repo
            )

        _git(["--git-dir", str(validated.gitdir), "read-tree", tip])
        blob = _git(["hash-object", "tracked.txt"], cwd=repo)
        symlink_index = subprocess.run(
            ["git", "--git-dir", str(validated.gitdir), "update-index", "--index-info"],
            input=f"120000 {blob}\tsymlink.txt\n",
            capture_output=True,
            text=True,
            check=False,
        )
        assert symlink_index.returncode == 0, symlink_index.stderr
        with pytest.raises(WorkspaceGitTransportError, match="unsupported mode"):
            validate_workspace_index(
                validated.gitdir, worktree=repo, trusted_repository=repo
            )
    finally:
        validated.cleanup()
