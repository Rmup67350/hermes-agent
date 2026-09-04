"""Docker execution environment for sandboxed command execution.

Security hardened (cap-drop ALL, no-new-privileges, PID limits),
configurable resource limits (CPU, memory, disk), and optional filesystem
persistence via bind mounts.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.environments.base import (
    BaseEnvironment,
    EnvironmentConnectionError,
    _popen_bash,
)
from tools.environments.path_utils import sanitize_task_id_for_path
from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)


# Common Docker Desktop install paths checked when 'docker' is not in PATH.
# macOS Intel: /usr/local/bin, macOS Apple Silicon (Homebrew): /opt/homebrew/bin,
# Docker Desktop app bundle: /Applications/Docker.app/Contents/Resources/bin
_DOCKER_SEARCH_PATHS = [
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
]

_docker_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EGRESS_LABEL_KEY = "hermes-egress"


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


@dataclass(frozen=True)
class _WorkspaceGitMetadata:
    workspace: Path
    dotgit: Path
    gitdir: Path
    common: Path
    branch_ref: str
    head: str


def _require_regular_metadata(path: Path, *, kind: str, directory: bool = False) -> None:
    """Reject symlinked or non-regular Git metadata components."""
    if path.is_symlink():
        raise ValueError(f"The selected Git worktree {kind} must not be a symlink")
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        expected = "directory" if directory else "regular file"
        raise ValueError(f"The selected Git worktree {kind} is not a {expected}")


def _canonical_metadata_path(path: Path, *, kind: str) -> Path:
    """Resolve metadata only after proving no path component is a symlink."""
    normalized = Path(os.path.abspath(path))
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"The selected Git worktree {kind} contains a symlinked component"
            )
    try:
        resolved = normalized.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"The selected Git worktree {kind} is unavailable") from exc
    if resolved != normalized:
        raise ValueError(f"The selected Git worktree {kind} is non-canonical")
    return resolved


def _workspace_git_metadata(workspace: Path) -> _WorkspaceGitMetadata:
    """Validate and identify only the metadata owned by an exact worktree."""
    dotgit = workspace / ".git"
    if dotgit.is_symlink():
        raise ValueError("The selected Git worktree .git must not be a symlink")
    if dotgit.is_dir():
        gitdir = _canonical_metadata_path(dotgit, kind=".git directory")
        common = gitdir
    elif dotgit.is_file():
        _require_regular_metadata(dotgit, kind=".git marker")
        try:
            marker = dotgit.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError("Could not read the selected Git worktree metadata") from exc
        if not marker.startswith("gitdir: ") or "\n" in marker or "\x00" in marker:
            raise ValueError("The selected Git worktree has an invalid .git marker")

        raw_gitdir = Path(marker.removeprefix("gitdir: ").strip())
        candidate_gitdir = raw_gitdir if raw_gitdir.is_absolute() else dotgit.parent / raw_gitdir
        gitdir = _canonical_metadata_path(candidate_gitdir, kind="admin directory")
        _require_regular_metadata(gitdir, kind="admin metadata", directory=True)

        commondir_file = gitdir / "commondir"
        if commondir_file.exists() or commondir_file.is_symlink():
            _require_regular_metadata(commondir_file, kind="commondir marker")
            raw_common = Path(commondir_file.read_text(encoding="utf-8").strip())
            candidate_common = raw_common if raw_common.is_absolute() else gitdir / raw_common
            common = _canonical_metadata_path(candidate_common, kind="common directory")
        else:
            common = gitdir
    else:
        raise ValueError(
            "terminal.docker_workspace_only requires terminal.cwd to be a Git worktree"
        )
    _require_regular_metadata(common, kind="common metadata", directory=True)
    if common.name != ".git":
        raise ValueError("The selected Git worktree common directory is invalid")
    try:
        relative_admin = gitdir.relative_to(common)
    except ValueError as exc:
        raise ValueError("Git worktree admin metadata escapes its common directory") from exc
    if gitdir != common and (
        len(relative_admin.parts) != 2 or relative_admin.parts[0] != "worktrees"
    ):
        raise ValueError("Git worktree admin metadata has an unexpected layout")

    reverse_marker = gitdir / "gitdir"
    if gitdir != common:
        _require_regular_metadata(reverse_marker, kind="reverse marker")
        reverse_path = Path(reverse_marker.read_text(encoding="utf-8").strip())
        if not reverse_path.is_absolute():
            reverse_path = gitdir / reverse_path
        canonical_reverse = _canonical_metadata_path(reverse_path, kind="reverse path")
        if canonical_reverse != _canonical_metadata_path(dotgit, kind=".git marker"):
            raise ValueError("Git worktree reverse metadata does not match terminal.cwd")

    head_file = gitdir / "HEAD"
    _require_regular_metadata(head_file, kind="HEAD")
    head_marker = head_file.read_text(encoding="utf-8").strip()
    if not head_marker.startswith("ref: refs/heads/") or "\n" in head_marker:
        raise ValueError("docker_workspace_only requires an owned local branch")
    branch_ref = head_marker.removeprefix("ref: ")
    head = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", head.stdout.strip()):
        raise ValueError("The selected Git worktree HEAD is unavailable")

    index = gitdir / "index"
    if index.exists() or index.is_symlink():
        _require_regular_metadata(index, kind="index")
    return _WorkspaceGitMetadata(
        workspace=workspace,
        dotgit=dotgit,
        gitdir=gitdir,
        common=common,
        branch_ref=branch_ref,
        head=head.stdout.strip(),
    )


def _validate_workspace_only_config(
    *,
    cwd: str,
    host_cwd: str | None,
    auto_mount_cwd: bool,
    volumes: list | None,
    forward_env: list[str] | None,
    env: dict | None,
    extra_args: list | None,
    workspace_identity: dict | None = None,
    workspace_transport: str | None = None,
) -> tuple[Path, _WorkspaceGitMetadata]:
    """Validate the fail-closed host boundary before contacting Docker."""
    if cwd != "/workspace":
        raise ValueError(
            "terminal.docker_workspace_only requires the container cwd /workspace"
        )
    if not auto_mount_cwd:
        raise ValueError(
            "terminal.docker_workspace_only requires docker_mount_cwd_to_workspace=true"
        )
    if not host_cwd:
        raise ValueError(
            "terminal.docker_workspace_only requires terminal.cwd to name the exact worktree"
        )
    if volumes:
        raise ValueError("terminal.docker_workspace_only forbids docker_volumes")
    if forward_env:
        raise ValueError("terminal.docker_workspace_only forbids docker_forward_env")
    if env:
        raise ValueError("terminal.docker_workspace_only forbids docker_env")
    if extra_args:
        raise ValueError("terminal.docker_workspace_only forbids docker_extra_args")

    if workspace_identity is not None:
        if workspace_transport != "volume":
            from tools.workspace_bootstrap import WorkspaceBootstrapError

            raise WorkspaceBootstrapError(
                "Docker path-based bind mounts cannot guarantee the pinned inode"
            )
        from tools.workspace_bootstrap import (
            revalidate_workspace_identity,
        )
        revalidate_workspace_identity({
            "env_type": "docker",
            "docker_workspace_only": True,
            "workspace_bootstrap": {"pinned": True},
            "host_cwd": host_cwd,
            "workspace_identity": workspace_identity,
        })
    try:
        workspace = Path(host_cwd).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("terminal.cwd is unavailable for docker_workspace_only") from exc
    if not workspace.is_dir():
        raise ValueError("terminal.cwd must be a directory for docker_workspace_only")
    return workspace, _workspace_git_metadata(workspace)


def _git_broker_env() -> dict[str, str]:
    """Return an environment isolated from user/system Git configuration."""
    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=False)
    for key in tuple(env):
        if key.startswith("GIT_"):
            env.pop(key, None)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _run_git_checked(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "diff.external=",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_git_broker_env(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git command failed").strip()
        raise RuntimeError(detail)
    return result


def _file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    _require_regular_metadata(path, kind="index")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_workspace_git_broker(metadata: _WorkspaceGitMetadata) -> tuple[Path, Path]:
    """Create a private single-branch Git database for the container."""
    broker_root = Path(
        tempfile.mkdtemp(prefix=".hermes-git-broker-", dir=str(metadata.workspace.parent))
    )
    private_git = broker_root / "repo.git"
    branch = metadata.branch_ref.removeprefix("refs/heads/")
    try:
        _run_git_checked(
            [
                "clone",
                "--bare",
                "--no-local",
                "--no-hardlinks",
                "--no-tags",
                "--single-branch",
                "--branch",
                branch,
                str(metadata.workspace),
                str(private_git),
            ]
        )
        _run_git_checked(["--git-dir", str(private_git), "config", "core.bare", "false"])
        _run_git_checked(
            ["--git-dir", str(private_git), "config", "core.worktree", "/workspace"]
        )
        _run_git_checked(
            ["--git-dir", str(private_git), "symbolic-ref", "HEAD", metadata.branch_ref]
        )
        _run_git_checked(["--git-dir", str(private_git), "read-tree", metadata.head])

        identity = _run_git_checked(
            ["-C", str(metadata.workspace), "show", "-s", "--format=%an%x1f%ae", "HEAD"]
        ).stdout.strip().split("\x1f", 1)
        if len(identity) == 2 and all(identity):
            _run_git_checked(
                ["--git-dir", str(private_git), "config", "user.name", identity[0]]
            )
            _run_git_checked(
                ["--git-dir", str(private_git), "config", "user.email", identity[1]]
            )
        _run_git_checked(
            ["--git-dir", str(private_git), "config", "--remove-section", "remote.origin"]
        )
        shutil.rmtree(private_git / "hooks", ignore_errors=True)
        (private_git / "hooks").mkdir(mode=0o700)
        return broker_root, private_git
    except Exception:
        shutil.rmtree(broker_root, ignore_errors=True)
        raise


def _workspace_publication_journal(metadata: _WorkspaceGitMetadata) -> Path:
    digest = hashlib.sha256(str(metadata.workspace).encode("utf-8")).hexdigest()[:24]
    return metadata.workspace.parent / f".hermes-publication-{digest}"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_journal_state_path(journal: Path, state: str) -> None:
    temporary = journal / f"state.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as output:
        output.write((state + "\n").encode("ascii"))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, journal / "state")
    _fsync_directory(journal)


def _write_journal_state(metadata: _WorkspaceGitMetadata, state: str) -> None:
    _write_journal_state_path(_workspace_publication_journal(metadata), state)


_WORKSPACE_PUBLICATION_STATES = frozenset(
    {"prepared", "publishing", "metadata_published", "committed"}
)


def _read_journal_state(journal: Path) -> str:
    state_path = journal / "state"
    try:
        path_metadata = state_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(path_metadata.st_mode) or path_metadata.st_nlink != 1:
            raise RuntimeError("Workspace publication journal state is unsafe")
        descriptor = os.open(
            state_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeError("Workspace publication journal state is unsafe") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_nlink != 1
            or (opened_metadata.st_dev, opened_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise RuntimeError("Workspace publication journal state is unsafe")
        encoded = os.read(descriptor, 64)
    finally:
        os.close(descriptor)
    allowed = {
        (state + "\n").encode("ascii"): state
        for state in _WORKSPACE_PUBLICATION_STATES
    }
    if encoded not in allowed:
        raise RuntimeError("Workspace publication journal state is invalid")
    return allowed[encoded]


def _copy_journal_payload(source, destination: Path) -> str:
    digest = hashlib.sha256()
    with destination.open("xb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def _begin_workspace_publication_journal(
    metadata: _WorkspaceGitMetadata,
    *,
    candidate_workspace: Path,
    candidate_git: Path,
    expected_head: str,
    candidate_head: str,
) -> Path:
    """Durably persist both sides before destructive multi-file publication."""
    from tools.workspace_staging import archive_tree, capture_identity

    journal = _workspace_publication_journal(metadata)
    if journal.exists() or journal.is_symlink():
        if not _recover_workspace_publication(metadata):
            raise RuntimeError("Workspace publication journal could not be recovered")
    prepare_prefix = journal.name + ".prepare-"
    for stale in journal.parent.glob(prepare_prefix + "*"):
        if stale.is_symlink():
            stale.unlink(missing_ok=True)
        elif stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
    preparing = journal.parent / f"{prepare_prefix}{uuid.uuid4().hex}"
    preparing.mkdir(mode=0o700)
    try:
        backup = archive_tree(metadata.workspace, exclude_root_names={".git"})
        candidate = archive_tree(candidate_workspace, exclude_root_names={".git"})
        try:
            backup_digest = _copy_journal_payload(backup, preparing / "backup.tar")
            candidate_digest = _copy_journal_payload(candidate, preparing / "candidate.tar")
        finally:
            backup.close()
            candidate.close()
        host_index = metadata.gitdir / "index"
        candidate_index = candidate_git / "index"
        old_index_present = host_index.exists()
        old_index_digest = None
        if old_index_present:
            _require_regular_metadata(host_index, kind="index")
            with host_index.open("rb") as source:
                old_index_digest = _copy_journal_payload(source, preparing / "old.index")
        _require_regular_metadata(candidate_index, kind="candidate index")
        with candidate_index.open("rb") as source:
            candidate_index_digest = _copy_journal_payload(source, preparing / "candidate.index")
        identity = capture_identity(metadata.workspace)
        payload = {
            "version": 2,
            "index_lock_token": uuid.uuid4().hex,
            "workspace": str(metadata.workspace),
            "device": identity.device,
            "inode": identity.inode,
            "branch_ref": metadata.branch_ref,
            "expected_head": expected_head,
            "candidate_head": candidate_head,
            "old_index_present": old_index_present,
            "old_index_digest": old_index_digest,
            "candidate_index_digest": candidate_index_digest,
            "backup_digest": backup_digest,
            "candidate_digest": candidate_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with (preparing / "metadata.json").open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        _write_journal_state_path(preparing, "prepared")
        _fsync_directory(preparing)
        os.replace(preparing, journal)
        _fsync_directory(journal.parent)
        return journal
    except BaseException:
        shutil.rmtree(preparing, ignore_errors=True)
        raise


def _discard_workspace_publication_journal(metadata: _WorkspaceGitMetadata) -> None:
    journal = _workspace_publication_journal(metadata)
    shutil.rmtree(journal, ignore_errors=True)
    try:
        _fsync_directory(journal.parent)
    except OSError:
        pass


def _verified_journal_payload(path: Path, expected_digest: str) -> None:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("Workspace publication journal payload is unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise RuntimeError("Workspace publication journal payload changed")


def _replace_host_index(
    metadata: _WorkspaceGitMetadata,
    payload: Path | None,
    *,
    publication_token: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", publication_token):
        raise RuntimeError("Workspace publication journal has an invalid index lock token")
    host_index = metadata.gitdir / "index"
    lock = metadata.gitdir / "index.lock"
    owned = metadata.gitdir / f"index.hermes-{publication_token}"

    if payload is None:
        expected_bytes = b""
    else:
        expected_bytes = payload.read_bytes()
    if owned.exists():
        owned_metadata = owned.stat(follow_symlinks=False)
        if not stat.S_ISREG(owned_metadata.st_mode) or owned.read_bytes() != expected_bytes:
            raise RuntimeError("Workspace publication owned index file is unsafe")
    else:
        fd = os.open(
            owned,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(fd, "wb", closefd=True) as target:
                target.write(expected_bytes)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            owned.unlink(missing_ok=True)
            raise

    if lock.exists():
        try:
            owned_lock = os.path.samefile(owned, lock)
        except OSError:
            owned_lock = False
        if not owned_lock:
            raise RuntimeError("Workspace publication found a concurrent Git index lock")
    else:
        try:
            os.link(owned, lock, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeError("Workspace publication found a concurrent Git index lock") from exc

    if payload is None:
        host_index.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    else:
        os.replace(lock, host_index)
    owned.unlink(missing_ok=True)
    _fsync_directory(metadata.gitdir)


def _recover_workspace_publication(metadata: _WorkspaceGitMetadata) -> bool:
    """Converge an interrupted publication to its exact old or candidate state."""
    from tools.workspace_staging import capture_identity, extract_archive, manifest_tree, publish_tree

    journal = _workspace_publication_journal(metadata)
    if not journal.exists() and not journal.is_symlink():
        return False
    journal_meta = journal.stat(follow_symlinks=False)
    if not stat.S_ISDIR(journal_meta.st_mode) or journal.is_symlink():
        raise RuntimeError("Workspace publication journal is unsafe")
    journal_state = _read_journal_state(journal)
    metadata_path = journal / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("Workspace publication journal is invalid") from exc
    required = {
        "version", "workspace", "device", "inode", "branch_ref", "expected_head",
        "candidate_head", "old_index_present", "old_index_digest",
        "candidate_index_digest", "backup_digest", "candidate_digest",
        "index_lock_token",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["version"] != 2:
        raise RuntimeError("Workspace publication journal is invalid")
    identity = capture_identity(metadata.workspace)
    if (
        payload["workspace"] != str(metadata.workspace)
        or payload["device"] != identity.device
        or payload["inode"] != identity.inode
        or payload["branch_ref"] != metadata.branch_ref
    ):
        raise RuntimeError("Workspace publication journal identity changed")
    expected_head = payload["expected_head"]
    candidate_head = payload["candidate_head"]
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40,64}", value) for value in (expected_head, candidate_head)):
        raise RuntimeError("Workspace publication journal head is invalid")
    backup_path = journal / "backup.tar"
    candidate_path = journal / "candidate.tar"
    candidate_index_path = journal / "candidate.index"
    _verified_journal_payload(backup_path, payload["backup_digest"])
    _verified_journal_payload(candidate_path, payload["candidate_digest"])
    _verified_journal_payload(candidate_index_path, payload["candidate_index_digest"])
    old_index_path = journal / "old.index" if payload["old_index_present"] else None
    if old_index_path is not None:
        _verified_journal_payload(old_index_path, payload["old_index_digest"])

    current_head = _run_git_checked(
        ["-C", str(metadata.workspace), "rev-parse", "--verify", metadata.branch_ref]
    ).stdout.strip()
    if current_head not in {expected_head, candidate_head}:
        raise RuntimeError("Workspace publication recovery found concurrent branch changes")
    current_index_digest = _file_digest(metadata.gitdir / "index")
    old_index_digest = payload["old_index_digest"]
    candidate_index_digest = payload["candidate_index_digest"]
    if current_index_digest not in {old_index_digest, candidate_index_digest}:
        raise RuntimeError("Workspace publication recovery found concurrent index changes")
    index_transitioned = (
        old_index_digest != candidate_index_digest
        and current_index_digest == candidate_index_digest
    )
    head_transitioned = (
        candidate_head != expected_head and current_head == candidate_head
    )
    candidate_started = (
        head_transitioned or index_transitioned
        or journal_state in {"metadata_published", "committed"}
    )
    desired_head = candidate_head if candidate_started else expected_head
    desired_index = candidate_index_path if candidate_started else old_index_path
    desired_archive = candidate_path if candidate_started else backup_path
    if candidate_started:
        _run_git_checked(
            ["-C", str(metadata.workspace), "cat-file", "-e", f"{candidate_head}^{{commit}}"]
        )

    stage = Path(tempfile.mkdtemp(prefix=".hermes-publication-recovery-"))
    try:
        with desired_archive.open("rb") as archive:
            extract_archive(archive, stage)
        current_manifest = manifest_tree(metadata.workspace, exclude_root_names={".git"})

        def finalize_recovery() -> None:
            observed = _run_git_checked(
                ["-C", str(metadata.workspace), "rev-parse", "--verify", metadata.branch_ref]
            ).stdout.strip()
            if observed != desired_head:
                _run_git_checked(
                    [
                        "-C", str(metadata.workspace), "update-ref", metadata.branch_ref,
                        desired_head, observed,
                    ]
                )
            _replace_host_index(
                metadata,
                desired_index,
                publication_token=payload["index_lock_token"],
            )

        publish_tree(
            stage,
            metadata.workspace,
            identity=identity,
            expected_manifest=current_manifest,
            exclude_root_names={".git"},
            finalize=finalize_recovery,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    _discard_workspace_publication_journal(metadata)
    return True


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


# Docker label values must match [a-zA-Z0-9_.-] and stay ≤63 chars to round-trip
# safely through `docker ps --filter label=key=value`. Profile and task names
# can technically contain other characters; sanitize defensively.
_LABEL_VALUE_OK_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_label_value(value: str) -> str:
    """Coerce *value* into a Docker label-safe form (alnum + ``_.-``, ≤63 chars).

    Empty or all-invalid inputs collapse to ``"unknown"`` so the resulting
    label is always queryable. Used at container-create time; never round-trip
    a sanitized value back into application logic.
    """
    if not isinstance(value, str) or not value:
        return "unknown"
    cleaned = _LABEL_VALUE_OK_RE.sub("_", value)
    cleaned = cleaned[:63] or "unknown"
    return cleaned


# The task_id -> host-directory-name mapping is shared with every backend
# that persists per-task state on the host filesystem (Singularity overlays
# use the same helper), so the whole bug class is fixed in one place:
# tools.environments.base.sanitize_task_id_for_path.
_sandbox_dir_name = sanitize_task_id_for_path

def _process_start_fingerprint(pid: int) -> int | None:
    """Return a PID-reuse-safe process identity for Docker lifecycle labels."""
    try:
        import psutil

        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``"default"`` on any error.

    Resolved at container-create time so a single container is permanently
    tagged with the profile that created it. Profile switches inside the
    same process don't retroactively relabel running containers.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _container_identity(shared_key: str = "") -> str:
    """Return the profile label used for reuse and orphan reaping.

    Profiles remain isolated by default. An explicit shared key lets trusted
    profiles that intentionally share a workspace use one Docker identity.

    Shared keys are made collision-resistant the same way
    :func:`sanitize_task_id_for_path` is: label sanitization is lossy
    (``team/workspace`` and ``team_workspace`` both sanitize to
    ``team_workspace``, and >63-char keys truncate), and since container
    reuse is label-keyed, two DIFFERENT keys colliding after sanitization
    would silently attach to the same running container. A digest of the
    raw key disambiguates; identical raw keys still map to identical
    labels across processes. Plain profile names keep their historical
    un-suffixed labels for backward compatibility with existing containers.
    """
    if not shared_key:
        return _sanitize_label_value(_get_active_profile_name())
    digest = hashlib.sha256(shared_key.encode("utf-8")).hexdigest()[:12]
    stem = _sanitize_label_value(shared_key)[:50]
    return f"{stem}-{digest}"


def reap_orphan_containers(
    *,
    max_age_seconds: int = 600,
    profile_filter: str | None = None,
    docker_exe: str | None = None,
) -> int:
    """Remove stale hermes-tagged containers left behind by prior processes.

    Targets containers that match all of:

    * ``label=hermes-agent=1`` (created by this codebase)
    * ``status=exited`` (running containers are NEVER reaped — they may
      belong to a sibling Hermes process whose reuse path will pick them
      up; killing them would crash the sibling mid-command)
    * (optional) ``label=hermes-profile=<profile_filter>`` (sweep only the
      caller's profile by default; a hermes process in profile A must not
      tear down profile B's containers)
    * ``State.FinishedAt`` older than *max_age_seconds* ago (so a sibling
      process that just exited and is about to be replaced doesn't get
      its container yanked out from under it)

    Returns the number of containers removed. Best-effort: any failure
    (docker daemon unreachable, slow inspect, parse error) is logged at
    debug level and the function returns whatever it managed before the
    failure. Safe to call repeatedly; idempotent.

    Issue #20561 — this is the safety net for SIGKILL / OOM / crashed
    terminal exits that bypass the ``atexit`` cleanup hook. Without it,
    even with the cleanup-fix in the prior commit, a hard-killed Hermes
    process leaves its container behind permanently because there's no
    subsequent Hermes process scheduled to reuse that exact (task, profile)
    pair.
    """
    docker = docker_exe or find_docker() or "docker"
    filters = ["--filter", "label=hermes-agent=1", "--filter", "status=exited"]
    if profile_filter:
        filters.extend(["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"])

    try:
        listing = subprocess.run(
            [docker, "ps", "-a", *filters, "--format", "{{.ID}}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker ps failed: %s", e)
        return 0
    if listing.returncode != 0:
        logger.debug(
            "orphan reaper docker ps returned %d: %s",
            listing.returncode, listing.stderr.strip(),
        )
        return 0

    exited_ids = {ln.strip() for ln in listing.stdout.splitlines() if ln.strip()}
    private_filters = ["--filter", "label=hermes-workspace-only=1"]
    if profile_filter:
        private_filters.extend(
            ["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"]
        )
    try:
        private_listing = subprocess.run(
            [docker, "ps", "-a", *private_filters, "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError):
        private_listing = None
    private_ids = {
        line.strip()
        for line in (private_listing.stdout.splitlines() if private_listing and private_listing.returncode == 0 else [])
        if line.strip()
    }
    dead_private_ids = {
        container_id
        for container_id in private_ids
        if _private_container_owner_is_dead(docker, container_id)
    }
    candidate_ids = sorted(exited_ids | dead_private_ids)
    if not candidate_ids:
        _reap_orphan_private_volumes(docker, profile_filter=profile_filter)
        return 0

    # Inspect each candidate to get FinishedAt; reap only those exited
    # long enough ago.  Doing this per-container (rather than bulk inspect)
    # keeps the failure blast radius to one container at a time.
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for cid in candidate_ids:
        if cid in dead_private_ids:
            age = max_age_seconds
        else:
            finished_at = _container_finished_at(docker, cid)
            if finished_at is None:
                # Couldn't determine age — be conservative and leave it alone.
                continue
            age = (now - finished_at).total_seconds()
            if age < max_age_seconds:
                continue
        private_volumes = _container_private_volumes(docker, cid)
        try:
            result = subprocess.run(
                [docker, "rm", "-f", cid],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                removed += 1
                for volume in private_volumes:
                    volume_result = subprocess.run(
                        [docker, "volume", "rm", "-f", volume],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=30,
                        check=False,
                        stdin=subprocess.DEVNULL,
                    )
                    if volume_result.returncode != 0:
                        logger.debug(
                            "docker volume rm -f %s failed: %s",
                            volume,
                            volume_result.stderr.strip(),
                        )
                logger.info(
                    "Reaped orphan container %s (exited %d seconds ago)",
                    cid[:12], int(age),
                )
            else:
                logger.debug(
                    "docker rm -f %s failed: %s",
                    cid[:12], result.stderr.strip(),
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("orphan reaper docker rm %s failed: %s", cid[:12], e)
    _reap_orphan_private_volumes(docker, profile_filter=profile_filter)
    return removed


def _container_labels(docker_exe: str, container_id: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{json .Config.Labels}}", container_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return {}
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        return {}
    return payload


def _recorded_owner_is_dead(labels: dict[str, str]) -> bool:
    try:
        pid = int(labels["hermes-owner-pid"])
        expected_start = int(labels["hermes-owner-start"])
        if pid <= 0 or expected_start <= 0:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return True
        observed_start = int(round(process.create_time() * 100))
    except psutil.NoSuchProcess:
        return True
    except Exception:
        return False
    return observed_start != expected_start


def _private_container_owner_is_dead(docker_exe: str, container_id: str) -> bool:
    labels = _container_labels(docker_exe, container_id)
    if labels.get("hermes-agent") != "1" or labels.get("hermes-workspace-only") != "1":
        return False
    return _recorded_owner_is_dead(labels)


def _volume_labels(docker_exe: str, volume_name: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            [docker_exe, "volume", "inspect", "--format", "{{json .Labels}}", volume_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return {}
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        return {}
    return payload


def _reap_orphan_private_volumes(
    docker_exe: str,
    *,
    profile_filter: str | None,
) -> None:
    filters = ["--filter", "label=hermes-workspace-only=1"]
    if profile_filter:
        filters.extend(
            ["--filter", f"label=hermes-profile={_sanitize_label_value(profile_filter)}"]
        )
    try:
        listing = subprocess.run(
            [docker_exe, "volume", "ls", *filters, "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if listing.returncode != 0:
        return
    for volume_name in {line.strip() for line in listing.stdout.splitlines() if line.strip()}:
        labels = _volume_labels(docker_exe, volume_name)
        kind = labels.get("hermes-private-kind")
        expected_prefix = "hermes-workspace-" if kind == "workspace" else "hermes-git-" if kind == "git" else None
        if (
            expected_prefix is None
            or not volume_name.startswith(expected_prefix)
            or labels.get("hermes-agent") != "1"
            or labels.get("hermes-workspace-only") != "1"
            or not labels.get("hermes-task-id")
            or not labels.get("hermes-profile")
            or not _recorded_owner_is_dead(labels)
        ):
            continue
        subprocess.run(
            [docker_exe, "volume", "rm", "-f", volume_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            stdin=subprocess.DEVNULL,
        )


def _container_private_volumes(docker_exe: str, container_id: str) -> tuple[str, ...]:
    """Return only named volumes whose labels match their owning container."""
    container_labels = _container_labels(docker_exe, container_id)
    task_id = container_labels.get("hermes-task-id")
    profile = container_labels.get("hermes-profile")
    if (
        container_labels.get("hermes-agent") != "1"
        or container_labels.get("hermes-workspace-only") != "1"
        or not task_id
        or not profile
    ):
        return ()
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{json .Mounts}}", container_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ()
    if result.returncode != 0:
        return ()
    try:
        mounts = json.loads(result.stdout)
    except (TypeError, ValueError):
        return ()
    if not isinstance(mounts, list):
        return ()
    names: set[str] = set()
    for mount in mounts:
        if not isinstance(mount, dict) or mount.get("Type") != "volume":
            continue
        name = mount.get("Name")
        if not isinstance(name, str):
            continue
        if name.startswith("hermes-workspace-"):
            expected_kind = "workspace"
        elif name.startswith("hermes-git-"):
            expected_kind = "git"
        else:
            continue
        labels = _volume_labels(docker_exe, name)
        if (
            labels.get("hermes-agent") == "1"
            and labels.get("hermes-workspace-only") == "1"
            and labels.get("hermes-private-kind") == expected_kind
            and labels.get("hermes-task-id") == task_id
            and labels.get("hermes-profile") == profile
        ):
            names.add(name)
    return tuple(sorted(names))


def _container_finished_at(docker_exe: str, container_id: str):
    """Parse ``docker inspect`` FinishedAt for *container_id*.

    Returns a timezone-aware datetime, or ``None`` if the field is missing,
    unparseable, or the zero-value ``0001-01-01T00:00:00Z`` Docker emits
    for never-finished containers. ``None`` means "don't reap" — the caller
    leaves the container alone.
    """
    try:
        result = subprocess.run(
            [docker_exe, "inspect", "--format", "{{.State.FinishedAt}}", container_id],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("orphan reaper docker inspect %s failed: %s", container_id[:12], e)
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    # Docker emits RFC3339 with nanoseconds (e.g. "2026-05-28T13:45:00.123456789Z").
    # Python's fromisoformat handles microseconds but not nanoseconds; trim.
    import re as _re
    raw = _re.sub(r"(\.\d{6})\d+", r"\1", raw)
    raw = raw.replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        logger.debug("could not parse FinishedAt %r for %s: %s", raw, container_id[:12], e)
        return None


def find_docker() -> Optional[str]:
    """Locate the docker (or podman) CLI binary.

    Resolution order:
    1. ``HERMES_DOCKER_BINARY`` env var — explicit override (e.g. ``/usr/bin/podman``)
    2. ``docker`` on PATH via ``shutil.which``
    3. ``podman`` on PATH via ``shutil.which``
    4. Well-known macOS Docker Desktop install locations

    Returns the absolute path, or ``None`` if neither runtime can be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable

    # 1. Explicit override via env var (e.g. for Podman on immutable distros)
    override = os.getenv("HERMES_DOCKER_BINARY")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _docker_executable = override
        logger.info("Using HERMES_DOCKER_BINARY override: %s", override)
        return override

    # 2. docker on PATH
    found = shutil.which("docker")
    if found:
        _docker_executable = found
        return found

    # 3. podman on PATH (drop-in compatible for our use case)
    found = shutil.which("podman")
    if found:
        _docker_executable = found
        logger.info("Using podman as container runtime: %s", found)
        return found

    # 4. Well-known macOS Docker Desktop locations
    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info("Found docker at non-PATH location: %s", path)
            return path

    return None


# Security flags applied to every container.
# The container itself is the security boundary (isolated from host).
# We drop all capabilities then add back the minimum needed:
#   DAC_OVERRIDE - root can write to bind-mounted dirs owned by host user
#   CHOWN/FOWNER - package managers (pip, npm, apt) need to set file ownership
#   SETUID/SETGID - the image's init drops from root to the 'hermes'
#       user (via `s6-setuidgid` in the bundled image, or whatever
#       privilege-drop helper a user image uses), which requires these
#       caps. Combined with `no-new-privileges`, the dropped process
#       still cannot escalate back to root, so the security posture is
#       preserved. Omitted entirely when the container starts as a
#       non-root user via --user, since no privilege drop is needed
#       in that mode.
# Block privilege escalation.
# /tmp is size-limited and nosuid but allows exec (needed by pip/npm builds).
#
# Note: ``--pids-limit`` is *not* in this list — it lives in ``resource_args``
# and is gated on ``_cgroup_limits_available(image)`` because it requires the
# ``pids`` cgroup controller to be delegated, which is not the case on hosts
# such as unprivileged LXCs. ``--cpus``/``--memory`` are gated for the same
# reason.
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--tmpfs", "/tmp:rw,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
]

# Default per-container PID limit. Applied as ``--pids-limit`` only when the
# cgroup ``pids`` controller is available (see ``_cgroup_limits_available``).
_DEFAULT_PIDS_LIMIT = "256"

# Default /dev/shm size. Docker's built-in default is a tiny 64 MB, which
# silently breaks shared-memory-hungry workloads inside the sandbox: Chromium /
# Playwright renderers crash tabs, and PyTorch DataLoader workers die with
# "bus error" / "insufficient shared memory" once they exceed it. tmpfs is
# lazily allocated, so a 1g ceiling costs nothing until actually used (and
# usage still counts against the container's --memory cgroup limit).
# Configurable via ``terminal.docker_shm_size`` in config.yaml; an empty value
# (or "0") omits the flag and falls back to Docker's 64 MB default.
# Ported from nanocoai/nanoclaw#2748.
_DEFAULT_SHM_SIZE = "1g"


def _extra_args_set_shm_size(extra_args: list) -> bool:
    """True when user-supplied docker_extra_args already set ``--shm-size``.

    In that case we skip our default so the user's value is unambiguous
    (rather than relying on flag-ordering / last-wins behavior).
    """
    return any(
        isinstance(a, str) and (a == "--shm-size" or a.startswith("--shm-size="))
        for a in (extra_args or [])
    )

# /run is split out from _BASE_SECURITY_ARGS because s6-overlay images need it
# mounted ``exec``: s6 stage0 later runs ``exec /run/s6/basedir/bin/init``, which
# fails with "Permission denied" (exit 126) on a ``noexec`` mount. For all other
# images we keep the hardened ``noexec`` default.
_RUN_TMPFS_NOEXEC = "--tmpfs", "/run:rw,noexec,nosuid,size=64m"
_RUN_TMPFS_EXEC = "--tmpfs", "/run:rw,exec,nosuid,size=64m"

# Extra caps needed when the container starts as root and an init/entrypoint
# must drop privileges (via `s6-setuidgid`, `gosu`, `su`, or similar).
# Skipped when --user is passed because the container already starts
# unprivileged and never needs to switch.
_PRIVDROP_CAP_ARGS = [
    "--cap-add", "SETUID",
    "--cap-add", "SETGID",
]


def _egress_proxy_args_for_docker() -> tuple[list[str], dict[str, str], list[str]]:
    """Build the docker mount/env/host args needed to route a sandbox through
    the iron-proxy egress firewall.

    Returns ``(volume_args, env_overrides, host_args)``:

    * ``volume_args`` — read-only bind mount of the CA cert into the container
      (extends docker's ``-v`` argv list)
    * ``env_overrides`` — env vars to set on container creation: ``HTTPS_PROXY``,
      ``HTTP_PROXY``, ``NO_PROXY`` (loopback only), Python/Node/curl CA-bundle
      paths, and one ``HERMES_PROXY_TOKEN_<NAME>`` per minted mapping
    * ``host_args`` — extra ``--add-host`` flags so the container can reach the
      host-side proxy (Linux needs ``host.docker.internal:host-gateway``;
      Docker Desktop populates this automatically on macOS/Windows)

    Returns three empty containers when the proxy is disabled, not yet set up,
    or not currently running.  If ``proxy.enforce_on_docker`` is true and the
    proxy is enabled-but-not-running, raises ``RuntimeError`` so the docker
    backend refuses to start the sandbox.
    """

    # Narrow except: ImportError is the only legitimate failure here.
    # Bare ``except Exception`` would hide AttributeError, SyntaxError in
    # the config module, etc. and silently start the sandbox without
    # proxy enforcement.  We let unexpected exceptions propagate so the
    # docker backend visibly fails rather than degrading silently.
    try:
        from hermes_cli.config import load_config
        from agent.proxy_sources import iron_proxy as ip
    except ImportError as exc:
        logger.debug("Egress proxy plumbing unavailable: %s", exc)
        return ([], {}, [])

    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        return ([], {}, [])

    status = ip.get_status()
    enforce = bool(proxy_cfg.get("enforce_on_docker", True))

    if not status.configured:
        msg = (
            "proxy.enabled is true but iron-proxy is not configured. "
            "Run `hermes egress setup` to mint tokens and write proxy.yaml."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    if not (status.pid and status.listening):
        msg = (
            f"iron-proxy is enabled but not running on port {status.tunnel_port}. "
            "Start it with `hermes egress start`."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    if status.ca_cert_path is None or not status.ca_cert_path.exists():
        # status.configured was True a moment ago but the CA file has
        # disappeared.  Treat this with the same enforce semantics as the
        # other failure branches — silently dropping the CA mount would
        # leave the sandbox with proxy env vars pointing at iron-proxy
        # but no trust anchor, so every TLS handshake would 5xx; or
        # worse, with enforce_on_docker=false we'd drop both the proxy
        # vars AND any other isolation, opening the sandbox.
        msg = (
            f"iron-proxy CA cert vanished from {status.ca_cert_path}. "
            "Re-run `hermes egress setup` to regenerate it."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    # Corrupt or empty mappings.json is a silent failure mode that's
    # indistinguishable from an upstream outage from inside the sandbox
    # (every request returns 403).  Refuse to mount with empty mappings
    # rather than ship a broken sandbox.
    mappings = ip.load_mappings()
    if not mappings:
        msg = (
            "iron-proxy is configured but mappings.json is empty or "
            "corrupt.  Re-run `hermes egress setup` to mint provider "
            "tokens before starting a sandbox."
        )
        if enforce:
            raise RuntimeError(msg)
        logger.warning("%s — continuing without proxy (enforce_on_docker=false).", msg)
        return ([], {}, [])

    container_ca = "/etc/ssl/certs/hermes-egress-ca.crt"
    volume_args = ["-v", f"{status.ca_cert_path}:{container_ca}:ro"]

    # tunnel_port serves CONNECT (HTTPS); the plain-HTTP forward listener
    # is on tunnel_port + 1 (see build_proxy_config's listener-role notes).
    proxy_url = f"http://host.docker.internal:{status.tunnel_port}"
    plain_http_url = f"http://host.docker.internal:{status.tunnel_port + 1}"
    env_overrides: dict[str, str] = {
        # HTTPS_PROXY / HTTP_PROXY are respected by curl, requests, urllib,
        # httpx, node fetch, go default transport, etc.  Lowercase variants
        # are also set because some tools only look at one casing.
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": plain_http_url,
        "http_proxy": plain_http_url,
        # Loopback-only NO_PROXY so localhost dev servers inside the sandbox
        # (test fixtures, local LLMs) don't get sent through the proxy.
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        # CA bundle locations for the major language runtimes.  iron-proxy
        # presents a leaf cert signed by our CA on every MITM'd connection.
        #
        # CRITICAL ASYMMETRY: Python (REQUESTS_CA_BUNDLE / SSL_CERT_FILE)
        # and curl (CURL_CA_BUNDLE) REPLACE the system CA store.
        # NODE_EXTRA_CA_CERTS ADDS to it.  A Node.js process that
        # bypasses HTTPS_PROXY by using a raw socket would still see the
        # system CA store and succeed where Python/curl fail validation.
        # We additionally set NODE_OPTIONS=--use-openssl-ca to force Node
        # through the OpenSSL store that SSL_CERT_FILE controls, narrowing
        # the asymmetry.  Not a complete fix — see the docs caveat — but
        # closes the easy case.
        "REQUESTS_CA_BUNDLE": container_ca,   # Python `requests`
        "SSL_CERT_FILE": container_ca,         # Python ssl module / OpenSSL
        "CURL_CA_BUNDLE": container_ca,        # curl
        "NODE_EXTRA_CA_CERTS": container_ca,   # Node.js: adds to system store
        # NOTE: NODE_OPTIONS is intentionally NOT placed in env_overrides
        # here as a flat assignment.  We need to APPEND --use-openssl-ca
        # to whatever the user already has in NODE_OPTIONS (e.g.
        # --max-old-space-size=4096), not clobber it.  The append-merge
        # happens in DockerEnvironment._merge_node_options below.
        # For the agent inside the sandbox to identify itself as proxy-aware.
        "HERMES_EGRESS_PROXY": "1",
        # Sentinel that DockerEnvironment uses to do the NODE_OPTIONS
        # append-merge.  Stripped from the final env before docker run.
        "_HERMES_EGRESS_NODE_OPTIONS_APPEND": "--use-openssl-ca",
    }

    # Surface the per-provider proxy tokens under the standard provider env
    # names so existing SDKs and provider clients work unchanged inside the
    # sandbox.  Alias env names (e.g. GOOGLE_API_KEY for GEMINI_API_KEY)
    # receive the same token so SDKs reading either name authenticate
    # through the proxy.  Keep the HERMES_PROXY_TOKEN_* aliases for
    # diagnostics.
    for m in mappings:
        env_overrides[m.real_env_name] = m.proxy_token
        env_overrides[f"HERMES_PROXY_TOKEN_{m.real_env_name}"] = m.proxy_token
        for alias in getattr(m, "alias_env_names", ()) or ():
            env_overrides[alias] = m.proxy_token

    # On Linux, host.docker.internal isn't populated by default — Docker Desktop
    # adds it on macOS/Windows; on Linux we need an explicit --add-host with
    # host-gateway.  On Desktop this is a no-op (harmless duplicate).
    host_args: list[str] = ["--add-host", "host.docker.internal:host-gateway"]

    return (volume_args, env_overrides, host_args)


def _egress_reuse_fingerprint(
    volume_args: list[str],
    env_overrides: dict[str, str],
    host_args: list[str],
) -> str:
    """Stable Docker-label value for the egress posture of a container."""
    if not (volume_args or env_overrides or host_args):
        return "off"
    payload = json.dumps(
        {
            "volume_args": volume_args,
            "env_overrides": env_overrides,
            "host_args": host_args,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _egress_enforce_on_docker(default: bool = True) -> bool:
    """Read proxy.enforce_on_docker with fail-safe defaulting."""
    try:
        from hermes_cli.config import load_config as _load_cfg

        return bool((_load_cfg().get("proxy") or {}).get("enforce_on_docker", default))
    except (ImportError, OSError):
        return default
    except Exception:
        return default


def _critical_egress_env_names(env_overrides: dict[str, str]) -> set[str]:
    """Env names that would weaken or bypass enforced egress if overridden."""
    critical = {
        "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
        "NO_PROXY", "no_proxy",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS", "NODE_OPTIONS",
    }
    critical.update(
        key for key in env_overrides
        if key.endswith("_API_KEY") or key.endswith("_TOKEN")
    )
    return critical


def _extra_args_egress_collisions(
    extra_args: list[str], critical_names: set[str],
) -> list[str]:
    """Return docker_extra_args entries that can override egress controls."""
    collisions: list[str] = []
    env_flags = {"-e", "--env", "--env-file"}
    network_flags = {"--network", "--net"}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        nxt = extra_args[i + 1] if i + 1 < len(extra_args) else ""
        if arg in env_flags:
            if arg == "--env-file":
                collisions.append(arg)
            else:
                name = nxt.split("=", 1)[0]
                if name in critical_names:
                    collisions.append(name)
            i += 2
            continue
        if any(arg.startswith(f"{flag}=") for flag in env_flags):
            if arg.startswith("--env-file="):
                collisions.append("--env-file")
            else:
                name = arg.split("=", 1)[1].split("=", 1)[0]
                if name in critical_names:
                    collisions.append(name)
        elif arg in network_flags or any(arg.startswith(f"{flag}=") for flag in network_flags):
            collisions.append(arg)
        i += 1
    return sorted(set(collisions))


def _build_security_args(run_as_host_user: bool, run_exec: bool = False) -> list[str]:
    """Return the security/cap/tmpfs args tailored to the privilege mode.

    ``run_exec`` mounts ``/run`` with ``exec`` instead of the hardened
    ``noexec`` default. This is required for s6-overlay images whose ``/init``
    entrypoint execs ``/run/s6/basedir/bin/init`` during startup; see
    ``_image_uses_init_entrypoint``.
    """
    run_tmpfs = list(_RUN_TMPFS_EXEC if run_exec else _RUN_TMPFS_NOEXEC)
    args = list(_BASE_SECURITY_ARGS) + run_tmpfs
    if run_as_host_user:
        return args
    return args + list(_PRIVDROP_CAP_ARGS)


def _image_uses_init_entrypoint(docker_exe: str, image: str) -> bool:
    """Return True if ``image``'s entrypoint is the s6-overlay ``/init``.

    Such images (e.g. anything built on ``s6-overlay``, including
    ``hermes-agent:latest``) already provide their own PID-1 init and execute
    ``/run/s6/basedir/bin/init`` during stage0 startup. They are incompatible
    with Docker's ``--init`` (two competing PID-1 inits) and with a ``noexec``
    ``/run`` mount. Detection is best-effort: on any inspection failure we
    return False and keep the hardened defaults.
    """
    try:
        result = subprocess.run(
            [docker_exe, "image", "inspect", image,
             "--format", "{{json .Config.Entrypoint}}"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Docker: could not inspect entrypoint for %s: %s", image, e)
        return False
    if result.returncode != 0:
        # Image may not be pulled yet; the run will pull it. Defaults are safe
        # for non-s6 images, so don't block on this.
        logger.debug(
            "Docker: image inspect for %s returned %d (stderr=%s)",
            image, result.returncode, result.stderr.strip(),
        )
        return False
    raw = (result.stdout or "").strip()
    if not raw or raw == "null":
        return False
    try:
        entrypoint = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if not isinstance(entrypoint, list) or not entrypoint:
        return False
    first = str(entrypoint[0]).strip()
    return first in ("/init", "/package/admin/s6-overlay/command/init")


def _ensure_image_has_no_declared_volumes(docker_exe: str, image: str) -> None:
    """Reject image-declared anonymous volumes for workspace-only sessions."""
    try:
        result = subprocess.run(
            [
                docker_exe,
                "image",
                "inspect",
                image,
                "--format",
                "{{json .Config.Volumes}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(
            "Could not inspect Docker image volumes for docker_workspace_only"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            "Docker image must be present and inspectable for docker_workspace_only"
        )
    raw = (result.stdout or "").strip()
    try:
        declared = None if raw == "null" else json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Docker image returned invalid volume metadata for docker_workspace_only"
        ) from exc
    if declared is not None and not isinstance(declared, dict):
        raise RuntimeError(
            "Docker image returned invalid volume metadata for docker_workspace_only"
        )
    if declared:
        paths = ", ".join(sorted(str(path) for path in declared))
        raise ValueError(
            "Docker image declares implicit writable volumes, which "
            f"docker_workspace_only forbids: {paths}"
        )


def _resolve_host_user_spec() -> Optional[str]:
    """Return ``<uid>:<gid>`` for the current host user, or ``None`` on platforms
    where this is not meaningful (e.g. Windows without posix ids).

    We intentionally read ``os.getuid()``/``os.getgid()`` directly rather than
    going through ``getpass``/``pwd`` so this stays cheap and never raises on
    nameless UIDs (nss lookups can fail inside sandboxed launchers).
    """
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if get_uid is None or get_gid is None:
        return None
    try:
        return f"{get_uid()}:{get_gid()}"
    except Exception:  # pragma: no cover - defensive
        return None


_storage_opt_ok: Optional[bool] = None  # cached result across instances
_cgroup_limits_ok: Optional[bool] = None  # cached result across instances


def _cgroup_limits_available(image: str) -> bool:
    """Probe whether cgroup resource limits work in this environment.

    Tests ``--cpus``, ``--memory`` and ``--pids-limit`` together by spawning
    a throwaway container from *image* (the same sandbox image we are about
    to use for real, so no extra pull and no dependency on a public
    registry). The container runs ``sleep 0`` — sleep is guaranteed to be
    present because the sandbox itself uses ``sleep 2h`` as its long-lived
    entrypoint.

    On hosts where the corresponding cgroup controllers are not delegated
    to this process (typical inside unprivileged LXCs and some rootless
    setups) these flags cause every container start to fail with ``OCI
    runtime error`` / exit 126. The probe runs once per process and the
    result — which is host-wide, not image-specific — is cached.
    """
    global _cgroup_limits_ok
    if _cgroup_limits_ok is not None:
        return _cgroup_limits_ok

    docker_exe = find_docker()
    if not docker_exe or not image:
        _cgroup_limits_ok = False
        return False

    try:
        result = subprocess.run(
            [docker_exe, "run", "--rm",
             "--cpus", "0.5", "--memory", "64m", "--pids-limit", "32",
             image, "sleep", "0"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        _cgroup_limits_ok = result.returncode == 0
        if not _cgroup_limits_ok:
            logger.warning(
                "Cgroup resource limits (--cpus/--memory/--pids-limit) not "
                "available in this environment. Containers will run without "
                "CPU, memory or PID limits. To enable, delegate the cpu, "
                "memory and pids cgroup controllers to this container. "
                "Probe stderr: %s",
                (result.stderr or "").strip()[:500],
            )
    except Exception as e:
        _cgroup_limits_ok = False
        logger.warning("Cgroup limit probe failed; disabling resource limits: %s", e)

    return _cgroup_limits_ok


def _ensure_docker_available() -> None:
    """Best-effort check that the docker CLI is available before use.

    Reuses ``find_docker()`` so this preflight stays consistent with the rest of
    the Docker backend, including known non-PATH Docker Desktop locations.
    """
    docker_exe = find_docker()
    if not docker_exe:
        logger.error(
            "Docker backend selected but no docker executable was found in PATH "
            "or known install locations. Install Docker Desktop and ensure the "
            "CLI is available."
        )
        raise EnvironmentConnectionError(
            "Docker executable not found in PATH or known install locations. "
            "Install Docker and ensure the 'docker' command is available.",
            retry_hint=(
                "Install Docker (or fix PATH) and retry, or switch "
                "terminal.backend to 'local'."
            ),
        )

    try:
        result = subprocess.run(
            [docker_exe, "version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error(
            "Docker backend selected but the resolved docker executable '%s' could "
            "not be executed.",
            docker_exe,
            exc_info=True,
        )
        raise EnvironmentConnectionError(
            "Docker executable could not be executed. Check your Docker installation.",
            retry_hint="Repair the Docker installation and retry.",
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Docker backend selected but '%s version' timed out. "
            "The Docker daemon may not be running.",
            docker_exe,
            exc_info=True,
        )
        raise EnvironmentConnectionError(
            "Docker daemon is not responding. Ensure Docker is running and try again.",
            retry_hint=(
                "Start the Docker daemon (e.g. `systemctl start docker` or "
                "launch Docker Desktop), then retry the same command."
            ),
        )
    except Exception:
        logger.error(
            "Unexpected error while checking Docker availability.",
            exc_info=True,
        )
        raise
    else:
        if result.returncode != 0:
            logger.error(
                "Docker backend selected but '%s version' failed "
                "(exit code %d, stderr=%s)",
                docker_exe,
                result.returncode,
                result.stderr.strip(),
            )
            raise EnvironmentConnectionError(
                "Docker command is available but 'docker version' failed. "
                "Check your Docker installation.",
                retry_hint=(
                    "The Docker daemon may be down or the current user lacks "
                    "permission (docker group). Fix and retry."
                ),
            )


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution with resource limits and persistence.

    Security: all capabilities dropped, no privilege escalation, PID limits,
    size-limited tmpfs for scratch dirs. The container itself is the security
    boundary — the filesystem inside is writable so agents can install packages
    (pip, npm, apt) as needed. Writable workspace via tmpfs or bind mounts.

    Persistence: when enabled, bind mounts preserve /workspace and /root
    across container restarts.
    """

    _profile_scoped_passthrough = True

    def _additional_profile_scoped_passthrough_names(self) -> tuple[str, ...]:
        """Keep explicit docker_forward_env values out of shared snapshots."""
        return tuple(self._forward_env)

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        host_cwd: Optional[str] = None,
        auto_mount_cwd: bool = False,
        run_as_host_user: bool = False,
        extra_args: list = None,
        persist_across_processes: bool = True,
        shm_size: str = _DEFAULT_SHM_SIZE,
        shared_container_key: str = "",
        workspace_only: bool = False,
        workspace_identity: dict | None = None,
        workspace_transport: str | None = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        workspace_root: Path | None = None
        git_metadata: _WorkspaceGitMetadata | None = None
        if workspace_only:
            workspace_root, git_metadata = _validate_workspace_only_config(
                cwd=cwd,
                host_cwd=host_cwd,
                auto_mount_cwd=auto_mount_cwd,
                volumes=volumes,
                forward_env=forward_env,
                env=env,
                extra_args=extra_args,
                workspace_identity=workspace_identity,
                workspace_transport=workspace_transport,
            )
            if workspace_identity is not None and git_metadata is not None:
                _recover_workspace_publication(git_metadata)
                git_metadata = _workspace_git_metadata(workspace_root)
            host_cwd = str(workspace_root)
            persistent_filesystem = False
            persist_across_processes = False
            run_as_host_user = True
            network = False
        self._workspace_only = workspace_only
        self._private_workspace = bool(workspace_only and workspace_identity is not None)
        self._workspace_git_metadata = git_metadata
        self._git_broker_root: Path | None = None
        self._git_broker_gitdir: Path | None = None
        self._git_broker_head = git_metadata.head if git_metadata else None
        self._git_broker_index_digest: str | None = None
        self._git_broker_sync_lock = threading.Lock()
        self._workspace_volume: str | None = None
        self._git_volume: str | None = None
        self._workspace_stage_identity = None
        self._workspace_manifest = None
        self._host_index_digest = (
            _file_digest(git_metadata.gitdir / "index") if git_metadata else None
        )
        self._persistent = persistent_filesystem
        self._persist_across_processes = persist_across_processes
        # Set by terminal_tool._create_environment when this container is
        # scoped to a single session (docker + container_persistent: false):
        # survives between turns, removed at session close / idle timeout.
        self._session_scoped = False
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._init_unset_passthrough_names: tuple[str, ...] = ()
        self._container_id: Optional[str] = None
        self._labels: dict[str, str] = {}
        self._owner_pid = os.getpid()
        self._owner_start = _process_start_fingerprint(self._owner_pid)
        if self._private_workspace and self._owner_start is None:
            raise RuntimeError("Dynamic workspace cannot prove its owner process identity")
        self._image: str = ""
        self._container_name: str = ""
        self._image_uses_s6_init: bool = False
        self._all_run_args: list[str] = []
        self._run_env_values: dict[str, str] = {}
        self._init_env_values: dict[str, str] = {}
        logger.info("DockerEnvironment volumes: %s", volumes)
        # Ensure volumes is a list (config.yaml could be malformed)
        if volumes is not None and not isinstance(volumes, list):
            logger.warning("docker_volumes config is not a list: %r", volumes)
            volumes = []

        # Fail fast if Docker is not available.
        _ensure_docker_available()
        if workspace_only:
            _ensure_image_has_no_declared_volumes(find_docker() or "docker", image)

        # Build resource limit args (gated by cgroup availability probe so
        # they degrade gracefully on hosts without controller delegation,
        # e.g. unprivileged LXCs). The probe runs once per process and is
        # cached host-wide.
        resource_args = []
        if cpu > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0 and _cgroup_limits_available(image):
            resource_args.extend(["--memory", f"{memory}m"])
        if _cgroup_limits_available(image):
            resource_args.extend(["--pids-limit", _DEFAULT_PIDS_LIMIT])
        # /dev/shm size (not cgroup-gated: --shm-size is a tmpfs mount option,
        # no controller delegation required). Skip when the user already sets
        # it via docker_extra_args, or opted out with an empty/"0" value.
        shm = str(shm_size or "").strip()
        if shm and shm != "0" and not _extra_args_set_shm_size(extra_args):
            resource_args.extend(["--shm-size", shm])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")
        if workspace_only:
            resource_args.append("--read-only")

        # Persistent workspace via bind mounts from a configurable host directory
        # (TERMINAL_SANDBOX_DIR, default ~/.hermes/sandboxes/). Non-persistent
        # mode uses tmpfs (ephemeral, fast, gone on cleanup).
        from tools.environments.base import get_sandbox_dir

        # User-configured volume mounts (from config.yaml docker_volumes)
        volume_args = []
        workspace_explicitly_mounted = False
        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning("Docker volume entry is not a string: %r", vol)
                continue
            vol = vol.strip()
            if not vol:
                continue
            if ":" in vol:
                volume_args.extend(["-v", vol])
                if ":/workspace" in vol:
                    workspace_explicitly_mounted = True
            else:
                logger.warning("Docker volume '%s' missing colon, skipping", vol)

        host_cwd_abs = os.path.abspath(os.path.expanduser(host_cwd)) if host_cwd else ""
        bind_host_cwd = (
            auto_mount_cwd
            and bool(host_cwd_abs)
            and os.path.isdir(host_cwd_abs)
            and not workspace_explicitly_mounted
            and not self._private_workspace
        )
        if auto_mount_cwd and host_cwd and not os.path.isdir(host_cwd_abs):
            logger.debug("Skipping docker cwd mount: host_cwd is not a valid directory: %s", host_cwd)

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        writable_args = []
        if self._persistent:
            # _sandbox_dir_name(): a raw session-key task_id carries colons,
            # which `-v` reads as extra spec fields (exit 125).
            sandbox = get_sandbox_dir() / "docker" / _sandbox_dir_name(task_id)
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args.extend([
                "-v", f"{self._home_dir}:/root",
            ])
            if not bind_host_cwd and not workspace_explicitly_mounted:
                self._workspace_dir = str(sandbox / "workspace")
                os.makedirs(self._workspace_dir, exist_ok=True)
                writable_args.extend([
                    "-v", f"{self._workspace_dir}:/workspace",
                ])
        else:
            if (
                not bind_host_cwd
                and not workspace_explicitly_mounted
                and not self._private_workspace
            ):
                writable_args.extend([
                    "--tmpfs", "/workspace:rw,exec,size=10g",
                ])
            writable_args.extend([
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ])
            if workspace_only:
                writable_args.extend([
                    "--tmpfs", "/cache:rw,exec,nosuid,size=2g",
                ])

        if bind_host_cwd:
            logger.info("Mounting configured host cwd to /workspace: %s", host_cwd_abs)
            workspace_mode = ":rw" if workspace_only else ""
            volume_args = ["-v", f"{host_cwd_abs}:/workspace{workspace_mode}", *volume_args]
            if workspace_only:
                assert git_metadata is not None
                broker_root, private_git = _prepare_workspace_git_broker(git_metadata)
                self._git_broker_root = broker_root
                self._git_broker_gitdir = private_git
                self._git_broker_index_digest = _file_digest(private_git / "index")
                volume_args.extend(["-v", f"{private_git}:/hermes-git:rw"])
                dotgit_mask = broker_root / "workspace-dotgit"
                if git_metadata.dotgit.is_dir():
                    dotgit_mask.mkdir(mode=0o500)
                else:
                    dotgit_mask.write_text("gitdir: /hermes-git\n", encoding="utf-8")
                    dotgit_mask.chmod(0o400)
                volume_args.extend(["-v", f"{dotgit_mask}:/workspace/.git:ro"])
        elif self._private_workspace:
            assert git_metadata is not None
            from tools.workspace_staging import capture_identity, manifest_tree

            broker_root, private_git = _prepare_workspace_git_broker(git_metadata)
            self._git_broker_root = broker_root
            self._git_broker_gitdir = private_git
            self._git_broker_index_digest = _file_digest(private_git / "index")
            self._workspace_stage_identity = capture_identity(git_metadata.workspace)
            self._workspace_manifest = manifest_tree(
                git_metadata.workspace, exclude_root_names={".git"}
            )
            volume_suffix = uuid.uuid4().hex
            self._workspace_volume = f"hermes-workspace-{volume_suffix}"
            self._git_volume = f"hermes-git-{volume_suffix}"
            volume_args.extend(
                [
                    "-v", f"{self._workspace_volume}:/workspace:rw",
                    "-v", f"{self._git_volume}:/hermes-git:rw",
                ]
            )
        elif workspace_explicitly_mounted:
            logger.debug("Skipping docker cwd mount: /workspace already mounted by user config")

        # Mount credential files (OAuth tokens, etc.) declared by skills.
        # Read-only so the container can authenticate but not modify host creds.
        try:
            if workspace_only:
                get_credential_file_mounts = lambda: []
                get_skills_directory_mount = lambda: []
                get_cache_directory_mounts = lambda: []
            else:
                from tools.credential_files import (
                    get_credential_file_mounts,
                    get_skills_directory_mount,
                    get_cache_directory_mounts,
                )

            for mount_entry in get_credential_file_mounts():
                src = Path(mount_entry["host_path"])
                if src.is_dir():
                    # Docker-in-Docker: Docker auto-created the source path as
                    # a directory when it didn't exist on the host.  Mounting a
                    # directory over a file destination causes exit 125.
                    logger.warning(
                        "Docker: skipping credential mount — source is a directory "
                        "(likely Docker-in-Docker auto-creation): %s",
                        src,
                    )
                    continue
                if not src.is_file():
                    logger.warning(
                        "Docker: skipping credential mount — source not found: %s", src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount skill directories (local + external) so skill
            # scripts/templates are available inside the container.
            for skills_mount in get_skills_directory_mount():
                src = Path(skills_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping skills mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            # Mount host-side cache directories (documents, images, audio,
            # screenshots) so the agent can access uploaded files and other
            # cached media from inside the container.  Read-only — the
            # container reads these but the host gateway manages writes.
            for cache_mount in get_cache_directory_mounts():
                src = Path(cache_mount["host_path"])
                if not src.is_dir():
                    logger.warning(
                        "Docker: skipping cache mount — source is not a directory: %s",
                        src,
                    )
                    continue
                volume_args.extend([
                    "-v",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}:ro",
                ])
                logger.info(
                    "Docker: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Docker: could not load credential file mounts: %s", e)

        # Egress credential-injection proxy (iron-proxy) — when configured,
        # mount the CA cert into the sandbox and set HTTPS_PROXY + CA-bundle
        # env vars so outbound traffic routes through the host-side proxy.
        # The sandbox receives PROXY tokens instead of real API keys.
        if workspace_only:
            egress_volume_args, egress_env_overrides, egress_host_args = [], {}, []
            egress_label = "off"
        else:
            egress_volume_args, egress_env_overrides, egress_host_args = (
                _egress_proxy_args_for_docker()
            )
            egress_label = _egress_reuse_fingerprint(
                egress_volume_args, egress_env_overrides, egress_host_args,
            )
        _enforce_egress = _egress_enforce_on_docker()
        _critical_egress_names = _critical_egress_env_names(egress_env_overrides)
        if egress_env_overrides:
            _forward_collisions = sorted(
                key for key in self._forward_env if key in _critical_egress_names
            )
            if _forward_collisions:
                _msg = (
                    f"docker_forward_env would inject real egress-protected "
                    f"variables {_forward_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these names from docker_forward_env "
                        "or disable enforce_on_docker to opt out of egress isolation."
                    )
                logger.warning(
                    "%s  Explicit docker_forward_env values will override egress tokens.",
                    _msg,
                )
        volume_args.extend(egress_volume_args)
        # egress env overrides are merged in further below alongside the
        # other env_args computation.

        # Explicit environment variables (docker_env config) — set at container
        # creation so they're available to all processes (including entrypoint).
        # Egress proxy env vars (HTTPS_PROXY, CA-bundle paths, proxy tokens)
        # are merged below.  Precedence policy:
        #
        # - When egress enforcement is on AND the user's docker_env tries
        #   to override one of the proxy-control vars (HTTPS_PROXY,
        #   SSL_CERT_FILE, etc.), fail-loud rather than silently inverting
        #   the isolation.  The CA mount + tokens would still ship while
        #   traffic leaves the sandbox direct with real credentials —
        #   exactly what enforce_on_docker is meant to prevent.
        # - When enforcement is off, the user's docker_env wins (current
        #   behavior) but we log a warning naming both config sources.
        # - When the user override is identical to the egress value, no-op.
        if egress_env_overrides:
            try:
                from hermes_cli.config import load_config as _load_cfg_for_collision
                _proxy_cfg = (_load_cfg_for_collision().get("proxy") or {})
            except (ImportError, OSError):
                _proxy_cfg = {}
            except Exception as _e:  # noqa: BLE001 — narrowed below via yaml import
                # yaml.YAMLError from a malformed config.yaml.  We import
                # lazily because PyYAML is a soft dep in some test envs.
                try:
                    import yaml  # noqa: F401
                except ImportError:
                    raise
                logger.warning(
                    "Could not read proxy config for egress collision check: %s",
                    _e,
                )
                _proxy_cfg = {}
            _enforce_egress = bool(_proxy_cfg.get("enforce_on_docker", True))
            # Egress-controlling env vars that affect the proxy posture.
            _critical_proxy_control = {
                "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "NO_PROXY", "no_proxy",
                "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
                "NODE_EXTRA_CA_CERTS",
            }
            # stephenschoettler #2: also block docker_env from injecting
            # real provider keys.  `docker_env: {OPENROUTER_API_KEY: sk-real}`
            # in config.yaml puts the live secret into the sandbox while
            # egress is nominally enforced — defeats the entire feature.
            # Pull the mapped real_env_name from each token mapping at
            # call time so this stays in sync with whatever the operator
            # has configured.
            _critical_provider_keys: set[str] = set()
            try:
                from agent.proxy_sources import iron_proxy as _ip_for_mappings
                _critical_provider_keys = {
                    m.real_env_name for m in _ip_for_mappings.load_mappings()
                }
            except Exception:  # noqa: BLE001 — best-effort collision check
                pass
            _critical = _critical_proxy_control | _critical_provider_keys
            _collisions = sorted(
                k for k in _critical
                if k in self._env
                and (
                    k not in egress_env_overrides
                    or self._env[k] != egress_env_overrides[k]
                )
                # For provider keys, ANY override is a collision (the egress
                # path mints proxy tokens; a real key in docker_env bypasses
                # the swap regardless of whether the egress dict happens to
                # carry it).
                and (
                    k in _critical_provider_keys
                    or (k in egress_env_overrides
                        and self._env[k] != egress_env_overrides[k])
                )
            )
            if _collisions:
                _msg = (
                    f"docker_env in config.yaml overrides egress-proxy "
                    f"variables {_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these keys from docker_env or "
                        "disable enforce_on_docker to opt out of egress "
                        "isolation."
                    )
                logger.warning(
                    "%s  Falling back to docker_env values; sandbox traffic "
                    "will NOT route through the proxy.", _msg,
                )

        # When enforce_on_docker is true, egress overrides win.  When
        # false, docker_env wins (back-compat for users who deliberately
        # opt out).  In both cases the collision check above has already
        # surfaced any disagreement.
        try:
            from hermes_cli.config import load_config as _load_cfg_for_precedence
            _enforce_egress_merge = bool(
                (_load_cfg_for_precedence().get("proxy") or {})
                .get("enforce_on_docker", True)
            )
        except (ImportError, OSError):
            _enforce_egress_merge = True
        except Exception:  # noqa: BLE001 — yaml.YAMLError or similar
            # Malformed config.yaml; fail-safe to enforced.
            _enforce_egress_merge = True

        if _enforce_egress_merge and egress_env_overrides:
            merged_env = dict(self._env)
            merged_env.update(egress_env_overrides)
        else:
            merged_env = dict(egress_env_overrides)
            merged_env.update(self._env)

        # arshkumarsingh #1: NODE_OPTIONS append-merge.  The egress path
        # wants ``--use-openssl-ca`` so Node routes through the OpenSSL
        # CA store ``SSL_CERT_FILE`` controls.  But the operator's
        # ``docker_env: {NODE_OPTIONS: "--max-old-space-size=8192"}``
        # MUST be preserved — replacing it would silently drop their
        # tuning.  We carry the egress flag in a sentinel key
        # ``_HERMES_EGRESS_NODE_OPTIONS_APPEND`` and merge here.
        _egress_node_append = merged_env.pop(
            "_HERMES_EGRESS_NODE_OPTIONS_APPEND", None,
        )
        if _egress_node_append:
            existing_node = merged_env.get("NODE_OPTIONS", "")
            existing_tokens = existing_node.split()
            # maxpetrusenko P1: dedupe is not enough — the operator may have set
            # a CONFLICTING CA-mode flag (e.g. --use-bundled-ca) that would
            # otherwise survive alongside our --use-openssl-ca, leaving Node's
            # final trust behavior dependent on option order / Node parsing.
            # Egress isolation requires our flag to win deterministically, so
            # strip any known-conflicting CA-mode flags before appending.
            _CA_MODE_FLAGS = {"--use-openssl-ca", "--use-bundled-ca"}
            append_token = _egress_node_append.strip()
            if append_token in _CA_MODE_FLAGS:
                dropped = [t for t in existing_tokens if t in _CA_MODE_FLAGS and t != append_token]
                if dropped:
                    logger.warning(
                        "Overriding conflicting NODE_OPTIONS CA-mode flag(s) %s "
                        "with egress-required %s to keep Node routed through the "
                        "egress CA store.", dropped, append_token,
                    )
                existing_tokens = [t for t in existing_tokens if t not in _CA_MODE_FLAGS or t == append_token]
            # De-dup: only add if not already present (the operator may
            # have set the same flag themselves).
            if append_token not in existing_tokens:
                existing_tokens.append(append_token)
            merged_env["NODE_OPTIONS"] = " ".join(existing_tokens).strip()
            if not merged_env["NODE_OPTIONS"]:
                merged_env.pop("NODE_OPTIONS", None)

        # Name-only -e flags: the docker CLI resolves a valueless `--env KEY`
        # from its own process environment, so the secret values travel via
        # the spawned client's env (owner-readable /proc/*/environ) instead of
        # its argv (world-readable /proc/*/cmdline). See issue #96268.
        env_args = []
        for key in sorted(merged_env):
            env_args.extend(["-e", key])
        # Values injected into the docker-client subprocess env at run time
        # (also reused verbatim by the container-recreation recovery path).
        self._run_env_values = dict(merged_env)
        if workspace_only:
            workspace_env = {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_DIR": "/hermes-git",
                "GIT_WORK_TREE": "/workspace",
                "HOME": "/home",
                "NPM_CONFIG_CACHE": "/cache/npm",
                "PIP_CACHE_DIR": "/cache/pip",
                "TMPDIR": "/tmp",
                "UV_CACHE_DIR": "/cache/uv",
                "XDG_CACHE_HOME": "/cache",
            }
            for key in sorted(workspace_env):
                env_args.extend(["-e", f"{key}={workspace_env[key]}"])

        # Optional: run the container as the host user so files written into
        # bind-mounted dirs (/workspace, /root, docker_volumes entries) are
        # owned by that user on the host instead of by root. Skip cleanly on
        # platforms without POSIX uid/gid (e.g. native Windows Docker).
        user_args: list[str] = []
        user_spec = None
        if run_as_host_user:
            user_spec = _resolve_host_user_spec()
            if user_spec is not None:
                user_args = ["--user", user_spec]
                logger.info("Docker: running container as host user %s", user_spec)
            else:
                if workspace_only:
                    raise ValueError(
                        "terminal.docker_workspace_only requires a POSIX host uid/gid"
                    )
                logger.warning(
                    "docker_run_as_host_user is enabled but this platform does "
                    "not expose POSIX uid/gid; container will start as its "
                    "image default user."
                )
                # Fall back to the full cap set — without --user, an image's
                # init may still need s6-setuidgid/gosu/su to drop privileges.
        self._container_user_spec = user_spec if user_args else None

        # Resolve the docker executable once so it works even when
        # /usr/local/bin is not in PATH (common on macOS gateway/service).
        self._docker_exe = find_docker() or "docker"
        if self._private_workspace:
            self._create_private_volumes()

        # s6-overlay images (e.g. hermes-agent:latest) already use /init as PID 1
        # and exec /run/s6/basedir/bin/init during startup. For those images we
        # must (a) skip Docker's --init (two competing PID-1 inits) and (b) mount
        # /run with exec instead of noexec, or s6 stage0 dies with exit 126
        # "Permission denied". Detected once here; defaults are kept on any
        # inspection failure. See issue #34628.
        image_uses_s6_init = _image_uses_init_entrypoint(self._docker_exe, image)
        if image_uses_s6_init:
            logger.info(
                "Docker: image %s uses /init (s6-overlay) as entrypoint — "
                "skipping --init and mounting /run with exec.",
                image,
            )
        security_args = _build_security_args(
            run_as_host_user and bool(user_args),
            run_exec=image_uses_s6_init,
        )

        logger.info("Docker volume_args: %s", volume_args)
        # User-supplied extra docker run flags (docker_extra_args in config.yaml).
        # Appended last so they can override defaults if needed.
        validated_extra = []
        for arg in (extra_args or []):
            if not isinstance(arg, str):
                logger.warning("Ignoring non-string docker_extra_args entry: %r", arg)
                continue
            validated_extra.append(arg)
        if egress_env_overrides:
            _extra_collisions = _extra_args_egress_collisions(
                validated_extra, _critical_egress_names,
            )
            if _extra_collisions:
                _msg = (
                    f"docker_extra_args would override egress-proxy controls "
                    f"{_extra_collisions}; enforce_on_docker is "
                    f"{'enabled' if _enforce_egress else 'disabled'}."
                )
                if _enforce_egress:
                    raise RuntimeError(
                        f"{_msg}  Remove these args or disable enforce_on_docker "
                        "to opt out of egress isolation."
                    )
                logger.warning(
                    "%s  Extra Docker args may bypass egress isolation.", _msg,
                )

        all_run_args = (
            security_args
            + user_args
            + writable_args
            + resource_args
            + egress_host_args
            + volume_args
            + env_args
            + validated_extra
        )
        logger.info("Docker run_args: %s", all_run_args)

        # Start the container directly via `docker run -d`.
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"
        # Labels make hermes-created containers identifiable to:
        #   * the orphan reaper (`hermes-agent=1` for the global sweep filter)
        #   * future cross-process reuse (`hermes-task-id`, `hermes-profile`)
        #   * operators running `docker ps --filter label=hermes-agent=1`
        # Values are limited to the safe character set defined by
        # _sanitize_label_value(); the configured reuse identity is captured at
        # container-start time and never changes for the container's lifetime.
        profile_name = _container_identity(shared_container_key)
        task_label = _sanitize_label_value(task_id)
        label_args = [
            "--label", "hermes-agent=1",
            "--label", f"hermes-task-id={task_label}",
            "--label", f"hermes-profile={profile_name}",
            "--label", f"{_EGRESS_LABEL_KEY}={egress_label}",
        ]
        if self._private_workspace:
            label_args.extend(
                [
                    "--label", "hermes-workspace-only=1",
                    "--label", f"hermes-owner-pid={self._owner_pid}",
                    "--label", f"hermes-owner-start={self._owner_start}",
                ]
            )
        # Save args for container recreation on "No such container" recovery.
        self._image = image
        self._container_name = container_name
        self._image_uses_s6_init = image_uses_s6_init
        self._all_run_args = all_run_args

        self._labels = {
            "hermes-agent": "1",
            "hermes-task-id": task_label,
            "hermes-profile": profile_name,
            _EGRESS_LABEL_KEY: egress_label,
        }
        if self._private_workspace:
            self._labels.update(
                {
                    "hermes-workspace-only": "1",
                    "hermes-owner-pid": str(self._owner_pid),
                    "hermes-owner-start": str(self._owner_start),
                }
            )

        # Cross-process container reuse (issue #20561 — docs claim "ONE long-lived
        # container shared across sessions").  If a prior Hermes process
        # already started a container for this (task_id, profile) and it
        # still exists, attach to it instead of starting a fresh one.  This
        # restores the documented contract; opt out via
        # ``terminal.docker_persist_across_processes: false``.
        #
        # Reuse matches on labels only.  The egress posture gets its own label
        # because env vars, CA mounts, and host mappings are immutable after
        # container creation — reusing a pre-egress or pre-rotation container
        # would silently bypass the credential firewall.
        reused = False
        if persist_across_processes:
            existing = self._find_reusable_container(
                task_label, profile_name, egress_label,
            )
            if existing is not None:
                container_id, state = existing
                # Network-mode guard: reuse must not silently defeat an
                # egress lockdown.  A container created before the operator
                # set ``docker_network: false`` keeps its original bridge
                # NetworkMode, so label-only reuse would hand the agent a
                # networked container despite the config.  On mismatch we
                # remove the stale container and start fresh — leaving it in
                # place would let the next label-based reuse pick it up again.
                # Only the lockdown direction is guarded: a ``none``-mode
                # container under a default-network config is left alone so
                # operators using ``docker_extra_args: ["--network=none"]``
                # don't get their container churned on every startup.
                mode_mismatch = False
                actual_mode = None
                if not network:
                    actual_mode = self._container_network_mode(container_id)
                    mode_mismatch = actual_mode != "none"
                if mode_mismatch:
                    logger.warning(
                        "Existing container %s has NetworkMode=%s but "
                        "docker_network=false requests an air-gapped "
                        "container — removing it and starting fresh "
                        "(task=%s, profile=%s).",
                        container_id[:12], actual_mode or "unknown",
                        task_label, profile_name,
                    )
                    try:
                        subprocess.run(
                            [self._docker_exe, "rm", "-f", container_id],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=30,
                            check=False,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.TimeoutExpired, OSError) as e:
                        logger.warning("Failed to remove mismatched container %s: %s", container_id[:12], e)
                    existing = None
            if existing is not None:
                container_id, state = existing
                self._container_id = container_id
                if state != "running":
                    try:
                        subprocess.run(
                            [self._docker_exe, "start", container_id],
                            capture_output=True,
                            text=True, encoding='utf-8', errors='replace',
                            timeout=30,
                            check=True,
                            stdin=subprocess.DEVNULL,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(
                            "Failed to start existing container %s (state=%s): "
                            "%s — falling back to a fresh container.",
                            container_id[:12], state, e,
                        )
                        self._container_id = None
                if self._container_id:
                    logger.info(
                        "Reusing container %s (task=%s, profile=%s, prior state=%s)",
                        container_id[:12], task_label, profile_name, state,
                    )
                    reused = True

        if not reused:
            # tini/catatonit as PID 1 reaps zombie children — but s6-overlay
            # images already provide their own /init PID 1, so adding --init
            # there creates two competing inits and breaks startup (#34628).
            init_args = [] if image_uses_s6_init else ["--init"]
            run_cmd = [
                self._docker_exe, "run", "-d",
                *init_args,
                "--name", container_name,
                *label_args,
                "-w", cwd,
                *all_run_args,
                image,
                "sleep", "infinity",  # no fixed lifetime — idle reaper handles cleanup
            ]
            logger.debug("Starting container: %s", ' '.join(run_cmd))
            try:
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=120,  # image pull may take a while
                    check=True,
                    stdin=subprocess.DEVNULL,
                    env=self._docker_client_env(self._run_env_values),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Docker may create the container object before `docker run`
                # fails to start it (e.g. exit code 125 when the daemon isn't
                # ready, or a timeout mid-pull). That orphan is left in
                # "Created" state — which the exited-only orphan reaper
                # (reap_orphan_containers, status=exited) never catches, so it
                # leaks permanently. Remove it by its known name before
                # re-raising. See #7439.
                logger.warning(
                    "docker run failed for %s, cleaning up orphaned container: %s",
                    container_name, e,
                )
                subprocess.run(
                    [self._docker_exe, "rm", "-f", container_name],
                    capture_output=True, timeout=10,
                    stdin=subprocess.DEVNULL,
                )
                if self._git_broker_root:
                    shutil.rmtree(self._git_broker_root, ignore_errors=True)
                    self._git_broker_root = None
                self._remove_private_volumes()
                raise
            self._container_id = result.stdout.strip()
            logger.info("Started container %s (%s)", container_name, self._container_id[:12])

        # Build the init-time env forwarding args used to seed the snapshot.
        if self._private_workspace:
            try:
                self._initialize_private_workspace()
            except Exception:
                subprocess.run(
                    [self._docker_exe, "rm", "-f", self._container_id],
                    capture_output=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
                self._container_id = None
                self._remove_private_volumes()
                if self._git_broker_root:
                    shutil.rmtree(self._git_broker_root, ignore_errors=True)
                    self._git_broker_root = None
                    self._git_broker_gitdir = None
                raise

        self._init_env_args = self._build_init_env_args()

        # Initialize session snapshot inside the container
        self.init_session()

    def _docker_client_env(self, values: dict[str, str]) -> dict[str, str] | None:
        """Env for a spawned docker-client subprocess carrying forwarded values.

        Name-only ``-e KEY`` flags make the docker CLI read each value from
        its own process environment, keeping secrets out of the client's
        world-readable ``/proc/<pid>/cmdline`` (issue #96268). Values live in
        ``/proc/<pid>/environ`` instead, which is owner/root-only.
        Returns ``None`` (inherit as before) when there is nothing to add.
        """
        if not values:
            return None
        client_env = dict(os.environ)
        client_env.update(values)
        return client_env

    def _create_private_volumes(self) -> None:
        """Create the two daemon-owned volumes used by dynamic workspaces."""
        created: list[str] = []
        try:
            for name, kind in (
                (self._workspace_volume, "workspace"),
                (self._git_volume, "git"),
            ):
                assert name is not None
                result = subprocess.run(
                    [
                        self._docker_exe,
                        "volume",
                        "create",
                        "--label",
                        "hermes-agent=1",
                        "--label",
                        "hermes-workspace-only=1",
                        "--label",
                        f"hermes-private-kind={kind}",
                        "--label",
                        f"hermes-task-id={_sanitize_label_value(self._task_id)}",
                        "--label",
                        f"hermes-profile={_sanitize_label_value(_get_active_profile_name())}",
                        "--label",
                        f"hermes-owner-pid={self._owner_pid}",
                        "--label",
                        f"hermes-owner-start={self._owner_start}",
                        name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Docker private {kind} volume creation failed: "
                        f"{(result.stderr or result.stdout).strip()}"
                    )
                created.append(name)
        except Exception:
            for name in created:
                subprocess.run(
                    [self._docker_exe, "volume", "rm", "-f", name],
                    capture_output=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
            raise

    def _remove_private_volumes(self) -> None:
        """Best-effort idempotent removal of dynamic workspace volumes."""
        volume_names = (
            getattr(self, "_workspace_volume", None),
            getattr(self, "_git_volume", None),
        )
        for name in volume_names:
            if not name:
                continue
            subprocess.run(
                [getattr(self, "_docker_exe", "docker"), "volume", "rm", "-f", name],
                capture_output=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
        self._workspace_volume = None
        self._git_volume = None

    def _stream_private_tree(
        self,
        source: Path,
        destination: str,
        *,
        exclude_root_names: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """Stream a validated host tree into a named volume without a bind."""
        from tools.workspace_staging import archive_tree

        assert self._container_id
        payload = archive_tree(source, exclude_root_names=exclude_root_names)
        try:
            result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    "-i",
                    "--user",
                    "0:0",
                    self._container_id,
                    "tar",
                    "-xpf",
                    "-",
                    "-C",
                    destination,
                ],
                stdin=payload,
                capture_output=True,
                timeout=120,
                check=False,
            )
        finally:
            payload.close()
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or b"tar import failed")
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RuntimeError(f"Docker private staging import failed: {detail.strip()}")

    def _initialize_private_workspace(self) -> None:
        """Populate named volumes from pinned host-side snapshots."""
        metadata = self._workspace_git_metadata
        private_git = self._git_broker_gitdir
        if metadata is None or private_git is None:
            raise RuntimeError("Dynamic workspace staging lacks Git metadata")
        self._stream_private_tree(
            metadata.workspace,
            "/workspace",
            exclude_root_names={".git"},
        )
        self._stream_private_tree(private_git, "/hermes-git")
        if self._container_user_spec:
            result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    "--user",
                    "0:0",
                    self._container_id,
                    "chown",
                    "-R",
                    self._container_user_spec,
                    "/workspace",
                    "/hermes-git",
                ],
                capture_output=True,
                timeout=60,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker private staging ownership initialization failed")

    def _export_private_tree(self, source: str):
        """Return a bounded tar stream exported from a named volume."""
        assert self._container_id
        output = tempfile.SpooledTemporaryFile(max_size=8 * 1024**2, mode="w+b")
        error_output = tempfile.SpooledTemporaryFile(max_size=1024**2, mode="w+b")
        timed_out = threading.Event()
        process = subprocess.Popen(
            [
                self._docker_exe,
                "exec",
                self._container_id,
                "tar",
                "-cpf",
                "-",
                "-C",
                source,
                ".",
            ],
            stdout=subprocess.PIPE,
            stderr=error_output,
            stdin=subprocess.DEVNULL,
        )

        def _kill_on_timeout() -> None:
            timed_out.set()
            process.kill()

        timer = threading.Timer(120, _kill_on_timeout)
        timer.daemon = True
        timer.start()
        exceeded = False
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                if output.tell() + len(chunk) > 2 * 1024**3:
                    exceeded = True
                    process.kill()
                    break
                output.write(chunk)
            process.stdout.close()
            process.wait()
        finally:
            timer.cancel()
        error_output.seek(0)
        stderr = error_output.read(1024 * 1024)
        error_output.close()
        if timed_out.is_set():
            output.close()
            raise RuntimeError("Docker private staging export timed out")
        if exceeded:
            output.close()
            raise RuntimeError("Docker private staging export exceeds size limit")
        if process.returncode != 0:
            output.close()
            raise RuntimeError(
                "Docker private staging export failed: "
                + (stderr or b"").decode("utf-8", errors="replace").strip()
            )
        output.seek(0)
        return output

    def _export_private_file(self, source: str, *, max_bytes: int) -> bytes:
        """Read one container file as a bounded raw stream."""
        assert self._container_id
        error_output = tempfile.SpooledTemporaryFile(max_size=1024**2, mode="w+b")
        process = subprocess.Popen(
            [self._docker_exe, "exec", self._container_id, "cat", "--", source],
            stdout=subprocess.PIPE,
            stderr=error_output,
            stdin=subprocess.DEVNULL,
        )
        timed_out = threading.Event()

        def _kill_on_timeout() -> None:
            timed_out.set()
            process.kill()

        timer = threading.Timer(60, _kill_on_timeout)
        timer.daemon = True
        timer.start()
        output = bytearray()
        exceeded = False
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                if len(output) + len(chunk) > max_bytes:
                    exceeded = True
                    process.kill()
                    break
                output.extend(chunk)
            process.stdout.close()
            process.wait()
        finally:
            timer.cancel()
        error_output.seek(0)
        stderr = error_output.read(1024 * 1024)
        error_output.close()
        if timed_out.is_set():
            raise RuntimeError("Docker private file export timed out")
        if exceeded:
            raise RuntimeError("Docker private file export exceeds size limit")
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Docker private file export failed: {detail}")
        return bytes(output)

    def _export_private_bundle(self) -> tuple[bytes, str]:
        """Export branch commits and index-only objects in one bounded Git bundle."""
        assert self._container_id
        metadata = self._workspace_git_metadata
        expected_head = self._git_broker_head
        if metadata is None or expected_head is None:
            raise RuntimeError("Dynamic workspace Git transport is not initialized")
        head_result = subprocess.run(
            [
                self._docker_exe,
                "exec",
                self._container_id,
                "git",
                "--git-dir=/hermes-git",
                "rev-parse",
                "--verify",
                metadata.branch_ref,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        private_head = head_result.stdout.strip()
        if head_result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", private_head):
            raise RuntimeError("Docker private Git head export failed")

        transport_ref = "refs/hermes/index-transport"
        bundle_path = f"/tmp/hermes-workspace-{uuid.uuid4().hex}.bundle"
        try:
            tree_result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    self._container_id,
                    "git",
                    "--git-dir=/hermes-git",
                    "write-tree",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            tree = tree_result.stdout.strip()
            if tree_result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", tree):
                raise RuntimeError("Docker private Git index tree export failed")
            commit_result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    "-i",
                    self._container_id,
                    "git",
                    "--git-dir=/hermes-git",
                    "commit-tree",
                    tree,
                    "-p",
                    private_head,
                ],
                input="Hermes staged index transport\n",
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            envelope = commit_result.stdout.strip()
            if commit_result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", envelope):
                raise RuntimeError("Docker private Git index envelope creation failed")
            update_result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    self._container_id,
                    "git",
                    "--git-dir=/hermes-git",
                    "update-ref",
                    transport_ref,
                    envelope,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if update_result.returncode != 0:
                raise RuntimeError("Docker private Git index envelope ref failed")
            result = subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    self._container_id,
                    "git",
                    "--git-dir=/hermes-git",
                    "bundle",
                    "create",
                    bundle_path,
                    transport_ref,
                    f"^{expected_head}",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker private Git bundle creation failed")
            return self._export_private_file(bundle_path, max_bytes=512 * 1024**2), private_head
        finally:
            subprocess.run(
                [
                    self._docker_exe,
                    "exec",
                    self._container_id,
                    "git",
                    "--git-dir=/hermes-git",
                    "update-ref",
                    "-d",
                    transport_ref,
                ],
                capture_output=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            subprocess.run(
                [self._docker_exe, "exec", self._container_id, "rm", "-f", "--", bundle_path],
                capture_output=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
            )

    def _export_private_index(self) -> bytes:
        """Export the separately bounded worktree index."""
        return self._export_private_file("/hermes-git/index", max_bytes=512 * 1024**2)

    def _sync_private_workspace(self) -> None:
        """Validate container exports in quarantine, then publish fail closed."""
        from tools.workspace_git_transport import (
            validate_workspace_bundle,
            validate_workspace_index,
        )
        from tools.workspace_staging import extract_archive, publish_tree

        metadata = self._workspace_git_metadata
        identity = self._workspace_stage_identity
        expected_manifest = self._workspace_manifest
        expected_head = self._git_broker_head
        if (
            metadata is None
            or identity is None
            or expected_manifest is None
            or expected_head is None
        ):
            raise RuntimeError("Dynamic workspace staging is not initialized")
        workspace_archive = self._export_private_tree("/workspace")
        bundle_bytes, private_head = self._export_private_bundle()
        index_bytes = self._export_private_index()
        workspace_stage = Path(tempfile.mkdtemp(prefix=".hermes-workspace-export-"))
        transport_root = Path(tempfile.mkdtemp(prefix=".hermes-git-transport-"))
        bundle_path = transport_root / "workspace.bundle"
        old_broker_root = self._git_broker_root
        old_private_git = self._git_broker_gitdir
        candidate_root: Path | None = None
        validated_bundle = None
        try:
            extract_archive(workspace_archive, workspace_stage)
            if (workspace_stage / ".git").exists():
                raise RuntimeError("Dynamic workspace export attempted to publish .git")
            bundle_path.write_bytes(bundle_bytes)
            validated_bundle = validate_workspace_bundle(
                bundle_path,
                trusted_repository=metadata.workspace,
                branch_ref=metadata.branch_ref,
                expected_head=expected_head,
                bundle_ref="refs/hermes/index-transport",
            )
            candidate_root = validated_bundle.root
            candidate_git = validated_bundle.gitdir
            envelope = validated_bundle.tip
            parents = _run_git_checked(
                ["--git-dir", str(candidate_git), "rev-list", "--parents", "-n", "1", envelope]
            ).stdout.split()
            if parents != [envelope, private_head]:
                raise RuntimeError("Docker private Git index envelope is invalid")
            _run_git_checked(
                ["--git-dir", str(candidate_git), "update-ref", metadata.branch_ref, private_head]
            )
            index_temp = candidate_git / "index.transport"
            with index_temp.open("xb") as target:
                target.write(index_bytes)
                target.flush()
                os.fsync(target.fileno())
            os.replace(index_temp, candidate_git / "index")
            validate_workspace_index(
                candidate_git,
                worktree=workspace_stage,
                trusted_repository=metadata.workspace,
            )
            _run_git_checked(
                ["--git-dir", str(candidate_git), "fsck", "--strict", "--full", "--no-reflogs"]
            )
            _run_git_checked(["--git-dir", str(candidate_git), "config", "core.bare", "false"])
            _run_git_checked(
                ["--git-dir", str(candidate_git), "config", "core.worktree", "/workspace"]
            )
            _run_git_checked(
                ["--git-dir", str(candidate_git), "symbolic-ref", "HEAD", metadata.branch_ref]
            )
            candidate_head = _run_git_checked(
                ["--git-dir", str(candidate_git), "rev-parse", "--verify", metadata.branch_ref]
            ).stdout.strip()
            _begin_workspace_publication_journal(
                metadata,
                candidate_workspace=workspace_stage,
                candidate_git=candidate_git,
                expected_head=expected_head,
                candidate_head=candidate_head,
            )
            _write_journal_state(metadata, "publishing")
            with self._git_broker_sync_lock:
                self._git_broker_root = candidate_root
                self._git_broker_gitdir = candidate_git
                try:
                    def finalize_publication() -> None:
                        self._sync_workspace_git_broker_locked()
                        _write_journal_state(metadata, "metadata_published")

                    self._workspace_manifest = publish_tree(
                        workspace_stage,
                        metadata.workspace,
                        identity=identity,
                        expected_manifest=expected_manifest,
                        exclude_root_names={".git"},
                        finalize=finalize_publication,
                    )
                except Exception:
                    self._git_broker_root = old_broker_root
                    self._git_broker_gitdir = old_private_git
                    raise
            _write_journal_state(metadata, "committed")
            _discard_workspace_publication_journal(metadata)
            if old_broker_root and old_broker_root != candidate_root:
                shutil.rmtree(old_broker_root, ignore_errors=True)
            # Ownership transferred to the environment; suppress quarantine cleanup.
            if validated_bundle is not None:
                validated_bundle = None
            candidate_root = None
        except Exception:
            if _workspace_publication_journal(metadata).exists():
                _recover_workspace_publication(metadata)
            if candidate_root is not None and self._git_broker_root != candidate_root:
                shutil.rmtree(candidate_root, ignore_errors=True)
            raise
        finally:
            workspace_archive.close()
            if validated_bundle is not None:
                validated_bundle.cleanup()
            shutil.rmtree(workspace_stage, ignore_errors=True)
            shutil.rmtree(transport_root, ignore_errors=True)

    def _build_init_env_args(self) -> list[str]:
        """Build name-only -e args for injecting host env vars into init_session.

        These are used during init_session() so that export -p captures the
        configured environment and the current profile's forwarded values.
        The VALUES intentionally do not appear in the argv — they are passed
        via the docker client subprocess env (see _docker_client_env and
        issue #96268); the flags here are name-only ``-e KEY``.
        """
        if getattr(self, "_workspace_only", False):
            return []
        passthrough_env, unset_names = self._resolve_passthrough_env()
        exec_env: dict[str, str] = dict(self._env)
        exec_env.update(passthrough_env)
        for name in unset_names:
            exec_env.pop(name, None)
        self._init_unset_passthrough_names = tuple(sorted(unset_names))
        self._init_env_values = dict(exec_env)

        args = []
        for key in sorted(exec_env):
            args.extend(["-e", key])
        return args

    def _build_passthrough_env(self) -> dict[str, str]:
        """Resolve forwarded host variables through the active profile scope."""
        return self._resolve_passthrough_env()[0]

    def _resolve_passthrough_env(self) -> tuple[dict[str, str], set[str]]:
        """Return forwarded values and scoped names that must be unset."""
        exec_env: dict[str, str] = {}
        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        resolve_passthrough_value = None
        multiplex_active = False
        is_global_env = lambda _name: False  # noqa: E731
        try:
            from tools.env_passthrough import (
                get_all_passthrough,
                resolve_passthrough_value,
            )
            from agent.secret_scope import _is_global_env, is_multiplex_active as _is_multiplex_active
            is_global_env = _is_global_env
            multiplex_active = _is_multiplex_active()
            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass
        # Explicit docker_forward_env entries are an intentional opt-in and must
        # win over the generic Hermes secret blocklist. Only implicit passthrough
        # keys are filtered. Also strip Hermes-internal dynamic secrets
        # (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_* auth) that the
        # name-based blocklist doesn't cover — see _is_hermes_internal_secret.
        _implicit_forward = {
            k for k in passthrough_keys if not _is_hermes_internal_secret(k)
        }
        forward_keys = explicit_forward_keys | (_implicit_forward - _HERMES_PROVIDER_ENV_BLOCKLIST)
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        unset_names: set[str] = set()
        for key in sorted(forward_keys):
            value = os.getenv(key) or hermes_env.get(key)
            if resolve_passthrough_value is not None:
                value = resolve_passthrough_value(key, value)
            if value is not None:
                exec_env[key] = value
            elif multiplex_active and not is_global_env(key) and _ENV_VAR_NAME_RE.fullmatch(key):
                unset_names.add(key)
        return exec_env, unset_names

    def _build_runtime_env_args_with_unsets(
        self,
    ) -> tuple[list[str], tuple[str, ...], dict[str, str]]:
        """Build runtime forwarding args, names absent from scope, and values.

        The returned args are name-only ``-e KEY`` flags (issue #96268); the
        values dict must be injected into the docker client subprocess env.
        """
        passthrough_env, unset_names = self._resolve_passthrough_env()
        args = []
        for key in sorted(passthrough_env):
            args.extend(["-e", key])
        return args, tuple(sorted(unset_names)), dict(passthrough_env)

    def _build_runtime_env_args(self) -> list[str]:
        """Build only dynamic forwarded names for a non-login command."""
        return self._build_runtime_env_args_with_unsets()[0]

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Docker container."""
        assert self._container_id, "Container not started"
        cmd = [self._docker_exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")

        # Init seeds the snapshot. Profile-scoped passthrough values are also
        # injected on every later command because this container can be shared
        # by multiple routed profiles in one gateway process.
        # Env flags are name-only; values travel via the client subprocess
        # env so they never hit world-readable /proc/*/cmdline (#96268).
        unset_names: tuple[str, ...] = ()
        env_values: dict[str, str] = {}
        if login:
            cmd.extend(self._init_env_args)
            env_values = dict(getattr(self, "_init_env_values", {}))
        elif self._profile_scoped_passthrough:
            runtime_args, unset_names, env_values = (
                self._build_runtime_env_args_with_unsets()
            )
            cmd.extend(runtime_args)

        if login:
            unset_names = getattr(self, "_init_unset_passthrough_names", ())
        if unset_names:
            quoted_names = " ".join(shlex.quote(name) for name in unset_names)
            cmd_string = f"unset {quoted_names} 2>/dev/null || true\n{cmd_string}"

        cmd.extend([self._container_id])

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        client_env = self._docker_client_env(env_values)
        if client_env is not None:
            return _popen_bash(cmd, stdin_data, env=client_env)
        return _popen_bash(cmd, stdin_data)

    # ------------------------------------------------------------------
    # "No such container" recovery (issue #36266)
    # ------------------------------------------------------------------

    _NO_CONTAINER_PATTERNS = (
        "No such container",
        "is not running",
        "no such container",
    )

    def _is_container_gone(self, output: str) -> bool:
        """Return True if the output indicates the container no longer exists."""
        return any(p in output for p in self._NO_CONTAINER_PATTERNS)

    def _recreate_container(self) -> bool:
        """Recreate the container after it was removed out-of-band.

        Tries label-based reuse first; if no existing container is found,
        starts a fresh one with the same image and run-args.  Returns True
        on success, False if recreation fails (caller should surface the
        original error).
        """
        old_id = (self._container_id or "")[:12]
        logger.warning(
            "Container %s appears to be gone — attempting recovery", old_id,
        )
        self._container_id = None

        # 1. Try label-based reuse (another process may have recreated it).
        task_label = self._labels.get("hermes-task-id", "")
        profile_label = self._labels.get("hermes-profile", "")
        existing = self._find_reusable_container(
            task_label, profile_label, self._labels.get(_EGRESS_LABEL_KEY, "off"),
        )
        if existing is not None:
            cid, state = existing
            if state == "running":
                self._container_id = cid
                logger.info("Recovery: reusing running container %s", cid[:12])
            else:
                try:
                    subprocess.run(
                        [self._docker_exe, "start", cid],
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, check=True,
                        stdin=subprocess.DEVNULL,
                    )
                    self._container_id = cid
                    logger.info("Recovery: restarted container %s", cid[:12])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning("Recovery: failed to start container %s: %s", cid[:12], e)

        # 2. No reusable container — create a fresh one.
        if not self._container_id:
            if not self._image:
                logger.error("Recovery: no saved image name, cannot recreate container")
                return False
            try:
                import uuid as _uuid
                new_name = f"hermes-{_uuid.uuid4().hex[:8]}"
                init_args = [] if self._image_uses_s6_init else ["--init"]
                label_args = []
                for k, v in self._labels.items():
                    label_args.extend(["--label", f"{k}={v}"])
                run_cmd = [
                    self._docker_exe, "run", "-d",
                    *init_args,
                    "--name", new_name,
                    *label_args,
                    "-w", self.cwd,
                    *self._all_run_args,
                    self._image,
                    "sleep", "infinity",
                ]
                result = subprocess.run(
                    run_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                    env=self._docker_client_env(self._run_env_values),
                )
                self._container_id = result.stdout.strip()
                self._container_name = new_name
                logger.info(
                    "Recovery: created fresh container %s (%s)",
                    new_name, self._container_id[:12],
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                logger.error("Recovery: failed to create new container: %s", e)
                return False

        # 3. Re-initialize session snapshot in the (re)created container.
        try:
            self._snapshot_ready = False
            self.init_session()
        except Exception as e:
            logger.error("Recovery: init_session failed in new container: %s", e)
            return False

        logger.info("Recovery successful — new container %s", (self._container_id or "")[:12])
        return True

    def _sync_workspace_git_broker(self) -> None:
        """Import private Git commits/index into the exact owned worktree only."""
        with self._git_broker_sync_lock:
            self._sync_workspace_git_broker_locked()

    def _sync_workspace_git_broker_locked(self) -> None:
        """Synchronize broker state while holding the per-environment lock."""
        metadata = self._workspace_git_metadata
        private_git = self._git_broker_gitdir
        expected_head = self._git_broker_head
        if metadata is None or private_git is None or expected_head is None:
            return

        private_index = private_git / "index"
        private_index_digest = _file_digest(private_index)
        private_index_changed = private_index_digest != self._git_broker_index_digest
        host_index = metadata.gitdir / "index"

        symbolic = _run_git_checked(
            ["--git-dir", str(private_git), "symbolic-ref", "HEAD"]
        ).stdout.strip()
        if symbolic != metadata.branch_ref:
            raise RuntimeError("Git broker refused a branch switch outside the owned branch")
        private_head = _run_git_checked(
            ["--git-dir", str(private_git), "rev-parse", "--verify", "HEAD"]
        ).stdout.strip()
        index_lock: Path | None = None
        if private_index_changed:
            index_lock = metadata.gitdir / "index.lock"
            try:
                lock_fd = os.open(
                    index_lock,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                raise RuntimeError(
                    "Git broker refused concurrent owned-index metadata changes"
                ) from exc
            try:
                with os.fdopen(lock_fd, "wb") as target, private_index.open("rb") as source:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                if _file_digest(host_index) != self._host_index_digest:
                    raise RuntimeError(
                        "Git broker refused concurrent owned-index metadata changes"
                    )
            except Exception:
                index_lock.unlink(missing_ok=True)
                raise

        ref_updated = False
        try:
            if private_head != expected_head:
                ancestry = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "-c",
                        "core.fsmonitor=false",
                        "-c",
                        "diff.external=",
                        "--git-dir",
                        str(private_git),
                        "merge-base",
                        "--is-ancestor",
                        expected_head,
                        private_head,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=_git_broker_env(),
                )
                if ancestry.returncode != 0:
                    raise RuntimeError("Git broker refused non-fast-forward history")
                host_head = _run_git_checked(
                    ["-C", str(metadata.workspace), "rev-parse", "--verify", metadata.branch_ref]
                ).stdout.strip()
                if host_head != expected_head:
                    raise RuntimeError("Git broker refused concurrent owned-branch metadata changes")
                _run_git_checked(
                    [
                        "-c",
                        "maintenance.auto=false",
                        "-C",
                        str(metadata.workspace),
                        "fetch",
                        "--no-tags",
                        "--no-write-fetch-head",
                        str(private_git),
                        private_head,
                    ]
                )
                _run_git_checked(
                    [
                        "-C",
                        str(metadata.workspace),
                        "update-ref",
                        metadata.branch_ref,
                        private_head,
                        expected_head,
                    ]
                )
                ref_updated = True

            if index_lock is not None:
                os.replace(index_lock, host_index)
                index_lock = None
        except Exception:
            if ref_updated:
                try:
                    _run_git_checked(
                        [
                            "-C",
                            str(metadata.workspace),
                            "update-ref",
                            metadata.branch_ref,
                            expected_head,
                            private_head,
                        ]
                    )
                except Exception as rollback_exc:
                    raise RuntimeError(
                        "Git broker failed to roll back the owned branch after index failure"
                    ) from rollback_exc
            raise
        finally:
            if index_lock is not None:
                index_lock.unlink(missing_ok=True)

        if private_head != expected_head:
            self._git_broker_head = private_head
        if private_index_changed:
            self._host_index_digest = private_index_digest
            self._git_broker_index_digest = private_index_digest

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Execute a command, auto-recovering from dead containers.

        If the container was removed out-of-band (idle reaper, docker prune,
        OOM kill, daemon restart), detect the error and recreate the container
        transparently before retrying once.
        """
        result = super().execute(command, cwd, **kwargs)
        if (
            result.get("returncode", 0) != 0
            and self._is_container_gone(result.get("output", ""))
            and self._persist_across_processes
        ):
            if self._recreate_container():
                result = super().execute(command, cwd, **kwargs)
        if getattr(self, "_private_workspace", False):
            try:
                self._sync_private_workspace()
            except Exception as exc:
                logger.error("Private workspace publication failed closed: %s", exc)
                output = result.get("output", "")
                result["output"] = (
                    f"{output}\nPrivate workspace refused host publication: {exc}"
                ).strip()
                result["returncode"] = 126
        elif getattr(self, "_workspace_only", False):
            try:
                self._sync_workspace_git_broker()
            except Exception as exc:
                logger.error("Workspace Git broker failed closed: %s", exc)
                output = result.get("output", "")
                result["output"] = f"{output}\nGit broker refused host sync: {exc}".strip()
                result["returncode"] = 126
        return result

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            docker = find_docker() or "docker"
            result = subprocess.run(
                [docker, "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
                stdin=subprocess.DEVNULL,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                [docker, "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run([docker, "rm", container_id],
                                   capture_output=True, timeout=5,
                                   stdin=subprocess.DEVNULL)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def _container_network_mode(self, container_id: str) -> Optional[str]:
        """Return the container's ``HostConfig.NetworkMode`` (e.g. ``bridge``,
        ``none``, ``host``), or ``None`` when inspection fails.

        Used by the reuse path to make sure a persisted container's network
        mode still matches the operator's ``docker_network`` setting; callers
        treat ``None`` (unknown) as a mismatch when lockdown was requested,
        so a failed inspect fails closed rather than open.
        """
        try:
            result = subprocess.run(
                [
                    self._docker_exe, "inspect",
                    "--format", "{{.HostConfig.NetworkMode}}",
                    container_id,
                ],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker inspect NetworkMode failed: %s", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker inspect NetworkMode returned %d: %s",
                result.returncode, result.stderr.strip(),
            )
            return None
        mode = result.stdout.strip()
        return mode or None

    def _find_reusable_container(
        self,
        task_label: str,
        profile_label: str,
        egress_label: str,
    ) -> Optional[tuple[str, str]]:
        """Look for an existing container labeled for this (task, profile).

        Returns ``(container_id, state)`` on hit, ``None`` on miss / on any
        failure (including ``docker ps`` itself failing). State is one of the
        values Docker reports via ``{{.State}}`` — e.g. ``running``, ``exited``,
        ``created``, ``paused``, ``restarting``, ``dead``. The caller decides
        whether the state warrants ``docker start`` before reuse.

        Restricted to the docker-stored label set this class creates; never
        matches containers that happened to be named ``hermes-*`` but were
        started by some other tool.
        """
        try:
            filters = [
                "--filter", "label=hermes-agent=1",
                "--filter", f"label=hermes-task-id={task_label}",
                "--filter", f"label=hermes-profile={profile_label}",
            ]
            if egress_label != "off":
                filters.extend(["--filter", f"label={_EGRESS_LABEL_KEY}={egress_label}"])
                fmt = "{{.ID}}\t{{.State}}"
            else:
                # When egress is off, we widen the probe to find any
                # task+profile container (regardless of egress label), then
                # post-filter in Python: reject containers whose
                # hermes-egress label is present and not "off".  Without
                # this, a container created with egress=on can be silently
                # reused after the operator runs "hermes egress disable",
                # preserving baked-in proxy env and CA mounts.
                fmt = '{{.ID}}\t{{.State}}\t{{.Label "' + _EGRESS_LABEL_KEY + '"}}'
            result = subprocess.run(
                [
                    self._docker_exe, "ps", "-a",
                    *filters,
                    "--format", fmt,
                ],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=10,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("docker ps probe failed: %s — will start a fresh container", e)
            return None
        if result.returncode != 0:
            logger.debug(
                "docker ps probe returned %d: %s — will start a fresh container",
                result.returncode, result.stderr.strip(),
            )
            return None
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        # Multiple matches are unusual (one (task, profile) should produce one
        # container) but can happen if a previous Hermes process crashed
        # mid-cleanup. Prefer a running one if present; otherwise pick the
        # first listed. Stale duplicates get reaped by the orphan-reaper in a
        # follow-up commit; we don't try to be heroic about them here.
        running = None
        first = None
        for ln in lines:
            if egress_label == "off":
                # Format: ID\tState\tEgressLabel — parse all three fields
                # and reject containers with a non-off egress label.
                parts = ln.split("\t", 2)
                if len(parts) < 3:
                    continue
                cid, state, egress_val = parts[0], parts[1].lower(), parts[2]
                if egress_val not in ("", "<no value>", "off"):
                    logger.debug(
                        "skipping container %s for egress=off reuse: "
                        "label %s=%r", cid, _EGRESS_LABEL_KEY, egress_val,
                    )
                    continue
            else:
                parts = ln.split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, state = parts[0], parts[1].lower()
            if first is None:
                first = (cid, state)
            if state == "running" and running is None:
                running = (cid, state)
        return running or first

    def cleanup(self, *, force_remove: bool = False):
        """Tear down the container according to persist mode and *force_remove*.

        Persist-mode (``persist_across_processes=True``, the default) leaves the
        container **running** untouched. The docs promise "ONE long-lived
        container shared across sessions" and stopping it on every Hermes exit
        breaks that promise:

        * Background processes inside the container (``npm run dev``, watchers,
          long-running pytest) get killed every time the user runs ``/quit``.
        * Every reuse requires ``docker start`` + waiting for the container to
          come back up, adding 1–2s to the first tool call of the new session.
        * The user-visible difference between "ONE long-lived container" and
          "a new container that happens to share state" is exactly this:
          processes survive in the former, die in the latter.

        Resource reclamation for the persist-mode case lives in the
        ``reap_orphan_containers()`` path (see issue #20561 commit 3): if no
        Hermes process touches a labeled container for ``2 × lifetime_seconds``
        it gets ``docker rm -f``'d at the next Hermes startup. That covers the
        SIGKILL / OOM / abandoned-laptop cases without us needing to stop the
        container on every graceful exit.

        Opt-out mode (``persist_across_processes=False``) still does
        ``docker stop`` + ``docker rm -f`` on every cleanup, matching the
        pre-PR behavior for users who explicitly want per-process isolation.

        ``force_remove=True`` overrides persist mode and always tears the
        container down (``docker stop`` + ``docker rm -f``). This is the
        explicit-teardown path for ``/reset``, ``cleanup_vm(task_id)``-driven
        resets, or any caller that wants a guaranteed fresh container on next
        ``DockerEnvironment(task_id=...)``. No current caller passes
        ``force_remove=True``; the parameter is here so the explicit-teardown
        semantics can be wired up later without changing this method's
        signature.

        Cleanup runs on a daemon thread with bounded ``subprocess.run`` calls
        (not the racy ``Popen(... &)`` pattern from before PR #33645). The
        atexit hook in ``tools/terminal_tool.py`` waits up to 15s for the
        thread to finish before the interpreter exits, so ``docker stop`` /
        ``docker rm`` actually completes when we do trigger it.
        """
        container_id = self._container_id
        if not container_id:
            # Still drop the bind-mount dirs if any were allocated and we're
            # NOT in persist mode (persist mode preserves them).
            if not self._persistent:
                for d in (self._workspace_dir, self._home_dir):
                    if d:
                        shutil.rmtree(d, ignore_errors=True)
            git_broker_root = getattr(self, "_git_broker_root", None)
            if git_broker_root:
                shutil.rmtree(git_broker_root, ignore_errors=True)
                self._git_broker_root = None
                self._git_broker_gitdir = None
            self._remove_private_volumes()
            return

        # Decide what to actually do. Three cases:
        #
        #   force_remove=True             → stop + rm (explicit teardown)
        #   persist_across_processes=True → no-op (leave container running)
        #   persist_across_processes=False → stop + rm (per-process isolation)
        #
        # The persist-mode no-op is the issue-#20561 contract: the container
        # outlives Hermes processes, processes inside it stay alive, and
        # reuse on next startup is instant.
        if force_remove:
            should_stop = True
            should_remove = True
        elif self._persist_across_processes:
            # No-op for the container. Drop the in-process handle so a fresh
            # __init__ will re-probe via labels (and find the running
            # container) instead of trying to reuse a stale Python reference.
            self._container_id = None
            return
        else:
            should_stop = True
            should_remove = True

        # Capture state needed by the worker before we null out the attrs —
        # the worker thread can outlive ``self``.
        docker_exe = self._docker_exe
        log_id = container_id[:12]
        git_broker_root = self._git_broker_root
        private_volumes = tuple(
            name for name in (self._workspace_volume, self._git_volume) if name
        )

        def _do_cleanup() -> None:
            if should_stop:
                try:
                    subprocess.run(
                        [docker_exe, "stop", "-t", "10", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker stop %s timed out / failed: %s", log_id, e)
            if should_remove:
                try:
                    subprocess.run(
                        [docker_exe, "rm", "-f", container_id],
                        capture_output=True, timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker rm -f %s failed: %s", log_id, e)
            if git_broker_root:
                shutil.rmtree(git_broker_root, ignore_errors=True)
            for name in private_volumes:
                try:
                    subprocess.run(
                        [docker_exe, "volume", "rm", "-f", name],
                        capture_output=True,
                        timeout=30,
                        stdin=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    logger.warning("docker volume rm -f %s failed: %s", name, e)

        # Daemon thread: doesn't block interpreter exit (atexit returns
        # promptly), but unlike the old ``Popen(... &)`` shell trick the
        # Python-level join semantics let the thread actually run to
        # completion if the interpreter is still alive. atexit registers
        # ``_atexit_cleanup`` in terminal_tool.py which waits up to ~60s for
        # outstanding cleanups, so most exits complete the work cleanly.
        t = threading.Thread(target=_do_cleanup, daemon=True, name=f"hermes-cleanup-{log_id}")
        t.start()
        self._cleanup_thread = t
        self._container_id = None
        self._git_broker_root = None
        self._git_broker_gitdir = None
        self._workspace_volume = None
        self._git_volume = None

        # Bind-mount dir teardown only runs when we actually removed the
        # container (the dirs are the container's filesystem state; keeping
        # them around with no container would orphan the data on disk).
        if should_remove and not self._persistent:
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)

    def wait_for_cleanup(self, timeout: float = 30.0) -> bool:
        """Block up to *timeout* seconds for the cleanup worker thread.

        Returns ``True`` if the thread finished (or no thread was started),
        ``False`` on timeout. The atexit hook in terminal_tool.py calls this
        on every active environment so docker stop/rm actually completes
        before the Python process exits — without this, ``hermes /quit``
        races the interpreter shutdown and leaves stopped containers behind.
        """
        thread = getattr(self, "_cleanup_thread", None)
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
