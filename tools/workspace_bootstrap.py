"""Trusted, host-side worktree bootstrap for workspace-only Docker."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import stat
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = frozenset({"registry", "policy", "profile", "agent", "key", "lane_sha256"})


class WorkspaceBootstrapError(RuntimeError):
    """A workspace-only environment cannot be safely bootstrapped."""


def _lane_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "factory_lane.py"


def validate_workspace_bootstrap_spec(
    value: Any, *, allow_empty: bool = False
) -> dict[str, str]:
    if allow_empty and value == {}:
        return {}
    if not isinstance(value, dict) or set(value) != _REQUIRED:
        raise WorkspaceBootstrapError("workspace_bootstrap must contain only the required trusted fields")
    if any(not isinstance(value[name], str) or not value[name] for name in _REQUIRED):
        raise WorkspaceBootstrapError("workspace_bootstrap fields must be non-empty strings")
    if not _SHA256_RE.fullmatch(value["lane_sha256"]):
        raise WorkspaceBootstrapError("workspace_bootstrap lane_sha256 must be a SHA-256 digest")
    return dict(value)


def _spec(value: Any) -> dict[str, str]:
    return validate_workspace_bootstrap_spec(value)


def _workspace_identity(path: str) -> dict[str, int | str]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise WorkspaceBootstrapError("bootstrap worktree identity is unavailable") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise WorkspaceBootstrapError("bootstrap worktree identity is not a directory")
    finally:
        os.close(fd)
    return {"path": path, "device": metadata.st_dev, "inode": metadata.st_ino}


def revalidate_workspace_identity(config: dict[str, Any]) -> None:
    """Fail closed if a bootstrapped worktree path no longer names the pinned inode."""
    if not uses_dynamic_workspace_bootstrap(config):
        return
    identity = config.get("workspace_identity")
    if not isinstance(identity, dict) or set(identity) != {"path", "device", "inode"}:
        raise WorkspaceBootstrapError("dynamic workspace binding has no pinned identity")
    if identity.get("path") != config.get("host_cwd"):
        raise WorkspaceBootstrapError("dynamic workspace binding path changed")
    current = _workspace_identity(identity["path"])
    if current != identity:
        raise WorkspaceBootstrapError("dynamic workspace binding identity changed")


def _verified_script(expected_hash: str) -> tuple[Path, bytes]:
    """Read and hash a regular, non-symlink lane script through one FD.

    The returned bytes, not the mutable pathname, are passed to the child Python
    process. This keeps the verified object and executed object identical even if
    another process replaces or rewrites the configured script after this call.
    """
    script = _lane_script()
    try:
        fd = os.open(script, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise WorkspaceBootstrapError("trusted factory lane script must be a regular file")
            with os.fdopen(fd, "rb", closefd=False) as source:
                contents = source.read()
        finally:
            os.close(fd)
    except WorkspaceBootstrapError:
        raise
    except OSError as exc:
        raise WorkspaceBootstrapError("trusted factory lane script is unavailable") from exc
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise WorkspaceBootstrapError("trusted factory lane script hash does not match configured lane_sha256")
    return script, contents


def _canonical_worktree(value: Any) -> str:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise WorkspaceBootstrapError("bootstrap returned an invalid worktree")
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise WorkspaceBootstrapError("bootstrap worktree is unavailable") from exc
    if path.is_symlink() or str(resolved) != value or not resolved.is_dir():
        raise WorkspaceBootstrapError("bootstrap worktree must be a canonical non-symlink directory")
    result = subprocess.run(
        ["git", "-C", value, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, timeout=10, shell=False,
    )
    if result.returncode != 0 or result.stdout.strip() != value:
        raise WorkspaceBootstrapError("bootstrap returned a non-canonical Git worktree")
    return value


def prepare_workspace_only_config(config: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    """Return a copied config with one verified bootstrap worktree injected.

    The caller invokes this only immediately before creating a Docker environment.
    Bootstrap executes a fixed argv, verifies the exact reviewed lane script hash,
    and validates the returned canonical Git top-level before Docker sees host_cwd.
    """
    prepared = copy.deepcopy(config)
    if not (prepared.get("env_type") == "docker" and prepared.get("docker_workspace_only")):
        return prepared
    raw = prepared.get("workspace_bootstrap")
    if raw in (None, {}):
        return prepared
    spec = _spec(raw)
    script, contents = _verified_script(spec["lane_sha256"])
    runner = (
        "import sys; "
        "__file__ = sys.argv[1]; "
        "sys.argv = [__file__, *sys.argv[2:]]; "
        "exec(compile(sys.stdin.buffer.read(), __file__, \"exec\"))"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner, str(script), "--registry", spec["registry"], "bootstrap", spec["key"],
         "--policy", spec["policy"], "--profile", spec["profile"], "--agent", spec["agent"],
         "--session", str(task_id), "--owner-pid", str(os.getpid())],
        input=contents, capture_output=True, text=False, timeout=45, shell=False,
    )
    if result.returncode != 0:
        raise WorkspaceBootstrapError("trusted workspace bootstrap failed closed")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise WorkspaceBootstrapError("trusted workspace bootstrap returned invalid JSON") from exc
    prepared["host_cwd"] = _canonical_worktree(payload.get("worktree"))
    prepared["workspace_identity"] = _workspace_identity(prepared["host_cwd"])
    prepared["workspace_transport"] = "volume"
    prepared["cwd"] = "/workspace"
    prepared["docker_mount_cwd_to_workspace"] = True
    return prepared


def uses_dynamic_workspace_bootstrap(config: dict[str, Any]) -> bool:
    """Whether an environment must remain isolated to its originating session.

    A workspace-only Docker container bind-mounts exactly one host worktree.
    Therefore a dynamically bootstrapped workspace cannot use the normal shared
    ``default`` container key without allowing another session to inherit that
    mount.
    """
    return bool(
        config.get("env_type") == "docker"
        and config.get("docker_workspace_only")
        and config.get("workspace_bootstrap") not in (None, {})
    )
