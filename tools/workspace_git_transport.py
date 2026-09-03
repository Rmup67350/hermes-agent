"""Host-side validation for Git data exported by Docker workspaces.

The container is untrusted.  Only a bounded Git bundle is admitted, seeded with
one trusted host commit, fscked in a host-created quarantine, and proven to be a
fast-forward of the exact owned branch before publication code may consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile

_DEFAULT_MAX_BUNDLE_BYTES = 512 * 1024**2
_DEFAULT_MAX_OBJECTS = 500_000
_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
_BRANCH_RE = re.compile(r"^refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*$")


class WorkspaceGitTransportError(RuntimeError):
    """Untrusted Git transport data violated an admission invariant."""


@dataclass
class ValidatedWorkspaceBundle:
    root: Path
    gitdir: Path
    tip: str
    branch_ref: str
    expected_head: str

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _git_env() -> dict[str, str]:
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


def _git(args: list[str], *, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "maintenance.auto=false",
            "-c",
            "diff.external=",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_git_env(),
    )
    if check and result.returncode != 0:
        raise WorkspaceGitTransportError("workspace Git bundle is invalid")
    return result


def _object_ids(gitdir: Path) -> set[str]:
    result = _git(
        [
            "--git-dir",
            str(gitdir),
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname)",
        ]
    )
    objects = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if any(not _SHA_RE.fullmatch(object_id) for object_id in objects):
        raise WorkspaceGitTransportError("workspace Git object inventory is invalid")
    return objects


def _validate_bundle_file(bundle: Path, max_bundle_bytes: int) -> None:
    try:
        metadata = bundle.stat(follow_symlinks=False)
    except OSError as exc:
        raise WorkspaceGitTransportError("workspace Git bundle is invalid") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WorkspaceGitTransportError("workspace Git bundle is invalid")
    if metadata.st_size <= 0:
        raise WorkspaceGitTransportError("workspace Git bundle is invalid")
    if metadata.st_size > max_bundle_bytes:
        raise WorkspaceGitTransportError("workspace Git bundle exceeds size limit")


def _index_entries(gitdir: Path, *, worktree: Path) -> list[tuple[str, str, str]]:
    try:
        output = _git(
            [
                "--git-dir",
                str(gitdir),
                "--work-tree",
                str(worktree),
                "ls-files",
                "--stage",
                "-z",
            ]
        ).stdout
    except (UnicodeError, WorkspaceGitTransportError) as exc:
        raise WorkspaceGitTransportError("workspace Git index is invalid") from exc
    entries: list[tuple[str, str, str]] = []
    for record in output.split("\0"):
        if not record:
            continue
        try:
            metadata, path = record.split("\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError as exc:
            raise WorkspaceGitTransportError("workspace Git index is invalid") from exc
        candidate = PurePosixPath(path)
        if (
            candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or any(part.casefold() == ".git" for part in candidate.parts)
        ):
            raise WorkspaceGitTransportError("workspace Git index path is unsafe")
        if stage != "0":
            raise WorkspaceGitTransportError("workspace Git index has unresolved stages")
        if not _SHA_RE.fullmatch(object_id):
            raise WorkspaceGitTransportError("workspace Git index object is invalid")
        entries.append((mode, object_id, path))
    return entries


def _batch_object_types(gitdir: Path, object_ids: set[str]) -> dict[str, str]:
    if not object_ids:
        return {}
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "--git-dir",
            str(gitdir),
            "cat-file",
            "--batch-check=%(objectname) %(objecttype)",
        ],
        input="".join(f"{object_id}\n" for object_id in sorted(object_ids)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_git_env(),
    )
    if result.returncode != 0:
        raise WorkspaceGitTransportError("workspace Git index object check failed")
    checked: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[1] == "missing":
            continue
        checked[fields[0]] = fields[1]
    return checked


def validate_workspace_index(
    gitdir: Path,
    *,
    worktree: Path,
    trusted_repository: Path,
    max_entries: int = 500_000,
) -> None:
    """Reject a container index that could poison the owned host worktree."""
    if max_entries <= 0:
        raise WorkspaceGitTransportError("workspace Git index limit is invalid")
    entries = _index_entries(Path(gitdir), worktree=Path(worktree))
    if len(entries) > max_entries:
        raise WorkspaceGitTransportError("workspace Git index exceeds entry limit")
    trusted_gitdir = _git(
        ["-C", str(trusted_repository), "rev-parse", "--absolute-git-dir"]
    ).stdout.strip()
    trusted_entries = set(
        _index_entries(Path(trusted_gitdir), worktree=Path(trusted_repository))
    )
    regular_objects: set[str] = set()
    for entry in entries:
        mode, object_id, _path = entry
        if mode in {"100644", "100755"}:
            regular_objects.add(object_id)
        elif mode == "160000" and entry in trusted_entries:
            # Submodule commits often do not exist in the superproject object DB;
            # preserve only an exact trusted gitlink, never a container-created one.
            continue
        else:
            raise WorkspaceGitTransportError("workspace Git index has unsupported mode")
    object_types = _batch_object_types(Path(gitdir), regular_objects)
    if any(object_types.get(object_id) != "blob" for object_id in regular_objects):
        raise WorkspaceGitTransportError("workspace Git index has missing index object")


def validate_workspace_bundle(
    bundle: Path,
    *,
    trusted_repository: Path,
    branch_ref: str,
    expected_head: str,
    max_bundle_bytes: int = _DEFAULT_MAX_BUNDLE_BYTES,
    max_objects: int = _DEFAULT_MAX_OBJECTS,
    bundle_ref: str | None = None,
) -> ValidatedWorkspaceBundle:
    """Return a host-created quarantine containing one validated branch tip."""
    bundle = Path(bundle)
    trusted_repository = Path(trusted_repository)
    if not _BRANCH_RE.fullmatch(branch_ref) or ".." in branch_ref.split("/"):
        raise WorkspaceGitTransportError("workspace Git branch is invalid")
    admitted_ref = bundle_ref or branch_ref
    if admitted_ref != branch_ref and admitted_ref != "refs/hermes/index-transport":
        raise WorkspaceGitTransportError("workspace Git transport ref is invalid")
    if not _SHA_RE.fullmatch(expected_head):
        raise WorkspaceGitTransportError("workspace Git expected head is invalid")
    if max_bundle_bytes <= 0 or max_objects <= 0:
        raise WorkspaceGitTransportError("workspace Git transport limits are invalid")
    _validate_bundle_file(bundle, max_bundle_bytes)

    heads = _git(["bundle", "list-heads", str(bundle)]).stdout.splitlines()
    parsed = [line.split(maxsplit=1) for line in heads if line.strip()]
    if len(parsed) != 1 or len(parsed[0]) != 2 or parsed[0][1] != admitted_ref:
        raise WorkspaceGitTransportError(
            "workspace Git bundle must contain exactly the owned branch"
        )
    tip = parsed[0][0]
    if not _SHA_RE.fullmatch(tip):
        raise WorkspaceGitTransportError("workspace Git bundle is invalid")

    root = Path(tempfile.mkdtemp(prefix=".hermes-git-quarantine-"))
    gitdir = root / "repo.git"
    try:
        _git(["init", "--bare", "-q", str(gitdir)])
        expected_ref = "refs/hermes/expected"
        _git(
            [
                "--git-dir",
                str(gitdir),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(trusted_repository),
                f"{expected_head}:{expected_ref}",
            ]
        )
        seeded = _git(
            ["--git-dir", str(gitdir), "rev-parse", "--verify", expected_ref]
        ).stdout.strip()
        if seeded != expected_head:
            raise WorkspaceGitTransportError("trusted Git seed changed")
        trusted_objects = _object_ids(gitdir)
        _git(["--git-dir", str(gitdir), "bundle", "verify", str(bundle)])
        incoming_ref = "refs/hermes/incoming"
        _git(
            [
                "--git-dir",
                str(gitdir),
                "-c",
                "fetch.fsckObjects=true",
                "-c",
                "transfer.fsckObjects=true",
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(bundle),
                f"{admitted_ref}:{incoming_ref}",
            ]
        )
        fetched_tip = _git(
            ["--git-dir", str(gitdir), "rev-parse", "--verify", incoming_ref]
        ).stdout.strip()
        if fetched_tip != tip:
            raise WorkspaceGitTransportError("workspace Git bundle tip changed")
        ancestry = _git(
            [
                "--git-dir",
                str(gitdir),
                "merge-base",
                "--is-ancestor",
                expected_head,
                tip,
            ],
            check=False,
        )
        if ancestry.returncode != 0:
            raise WorkspaceGitTransportError("workspace Git bundle is not a fast-forward")
        _git(["--git-dir", str(gitdir), "fsck", "--strict", "--full", "--no-reflogs"])
        reachable = {
            line.split(maxsplit=1)[0]
            for line in _git(
                ["--git-dir", str(gitdir), "rev-list", "--objects", f"{expected_head}..{tip}"]
            ).stdout.splitlines()
            if line.strip()
        }
        imported_objects = _object_ids(gitdir) - trusted_objects
        if not imported_objects.issubset(reachable):
            raise WorkspaceGitTransportError("workspace Git bundle contains hidden objects")
        if len(imported_objects) > max_objects:
            raise WorkspaceGitTransportError("workspace Git bundle exceeds object limit")
        _git(["--git-dir", str(gitdir), "symbolic-ref", "HEAD", branch_ref])
        _git(["--git-dir", str(gitdir), "update-ref", branch_ref, tip])
        _git(["--git-dir", str(gitdir), "update-ref", "-d", incoming_ref])
        _git(["--git-dir", str(gitdir), "update-ref", "-d", expected_ref])
        return ValidatedWorkspaceBundle(root, gitdir, tip, branch_ref, expected_head)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
