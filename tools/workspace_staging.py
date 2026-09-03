"""Fail-closed private staging for dynamically owned Docker workspaces.

The container runtime only receives tar streams and named volumes. Host paths are
opened with ``O_NOFOLLOW`` and all publication writes stay anchored to a pinned
workspace directory descriptor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from typing import BinaryIO

_DEFAULT_MAX_FILES = 100_000
_DEFAULT_MAX_TOTAL_BYTES = 2 * 1024**3
_DEFAULT_MAX_FILE_BYTES = 512 * 1024**2
_COPY_CHUNK_BYTES = 1024 * 1024


class WorkspaceStagingError(RuntimeError):
    """Workspace staging or publication violated a safety invariant."""


@dataclass(frozen=True)
class WorkspaceIdentity:
    path: str
    device: int
    inode: int
    owner: int


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory(path: os.PathLike[str] | str) -> int:
    try:
        descriptor = os.open(os.fspath(path), _directory_flags())
    except OSError as exc:
        raise WorkspaceStagingError("workspace identity is unavailable") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise WorkspaceStagingError("workspace identity is not a directory")
    return descriptor


def capture_identity(path: os.PathLike[str] | str) -> WorkspaceIdentity:
    canonical = os.path.realpath(os.fspath(path))
    if canonical != os.fspath(path):
        raise WorkspaceStagingError("workspace identity path is not canonical")
    descriptor = _open_directory(path)
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if metadata.st_uid != os.geteuid():
        raise WorkspaceStagingError("workspace identity has a foreign owner")
    return WorkspaceIdentity(canonical, metadata.st_dev, metadata.st_ino, metadata.st_uid)


def _check_identity(path: os.PathLike[str] | str, identity: WorkspaceIdentity) -> int:
    if os.fspath(path) != identity.path:
        raise WorkspaceStagingError("workspace identity path changed")
    descriptor = _open_directory(path)
    metadata = os.fstat(descriptor)
    current = (metadata.st_dev, metadata.st_ino, metadata.st_uid)
    expected = (identity.device, identity.inode, identity.owner)
    if current != expected:
        os.close(descriptor)
        raise WorkspaceStagingError("workspace identity changed")
    return descriptor


def _safe_names(descriptor: int) -> list[str]:
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise WorkspaceStagingError("workspace directory cannot be enumerated") from exc
    if any(not name or name in {".", ".."} or "/" in name or "\x00" in name for name in names):
        raise WorkspaceStagingError("workspace contains an invalid entry name")
    return sorted(names)


def _open_child_directory(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise WorkspaceStagingError(f"workspace directory changed while reading: {name}") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise WorkspaceStagingError(f"workspace entry is not a directory: {name}")
    return descriptor


def _open_regular(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise WorkspaceStagingError(f"workspace entry is a symlink or unavailable: {name}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise WorkspaceStagingError(f"workspace entry is not a regular file: {name}")
    if metadata.st_nlink != 1:
        os.close(descriptor)
        raise WorkspaceStagingError(f"workspace entry is a hardlink: {name}")
    if metadata.st_size > _DEFAULT_MAX_FILE_BYTES:
        os.close(descriptor)
        raise WorkspaceStagingError(f"workspace file exceeds size limit: {name}")
    return descriptor, metadata


def _walk_files(
    descriptor: int,
    *,
    prefix: PurePosixPath = PurePosixPath(),
    exclude_root_names: frozenset[str],
) -> Iterable[tuple[str, str, int, int | None]]:
    for name in _safe_names(descriptor):
        if not prefix.parts and name in exclude_root_names:
            continue
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise WorkspaceStagingError(f"workspace entry changed while reading: {name}") from exc
        relative = str(prefix / name)
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkspaceStagingError(f"workspace entry is a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_child_directory(descriptor, name)
            try:
                yield ("directory", relative, metadata.st_mode & 0o777, None)
                yield from _walk_files(
                    child,
                    prefix=prefix / name,
                    exclude_root_names=exclude_root_names,
                )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceStagingError(f"workspace entry is not a regular file: {relative}")
        file_fd, opened = _open_regular(descriptor, name)
        try:
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise WorkspaceStagingError(f"workspace file identity changed: {relative}")
            yield ("file", relative, opened.st_mode & 0o777, file_fd)
        finally:
            os.close(file_fd)


def _manifest_fd(descriptor: int, *, exclude_root_names: frozenset[str]) -> dict[str, tuple]:
    manifest: dict[str, tuple] = {}
    count = 0
    total = 0
    for kind, relative, mode, file_fd in _walk_files(
        descriptor, exclude_root_names=exclude_root_names
    ):
        count += 1
        if count > _DEFAULT_MAX_FILES:
            raise WorkspaceStagingError("workspace file count limit exceeded")
        if kind == "directory":
            manifest[relative] = (kind, mode)
            continue
        assert file_fd is not None
        digest = hashlib.sha256()
        size = 0
        os.lseek(file_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(file_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            total += len(chunk)
            if total > _DEFAULT_MAX_TOTAL_BYTES:
                raise WorkspaceStagingError("workspace total size limit exceeded")
            digest.update(chunk)
        manifest[relative] = (kind, mode, size, digest.hexdigest())
    return manifest


def manifest_tree(
    path: os.PathLike[str] | str, *, exclude_root_names: set[str] | frozenset[str] = frozenset()
) -> dict[str, tuple]:
    descriptor = _open_directory(path)
    try:
        return _manifest_fd(descriptor, exclude_root_names=frozenset(exclude_root_names))
    finally:
        os.close(descriptor)


def _archive_fd(descriptor: int, *, exclude_root_names: frozenset[str]) -> BinaryIO:
    output = tempfile.SpooledTemporaryFile(max_size=8 * 1024**2, mode="w+b")
    count = 0
    total = 0
    try:
        with tarfile.open(fileobj=output, mode="w") as archive:
            for kind, relative, mode, file_fd in _walk_files(
                descriptor, exclude_root_names=exclude_root_names
            ):
                count += 1
                if count > _DEFAULT_MAX_FILES:
                    raise WorkspaceStagingError("workspace file count limit exceeded")
                info = tarfile.TarInfo(relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = mode & 0o777
                if kind == "directory":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                    continue
                assert file_fd is not None
                size = os.fstat(file_fd).st_size
                total += size
                if total > _DEFAULT_MAX_TOTAL_BYTES:
                    raise WorkspaceStagingError("workspace total size limit exceeded")
                info.size = size
                os.lseek(file_fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(file_fd), "rb") as source:
                    archive.addfile(info, source)
        output.seek(0)
        return output
    except BaseException:
        output.close()
        raise


def archive_tree(
    path: os.PathLike[str] | str, *, exclude_root_names: set[str] | frozenset[str] = frozenset()
) -> BinaryIO:
    descriptor = _open_directory(path)
    try:
        return _archive_fd(descriptor, exclude_root_names=frozenset(exclude_root_names))
    finally:
        os.close(descriptor)


def _normalized_member_name(name: str) -> tuple[str, ...]:
    while name.startswith("./"):
        name = name[2:]
    candidate = PurePosixPath(name)
    if not name or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise WorkspaceStagingError(f"archive member escapes workspace: {name!r}")
    return candidate.parts


def _ensure_directory(root_fd: int, parts: tuple[str, ...], mode: int = 0o700) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode & 0o777, dir_fd=current)
            except FileExistsError:
                pass
            child = _open_child_directory(current, part)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _extract_to_fd(
    archive_file: BinaryIO,
    root_fd: int,
    *,
    max_files: int,
    max_total_bytes: int,
) -> None:
    archive_file.seek(0)
    seen: set[tuple[str, ...]] = set()
    count = 0
    total = 0
    try:
        archive = tarfile.open(fileobj=archive_file, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise WorkspaceStagingError("workspace archive is invalid") from exc
    with archive:
        for member in archive:
            parts = _normalized_member_name(member.name)
            if parts in seen:
                raise WorkspaceStagingError(f"duplicate archive member: {member.name}")
            seen.add(parts)
            count += 1
            if count > max_files:
                raise WorkspaceStagingError("workspace archive file count limit exceeded")
            if not (member.isdir() or member.isreg()):
                raise WorkspaceStagingError(f"workspace archive member is not regular: {member.name}")
            if member.isdir():
                directory = _ensure_directory(root_fd, parts, member.mode)
                os.close(directory)
                continue
            if member.size < 0 or member.size > _DEFAULT_MAX_FILE_BYTES:
                raise WorkspaceStagingError(f"workspace archive member exceeds size limit: {member.name}")
            total += member.size
            if total > max_total_bytes:
                raise WorkspaceStagingError("workspace archive total size limit exceeded")
            parent = _ensure_directory(root_fd, parts[:-1])
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                try:
                    target_fd = os.open(parts[-1], flags, member.mode & 0o777, dir_fd=parent)
                except OSError as exc:
                    raise WorkspaceStagingError(f"workspace archive target is unsafe: {member.name}") from exc
                source = archive.extractfile(member)
                if source is None:
                    os.close(target_fd)
                    raise WorkspaceStagingError(f"workspace archive member has no data: {member.name}")
                written = 0
                try:
                    with os.fdopen(target_fd, "wb") as target:
                        while True:
                            chunk = source.read(_COPY_CHUNK_BYTES)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > member.size:
                                raise WorkspaceStagingError(
                                    f"workspace archive member grew while reading: {member.name}"
                                )
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                finally:
                    source.close()
                if written != member.size:
                    raise WorkspaceStagingError(
                        f"workspace archive member size changed: {member.name}"
                    )
            finally:
                os.close(parent)


def extract_archive(
    archive: BinaryIO,
    destination: os.PathLike[str] | str,
    *,
    max_files: int = _DEFAULT_MAX_FILES,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
) -> None:
    destination_path = Path(destination)
    created = not destination_path.exists()
    destination_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = _open_directory(destination_path)
    try:
        if _safe_names(descriptor):
            raise WorkspaceStagingError("workspace archive destination must be empty")
        _extract_to_fd(
            archive,
            descriptor,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )
    except BaseException:
        if created:
            shutil.rmtree(destination_path, ignore_errors=True)
        raise
    finally:
        os.close(descriptor)


def _remove_entry(parent_fd: int, name: str) -> None:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode):
        raise WorkspaceStagingError(f"workspace changed to a symlink during publication: {name}")
    if stat.S_ISDIR(metadata.st_mode):
        child = _open_child_directory(parent_fd, name)
        try:
            for nested in _safe_names(child):
                _remove_entry(child, nested)
        finally:
            os.close(child)
        os.rmdir(name, dir_fd=parent_fd)
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WorkspaceStagingError(f"workspace changed to an unsafe file during publication: {name}")
    os.unlink(name, dir_fd=parent_fd)


def _clear_fd(descriptor: int, *, exclude_root_names: frozenset[str]) -> None:
    for name in _safe_names(descriptor):
        if name in exclude_root_names:
            continue
        _remove_entry(descriptor, name)


def publish_tree(
    staged: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
    *,
    identity: WorkspaceIdentity,
    expected_manifest: dict[str, tuple],
    exclude_root_names: set[str] | frozenset[str] = frozenset(),
    finalize: Callable[[], None] | None = None,
) -> dict[str, tuple]:
    exclusions = frozenset(exclude_root_names)
    workspace_fd = _check_identity(workspace, identity)
    staged_fd = _open_directory(staged)
    backup: BinaryIO | None = None
    staged_archive: BinaryIO | None = None
    try:
        current = _manifest_fd(workspace_fd, exclude_root_names=exclusions)
        if current != expected_manifest:
            raise WorkspaceStagingError("workspace changed concurrently before publication")
        backup = _archive_fd(workspace_fd, exclude_root_names=exclusions)
        staged_archive = _archive_fd(staged_fd, exclude_root_names=frozenset())
        try:
            _clear_fd(workspace_fd, exclude_root_names=exclusions)
            _extract_to_fd(
                staged_archive,
                workspace_fd,
                max_files=_DEFAULT_MAX_FILES,
                max_total_bytes=_DEFAULT_MAX_TOTAL_BYTES,
            )
            if capture_identity(workspace) != identity:
                raise WorkspaceStagingError(
                    "workspace identity changed before publication finalized"
                )
            if finalize is not None:
                finalize()
            if capture_identity(workspace) != identity:
                raise WorkspaceStagingError(
                    "workspace identity changed before publication completed"
                )
        except BaseException:
            _clear_fd(workspace_fd, exclude_root_names=exclusions)
            _extract_to_fd(
                backup,
                workspace_fd,
                max_files=_DEFAULT_MAX_FILES,
                max_total_bytes=_DEFAULT_MAX_TOTAL_BYTES,
            )
            raise
        return _manifest_fd(workspace_fd, exclude_root_names=exclusions)
    finally:
        if backup is not None:
            backup.close()
        if staged_archive is not None:
            staged_archive.close()
        os.close(staged_fd)
        os.close(workspace_fd)
