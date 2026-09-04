from __future__ import annotations

import io
import os
import tarfile

import pytest

from tools.workspace_staging import (
    WorkspaceStagingError,
    archive_tree,
    capture_identity,
    extract_archive,
    manifest_tree,
    publish_tree,
)


def _rewind(archive):
    archive.seek(0)
    return archive


def test_archive_round_trip_excludes_git_and_publication_reconciles_create_edit_delete(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").write_text("host metadata\n", encoding="utf-8")
    (workspace / "edit.txt").write_text("before\n", encoding="utf-8")
    (workspace / "delete.txt").write_text("delete me\n", encoding="utf-8")
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "keep.txt").write_text("keep\n", encoding="utf-8")

    identity = capture_identity(workspace)
    baseline = manifest_tree(workspace, exclude_root_names={".git"})
    staged = tmp_path / "staged"
    staged.mkdir()
    extract_archive(_rewind(archive_tree(workspace, exclude_root_names={".git"})), staged)

    assert not (staged / ".git").exists()
    (staged / "edit.txt").write_text("after\n", encoding="utf-8")
    (staged / "delete.txt").unlink()
    (staged / "created.txt").write_text("created\n", encoding="utf-8")

    published = publish_tree(
        staged,
        workspace,
        identity=identity,
        expected_manifest=baseline,
        exclude_root_names={".git"},
    )

    assert (workspace / ".git").read_text(encoding="utf-8") == "host metadata\n"
    assert (workspace / "edit.txt").read_text(encoding="utf-8") == "after\n"
    assert not (workspace / "delete.txt").exists()
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "created\n"
    assert published == manifest_tree(workspace, exclude_root_names={".git"})


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "fifo"])
def test_archive_rejects_non_regular_or_multiply_linked_workspace_entries(tmp_path, unsafe_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    regular = workspace / "regular.txt"
    regular.write_text("data\n", encoding="utf-8")
    unsafe = workspace / "unsafe"
    if unsafe_kind == "symlink":
        unsafe.symlink_to(regular)
    elif unsafe_kind == "hardlink":
        os.link(regular, unsafe)
    else:
        os.mkfifo(unsafe)

    with pytest.raises(WorkspaceStagingError, match="symlink|hardlink|regular"):
        archive_tree(workspace)


def test_extract_rejects_traversal_absolute_links_and_devices(tmp_path):
    for member in (
        tarfile.TarInfo("../escape"),
        tarfile.TarInfo("/absolute"),
        tarfile.TarInfo("linked"),
        tarfile.TarInfo("device"),
    ):
        if member.name == "linked":
            member.type = tarfile.SYMTYPE
            member.linkname = "target"
        elif member.name == "device":
            member.type = tarfile.CHRTYPE
        else:
            member.size = 1
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            archive.addfile(member, io.BytesIO(b"x") if member.isreg() else None)
        payload.seek(0)

        destination = tmp_path / member.name.replace("/", "_").replace("..", "dotdot")
        destination.mkdir()
        with pytest.raises(WorkspaceStagingError):
            extract_archive(payload, destination)

    assert not (tmp_path / "escape").exists()


def test_publish_refuses_workspace_inode_swap(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "owned.txt").write_text("owned\n", encoding="utf-8")
    identity = capture_identity(workspace)
    baseline = manifest_tree(workspace)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "owned.txt").write_text("changed\n", encoding="utf-8")

    displaced = tmp_path / "displaced"
    workspace.rename(displaced)
    workspace.mkdir()
    (workspace / "foreign.txt").write_text("foreign\n", encoding="utf-8")

    with pytest.raises(WorkspaceStagingError, match="identity"):
        publish_tree(staged, workspace, identity=identity, expected_manifest=baseline)

    assert (workspace / "foreign.txt").read_text(encoding="utf-8") == "foreign\n"
    assert (displaced / "owned.txt").read_text(encoding="utf-8") == "owned\n"


def test_publish_rolls_back_files_when_finalize_fails(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "before.txt").write_text("before\n", encoding="utf-8")
    identity = capture_identity(workspace)
    baseline = manifest_tree(workspace)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "after.txt").write_text("after\n", encoding="utf-8")

    def fail_finalize():
        raise RuntimeError("index.lock conflict")

    with pytest.raises(RuntimeError, match="index.lock conflict"):
        publish_tree(
            staged,
            workspace,
            identity=identity,
            expected_manifest=baseline,
            finalize=fail_finalize,
        )

    assert manifest_tree(workspace) == baseline
    assert (workspace / "before.txt").read_text(encoding="utf-8") == "before\n"
    assert not (workspace / "after.txt").exists()


def test_publish_rolls_back_pinned_inode_when_path_swaps_during_finalize(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tracked.txt").write_text("before", encoding="utf-8")
    identity = capture_identity(workspace)
    baseline = manifest_tree(workspace)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "tracked.txt").write_text("after", encoding="utf-8")
    displaced = tmp_path / "displaced"

    def swap_workspace_path():
        workspace.rename(displaced)
        workspace.mkdir()
        (workspace / "attacker.txt").write_text("untouched", encoding="utf-8")

    with pytest.raises(WorkspaceStagingError, match="identity changed"):
        publish_tree(
            staged,
            workspace,
            identity=identity,
            expected_manifest=baseline,
            finalize=swap_workspace_path,
        )

    assert (displaced / "tracked.txt").read_text(encoding="utf-8") == "before"
    assert (workspace / "attacker.txt").read_text(encoding="utf-8") == "untouched"


def test_extract_enforces_file_count_and_total_size_limits(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in ("one", "two"):
            info = tarfile.TarInfo(name)
            info.size = 2
            archive.addfile(info, io.BytesIO(b"xx"))
    payload.seek(0)

    with pytest.raises(WorkspaceStagingError, match="count"):
        extract_archive(payload, tmp_path / "count", max_files=1)
    payload.seek(0)
    with pytest.raises(WorkspaceStagingError, match="size"):
        extract_archive(payload, tmp_path / "size", max_total_bytes=3)
