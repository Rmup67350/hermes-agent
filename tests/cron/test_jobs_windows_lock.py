"""Windows-native regression coverage for the jobs.json cross-process lock."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import cron.jobs as jobs_mod

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows msvcrt semantics only")


def _make_junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr or result.stdout}")


def test_windows_junction_parent_is_rejected_before_lock_file_write(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    junction = tmp_path / "cron-link"
    _make_junction(junction, target)

    with pytest.raises(jobs_mod.CronJobsLockError, match="reparse point"):
        jobs_mod._open_windows_jobs_lock_file(junction / ".jobs.lock")

    assert not (target / ".jobs.lock").exists()


def test_windows_msvcrt_contention_times_out_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import msvcrt

    lock_path = tmp_path / ".jobs.lock"
    lock_path.write_bytes(b"\0")
    holder = open(lock_path, "r+b")
    holder.seek(0)
    locking = getattr(msvcrt, "locking")
    lock_mode = getattr(msvcrt, "LK_NBLCK")
    unlock_mode = getattr(msvcrt, "LK_UNLCK")
    locking(holder.fileno(), lock_mode, 1)
    monkeypatch.setattr(jobs_mod, "fcntl", None)
    monkeypatch.setattr(jobs_mod, "msvcrt", msvcrt)
    monkeypatch.setattr(jobs_mod, "ensure_dirs", lambda: None)
    monkeypatch.setattr(jobs_mod, "_jobs_lock_file", lambda: lock_path)
    monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_POLL_SECONDS", 0.01)
    try:
        with pytest.raises(jobs_mod.CronJobsLockError, match="Timed out"):
            with jobs_mod._jobs_lock():
                pytest.fail("critical section must remain closed")
    finally:
        holder.seek(0)
        locking(holder.fileno(), unlock_mode, 1)
        holder.close()


def test_windows_msvcrt_lock_is_released_after_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import msvcrt

    lock_path = tmp_path / ".jobs.lock"
    monkeypatch.setattr(jobs_mod, "fcntl", None)
    monkeypatch.setattr(jobs_mod, "msvcrt", msvcrt)
    monkeypatch.setattr(jobs_mod, "ensure_dirs", lambda: None)
    monkeypatch.setattr(jobs_mod, "_jobs_lock_file", lambda: lock_path)

    with jobs_mod._jobs_lock():
        pass

    with open(lock_path, "r+b") as probe:
        probe.seek(0)
        locking = getattr(msvcrt, "locking")
        lock_mode = getattr(msvcrt, "LK_NBLCK")
        unlock_mode = getattr(msvcrt, "LK_UNLCK")
        locking(probe.fileno(), lock_mode, 1)
        probe.seek(0)
        locking(probe.fileno(), unlock_mode, 1)
