"""Windows-native regression coverage for the jobs.json kernel mutex."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

import cron.jobs as jobs_mod

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel mutex semantics only")


def _make_junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation unavailable: {result.stderr or result.stdout}")


def test_windows_junction_path_never_creates_a_filesystem_lock(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    junction = tmp_path / "cron-link"
    _make_junction(junction, target)
    lock_path = junction / ".jobs.lock"

    mutex = jobs_mod._acquire_windows_jobs_mutex(lock_path, 1.0)
    jobs_mod._release_windows_jobs_mutex(mutex)

    assert not lock_path.exists()
    assert not (target / ".jobs.lock").exists()


def test_windows_named_mutex_contention_times_out_fail_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / ".jobs.lock"
    owner = jobs_mod._acquire_windows_jobs_mutex(lock_path, 1.0)
    result: list[BaseException | str] = []

    def contender() -> None:
        try:
            second = jobs_mod._acquire_windows_jobs_mutex(lock_path, 0.05)
        except BaseException as exc:
            result.append(exc)
        else:
            result.append("unexpected-acquisition")
            jobs_mod._release_windows_jobs_mutex(second)

    thread = threading.Thread(target=contender)
    thread.start()
    thread.join(timeout=5)
    jobs_mod._release_windows_jobs_mutex(owner)

    assert not thread.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], jobs_mod.CronJobsLockError)
    assert "Timed out" in str(result[0])
    assert not lock_path.exists()


def test_windows_named_mutex_is_released_after_context(tmp_path: Path) -> None:
    lock_path = tmp_path / ".jobs.lock"
    first = jobs_mod._acquire_windows_jobs_mutex(lock_path, 1.0)
    jobs_mod._release_windows_jobs_mutex(first)

    second = jobs_mod._acquire_windows_jobs_mutex(lock_path, 1.0)
    jobs_mod._release_windows_jobs_mutex(second)
    assert not lock_path.exists()
