# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for CustomOpBuilder FileLocker and ExtensionBuilder robustness."""

import errno
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from mindspore.ops.operations._custom_ops_utils import FileLocker, ExtensionBuilder, _MAX_BUILD_ATTEMPTS


@contextmanager
def _temp_build_dir():
    """Create a temporary build directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _hold_lock_and_crash(build_path):
    """Acquire the build lock and die without releasing it (simulates a crash)."""
    locker = FileLocker(build_path)
    assert locker.try_lock() is True
    os.kill(os.getpid(), signal.SIGKILL)


def test_file_locker_acquire_release_writes_pid():
    """
    Feature: Custom op build.lock
    Description: Successful lock stores the holder PID and release frees the lock
    Expectation: lock file contains PID while held and can be re-acquired after release
    """
    with _temp_build_dir() as build_path:
        locker = FileLocker(build_path)
        assert locker.try_lock() is True
        lock_path = os.path.join(build_path, "build.lock")
        assert os.path.isfile(lock_path)
        with open(lock_path, "r", encoding="utf-8") as lock_file:
            pid = int(lock_file.read().strip())
        assert pid == os.getpid()
        locker.release_lock()
        # The lock file itself is kept on disk; the lock must be free.
        contender = FileLocker(build_path)
        assert contender.try_lock() is True
        contender.release_lock()


def test_file_locker_recovers_after_holder_crash():
    """
    Feature: Custom op build.lock
    Description: Lock held by a process that dies without releasing is freed by the kernel
    Expectation: try_lock succeeds immediately after the holder process is killed
    """
    with _temp_build_dir() as build_path:
        proc = multiprocessing.Process(target=_hold_lock_and_crash, args=(build_path,))
        proc.start()
        proc.join(timeout=30)
        assert proc.exitcode == -signal.SIGKILL

        locker = FileLocker(build_path)
        assert locker.try_lock() is True
        locker.release_lock()


def test_file_locker_wait_timeout():
    """
    Feature: Custom op build.lock
    Description: wait() raises when a live holder never releases within timeout
    Expectation: RuntimeError mentioning timeout
    """
    with _temp_build_dir() as build_path:
        holder = FileLocker(build_path)
        assert holder.try_lock() is True
        waiter = FileLocker(build_path)
        with pytest.raises(RuntimeError, match="Timeout waiting for custom op build lock"):
            waiter.wait(timeout=1.0)
        holder.release_lock()


def test_file_locker_second_acquire_fails_while_held():
    """
    Feature: Custom op build.lock
    Description: Concurrent try_lock fails while another locker holds the lock
    Expectation: second try_lock returns False
    """
    with _temp_build_dir() as build_path:
        holder = FileLocker(build_path)
        assert holder.try_lock() is True
        contender = FileLocker(build_path)
        assert contender.try_lock() is False
        holder.release_lock()


def test_file_locker_ignores_leftover_lock_file():
    """
    Feature: Custom op build.lock
    Description: A leftover lock file with any content and age does not block acquisition
    Expectation: try_lock succeeds and overwrites the recorded PID
    """
    with _temp_build_dir() as build_path:
        lock_path = os.path.join(build_path, "build.lock")
        with open(lock_path, "w", encoding="utf-8"):
            pass
        old = time.time() - 3600
        os.utime(lock_path, (old, old))

        locker = FileLocker(build_path)
        assert locker.try_lock() is True
        with open(lock_path, "r", encoding="utf-8") as lock_file:
            assert int(lock_file.read().strip()) == os.getpid()
        locker.release_lock()


def test_file_locker_pid_write_failure_still_holds_lock():
    """
    Feature: Custom op build.lock
    Description: Failure to record the diagnostic PID does not fail lock acquisition
    Expectation: try_lock returns True, the fd is kept, and mutual exclusion still works
    """
    with _temp_build_dir() as build_path:
        locker = FileLocker(build_path)
        with patch("os.ftruncate", side_effect=OSError("No space left on device")):
            assert locker.try_lock() is True
        assert locker.lock_fd is not None
        contender = FileLocker(build_path)
        assert contender.try_lock() is False
        locker.release_lock()
        assert locker.lock_fd is None


def test_file_locker_windows_branch_locking():
    """
    Feature: Custom op build.lock on Windows
    Description: The msvcrt byte-range lock branch reports contention and unlocks explicitly
    Expectation: Contention returns False, release unlocks the byte range, lock can be re-acquired
    """
    state = {"held": False, "unlock_calls": 0}

    def _fake_locking(fd, mode, nbytes):
        """Simulate msvcrt.locking with a single-byte region lock."""
        del fd, nbytes
        if mode == 1:  # LK_NBLCK
            if state["held"]:
                raise OSError(errno.EDEADLK, "Resource deadlock avoided")
            state["held"] = True
        else:  # LK_UNLCK
            state["unlock_calls"] += 1
            state["held"] = False

    fake_msvcrt = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=0, locking=_fake_locking)
    locker_module = sys.modules[FileLocker.__module__]
    with patch("platform.system", return_value="Windows"), \
         patch.object(locker_module, "msvcrt", fake_msvcrt, create=True):
        with _temp_build_dir() as build_path:
            locker = FileLocker(build_path)
            assert locker.try_lock() is True
            contender = FileLocker(build_path)
            assert contender.try_lock() is False
            locker.release_lock()
            assert state["unlock_calls"] == 1
            assert contender.try_lock() is True
            contender.release_lock()


def test_so_ready_requires_elf_magic():
    """
    Feature: Extension artifact verification
    Description: _so_ready rejects missing, empty, and non-ELF files
    Expectation: True only for a non-empty file starting with the ELF magic
    """
    # pylint: disable=protected-access  # internal artifact check
    with _temp_build_dir() as build_path:
        so_file = os.path.join(build_path, "libx.so")
        assert ExtensionBuilder._so_ready(so_file) is False
        with open(so_file, "wb") as file:
            pass
        assert ExtensionBuilder._so_ready(so_file) is False
        with open(so_file, "wb") as file:
            file.write(b"not an elf")
        assert ExtensionBuilder._so_ready(so_file) is False
        with open(so_file, "wb") as file:
            file.write(b"\x7fELF" + b"\x00" * 16)
        assert ExtensionBuilder._so_ready(so_file) is True


def test_write_ninja_file_targets_tmp_for_atomic_publish():
    """
    Feature: Atomic publish of built extension
    Description: The ninja link target is the .so.tmp file, renamed only after a successful build
    Expectation: build.ninja declares the .so.tmp target
    """
    # pylint: disable=protected-access  # internal file generation
    with _temp_build_dir() as build_path:
        ninja_file = os.path.join(build_path, "build.ninja")
        ExtensionBuilder._write_ninja_file(ninja_file, "mod", ["x.cpp"], [], [], [])
        with open(ninja_file, "r", encoding="utf-8") as file:
            content = file.read()
        assert "build mod.so.tmp: link" in content
        assert "default mod.so.tmp" in content


def test_extension_builder_compile_retries_after_wait():
    """
    Feature: Custom op extension build
    Description: When lock is held and .so is missing, wait is invoked and then retried
    Expectation: Build succeeds after the initial lock contention is resolved
    """
    with _temp_build_dir() as build_path:
        name = "ms_file_locker_retry_ext"
        so_file = os.path.join(build_path, f"{name}.so")

        def _fake_build(self, module_name, sources, cflags, ldflags, include_paths):
            """Write a dummy artifact in place of the real ninja build."""
            del self, module_name, sources, cflags, ldflags, include_paths
            with open(so_file, "w", encoding="utf-8") as f:
                f.write("dummy")

        with patch.object(FileLocker, "try_lock", side_effect=[False, True]), \
             patch.object(FileLocker, "wait") as mock_wait, \
             patch.object(ExtensionBuilder, "_write_ninja_file_and_build_library", _fake_build):
            builder = ExtensionBuilder(build_path)
            builder._compile(name, [], [], [], [])  # pylint: disable=protected-access  # internal retry

        assert mock_wait.called
        assert os.path.isfile(so_file)
        assert os.path.getsize(so_file) > 0


def test_extension_builder_compile_raises_after_max_attempts():
    """
    Feature: Custom op extension build
    Description: If the lock is never released and no usable .so appears, all attempts fail
    Expectation: RuntimeError is raised after _MAX_BUILD_ATTEMPTS attempts
    """
    with _temp_build_dir() as build_path:
        name = "ms_file_locker_max_attempts_ext"

        with patch.object(FileLocker, "try_lock", return_value=False), \
             patch.object(FileLocker, "wait") as mock_wait, \
             patch.object(ExtensionBuilder, "_write_ninja_file_and_build_library"):
            builder = ExtensionBuilder(build_path)
            with pytest.raises(RuntimeError, match="Failed to build or locate extension module"):
                builder._compile(name, [], [], [], [])  # pylint: disable=protected-access  # retry limit

        assert mock_wait.call_count == _MAX_BUILD_ATTEMPTS
