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

import os
import tempfile
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from mindspore.ops.operations._custom_ops_utils import FileLocker, ExtensionBuilder, _MAX_BUILD_ATTEMPTS


@contextmanager
def _temp_build_dir():
    """Create a temporary build directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_file_locker_acquire_release_writes_pid():
    """
    Feature: Custom op build.lock
    Description: Successful lock stores the holder PID and release removes the file
    Expectation: lock file contains PID and is removed after release
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
        assert not os.path.exists(lock_path)


def test_file_locker_reclaims_stale_lock_of_dead_pid():
    """
    Feature: Custom op build.lock
    Description: Lock held by a non-existent PID is treated as stale and reclaimed
    Expectation: try_lock succeeds after removing the stale lock
    """
    with _temp_build_dir() as build_path:
        lock_path = os.path.join(build_path, "build.lock")
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            # PID 1<<30 is extremely unlikely to be alive on Linux.
            lock_file.write(f"{1 << 30}\n")
        # Make the lock look older than the PID-write grace period.
        old = time.time() - 60
        os.utime(lock_path, (old, old))

        locker = FileLocker(build_path)
        assert locker.try_lock() is True
        locker.release_lock()
        assert not os.path.exists(lock_path)


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
    Description: Concurrent try_lock fails while another process holds a live lock
    Expectation: second try_lock returns False
    """
    with _temp_build_dir() as build_path:
        holder = FileLocker(build_path)
        assert holder.try_lock() is True
        contender = FileLocker(build_path)
        assert contender.try_lock() is False
        holder.release_lock()


def test_file_locker_empty_lock_within_grace_returns_false():
    """
    Feature: Custom op build.lock
    Description: Empty lock file younger than PID_WRITE_GRACE is not reclaimed
    Expectation: try_lock returns False and leaves the existing lock file in place
    """
    with _temp_build_dir() as build_path:
        lock_path = os.path.join(build_path, "build.lock")
        with open(lock_path, "w", encoding="utf-8"):
            pass
        # Lock file is fresh; age is well below the grace period.
        locker = FileLocker(build_path)
        assert locker.try_lock() is False
        assert os.path.isfile(lock_path)


def test_file_locker_empty_lock_past_grace_is_reclaimed():
    """
    Feature: Custom op build.lock
    Description: Empty lock file older than PID_WRITE_GRACE is treated as stale
    Expectation: try_lock removes the stale lock and returns True
    """
    with _temp_build_dir() as build_path:
        lock_path = os.path.join(build_path, "build.lock")
        with open(lock_path, "w", encoding="utf-8"):
            pass
        old = time.time() - 60
        os.utime(lock_path, (old, old))

        locker = FileLocker(build_path)
        assert locker.try_lock() is True
        locker.release_lock()
        assert not os.path.exists(lock_path)


def test_extension_builder_compile_retries_after_wait():
    """
    Feature: Custom op extension build
    Description: When lock is held and .so is missing, wait is invoked and then retried
    Expectation: Build succeeds after the initial lock contention is resolved
    """
    with _temp_build_dir() as build_path:
        name = "ms_file_locker_retry_ext"
        so_file = os.path.join(build_path, f"{name}.so")

        def _fake_build(_self, _module_name, _sources, _cflags, _ldflags, _include_paths):
            with open(so_file, "w", encoding="utf-8") as f:
                f.write("dummy")

        with patch.object(FileLocker, "try_lock", side_effect=[False, True]), \
             patch.object(FileLocker, "wait") as mock_wait, \
             patch.object(ExtensionBuilder, "_write_ninja_file_and_build_library", _fake_build):
            builder = ExtensionBuilder(build_path)
            builder._compile(name, [], [], [], [])  # pylint: disable=protected-access # internal retry

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
                builder._compile(name, [], [], [], [])  # pylint: disable=protected-access # retry limit

        assert mock_wait.call_count == _MAX_BUILD_ATTEMPTS
