# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Build utilities for mock op plugin."""

import os
import platform
import stat
import subprocess
import time
from pathlib import Path
import pytest
import mindspore as ms


class FileLocker:
    """File-based lock to prevent concurrent builds in multi-processing.

    Cross-platform file locking using atomic file creation (O_EXCL on Unix,
    O_CREAT | O_EXCL on Windows).
    """

    def __init__(self, lock_file_path):
        """Initialize file locker."""
        self.lock_file_path = lock_file_path
        self.lock_fd = None

    def try_lock(self):
        """Try to acquire a file-based lock. Returns True if successful.

        Uses atomic file creation (O_EXCL) which works on both Unix/Linux and Windows.
        On Unix: O_EXCL provides true atomic locking.
        On Windows: O_EXCL is supported via Python's os module.
        """
        try:
            mode = stat.S_IRUSR | stat.S_IWUSR
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            self.lock_fd = os.open(self.lock_file_path, flags, mode)
            return True
        except (FileExistsError, OSError):
            return False

    def release_lock(self):
        """Release the file-based lock."""
        if self.lock_fd is not None:
            try:
                os.close(self.lock_fd)
            except OSError:
                pass  # File descriptor may already be closed
            self.lock_fd = None
        try:
            if os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
        except OSError:
            pass  # Lock file may have been removed by another process

    def wait(self, timeout=200):
        """Wait until lock is released or timeout."""
        elapsed = 0
        while os.path.exists(self.lock_file_path):
            if elapsed >= timeout:
                raise RuntimeError(f"Timeout waiting for build lock after {timeout} seconds")
            time.sleep(0.5)
            elapsed += 0.5


def _get_library_path(build_dir, system):
    """Locate the built library file."""
    exts = {
        "linux": ".so",
        "darwin": ".dylib",
        "windows": ".dll",
    }
    target_name = "mindspore_op_plugin_mock"
    target_ext = exts.get(system, ".so")

    candidates = []
    for p in build_dir.rglob(f"*{target_name}*{target_ext}"):
        # Prefer non-import libraries on Windows (exclude .lib/.exp)
        if p.suffix.lower() == target_ext:
            candidates.append(p)
    if not candidates:
        return None

    # Heuristic: pick the shortest path (usually the actual artifact, not intermediates)
    lib_path = str(sorted(candidates, key=lambda x: len(str(x)))[0])
    return lib_path


def _build(plugin_src_dir, build_dir, system):
    """Build the mock plugin using CMake."""
    repo_root = ms.__path__[0]
    # include path for custom_kernel_input_info.h
    include_dir = os.path.join(repo_root, "include", "mindspore", "ops", "kernel", "cpu", "op_plugin")

    cmake_args = [
        "cmake",
        "-S",
        plugin_src_dir,
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if system == "windows":
        include_flags = f"/I{include_dir}"
    else:
        include_flags = f"-I{include_dir}"
    cmake_args.append(f"-DCMAKE_CXX_FLAGS={include_flags}")

    # Configure
    result = subprocess.run(cmake_args, check=True)

    # Build
    build_cmd = ["cmake", "--build", build_dir]
    if system == "windows":
        build_cmd += ["--config", "Release"]
    result = subprocess.run(build_cmd, check=True)
    return result


def configure_and_build_mock_plugin() -> str:
    """Configure and build the mock op plugin and return the built library path."""
    system = platform.system().lower()
    if system == "windows" or system == "darwin":  # windows and macos are not supported for now
        return ""

    this_dir = Path(__file__).resolve().parent
    plugin_src_dir = this_dir / "mock_op_plugin"
    build_dir = plugin_src_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Use file-based locking to prevent concurrent builds.
    #
    # Important for parallel ST runners (e.g. tests/st/scripts/run_st.py):
    # - Another process may be building and a partially-produced library file may already exist.
    # - If we return early purely based on file existence, we can race and dlopen a half-written .so.
    lock_file_path = build_dir / "build.lock"
    locker = FileLocker(str(lock_file_path))

    # If another build is in progress, wait first, then re-check artifact.
    if os.path.exists(lock_file_path):
        locker.wait()

    # Fast path: library exists and no one is building.
    lib_path = _get_library_path(build_dir, system)
    if lib_path and os.path.exists(lib_path):
        return lib_path

    # Try to acquire lock
    acquired_lock = locker.try_lock()
    if not acquired_lock:
        # Another process is building, wait for it to complete
        locker.wait()
        # After waiting, check if library exists now
        lib_path = _get_library_path(build_dir, system)
        if lib_path and os.path.exists(lib_path):
            return lib_path
        raise RuntimeError("Build lock released but library not found")

    try:
        # Double-check library doesn't exist (another process might have built it while we waited)
        lib_path = _get_library_path(build_dir, system)
        if lib_path and os.path.exists(lib_path):
            return lib_path

        result = _build(plugin_src_dir, build_dir, system)
        if result.returncode != 0:
            print("Build failed.")
            return None

        # Locate built library
        lib_path = _get_library_path(build_dir, system)
        if not lib_path:
            raise RuntimeError("Failed to locate built mock op plugin library")

        return lib_path
    finally:
        locker.release_lock()


@pytest.fixture(scope="session", autouse=True)
def build_and_setup_mock_plugin():
    """Build mock plugin once per test session and set environment variable.

    This fixture ensures the plugin is built only once per test session,
    even when tests are run with multi-processing. File-based locking
    prevents concurrent builds across different processes.
    """
    lib_path = configure_and_build_mock_plugin()
    if lib_path:
        os.environ["MS_OP_PLUGIN_PATH"] = lib_path
    yield
