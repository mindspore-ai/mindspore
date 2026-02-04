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

"""
Custom pass ST test configuration and fixtures.

This module provides session-level fixtures for building custom pass plugins
and managing test resources.
"""

import os
import subprocess
import shutil
import logging
import pytest
from pathlib import Path

logger = logging.getLogger(__name__)


def get_temp_base():
    """Get base directory for temporary files.

    By default uses system /tmp for better resource management.
    Can be overridden with environment variables:
    - CUSTOM_PASS_TMP_DIR: specific directory path
    - USE_PROJECT_TMP=1: use project-local ./tmp directory

    Returns:
        Path: Base directory for temporary files
    """
    # Check for explicit directory override
    custom_dir = os.getenv("CUSTOM_PASS_TMP_DIR")
    if custom_dir:
        return Path(custom_dir)

    # Check for project-local tmp preference
    if os.getenv("USE_PROJECT_TMP", "").lower() in ("1", "true", "yes"):
        return Path(__file__).parent / "tmp"

    # Default: use system tmp with project namespace
    return Path("/tmp") / "mindspore_custom_pass_tests"


@pytest.fixture(scope="session")
def build_plugin():
    """Build custom pass plugin using cmake.

    This fixture creates a temporary build directory in the project path,
    configures cmake with MindSpore paths, and builds the pass plugin.
    All tests in the session share the same build.

    Build cache location: {temp_base}/session_{session_id}/custom_pass_build

    Returns:
        str: Path to the built plugin (.so file)
    """
    import time
    import threading

    # Create temporary build directory (shared across session)
    session_id = f"session_{int(time.time())}_{threading.current_thread().ident}"
    temp_base = get_temp_base()
    build_dir = temp_base / session_id / "custom_pass_build"

    # Ensure directories exist
    build_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Build cache location: %s", build_dir)
    logger.info("Temporary directory base: %s", temp_base)
    if str(temp_base).startswith("/tmp"):
        logger.info("Using system /tmp for better resource management")
    else:
        logger.info("Using project-local directory for debugging")

    # Get source directory (resources)
    source_dir = Path(__file__).parent / "resources"

    if not source_dir.exists():
        pytest.skip(f"Source directory not found: {source_dir}")

    logger.info("Building plugin from source: %s", source_dir)
    logger.info("Build directory: %s", build_dir)

    # Try to find MindSpore installation
    mindspore_root = find_mindspore_root()
    if mindspore_root:
        logger.info("Found MindSpore at: %s", mindspore_root)
    else:
        logger.warning("MindSpore installation not found, using default paths")

    # Configure cmake
    cmake_config_cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release"
    ]

    if mindspore_root:
        cmake_config_cmd.extend([f"-DMINDSPORE_ROOT={mindspore_root}"])

    logger.info("Running cmake configure: %s", ' '.join(cmake_config_cmd))

    try:
        result = subprocess.run(
            cmake_config_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        logger.info("CMake configure completed successfully")
        if result.stdout:
            logger.debug("CMake stdout: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("CMake configure failed: %s", e)
        logger.error("CMake stderr: %s", e.stderr)
        pytest.fail(f"CMake configure failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        pytest.fail("CMake configure timed out after 300 seconds")

    # Build the plugin
    cmake_build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--target", "pass",
        "--parallel"
    ]

    logger.info("Running cmake build: %s", ' '.join(cmake_build_cmd))

    try:
        result = subprocess.run(
            cmake_build_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=True
        )
        logger.info("CMake build completed successfully")
        if result.stdout:
            logger.debug("CMake build stdout: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("CMake build failed: %s", e)
        logger.error("CMake build stderr: %s", e.stderr)
        pytest.fail(f"CMake build failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        pytest.fail("CMake build timed out after 600 seconds")

    # Find the built plugin
    plugin_path = build_dir / "libpass.so"

    if not plugin_path.exists():
        # Try alternative locations
        for candidate in build_dir.rglob("libpass.so"):
            plugin_path = candidate
            break

    if not plugin_path.exists():
        pytest.fail(f"Plugin not found after build. Expected: {plugin_path}")

    logger.info("Plugin built successfully: %s", plugin_path)
    logger.info("Plugin size: %s bytes", plugin_path.stat().st_size)

    # Verify plugin is readable
    if not os.access(plugin_path, os.R_OK):
        pytest.fail(f"Plugin is not readable: {plugin_path}")

    plugin_path_str = str(plugin_path)

    # Ensure the runtime can locate a libstdc++ version compatible with the toolchain
    try:
        libstdcxx_path = subprocess.check_output(
            ["g++", "-print-file-name=libstdc++.so.6"],
            text=True,
            timeout=10
        ).strip()
        if libstdcxx_path:
            libstdcxx_dir = os.path.dirname(os.path.realpath(libstdcxx_path))
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            path_entries = current_ld_path.split(":") if current_ld_path else []
            if libstdcxx_dir not in path_entries:
                new_ld_path = f"{libstdcxx_dir}:{current_ld_path}" if current_ld_path else libstdcxx_dir
                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                os.putenv("LD_LIBRARY_PATH", new_ld_path)
                logger.info(
                    "Prepended libstdc++ runtime directory to LD_LIBRARY_PATH for custom pass plugin: %s",
                    libstdcxx_dir
                )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Unable to augment LD_LIBRARY_PATH with libstdc++ runtime: %s", exc)

    # Track cleanup state to prevent double cleanup
    cleanup_done = False

    # Register cleanup function for session (runs on normal/abnormal exit)
    def cleanup_session_dir():
        nonlocal cleanup_done
        if cleanup_done:
            return

        try:
            session_temp_dir = build_dir.parent  # {temp_base}/session_xxx/
            if session_temp_dir.exists():
                shutil.rmtree(session_temp_dir)
                logger.info("Session cleanup completed: %s", session_temp_dir)
                cleanup_done = True
                # For system /tmp, also try to clean empty parent if we're the only user
                if str(temp_base).startswith("/tmp"):
                    try:
                        if temp_base.exists() and not any(temp_base.iterdir()):
                            temp_base.rmdir()
                            logger.info("Cleaned empty system temp base: %s", temp_base)
                    except (OSError, PermissionError):
                        # Multiple processes might be using it, ignore cleanup failure
                        pass
        except (OSError, PermissionError) as e:
            logger.warning("Session cleanup warning: %s", e)

    # Register cleanup only through atexit (avoid double cleanup)
    import atexit
    import signal

    # Store original signal handlers
    original_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Define safe cleanup that doesn't interfere with pytest
    def safe_cleanup_session():
        try:
            if not cleanup_done:
                cleanup_session_dir()
        except (OSError, PermissionError) as e:
            logger.warning("Session cleanup error: %s", e)

    # Register cleanup strategy based on temp directory location
    is_system_tmp = str(temp_base).startswith("/tmp")

    if is_system_tmp:
        # For system /tmp, register immediate cleanup on session end
        atexit.register(safe_cleanup_session)
        logger.info("Registered immediate cleanup for system /tmp directory")
    else:
        # For project-local tmp, less aggressive cleanup (keep for debugging)
        atexit.register(safe_cleanup_session)
        logger.info("Registered cleanup for project-local directory")

    try:
        yield plugin_path_str
    finally:
        # Restore original signal handlers to not interfere with pytest
        try:
            if original_sigint is not None:
                signal.signal(signal.SIGINT, original_sigint)
            if original_sigterm is not None:
                signal.signal(signal.SIGTERM, original_sigterm)
        except (OSError, ValueError):
            pass  # Signal handling restoration can fail in some environments

        # For system /tmp, also do immediate cleanup in finally block
        if is_system_tmp:
            try:
                safe_cleanup_session()
                logger.info("Immediate session cleanup completed")
            except (OSError, PermissionError) as e:
                logger.warning("Immediate cleanup warning: %s", e)


def find_mindspore_root():
    """Try to find MindSpore installation root directory.

    This function searches for MindSpore in several common locations:
    1. Environment variable MINDSPORE_ROOT
    2. Python site-packages (for pip installed MindSpore)
    3. Common system paths

    Returns:
        str or None: Path to MindSpore root directory
    """
    # Check environment variable
    if "MINDSPORE_ROOT" in os.environ:
        mindspore_root = os.environ["MINDSPORE_ROOT"]
        if os.path.exists(mindspore_root):
            return mindspore_root

    # Try to find through Python import
    try:
        import mindspore
        import inspect
        mindspore_path = os.path.dirname(inspect.getfile(mindspore))

        # Check if include and lib directories exist in mindspore package
        include_path = os.path.join(mindspore_path, "include")
        lib_path = os.path.join(mindspore_path, "lib")

        if os.path.exists(include_path) and os.path.exists(lib_path):
            return mindspore_path

        # Look for include directory in parent directories
        current_path = mindspore_path
        for _ in range(5):  # Search up to 5 levels up
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # Reached root
                break

            include_path = os.path.join(parent_path, "include")
            lib_path = os.path.join(parent_path, "lib")

            if os.path.exists(include_path) and os.path.exists(lib_path):
                return parent_path

            current_path = parent_path
    except ImportError:
        pass

    # Try common installation paths
    common_paths = [
        "/usr/local",
        "/opt/mindspore",
        "/usr",
    ]

    for path in common_paths:
        include_path = os.path.join(path, "include", "mindspore")
        if os.path.exists(include_path):
            return path

    return None


@pytest.fixture(scope="function", name="graphs_dir_fixture")
def graphs_temp_dir():
    """Create temporary directory for graph dumps.

    Environment variables:
    - KEEP_GRAPHS=1: Keep graph files after test completion
    - CUSTOM_PASS_TMP_DIR: Custom temp directory base

    Returns:
        str: Path to graphs directory
    """
    import time
    import threading

    test_name = f"test_{int(time.time())}_{threading.current_thread().ident}"
    temp_base = get_temp_base()
    graphs_dir = temp_base / test_name / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Graphs directory: %s", graphs_dir)

    # Check if graphs should be kept for debugging
    keep_graphs = os.getenv("KEEP_GRAPHS", "").lower() in ("1", "true", "yes")
    if keep_graphs:
        logger.info("KEEP_GRAPHS=1, graph files will be preserved for debugging")

    # Function-scope fixture should use teardown, not atexit
    yield str(graphs_dir)

    # Cleanup graphs directory after test function (unless KEEP_GRAPHS=1)
    if not keep_graphs:
        try:
            if graphs_dir.exists():
                # Clean up the specific test directory
                test_temp_dir = graphs_dir.parent  # {temp_base}/test_xxx/
                shutil.rmtree(test_temp_dir)
                logger.debug("Cleaned up test directory: %s", test_temp_dir)
        except (OSError, PermissionError) as e:
            logger.warning("Cleanup warning for graphs: %s", e)
    else:
        logger.info("Graphs preserved at: %s", graphs_dir)


@pytest.fixture(autouse=True)
def setup_test_environment(graphs_dir_fixture):
    """Setup test environment for each test.

    This fixture runs before each test to ensure clean state.
    Respects KEEP_GRAPHS environment variable for debugging.
    """
    # Check if graphs should be kept for debugging
    keep_graphs = os.getenv("KEEP_GRAPHS", "").lower() in ("1", "true", "yes")

    # Clean graphs directory only if not in debug mode
    if not keep_graphs:
        if os.path.exists(graphs_dir_fixture):
            shutil.rmtree(graphs_dir_fixture)
        os.makedirs(graphs_dir_fixture, exist_ok=True)
    else:
        # In debug mode, just ensure directory exists without cleaning
        os.makedirs(graphs_dir_fixture, exist_ok=True)
        logger.info("KEEP_GRAPHS=1, preserving existing graphs in: %s", graphs_dir_fixture)

    # Set MindSpore context
    try:
        from mindspore import context

        context.set_context(
            mode=context.GRAPH_MODE,
            device_target="CPU",
            save_graphs=True,
            save_graphs_path=graphs_dir_fixture
        )
        logger.info("MindSpore context configured with graphs path: %s", graphs_dir_fixture)
    except ImportError:
        logger.warning("MindSpore not available, skipping context setup")

    yield

    # Cleanup after test (optional)
    # The graphs directory will be cleaned by the next test automatically
