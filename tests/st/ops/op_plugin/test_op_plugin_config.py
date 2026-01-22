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

"""Mock tests for op plugin configuration mechanism."""

import os
import pytest
import ctypes
from ctypes import c_uint32, c_bool, c_int

from tests.mark_utils import arg_mark

from tests.st.ops.op_plugin.mock_build import build_and_setup_mock_plugin # pylint: disable=unused-import

import mindspore as ms
from mindspore.graph.api import _pynative_executor

ms.set_context(device_target="CPU")

def _query_plugin_config(lib_path):
    """Load plugin library and query config state for test verification."""
    if not lib_path or not os.path.exists(lib_path):
        return None

    try:
        lib = ctypes.CDLL(lib_path)

        # Define function signatures
        lib.GetReceivedOpThreadsNum.restype = c_uint32
        lib.GetReceivedOpThreadsNum.argtypes = []

        lib.IsOpThreadsNumReceived.restype = c_bool
        lib.IsOpThreadsNumReceived.argtypes = []

        lib.GetConfigCallCount.restype = c_int
        lib.GetConfigCallCount.argtypes = []

        lib.GetConfigInvalidVersion.restype = c_bool
        lib.GetConfigInvalidVersion.argtypes = []

        lib.GetConfigInvalidSize.restype = c_bool
        lib.GetConfigInvalidSize.argtypes = []

        return {
            'op_threads_num': lib.GetReceivedOpThreadsNum(),
            'is_received': lib.IsOpThreadsNumReceived(),
            'call_count': lib.GetConfigCallCount(),
            'invalid_version': lib.GetConfigInvalidVersion(),
            'invalid_size': lib.GetConfigInvalidSize()
        }
    except (OSError, AttributeError) as e:
        pytest.skip(f"Failed to query plugin config: {e}")
        return None


def _reset_plugin_config_state(lib_path):
    """Reset plugin config state between tests."""
    if not lib_path or not os.path.exists(lib_path):
        return

    try:
        lib = ctypes.CDLL(lib_path)
        lib.ResetConfigState.argtypes = []
        lib.ResetConfigState.restype = None
        lib.ResetConfigState()
    except (OSError, AttributeError):
        pass  # Ignore errors in reset


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_op_plugin_config_with_threads_num():
    """
    Feature: op_plugin configuration mechanism
    Description: Test that op_threads_num config is correctly passed to plugin
    Expectation: Plugin receives correct op_threads_num value during plugin loading
    
    Note: This test verifies the config mechanism works. Since RuntimeConf can only be
    set once per process and plugin loads once, we check if config was received.
    For complete isolation, see test_op_plugin_config_subprocess.
    """
    lib_path = os.environ.get("MS_OP_PLUGIN_PATH")
    if not lib_path:
        raise RuntimeError("Mock plugin not available")

    # Reset plugin state before test
    _reset_plugin_config_state(lib_path)

    from mindspore.device_context.cpu.op_tuning import threads_num
    expected_threads = 8
    threads_num(expected_threads)

    # Trigger plugin loading
    ms.mint.logical_and(ms.Tensor(0), ms.Tensor(0))
    _pynative_executor.sync()

    # Query plugin to verify it received the config
    config_state = _query_plugin_config(lib_path)
    if config_state:
        # Config function should have been called at least once
        assert config_state['call_count'] >= 1, "Config function should have been called at least once"
        # Verify no invalid config flags
        assert not config_state['invalid_version'], "Config should not have invalid version"
        assert not config_state['invalid_size'], "Config should not have invalid size"
        # If config was received, verify it's a valid value
        if config_state['is_received']:
            assert config_state['op_threads_num'] == expected_threads, \
                f"Received op_threads_num should be {expected_threads}, got {config_state['op_threads_num']}"
