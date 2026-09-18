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
"""Test that DeviceCtx keeps the thread-local device stack balanced when entering fails."""

from unittest import mock

import pytest

from mindspore.common import device as device_module
from mindspore.common.device import DeviceCtx

# Reading the module-private stack is the point of this test: DeviceCtx keeps its bookkeeping there.
_DEVICE_STACK = device_module._DEVICE_STACK  # pylint: disable=protected-access


def _device_stack_depth():
    return len(_DEVICE_STACK.get_stack())


def _reject_device_id(device_id):
    """Build a DeviceManagerConf.set_device replacement that fails for one device id."""

    def _set_device(_name, target_device_id, _is_default):
        if target_device_id == device_id:
            raise RuntimeError(f"device {device_id} is not available")

    return _set_device


def test_nested_ctx_enter_failure_keeps_device_stack_balanced():
    """
    Feature: DeviceCtx thread-local device stack.
    Description: An inner DeviceCtx fails in __enter__ because the device is unavailable, and the caller catches it.
    Expectation: The failed context leaves no entry behind, so the enclosing context still pops its own entry.
    """
    depth_before = _device_stack_depth()
    fake_device_conf = mock.MagicMock()
    fake_device_conf.get_instance.return_value.set_device.side_effect = _reject_device_id(7)

    with mock.patch.object(device_module, "MSContext", mock.MagicMock()), \
            mock.patch.object(device_module, "DeviceManagerConf", fake_device_conf), \
            mock.patch.object(device_module, "_wait_pipeline", mock.MagicMock()):
        with DeviceCtx("CPU", 0):
            assert _device_stack_depth() == depth_before + 1

            with pytest.raises(RuntimeError, match="device 7 is not available"):
                with DeviceCtx("Ascend", 7):
                    pass

            assert _device_stack_depth() == depth_before + 1

    assert _device_stack_depth() == depth_before


def test_failed_set_device_target_leaves_device_stack_unchanged():
    """
    Feature: DeviceCtx thread-local device stack.
    Description: Switching the device target fails in DeviceCtx.__enter__, and the caller catches the error.
    Expectation: The failed context leaves the thread-local device stack unchanged.
    """
    depth_before = _device_stack_depth()
    fake_context = mock.MagicMock()
    fake_context.get_instance.return_value.set_device_target_inner.side_effect = \
        RuntimeError("the device target is not compiled into this package")

    with mock.patch.object(device_module, "MSContext", fake_context), \
            mock.patch.object(device_module, "DeviceManagerConf", mock.MagicMock()), \
            mock.patch.object(device_module, "_wait_pipeline", mock.MagicMock()):
        with pytest.raises(RuntimeError, match="not compiled into this package"):
            with DeviceCtx("Ascend", 0):
                pass

        assert _device_stack_depth() == depth_before
