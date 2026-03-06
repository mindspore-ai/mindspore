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
"""Tests for device synchronization in condition switch."""

import os

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, jit, ops

from tests.mark_utils import arg_mark


@jit(backend="ms_backend")
def condition_branch_func(x, y):
    """Build a branch graph containing CopyToHost and CopyToDevice."""
    copied_x = ops.auto_generate.CopyToHost()(x)
    if x > y:
        copied_device = ops.auto_generate.CopyToDevice()(copied_x)
        output = copied_device + 1
    else:
        copied_device = ops.auto_generate.CopyToDevice()(copied_x)
        output = copied_device + y + 1
    return output, copied_device


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=1.00,
)
def test_condition_branch_true():
    """
    Feature: Condition Switch Control Flow
    Description: Verify condition branch computation returns expected tensors.
    Expectation: Successfully execute the true branch and return expected tensors
    """
    x1 = Tensor(np.array([3.0]), ms.float32)
    x2 = Tensor(np.array([6.0]), ms.float32)

    output, copied = condition_branch_func(x1, x2)
    assert np.allclose(output.asnumpy(), np.array([10.0]))
    assert np.allclose(copied.asnumpy(), np.array([3.0]))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=1.00,
)
def test_device_sync_guard_feature():
    """
    Feature: Device Synchronization Guard
    Description: Verify CopyToDevice succeeds with device sync guard feature enabled.
    Expectation: CopyToDevice operation succeeds when device sync guard feature is present, fails otherwise
    """
    old_hyper_offload = os.environ.get("MS_DEV_HYPER_OFFLOAD")
    try:
        os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

        x = Tensor(np.array([3.0]), ms.float32)
        y = Tensor(np.array([6.0]), ms.float32)

        _, copied = condition_branch_func(x, y)
        result = ops.auto_generate.CopyToDevice()(copied)

        assert "Ascend" in result.device
    except RuntimeError as error:
        pytest.fail(f"Device sync guard feature missing. CopyToDevice operation failed: {str(error)}")
    finally:
        if old_hyper_offload is None:
            os.environ.pop("MS_DEV_HYPER_OFFLOAD", None)
        else:
            os.environ["MS_DEV_HYPER_OFFLOAD"] = old_hyper_offload
