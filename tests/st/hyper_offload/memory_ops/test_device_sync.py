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
"""Test device sync in condition switch"""
import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import jit, ops, Tensor
from tests.mark_utils import arg_mark

@jit(backend="ms_backend")
def condition_branch_func(x, y):

    x1 = ops.auto_generate.CopyToHost()(x)

    if x > y:
        x2 = ops.auto_generate.CopyToDevice()(x1)
        z = x2 + 1
    else:
        x2 = ops.auto_generate.CopyToDevice()(x1)
        z = x2 + y + 1

    return z, x2


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_condition_branch_true():
    """
    Feature: Condition Switch Control Flow
    Description: Test the true branch, including tensor operations and value calculations
    Expectation: Successfully execute the true branch and return expected tensors
    """
    x1 = Tensor(np.array([3.0]), ms.float32)
    x2 = Tensor(np.array([6.0]), ms.float32)

    output = condition_branch_func(x1, x2)
    expected = (Tensor(np.array([10.0]), ms.float32), Tensor(np.array([3.0]), ms.float32))

    assert output == expected


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_device_sync_guard_feature():
    """
    Feature: Device Synchronization Guard
    Description: Test device sync guard feature in condition branches
    Expectation: CopyToDevice operation succeeds when device sync guard feature is present, fails otherwise
    """

    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"

    try:
        x = Tensor(np.array([3.0]), ms.float32)
        y = Tensor(np.array([6.0]), ms.float32)

        _, output = condition_branch_func(x, y)

        result = ops.auto_generate.CopyToDevice()(output)

        assert "Ascend" in result.device, "Tensor not on Ascend device after CopyToDevice"
        print(f"CopyToDevice operation succeeded, tensor on device: {result.device}")

    except RuntimeError as e:
        pytest.fail(f"Device sync guard feature missing. CopyToDevice operation failed: {str(e)}")

    finally:
        os.environ["MS_DEV_HYPER_OFFLOAD"] = "0"
