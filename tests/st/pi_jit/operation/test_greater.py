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
""" test greater operator in pijit"""
import pytest
import numpy as np
from mindspore import ops, jit, context
import mindspore as ms
from tests.mark_utils import arg_mark


@jit(capture_mode="bytecode")
def greater_forward_func(x, y):
    return ops.greater(x, y)

@jit(capture_mode="bytecode")
def greater_backward_func(x, y):
    return ops.grad(greater_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_forward():
    """
    Feature: Ops.
    Description: test op greater.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([False, True, False])
    out = greater_forward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op greater.
    Expectation: expect correct result.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([0, 0, 0])
    out = greater_backward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_greater_backward_invalid():
    """
    Feature: Auto grad in pijit.
    Description: test auto grad of op greater.
    Expectation: expect correct result.
    """
    @jit(capture_mode="bytecode")
    def greater_backward_func(x, y):
        return ops.grad(greater_forward_func, (2,))(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    with pytest.raises(RuntimeError) as error_info:
        greater_backward_func(x, y)
    assert "Position index 2 is exceed input size" in str(error_info.value)
