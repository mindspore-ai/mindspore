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
"""Regression tests for Tensor.fill_ value gradients with mixed dtypes."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops
from tests.mark_utils import arg_mark


class FillNet(nn.Cell):
    """Fill a copy so the differentiated input is not modified."""

    def construct(self, input_x: Tensor, value: Tensor) -> Tensor:
        """Apply in-place fill to a fresh tensor."""
        return (input_x + 0).fill_(value)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", mem_peak=4.00)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("input_dtype", [ms.float16, ms.bfloat16, ms.float32])
@pytest.mark.parametrize("value_dtype", [ms.float16, ms.bfloat16, ms.float32])
def test_fill_value_gradient_dtype(mode, input_dtype, value_dtype):
    """
    Feature: Tensor.fill_ backward.
    Description: Fill a tensor with a scalar tensor of a different dtype.
    Expectation: Each input gradient has its input's dtype and the correct value.
    """
    ms.set_context(mode=mode, jit_level="O0")
    input_x = Tensor(np.zeros((2, 3), np.float32), input_dtype)
    value = Tensor(1.25, value_dtype)
    sensitivity = Tensor(np.array([[1, -2, 3], [4, -5, 0.5]], np.float32), input_dtype)
    grad_fn = ops.GradOperation(get_all=True, sens_param=True)(FillNet())
    input_grad, value_grad = grad_fn(input_x, value, sensitivity)

    assert input_grad.dtype == input_dtype
    assert value_grad.dtype == value_dtype
    assert value_grad.shape == value.shape
    np.testing.assert_array_equal(input_grad.float().asnumpy(), np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(value_grad.float().asnumpy(), np.array(1.5, np.float32))
