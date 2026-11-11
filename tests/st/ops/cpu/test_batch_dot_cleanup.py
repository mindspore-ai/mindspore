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
"""Protect shared contraction operators when removing the batch_dot API."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite
from mindspore.ops.function import math_func
from tests.mark_utils import arg_mark


class ContractionNet(nn.Cell):
    """Select an existing contraction operator before graph compilation."""

    def __init__(self, name):
        super().__init__()
        self.operation = getattr(ops, name)
        self.is_tensor_dot = name == 'tensor_dot'

    def construct(self, left, right):
        if self.is_tensor_dot:
            return self.operation(left, right, 2)
        return self.operation(left, right)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('execution_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('name', ['bmm', 'dot', 'tensor_dot'])
def test_retained_contractions_after_batch_dot_cleanup(execution_mode, name):
    """
    Feature: Retained tensor contractions.
    Description: Evaluate shared contraction paths with nonuniform inputs in graph and pynative modes.
    Expectation: Shapes, float32 dtype and numeric outputs match the corresponding NumPy operations.
    """
    assert composite.dot is math_func.dot
    assert composite.tensor_dot is math_func.tensor_dot
    original_mode = ms.get_context('mode')
    try:
        ms.set_context(mode=execution_mode)
        if name == 'bmm':
            left = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
            right = np.arange(40, dtype=np.float32).reshape(2, 4, 5) / 10
            expected = np.matmul(left, right)
        elif name == 'dot':
            left = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
            right = np.arange(40, dtype=np.float32).reshape(2, 4, 5) / 10
            expected = np.dot(left, right)
        else:
            left = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
            right = np.arange(60, dtype=np.float32).reshape(3, 4, 5) / 10
            expected = np.tensordot(left, right, axes=2)
        output = ContractionNet(name)(Tensor(left), Tensor(right)).asnumpy()
        assert output.shape == expected.shape
        assert output.dtype == np.float32
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    finally:
        ms.set_context(mode=original_mode)
