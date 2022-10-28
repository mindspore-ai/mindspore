# Copyright 2022 Huawei Technologies Co., Ltd
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
"""smoke tests for RowTensor operations"""

import pytest
import numpy as np

from mindspore import Tensor, nn, context
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.common import dtype as mstype


def compare_row(row1, row2):
    assert isinstance(row1, RowTensorInner)
    assert isinstance(row2, RowTensorInner)
    assert (row1.indices.asnumpy() == row1.indices.asnumpy()).all()
    assert (row2.values.asnumpy() == row2.values.asnumpy()).all()
    assert row1.dense_shape == row2.dense_shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_make_row():
    """
    Feature: Test RowTensor Constructor in Graph and PyNative.
    Description: Test RowTensorInner(indices, values, shape) and RowTensorInner(RowTensor)
    Expectation: Success.
    """
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([[1, 2], [3, 4]], dtype=mstype.float32)
    dense_shape = (3, 2)

    def test_pynative():
        return RowTensorInner(indices, values, dense_shape)

    row1 = test_pynative()
    compare_row(row1, row1)
    row2 = RowTensorInner(row_tensor=row1)
    compare_row(row1, row2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_row_tensor_with_control_if():
    """
    Feature: Test RowTensor in if.
    Description: Test RowTensor computation in while loop.
    Expectation: Success.
    """
    class RowTensorValuesDouble(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values * 2
            shape = x.dense_shape
            return RowTensorInner(indices, values, shape)

    class RowTensorValuesAdd2(nn.Cell):

        def construct(self, x):
            indices = x.indices
            values = x.values + 2
            shape = x.dense_shape
            return RowTensorInner(indices, values, shape)

    class RowTensorWithControlIf(nn.Cell):
        def __init__(self, shape):
            super(RowTensorWithControlIf, self).__init__()
            self.op1 = RowTensorValuesDouble()
            self.op2 = RowTensorValuesAdd2()
            self.shape = shape

        def construct(self, a, b, indices, values):
            x = RowTensorInner(indices, values, self.shape)
            if a > b:
                x = self.op1(x)
            else:
                x = self.op2(x)
            return x.indices, x.values, x.dense_shape
    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor(0, mstype.int32)
    b = Tensor(2, mstype.int32)
    indices = Tensor([0, 1], dtype=mstype.int32)
    values = Tensor([[1, 2], [3, 4]], dtype=mstype.float32)
    shape = (3, 2)
    net = RowTensorWithControlIf(shape)
    out = net(a, b, indices, values)
    assert np.allclose(out[0].asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose(out[1].asnumpy(), values.asnumpy() + 2, .0, .0)
    assert out[2] == shape
