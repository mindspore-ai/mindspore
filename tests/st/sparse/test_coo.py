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
"""smoke tests for COO operations"""

import pytest
import numpy as np

from mindspore import Tensor, COOTensor, ms_function, nn, context
from mindspore.common import dtype as mstype


context.set_context(mode=context.GRAPH_MODE)


def compare_coo(coo1, coo2):
    assert isinstance(coo1, COOTensor)
    assert isinstance(coo2, COOTensor)
    assert (coo1.indices.asnumpy() == coo2.indices.asnumpy()).all()
    assert (coo1.values.asnumpy() == coo2.values.asnumpy()).all()
    assert coo1.shape == coo2.shape


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_make_coo():
    """
    Feature: Test COOTensor Constructor in Graph and PyNative.
    Description: Test COOTensor(indices, values, shape) and COOTensor(COOTensor)
    Expectation: Success.
    """
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    dense_shape = (3, 4)

    def test_pynative():
        return COOTensor(indices, values, dense_shape)
    test_graph = ms_function(test_pynative)

    coo1 = test_pynative()
    coo2 = test_graph()
    compare_coo(coo1, coo2)
    coo3 = COOTensor(coo_tensor=coo2)
    compare_coo(coo3, coo2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coo_tensor_in_while():
    """
    Feature: Test COOTensor in while loop.
    Description: Test COOTensor computation in while loop.
    Expectation: Success.
    """
    class COOTensorWithControlWhile(nn.Cell):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        @ms_function
        def construct(self, a, b, indices, values):
            x = COOTensor(indices, values, self.shape)
            while a > b:
                x = COOTensor(indices, values, self.shape)
                b = b + 1
            return x
    a = Tensor(3, mstype.int32)
    b = Tensor(0, mstype.int32)
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=mstype.float32)
    shape = (3, 4)
    net = COOTensorWithControlWhile(shape)
    out = net(a, b, indices, values)
    assert np.allclose(out.indices.asnumpy(), indices.asnumpy(), .0, .0)
    assert np.allclose(out.values.asnumpy(), values.asnumpy(), .0, .0)
    assert out.shape == shape
