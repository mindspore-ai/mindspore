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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops.operations.nn_ops import SparseApplyProximalGradientDescent
from mindspore.common import dtype as mstype


class SparseApplyProximalGradientDescentNet(nn.Cell):
    def __init__(self):
        super(SparseApplyProximalGradientDescentNet, self).__init__()
        self.sparse_apply_proximal_gradient_descent = SparseApplyProximalGradientDescent()

    def construct(self, var, alpha, l1, l2, grad, indices):
        out = self.sparse_apply_proximal_gradient_descent(var, alpha, l1, l2, grad, indices)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_proximal_gradient_descent_float32():
    """
    Feature: test float32
    Description: when the dtype of var and grad is float32
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    var = Parameter(Tensor(np.array([[4.1, 7.2], [1.1, 3.0]]).astype(np.float32)))
    alpha = Tensor(1.0, mstype.float32)
    l1 = Tensor(1.0, mstype.float32)
    l2 = Tensor(0.0, mstype.float32)
    grad = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    sparse_apply_proximal_gradient_descent = SparseApplyProximalGradientDescentNet()
    output = sparse_apply_proximal_gradient_descent(var, alpha, l1, l2, grad, indices)
    expect = np.array([[2.1, 5.2], [0., 1.]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_apply_proximal_gradient_descent_float64():
    """
    Feature: test float64
    Description: when the dtype of var and grad is float64
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    var = Parameter(Tensor(np.array([[4.1, 7.2], [1.1, 3.0]]).astype(np.float64)))
    alpha = Tensor(1.0, mstype.float64)
    l1 = Tensor(1.0, mstype.float64)
    l2 = Tensor(0.0, mstype.float64)
    grad = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float64))
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    sparse_apply_proximal_gradient_descent = SparseApplyProximalGradientDescentNet()
    output = sparse_apply_proximal_gradient_descent(var, alpha, l1, l2, grad, indices)
    expect = np.array([[2.1, 5.2], [0., 1.]], dtype=np.float64)
    np.testing.assert_almost_equal(output.asnumpy(), expect)
    