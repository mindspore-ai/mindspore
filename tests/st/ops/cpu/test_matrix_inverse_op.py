# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either matrix_inverseress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from numpy.linalg import inv
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

np.random.seed(1)

class NetMatrixInverse(nn.Cell):
    def __init__(self):
        super(NetMatrixInverse, self).__init__()
        self.matrix_inverse = P.MatrixInverse()

    def construct(self, x):
        return self.matrix_inverse(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_matrix_inverse(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for MatrixInverse
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(-2, 2, (3, 4, 4)).astype(dtype)
    x0 = Tensor(x0_np)
    expect0 = inv(x0_np)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    matrix_inverse = NetMatrixInverse()
    output0 = matrix_inverse(x0).asnumpy()
    np.testing.assert_almost_equal(expect0, output0, decimal=5)
    assert output0.shape == expect0.shape

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    matrix_inverse = NetMatrixInverse()
    output0 = matrix_inverse(x0).asnumpy()
    np.testing.assert_almost_equal(expect0, output0, decimal=5)
    assert output0.shape == expect0.shape
