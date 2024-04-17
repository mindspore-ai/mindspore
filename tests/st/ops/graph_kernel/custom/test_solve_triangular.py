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
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.scipy.ops import SolveTriangular


class NetSolveTriangular(nn.Cell):
    def __init__(self):
        super(NetSolveTriangular, self).__init__()
        self.sta = SolveTriangular(lower=True, unit_diagonal=False, trans='N')

    def construct(self, a, b):
        return self.sta(a, b)


@pytest.mark.level1
@pytest.mark.env_onecard
def test_solve_triangular():
    """
    Feature: solve_triangular op use custom compile test on graph mode.
    Description: test solve_triangular op on graph mode
    Expectation: the result equal to expect.
    """
    np.random.seed(0)
    num = 32
    x1 = np.random.uniform(0.5, 1.0, (num, num))
    x1 = np.tril(x1).astype(np.float32)
    expect = np.ones((num, 16)).astype(np.float32)
    x2 = np.dot(x1, expect).astype(np.float32)

    x1 = Tensor(x1, mstype.float32)
    x2 = Tensor(x2, mstype.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = NetSolveTriangular()
    result = net(x1, x2)
    result = result.asnumpy()
    rtol = 0.05
    atol = 0.05
    assert np.allclose(result, expect, rtol, atol, equal_nan=True)
