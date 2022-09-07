# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net2Inputs(nn.Cell):
    def __init__(self):
        super(Net2Inputs, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y):
        return self.addn((x, y))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_two_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for AddN of two tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    addn_net = Net2Inputs()
    dtypes = (np.int32, np.float32, np.float64)
    for dtype in dtypes:
        output = addn_net(Tensor(x.astype(dtype)), Tensor(y.astype(dtype)))
        expect_result = (x + y).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)


class Net4Inputs(nn.Cell):
    def __init__(self):
        super(Net4Inputs, self).__init__()
        self.addn = P.AddN()

    def construct(self, x, y, m, n):
        return self.addn((x, y, m, n))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_four_tensors_add():
    """
    Feature: ALL To ALL
    Description: test cases for AddN of four tensors
    Expectation: the result match to numpy
    """
    x = np.arange(2 * 3).reshape((2, 3))
    y = np.arange(1, 2 * 3 + 1).reshape((2, 3))
    m = np.arange(2, 2 * 3 + 2).reshape((2, 3))
    n = np.arange(3, 2 * 3 + 3).reshape((2, 3))
    addn_net = Net4Inputs()
    dtypes = (np.int32, np.float32, np.float64)
    for dtype in dtypes:
        output = addn_net(Tensor(x.astype(dtype)), Tensor(y.astype(dtype)),
                          Tensor(m.astype(dtype)), Tensor(n.astype(dtype)))
        expect_result = (x + y + m + n).astype(dtype)
        assert output.asnumpy().dtype == expect_result.dtype
        assert np.array_equal(output.asnumpy(), expect_result)
