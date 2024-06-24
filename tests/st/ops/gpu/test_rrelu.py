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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, lower, upper):
        super(Net, self).__init__()
        self.rrelu = nn.RReLU(lower, upper)

    def construct(self, x):
        return self.rrelu(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rrelu_normal():
    """
    Feature: RReLU
    Description: Verify the result of RReLU, with l = u = 0.5
    Expectation: success
    """
    net = Net(0.5, 0.5)
    a = Tensor(np.array([[0, -1, -2], [-4, 5, 6]]).astype(np.float32))
    output = net(a)
    expected_output = Tensor(np.array([[0, -0.5, -1], [-2, 5, 6]]).astype(np.float32))
    assert np.array_equal(output, expected_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rrelu_negative_lu():
    """
    Feature: RReLU
    Description: Verify the result of RReLU, with l = u = -1
    Expectation: success
    """
    net = Net(-1, -1)
    a = Tensor(np.array([[0, -1, -2], [-4, 5, 6]]).astype(np.float32))
    output = net(a)
    expected_output = Tensor(np.array([[0, 1, 2], [4, 5, 6]]).astype(np.float32))
    assert np.array_equal(output, expected_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rrelu_zeros():
    """
    Feature: RReLU
    Description: Verify the result of RReLU, with zeros
    Expectation: success
    """
    net = Net(5, 5)
    a = Tensor(np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32))
    output = net(a)
    expected_output = Tensor(np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32))
    assert np.array_equal(output, expected_output)
