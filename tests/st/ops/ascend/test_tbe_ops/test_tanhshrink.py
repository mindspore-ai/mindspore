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
    def __init__(self):
        super(Net, self).__init__()
        self.tanhshrink = nn.Tanhshrink()

    def construct(self, x):
        return self.tanhshrink(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanhshrink_normal():
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with normal input
    Expectation: success
    """
    net = Net()
    a = Tensor(np.array([1, 2, 3, 2, 1]).astype(np.float16))
    output = net(a)
    expected_output = Tensor(np.array([0.2383, 1.036, 2.004, 1.036, 0.2383]).astype(np.float16))
    assert np.array_equal(output, expected_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanhshrink_negative():
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with negative input
    Expectation: success
    """
    net = Net()
    a = Tensor(np.array([-1, -2, -3, -2, -1]).astype(np.float16))
    output = net(a)
    expected_output = Tensor(np.array([-0.2383, -1.036, -2.004, -1.036, -0.2383]).astype(np.float16))
    assert np.array_equal(output, expected_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanhshrink_zeros():
    """
    Feature: Tanhshrink
    Description: Verify the result of Tanhshrink with zeros
    Expectation: success
    """
    net = Net()
    a = Tensor(np.array([0, 0, 0, 0, 0]).astype(np.float16))
    output = net(a)
    expected_output = Tensor(np.array([0, 0, 0, 0, 0]).astype(np.float16))
    assert np.array_equal(output, expected_output)
