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

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetGer(nn.Cell):
    """Net of ger."""

    def __init__(self):
        """Init."""
        super(NetGer, self).__init__()
        self.ger = P.Ger()

    def construct(self, x, y):
        """Construct."""
        return self.ger(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ger_float16():
    """
    Feature: Ger cpu kernel
    Description: test the rightness of Ger cpu kernel.
    Expectation: Success.
    """
    x_array = np.array([1, 2, 3, 4]).astype('float16')
    y_array = np.array([1, 2, 3]).astype('float16')
    input_x = Tensor(x_array)
    input_y = Tensor(y_array)
    net = NetGer()
    output = net(input_x, input_y)
    print(output)
    expect = x_array.reshape(4, 1) * y_array.reshape(1, 3)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ger_float32():
    """
    Feature: Ger cpu kernel
    Description: test the rightness of Ger cpu kernel.
    Expectation: Success.
    """
    x_array = np.array([1, 2, 3, 4]).astype('float32')
    y_array = np.array([1, 2, 3]).astype('float32')
    input_x = Tensor(x_array)
    input_y = Tensor(y_array)
    net = NetGer()
    output = net(input_x, input_y)
    print(output)
    expect = x_array.reshape(4, 1) * y_array.reshape(1, 3)
    assert np.allclose(output.asnumpy(), expect)
