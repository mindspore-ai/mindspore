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
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit

context.set_context(device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bidense = nn.BiDense(20, 30, 40)

    @jit
    def construct(self, x1, x2):
        return self.bidense(x1, x2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    """
    Feature: Assert BiDense output shape
    Description: test the output.shape == (128, 40).
    Expectation: match the shape.
    """
    x1 = np.random.randn(128, 20).astype(np.float32)
    x2 = np.random.randn(128, 30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (128, 40)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_nd():
    """
    Feature: Assert BiDense output shape for n-dimensional input
    Description: test the output.shape == (128, 4, 40).
    Expectation: match the shape.
    """
    x1 = np.random.randn(128, 4, 20).astype(np.float32)
    x2 = np.random.randn(128, 4, 30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (128, 4, 40)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_1d():
    """
    Feature: Assert BiDense output shape for 1-dimensional input
    Description: test the output.shape == (40,).
    Expectation: match the shape.
    """
    x1 = np.random.randn(20).astype(np.float32)
    x2 = np.random.randn(30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (40,)
