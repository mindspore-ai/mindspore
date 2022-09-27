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
from mindspore.ops.operations import nn_ops
from mindspore.ops import functional as F


class Net(nn.Cell):
    def __init__(self, padding, mode, paddings_contiguous, value=0):
        super(Net, self).__init__()
        self.ops = nn_ops.PadV3(mode, paddings_contiguous)
        self.padding = padding
        self.mode = mode
        if mode == "constant":
            self.value = value

    def construct(self, x):
        if self.mode == "constant":
            out = self.ops(x, self.padding, self.value)
        else:
            out = self.ops(x, self.padding)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_padv3_constant():
    """
    Feature: test padv3 constant mode
    Description: test padv3 constant mode
    Expectation: Success
    """
    mode = 'constant'
    context.set_context(device_target="GPU")
    x = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    padding = Tensor((-1, 1, 0, 0))
    value = 1.5
    net = Net(padding, mode, True, value)
    out = net(Tensor(x))
    res_ms = out.asnumpy()
    expect = [[[[1., 2., 1.5], [4., 5., 1.5], [7., 8., 1.5]]]]
    np.testing.assert_almost_equal(expect, res_ms)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_padv3_reflect():
    """
    Feature: test padv3 reflect mode
    Description: test padv3 reflect mode
    Expectation: Success
    """
    mode = 'reflect'
    context.set_context(device_target="GPU")
    x = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    padding = Tensor((-1, 1, 1, -1))
    net = Net(padding, mode, True)
    out = net(Tensor(x))
    res_ms = out.asnumpy()
    expect = [[[[4., 5., 4.], [1., 2., 1.], [4., 5., 4.]]]]
    np.testing.assert_almost_equal(expect, res_ms)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_padv3_edge():
    """
    Feature: test padv3 edge mode
    Description: test padv3 edge mode
    Expectation: Success
    """
    mode = 'edge'
    context.set_context(device_target="GPU")
    x = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    padding = Tensor((-1, 1, 1, -1))
    net = Net(padding, mode, True)
    out = net(Tensor(x))
    res_ms = out.asnumpy()
    expect = [[[[1., 2., 2.], [1., 2., 2.], [4., 5., 5.]]]]
    np.testing.assert_almost_equal(expect, res_ms)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_padv3_vmap():
    """
    Feature: test padv3 vmap feature.
    Description: test padv3 vmap feature.
    Expectation: Success.
    """
    mode = 'reflect'
    context.set_context(device_target="GPU")
    x = Tensor(np.arange(18).reshape(2, 1, 1, 3, 3).astype(np.float32))
    padding = Tensor((-1, 1, 1, -1))
    net = Net(padding, mode, True)
    output = F.vmap(net, 0, 0)(x)
    expect_out = [[[[[4., 5., 4.], [1., 2., 1.], [4., 5., 4.]]]],
                  [[[[13., 14., 13.], [10., 11., 10.], [13., 14., 13.]]]]]
    np.testing.assert_almost_equal(output.asnumpy(), expect_out)
