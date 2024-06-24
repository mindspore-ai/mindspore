# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, var):
        super(Net, self).__init__()
        self.var = Parameter(var, name="var")
        self.apply_gradient_descent = P.ApplyGradientDescent()

    def construct(self, alpha, delta):
        return self.apply_gradient_descent(self.var, alpha, delta)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_gradient_descent_float32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float32) / 10)
    net = Net(var)
    alpha = Tensor(np.array([0.0001]).astype(np.float32))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float32))
    output = net(alpha, delta)
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                       [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect)
    np.testing.assert_almost_equal(net.var.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float32) / 10)
    net = Net(var)
    alpha = Tensor(np.array([0.0001]).astype(np.float32))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float32))
    output = net(alpha, delta)
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                       [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect)
    np.testing.assert_almost_equal(net.var.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_gradient_descent_float16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float16) / 10)
    net = Net(var)
    alpha = Tensor(np.array([0.0001]).astype(np.float16))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float16))
    output = net(alpha, delta)
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                       [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]], dtype=np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=3)
    np.testing.assert_almost_equal(net.var.asnumpy(), expect, decimal=3)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float16) / 10)
    net = Net(var)
    alpha = Tensor(np.array([0.0001]).astype(np.float16))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float16))
    output = net(alpha, delta)
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.2964, 0.396],
                       [0.496, 0.596, 0.6963, 0.7954, 0.8955]], dtype=np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=3)
    np.testing.assert_almost_equal(net.var.asnumpy(), expect, decimal=3)
