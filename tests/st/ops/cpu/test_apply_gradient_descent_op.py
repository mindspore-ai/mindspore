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
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, var):
        super(Net, self).__init__()
        self.var = Parameter(var, name="var")
        self.apply_gradient_descent = P.ApplyGradientDescent()

    def construct(self, alpha, delta):
        return self.apply_gradient_descent(self.var, alpha, delta)


def run_net(var, alpha, delta, expect):
    net = Net(var)
    output = net(alpha, delta)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=3)
    np.testing.assert_almost_equal(net.var.asnumpy(), expect, decimal=3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_gradient_descent_float32():
    """
    Feature: ApplyGradientDescent cpu op.
    Description: test data type is float32 in both graph mode and pynative mode.
    Expectation: success.
    """
    # data preparation
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float32) / 10)
    alpha = Tensor(np.array([0.0001]).astype(np.float32))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float32))
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                       [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]], dtype=np.float32)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_net(var, alpha, delta, expect)

    # run in pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_net(var, alpha, delta, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_gradient_descent_float16():
    """
    Feature: ApplyGradientDescent cpu op.
    Description: test data type is float16 in both graph mode and pynative mode.
    Expectation: success.
    """
    # data preparation
    var = Tensor(np.arange(10).reshape(2, 5).astype(np.float16) / 10)
    alpha = Tensor(np.array([0.0001]).astype(np.float16))
    delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float16))
    expect = np.array([[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                       [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]], dtype=np.float16)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    run_net(var, alpha, delta, expect)

    # run in pynative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_net(var, alpha, delta, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_gradient_descent_wrong_dtype():
    """
    Feature: ApplyGradientDescent cpu op.
    Description: test invalid data type: 1) dtype of var is neither float16 nor float32.
                 2) delta is not a Tensor. 3) alpha is neither a Tensor nor a Number.
    Expectation: Failure and TypeError is caught.
    """

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    with pytest.raises(TypeError):
        var = Tensor(np.arange(10).reshape(2, 5).astype(np.float64) / 10)
        alpha = Tensor(np.array([0.0001]).astype(np.float16))
        delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float16))
        net = Net(var)
        _ = net(alpha, delta)

    with pytest.raises(TypeError):
        var = Tensor(np.arange(10).reshape(2, 5).astype(np.float32) / 10)
        alpha = Tensor(np.array([0.0001]).astype(np.float32))
        delta = np.arange(34, 44).reshape(2, 5).astype(np.float32)
        net = Net(var)
        _ = net(alpha, delta)

    with pytest.raises(TypeError):
        var = Tensor(np.arange(10).reshape(2, 5).astype(np.float32) / 10)
        alpha = np.array([0.0001]).astype(np.float32)
        delta = Tensor(np.arange(34, 44).reshape(2, 5).astype(np.float32))
        net = Net(var)
        _ = net(alpha, delta)
