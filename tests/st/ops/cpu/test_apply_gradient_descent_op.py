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
from mindspore.ops import functional as F
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


class DynamicShapeNet(nn.Cell):
    def __init__(self, var):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.var = Parameter(var, name="var")
        self.apply_gradient_descent = P.ApplyGradientDescent()

    def construct(self, alpha, delta, indices):
        unique_indices, _ = self.unique(indices)
        delta = self.gather(delta, unique_indices, 0)
        return self.apply_gradient_descent(self.var, alpha, delta)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_gradient_descent_dynamic_shape():
    """
    Feature: test ApplyGradientDescent dynamic_shape feature.
    Description: test ApplyGradientDescent dynamic_shape feature.
    Expectation: success.
    """
    # dynamic inputs
    indices_np = np.random.randint(0, 3, size=6)
    indices_ms = Tensor(indices_np)

    # data preparation
    var = Tensor(np.arange(20).reshape(4, 5).astype(np.float32) / 10)
    unique_indices, _ = P.Unique()(indices_ms)
    var = P.Gather()(var, unique_indices, 0)
    alpha = Tensor(np.array([0.0001]).astype(np.float32))
    delta = Tensor(np.arange(24, 44).reshape(4, 5).astype(np.float32))

    # dynamic shape
    delta_dyn = Tensor(shape=[None for _ in delta.shape], dtype=delta.dtype)
    dynamic_shape_net = DynamicShapeNet(var)
    dynamic_shape_net.set_inputs(alpha, delta_dyn, indices_ms)

    # run in graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    outputs = dynamic_shape_net(alpha, delta, indices_ms)
    expect_shape = var.asnumpy().shape
    assert outputs.asnumpy().shape == expect_shape


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


class VmapNet(nn.Cell):
    def __init__(self):
        super(VmapNet, self).__init__()
        self.apply_gradient_descent = P.ApplyGradientDescent()

    def construct(self, var, alpha, delta):
        return self.apply_gradient_descent(var, alpha, delta)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_apply_gradient_descent():
    """
    Feature: ApplyGradientDescent cpu op vmap.
    Description: test vmap feature for ApplyGradientDescent cpu op.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    var = Parameter(np.arange(30).reshape(3, 2, 5).astype(np.float32) / 10, name="var")
    alpha = Tensor(np.array([0.0001, 0.1, 3]).astype(np.float32))
    delta = Tensor(np.arange(34, 64).reshape(3, 2, 5).astype(np.float32))
    net = VmapNet()
    expect = np.array([[[[-0.0034, 0.0965, 0.1964, 0.29630002, 0.3962],
                         [0.4961, 0.596, 0.69589996, 0.79580003, 0.8957]],
                        [[-3.4, -3.4, -3.3999999, -3.4000003, -3.4],
                         [-3.4, -3.4, -3.3999999, -3.4000003, -3.4]],
                        [[-160, -162.9, -165.8, -168.7, -171.6],
                         [-174.5, -177.4, -180.3, -183.2, -186.1]]]]).astype(np.float32)
    out_vmap = F.vmap(net, in_axes=(0, 0, 0))(var, alpha, delta)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
