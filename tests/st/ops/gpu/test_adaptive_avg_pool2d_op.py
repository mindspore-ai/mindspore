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

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.ops import operations as P
from mindspore.common.api import jit
from mindspore.ops.functional import vmap
from mindspore.ops import composite as C

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.adaptive_avg_pool2d = P.AdaptiveAvgPool2D(output_size)

    @jit
    def construct(self, x):
        return self.adaptive_avg_pool2d(x)


class AdaptiveAvgPool2dGrad(nn.Cell):
    def __init__(self, forward):
        super(AdaptiveAvgPool2dGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, sens):
        return self.grad(self.forward)(x, sens)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_normal():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test output_size with tuple.
    Expectation: Expect correct result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = Net((3, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_single():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test output_size with int.
    Expectation: Expect correct result.
    """
    x = np.random.randn(1, 32, 7, 9)
    net = Net(5)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 5, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_none():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test output_size with None.
    Expectation: Expect correct result.
    """
    x = np.random.randn(1, 32, 7, 9)
    net = Net((None, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 7, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_value():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test forward and backward.
    Expectation: Expect correct result.
    """
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    net = Net((2, 2))
    output = net(Tensor(x))
    expect_shape = (3, 2, 2)
    expect_output = np.array([[[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]]])
    assert output.asnumpy().shape == expect_shape
    assert (output.asnumpy() == expect_output).all

    expect_dx = np.array([[[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]]])
    grad_net = AdaptiveAvgPool2dGrad(net)
    dx = grad_net(Tensor(x), output)[0]
    assert dx.asnumpy().shape == x.shape
    assert (dx.asnumpy() == expect_dx).all


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_graph_mode():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test forward and backward in graph mode.
    Expectation: Expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D((2, 2))
    output = adaptive_avg_pool_2d(Tensor(x, mindspore.float16))
    expect_shape = (3, 2, 2)
    expect_output = np.array([[[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]]])
    assert output.asnumpy().shape == expect_shape
    assert (output.asnumpy() == expect_output).all

    expect_dx = np.array([[[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]]])
    grad_net = AdaptiveAvgPool2dGrad(Net((2, 2)))
    dx = grad_net(Tensor(x, mindspore.float16), output)[0]
    assert dx.asnumpy().shape == x.shape
    assert (dx.asnumpy() == expect_dx).all


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_graph_mode_fp64():
    """
    Feature: Test AdaptiveAvgPool2D op.
    Description: Test forward and backward in graph mode.
    Expectation: Expect correct result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D((2, 2))
    output = adaptive_avg_pool_2d(Tensor(x, mindspore.float64))
    expect_shape = (3, 2, 2)
    expect_output = np.array([[[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]],
                              [[3.0, 4.0], [6.0, 7.0]]])
    assert output.asnumpy().shape == expect_shape
    assert (output.asnumpy() == expect_output).all

    expect_dx = np.array([[[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]],
                          [[0.75, 1.75, 1.0],
                           [2.25, 5.0, 2.75],
                           [1.5, 3.25, 1.75]]])
    grad_net = AdaptiveAvgPool2dGrad(Net((2, 2)))
    dx = grad_net(Tensor(x, mindspore.float64), output)[0]
    assert dx.asnumpy().shape == x.shape
    assert (dx.asnumpy() == expect_dx).all


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_adaptive_avgpool_vmap():
    """
    Feature: test vmap function.
    Description: test adaptive avgpool op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    x = Tensor(np.random.randn(1, 2, 9, 9, 1, 2).astype(np.float32))
    net = Net((3, 5))
    nest_vmap = vmap(vmap(net, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert out.shape == (2, 1, 1, 2, 3, 5)
