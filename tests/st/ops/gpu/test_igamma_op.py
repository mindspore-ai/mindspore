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
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations.math_ops import Igamma, Igammac
from mindspore.ops.operations._grad_ops import IgammaGradA
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class IgammaNet(Cell):

    def __init__(self):
        super().__init__()
        self.igamma = Igamma()

    def construct(self, a, x):
        return self.igamma(a, x)


class IgammaGradNet(Cell):

    def __init__(self, network):
        super(IgammaGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, a, x, grad_np):
        gout = self.grad(self.network)(a, x, grad_np)
        return gout


class IgammacNet(Cell):

    def __init__(self):
        super().__init__()
        self.igammac = Igammac()

    def construct(self, a, x):
        return self.igammac(a, x)


class IgammacGradNet(Cell):

    def __init__(self, network):
        super(IgammacGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, a, x, grad_np):
        gout = self.grad(self.network)(a, x, grad_np)
        return gout


class IgammaGradANet(Cell):

    def __init__(self):
        super().__init__()
        self.igammagrada = IgammaGradA()

    def construct(self, a, x):
        return self.igammagrada(a, x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igamma_fp32():
    """
    Feature: Igamma
    Description: test input data of float32
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    x_np = np.array([[10, 22], [20, 50]]).astype(np.float32)
    net = IgammaNet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    grad_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
    grad_net = IgammaGradNet(net)
    output_grad_ms = grad_net(Tensor(a_np), Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([[0.54207104, 0.5283601], [0.52974427, 0.5188052]])
    expect_grad_a = np.array([[-0.12721506, -0.1707564], [-0.2687388, -0.22605102]])
    expect_grad_x = np.array([[0.12511013, 0.16946727], [0.266507, 0.22529985]])
    assert np.allclose(output_ms.asnumpy(), expect_output, 1e-4, 1e-4)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_a, 1e-4, 1e-4)
    assert np.allclose(output_grad_ms[1].asnumpy(), expect_grad_x, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igamma_fp64():
    """
    Feature: Igamma
    Description: test input data of float64
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a_np = np.array([[1, 2], [2, 5]]).astype(np.float64)
    x_np = np.array([[1, 2], [2, 5]]).astype(np.float64)
    net = IgammaNet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    expect_output = np.array([[0.63212056, 0.59399415], [0.59399415, 0.55950671]])
    assert np.allclose(output_ms.asnumpy(), expect_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igammac_fp32():
    """
    Feature: Igammac
    Description: test input data of float32
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a_np = np.array([[1, 2], [2, 5]]).astype(np.float32)
    x_np = np.array([[1, 2], [2, 5]]).astype(np.float32)
    net = IgammacNet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    grad_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
    grad_net = IgammacGradNet(net)
    output_grad_ms = grad_net(Tensor(a_np), Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([[0.36787945, 0.40600586], [0.40600586, 0.44049338]])
    expect_grad_a = np.array([[0.4317296, 0.58800924], [0.88201386, 0.7256764]])
    expect_grad_x = np.array([[-0.36787945, -0.5413411], [-0.8120117, -0.70186955]])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_a, 1e-4, 1e-4)
    assert np.allclose(output_grad_ms[1].asnumpy(), expect_grad_x, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igammac_fp64():
    """
    Feature: Igammac
    Description: test input data of float64
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a_np = np.array([[1, 2], [3, 4]]).astype(np.float64)
    x_np = np.array([[1, 3], [2, 5]]).astype(np.float64)
    net = IgammacNet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    expect_output = np.array([[0.36787944, 0.19914827], [0.67667642, 0.26502592]])
    assert np.allclose(output_ms.asnumpy(), expect_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igammagrada_fp32():
    """
    Feature: Igammagrada
    Description: test input data of float32
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    a_np = np.array([[1, 2], [2, 5]]).astype(np.float32)
    x_np = np.array([[1, 2], [2, 5]]).astype(np.float32)
    net = IgammaGradANet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    expect_output = np.array([[-0.4317296, -0.29400462], [-0.29400462, -0.1814191]])
    assert np.allclose(output_ms.asnumpy(), expect_output, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_igammagrada_fp64():
    """
    Feature: Igammagrada
    Description: test input data of float64
    Expectation: success or throw assertion error.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    a_np = np.array([[0, 0], [0, 0]]).astype(np.float64)
    x_np = np.array([[0, 0], [0, 0]]).astype(np.float64)
    net = IgammaGradANet()
    output_ms = net(Tensor(a_np), Tensor(x_np))
    expect_output = np.array([[0, 0], [0, 0]])
    assert np.allclose(output_ms.asnumpy(), expect_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igamma_functional_api_modes(mode):
    """
    Feature: Test igamma functional api.
    Description: Test igamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = F.igamma(a, x)
    expected = np.array([0.593994, 0.35276785, 0.21486944, 0.13337152])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igamma_tensor_api_modes(mode):
    """
    Feature: Test igamma tensor api.
    Description: Test igamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = a.igamma(x)
    expected = np.array([0.593994, 0.35276785, 0.21486944, 0.13337152])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igammac_functional_api_modes(mode):
    """
    Feature: Test igamma functional api.
    Description: Test igamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = F.igammac(a, x)
    expected = np.array([0.40600586, 0.6472318, 0.7851304, 0.8666283])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igammac_tensor_api_modes(mode):
    """
    Feature: Test igamma tensor api.
    Description: Test igamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = a.igammac(x)
    expected = np.array([0.40600586, 0.6472318, 0.7851304, 0.8666283])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)
