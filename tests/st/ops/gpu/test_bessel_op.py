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
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations.math_ops import BesselJ0, BesselJ1, BesselK0, BesselK0e, BesselK1, BesselK1e, BesselI0, \
    BesselI0e, BesselI1, BesselI1e, BesselY0, BesselY1
from mindspore.nn import Cell

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class BesselI0Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_i0 = BesselI0()

    def construct(self, x):
        return self.bessel_i0(x)


class BesselI0GradNet(Cell):
    def __init__(self, network):
        super(BesselI0GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselI0eNet(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_i0e = BesselI0e()

    def construct(self, x):
        return self.bessel_i0e(x)


class BesselI0eGradNet(Cell):
    def __init__(self, network):
        super(BesselI0eGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselI1Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_i1 = BesselI1()

    def construct(self, x):
        return self.bessel_i1(x)


class BesselI1GradNet(Cell):
    def __init__(self, network):
        super(BesselI1GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselI1eNet(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_i1e = BesselI1e()

    def construct(self, x):
        return self.bessel_i1e(x)


class BesselI1eGradNet(Cell):
    def __init__(self, network):
        super(BesselI1eGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselJ0Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_j0 = BesselJ0()

    def construct(self, x):
        return self.bessel_j0(x)


class BesselJ0GradNet(Cell):
    def __init__(self, network):
        super(BesselJ0GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselJ1Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_j1 = BesselJ1()

    def construct(self, x):
        return self.bessel_j1(x)


class BesselJ1GradNet(Cell):
    def __init__(self, network):
        super(BesselJ1GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselK0Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_k0 = BesselK0()

    def construct(self, x):
        return self.bessel_k0(x)


class BesselK0GradNet(Cell):
    def __init__(self, network):
        super(BesselK0GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselK0eNet(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_k0e = BesselK0e()

    def construct(self, x):
        return self.bessel_k0e(x)


class BesselK0eGradNet(Cell):
    def __init__(self, network):
        super(BesselK0eGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselK1Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_k1 = BesselK1()

    def construct(self, x):
        return self.bessel_k1(x)


class BesselK1GradNet(Cell):
    def __init__(self, network):
        super(BesselK1GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselK1eNet(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_k1e = BesselK1e()

    def construct(self, x):
        return self.bessel_k1e(x)


class BesselK1eGradNet(Cell):
    def __init__(self, network):
        super(BesselK1eGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselY0Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_y0 = BesselY0()

    def construct(self, x):
        return self.bessel_y0(x)


class BesselY0GradNet(Cell):
    def __init__(self, network):
        super(BesselY0GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


class BesselY1Net(Cell):
    def __init__(self):
        super().__init__()
        self.bessel_y1 = BesselY1()

    def construct(self, x):
        return self.bessel_y1(x)


class BesselY1GradNet(Cell):
    def __init__(self, network):
        super(BesselY1GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, grad_np):
        gout = self.grad(self.network)(input_data, grad_np)
        return gout


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_y0_fp32():
    """
    Feature: BesselY0
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselY0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselY0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.08825696, 0.5103757, 0.3768501, -0.01694074])
    expect_grad_output = np.array([0.7812128, 0.21406485, -0.9740231, -1.5917029])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_y0_fp16():
    """
    Feature: BesselY0
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselY0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselY0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([-0.4446, 0.08826, 0.377, 0.3074]).astype(np.float16)
    expect_grad_output = np.array([0.736, 0.7812, -0.974, -1.186]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_y1_fp32():
    """
    Feature: BesselY1
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselY1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselY1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([-0.7812128, -0.10703243, 0.32467437, 0.397925735])
    expect_grad_output = np.array([0.86946976, 1.1277838, 0.80587596, -0.4656887])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_y1_fp16():
    """
    Feature: BesselY1
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselY1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselY1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([-1.472, -0.7812, 0.3247, 0.3706]).astype(np.float16)
    expect_grad_output = np.array([1.249, 0.8696, 0.8066, 0.613]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_j0_fp32():
    """
    Feature: BesselJ0
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselJ0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselJ0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.7651977, 0.22389078, -0.26005197, -0.3971498])
    expect_grad_output = np.array([-4.40050572e-01, -1.15344965e+00, -1.01717687e+00, 2.64173299e-01])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_j0_fp16():
    """
    Feature: BesselJ0
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselJ0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselJ0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.9385, 0.765, -0.26, -0.32]).astype(np.float16)
    expect_grad_output = np.array([-1.2115e-01, -4.3994e-01, -1.0176e+00, -8.3740e-01]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_j1_fp32():
    """
    Feature: BesselJ1
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselJ1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselJ1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.44005057, 0.5767248, 0.33905897, -0.06604332])
    expect_grad_output = np.array([3.25147122e-01, -1.28943264e-01, -1.11921477e+00, -1.52255583e+00])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_j1_fp16():
    """
    Feature: BesselJ1
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselJ1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselJ1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.2423, 0.44, 0.339, 0.2617]).astype(np.float16)
    expect_grad_output = np.array([2.2693e-01, 3.2520e-01, -1.1191e+00, -1.2852e+00]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k0_fp32():
    """
    Feature: BesselK0
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselK0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselK0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.42102438, 0.1138939, 0.03473951, 0.01115968])
    expect_grad_output = np.array([-6.01907313e-01, -2.79731750e-01, -1.20469294e-01, -4.99339961e-02])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k0_fp16():
    """
    Feature: BesselK0
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselK0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselK0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.9243, 0.4211, 0.03473, 0.02762]).astype(np.float16)
    expect_grad_output = np.array([-0.828, -0.602, -0.1205, -0.1013]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k0e_fp32():
    """
    Feature: BesselK0e
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselK0eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselK0eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.144463, 0.8415684, 0.69776165, 0.6092977])
    expect_grad_output = np.array([-0.49169075, -0.38381684, -0.32640553, -0.28911304])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k0e_fp16():
    """
    Feature: BesselK0e
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselK0eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselK0eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.524, 1.145, 0.6978, 0.6772]).astype(np.float16)
    expect_grad_output = np.array([-0.603, -0.4912, -0.3267, -0.3171]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k1_fp32():
    """
    Feature: BesselK1e
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselK1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselK1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.6019073, 0.13986588, 0.04015643, 0.0124835])
    expect_grad_output = np.array([-1.0229317, -0.36765367, -0.14437495, -0.05712221])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k1_fp16():
    """
    Feature: BesselK1
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselK1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselK1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.656, 0.602, 0.04016, 0.03168]).astype(np.float16)
    expect_grad_output = np.array([-2.12, -1.023, -0.1444, -0.1201]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k1e_fp32():
    """
    Feature: BesselK1e
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselK1eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselK1eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.6361537, 1.0334768, 0.8065635, 0.68157595])
    expect_grad_output = np.array([-1.144463, -0.64966, -0.48015815, -0.39246297])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_k1e_fp16():
    """
    Feature: BesselK1e
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselK1eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselK1eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([2.73, 1.636, 0.8066, 0.7764]).astype(np.float16)
    expect_grad_output = np.array([-2.127, -1.145, -0.479, -0.4592]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i0_fp32():
    """
    Feature: BesselI0
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselI0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselI0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.26606584e+00, 2.27958536e+00, 4.88079262e+00, 1.13019218e+01])
    expect_grad_output = np.array([5.65159082e-01, 3.18127370e+00, 1.18601112e+01, 3.90378609e+01])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i0_fp16():
    """
    Feature: BesselI0
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselI0Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselI0GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([1.0635e+00, 1.2656e+00, 4.8789e+00, 5.7422e+00]).astype(np.float16)
    expect_grad_output = np.array([1.2891e-01, 5.6494e-01, 1.1859e+01, 1.5133e+01]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i0e_fp32():
    """
    Feature: BesselI0e
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselI0eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselI0eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([4.65759605e-01, 3.08508337e-01, 2.43000358e-01, 2.07001925e-01])
    expect_grad_output = np.array([-2.57849187e-01, -1.86478108e-01, -1.38520941e-01, -1.13004327e-01])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i0e_fp16():
    """
    Feature: BesselI0e
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselI0eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselI0eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([6.4502e-01, 4.6582e-01, 2.4304e-01, 2.3425e-01]).astype(np.float16)
    expect_grad_output = np.array([-2.4438e-01, -2.5781e-01, -1.3879e-01, -1.3196e-01]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i1_fp32():
    """
    Feature: BesselI1
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselI1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselI1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([5.65159082e-01, 1.59063685e+00, 3.95337033e+00, 9.75946522e+00])
    expect_grad_output = np.array([7.00906754e-01, 2.96853399e+00, 1.06890078e+01, 3.54482231e+01])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i1_fp16():
    """
    Feature: BesselI1
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselI1Net()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselI1GradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([2.5781e-01, 5.6494e-01, 3.9531e+00, 4.7305e+00]).astype(np.float16)
    expect_grad_output = np.array([2.7393e-01, 7.0068e-01, 1.0688e+01, 1.3648e+01]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i1e_fp32():
    """
    Feature: BesselI1e
    Description: Test float32 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([1, 2, 3, 4]).astype(np.float32)
    net = BesselI1eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([1, 2, 3, 4]).astype(np.float32)
    grad_net = BesselI1eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.20791042, 0.21526928, 0.19682671, 0.17875084])
    expect_grad_output = np.array([0.04993877, -0.02879119, -0.05830577, -0.06574655])
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bessel_i1e_fp16():
    """
    Feature: BesselI1e
    Description: Test float16 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    net = BesselI1eNet()
    output_ms = net(Tensor(x_np))
    grad_np = np.array([0.5, 1, 3, 3.2]).astype(np.float16)
    grad_net = BesselI1eGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = np.array([0.1564, 0.2079, 0.1968, 0.193]).astype(np.float16)
    expect_grad_output = np.array([0.0879, 0.05005, -0.0575, -0.0613]).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), expect_output)
    assert np.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)
