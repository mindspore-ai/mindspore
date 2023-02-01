# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.common.api import jit
from mindspore.common import Parameter, ParameterTuple
import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE)


@jit
def ConvBnReLU(x):
    conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
    bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
    relu = nn.ReLU()

    x = conv(x)
    x = bn(x)
    x = relu(x)

    return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_call_single_func():
    inputs = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    out = ConvBnReLU(inputs)
    assert np.allclose(out[0][0][0][0].asnumpy(), 3.9999797, 0.0001, 0.0001)
    assert np.allclose(out[0][1][0][0].asnumpy(), 3.9999797, 0.0001, 0.0001)
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    out_grad = grad(ConvBnReLU)(inputs)
    assert np.allclose(out_grad[0][0][0][0][0][0].asnumpy(), 1.99998, 0.0001, 0.0001)


class CellConvBnReLU(nn.Cell):
    def __init__(self):
        super(CellConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    @jit
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_call_single_cell():
    inputs = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    # run forward
    net = CellConvBnReLU()
    out = net(inputs)
    assert np.allclose(out[0][0][0][0].asnumpy(), 3.9999797, 0.0001, 0.0001)
    assert np.allclose(out[0][1][0][0].asnumpy(), 3.9999797, 0.0001, 0.0001)
    # run grad twice
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_first[0][0][0][0][0][0].asnumpy(), 1.99998, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][0][0][0][0][0].asnumpy(), 0.99999, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][1][0].asnumpy(), 3.99997, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][2][0].asnumpy(), 1.00000, 0.0001, 0.0001)
    optimizer(grad_first[1])
    grad_second = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_second[0][0][0][0][0][0].asnumpy(), 1.07999, 0.0001, 0.0001)
    assert np.allclose(grad_second[1][0][0][0][0][0].asnumpy(), 0.59999, 0.0001, 0.0001)
    assert np.allclose(grad_second[1][1][0].asnumpy(), 3.59998, 0.0001, 0.0001)
    assert np.allclose(grad_second[1][2][0].asnumpy(), 1.00000, 0.0001, 0.0001)


class AddMulMul(nn.Cell):
    def __init__(self):
        super(AddMulMul, self).__init__()
        self.param = Parameter(Tensor(0.5, ms.float32))

    @jit
    def construct(self, x):
        x = x + x
        x = x * self.param
        x = x * x
        return x


class CellCallSingleCell(nn.Cell):
    def __init__(self):
        super(CellCallSingleCell, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()
        self.add_mul_mul = AddMulMul()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.add_mul_mul(x)
        x = self.relu(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cell_call_cell():
    inputs = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    # run forward
    net = CellCallSingleCell()
    out = net(inputs)
    assert np.allclose(out[0][0][0][0].asnumpy(), 15.9998, 0.0001, 0.0001)
    assert np.allclose(out[0][1][0][0].asnumpy(), 15.9998, 0.0001, 0.0001)
    # run grad twice
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_first[0][0][0][0][0][0].asnumpy(), 16.0, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][0][0][0][0][0].asnumpy(), 8.0, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][1][0].asnumpy(), 3.1999e+01, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][2][0].asnumpy(), 7.9999e+00, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][3].asnumpy(), 127.999, 0.0001, 0.0001)
    optimizer(grad_first[1])
    grad_second = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_second[0][0][0][0][0][0].asnumpy(), 2.726e+03, 1, 1)
    assert np.allclose(grad_second[1][0][0][0][0][0].asnumpy(), 6.816e+03, 1, 1)
    assert np.allclose(grad_second[1][1][0].asnumpy(), -2.477e+03, 1, 1)
    assert np.allclose(grad_second[1][2][0].asnumpy(), -3.097e+03, 1, 1)
    assert np.allclose(grad_second[1][3].asnumpy(), -1289, 1, 1)


class Mul(nn.Cell):
    def __init__(self):
        super(Mul, self).__init__()
        self.param = Parameter(Tensor(1.5, ms.float32))

    @jit
    def construct(self, x):
        x = x * self.param
        return x


class CallSameFunc(nn.Cell):
    def __init__(self):
        super(CallSameFunc, self).__init__()
        self.conv_bn_relu = CellConvBnReLU()
        self.mul = Mul()

    def construct(self, x):
        x = self.mul(x)
        x = self.mul(x)
        x = self.mul(x)
        x = self.conv_bn_relu(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_call_same_func():
    inputs = Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
    # run forward
    net = CallSameFunc()
    out = net(inputs)
    assert np.allclose(out[0][0][0][0].asnumpy(), 13.4999, 0.0001, 0.0001)
    assert np.allclose(out[0][1][0][0].asnumpy(), 13.4999, 0.0001, 0.0001)
    # run grad twice
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    grad_first = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_first[0][0][0][0][0][0].asnumpy(), 6.75, 0.01, 0.01)
    assert np.allclose(grad_first[1][0][0][0][0][0].asnumpy(), 3.375, 0.001, 0.001)
    assert np.allclose(grad_first[1][1][0].asnumpy(), 13.4999, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][2][0].asnumpy(), 1.0000, 0.0001, 0.0001)
    assert np.allclose(grad_first[1][3].asnumpy(), 54.0000, 0.0001, 0.0001)
    optimizer(grad_first[1])
    grad_second = grad(net, ParameterTuple(net.trainable_params()))(inputs)
    assert np.allclose(grad_second[0][0][0][0][0][0].asnumpy(), 27.5, 0.1, 0.1)
    assert np.allclose(grad_second[1][0][0][0][0][0].asnumpy(), 20.76, 0.01, 0.01)
    assert np.allclose(grad_second[1][1][0].asnumpy(), -157, 1, 1)
    assert np.allclose(grad_second[1][2][0].asnumpy(), 1.0000, 0.0001, 0.0001)
    assert np.allclose(grad_second[1][3].asnumpy(), -84.6, 0.1, 0.1)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function():
    class MsFunctionCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, ms.float32))

        @jit
        def construct(self, x):
            x = self.param * x
            return x

    class NetA(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, ms.float32))

        def construct(self, x):
            x = self.param * x
            x = x + x
            return x

    class NetB(nn.Cell):
        def __init__(self):
            super().__init__()
            self.ms_function_net = MsFunctionCell()

        def construct(self, x):
            x = self.ms_function_net(x)
            x = x + x
            return x

    net_a = NetA()
    params_a = ParameterTuple(net_a.trainable_params())
    net_b = NetB()
    params_b = ParameterTuple(net_b.trainable_params())
    input_data = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    # The first net run
    grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=False)
    out_a = grad(net_a, params_a)(input_data)
    out_b = grad(net_b, params_b)(input_data)
    assert np.allclose(out_a[0][0].asnumpy(), out_b[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_a[1][0].asnumpy(), out_b[1][0].asnumpy(), 0.0001, 0.0001)
    # The second net run
    out_a = grad(net_a, params_a)(input_data)
    out_b = grad(net_b, params_b)(input_data)
    assert np.allclose(out_a[0][0].asnumpy(), out_b[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out_a[1][0].asnumpy(), out_b[1][0].asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_mix_execute():
    """
    Feature: PyNative ms_function.
    Description: Mixed execution of PyNative and ms_function.
    Expectation: The calculation result is correct.
    """

    class Net(nn.Cell):
        @jit
        def test_ms_function(self, x, y):
            return x * y

        def construct(self, x, y):
            z = x * y
            return self.test_ms_function(z, x)

    net = Net()
    a = Tensor(2)
    b = Tensor(2)
    output = net(a, b)
    assert output == 8


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_empty_graph():
    """
    Feature: PyNative ms_function.
    Description: Empty ms_function graph.
    Expectation: The calculation result is correct.
    """

    class Net(nn.Cell):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y
            self.relu = P.ReLU()

        @jit
        def max(self):
            if self.x > self.y:
                return self.x
            return self.y

        def construct(self):
            a = self.max()
            return self.relu(a)

    net = Net(Tensor(5, ms.float32), Tensor(10, ms.float32))
    output = net()
    assert output.asnumpy() == 10


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_control_flow_if_break():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with control flow.
    Expectation: The calculation result is correct.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.add = P.TensorAdd()

        @jit
        def construct(self, x, y, z):
            out = z
            for _ in range(5):
                if 2 * x < y:
                    if 3 * x < y:
                        out = self.add(out, out)
                        x = x + 1
                    out = self.relu(out)
                if x + 6 == y:
                    break
            out = self.relu(out)
            return out

    net = Net()
    x = Tensor(2, ms.int32)
    y = Tensor(10, ms.int32)
    z = Tensor(np.ones([4, 4, 4]), ms.float32)
    output = net(x, y, z)
    assert (output.asnumpy() == z.asnumpy() * 4).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_with_dynamic_shape():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with dynamic shape.
    Expectation: The calculation result is correct.
    """

    @jit()
    def test(x):
        return ms.numpy.unique(x, return_inverse=True)

    x = Tensor([[1, 1, 2], [3, 3, 5]], ms.int32)
    output = test(x)
    assert (output[0].asnumpy() == np.array([1, 2, 3, 5])).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_with_tuple_inputs():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with tuple inputs.
    Expectation: The calculation result is correct.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.enable_tuple_broaden = True

        @jit
        def construct(self, grads):
            new_grads = []
            for grad in grads:
                new_grads.append(grad + 1)
            return new_grads

    x = Tensor(np.ones([2, 2]), dtype=ms.int32)
    y = Tensor(np.ones([2, 2]), dtype=ms.int32)
    net = Net()
    out = net((x, y))
    assert (out[0].asnumpy() == np.ones([2, 2]) + 1).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_with_optional_inputs():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with optional inputs.
    Expectation: The calculation result is correct.
    """

    @jit
    def foo(x, y=1):
        return x + y

    a = Tensor(3, dtype=ms.int32)
    assert foo(a).asnumpy() == 4
    assert foo(a, 2).asnumpy() == 5
    assert foo(a, y=3).asnumpy() == 6
    assert foo(x=a, y=4).asnumpy() == 7


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_with_args_inputs():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with *args.
    Expectation: The calculation result is correct.
    """

    @jit
    def foo(x, *args):
        return x + args[0] + args[1]

    x = Tensor(3, dtype=ms.int32)
    assert foo(x, 1, 2).asnumpy() == 6


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_with_kwargs_inputs():
    """
    Feature: PyNative ms_function.
    Description: PyNative ms_function with **kwargs.
    Expectation: Raise expected exception
    """

    @jit
    def foo(x, **kwargs):
        return x + kwargs.get('y')

    x = Tensor(3, dtype=ms.int32)
    data = {"y": 1}
    assert foo(x, **data).asnumpy() == [4]
