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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.common.api import ms_function
from mindspore.common import Parameter, ParameterTuple
import mindspore.context as context
context.set_context(mode=context.PYNATIVE_MODE)

@ms_function
def ConvBnReLU(x):
    conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
    bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
    relu = nn.ReLU()

    x = conv(x)
    x = bn(x)
    x = relu(x)

    return x

@pytest.mark.level0
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

    @ms_function
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@pytest.mark.level0
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

    @ms_function
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


@pytest.mark.level0
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

    @ms_function
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


@pytest.mark.level0
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


@pytest.mark.level0
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

        @ms_function
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
