# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""test high grad train in graph mode"""

import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor, Parameter, jit
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import Momentum
from mindspore.ops.composite import GradOperation
from mindspore.common import ParameterTuple, dtype
import mindspore.ops.functional as F

from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class _Grad(nn.Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.real_inputs_count is None or self.sens_param is False:
            if self.wrt_params:
                return self.grad(self.network, self.params)(*inputs)
            return self.grad(self.network)(*inputs)

        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        if self.wrt_params:
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class GradOfAllInputs(_Grad):
    """
    get grads of all inputs
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()
        self.add = ops.TensorAdd()
        weight_np = np.array([2]).astype(np.float32)
        bias_np = np.array([1]).astype(np.float32)
        self.weight = Parameter(Tensor(weight_np),
                                name='weight', requires_grad=True)
        self.bias = Parameter(Tensor(bias_np),
                              name="bias", requires_grad=True)

    def construct(self, x):
        xw = self.mul(x, self.weight)
        output = self.add(xw, self.bias)
        return output


class WithLossCellLocal(nn.Cell):
    def __init__(self, grad, loss):
        super().__init__(auto_prefix=False)
        self.grad = grad
        self.loss = loss

    def construct(self, data, label):
        out = self.grad(data)
        return self.loss(out, label)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_high_grad_train():
    x_pure = np.random.randint(-10, 100, 32)
    x_train = x_pure.astype(np.float32)
    y_noise = 3 * x_pure + 2 + np.random.randn(32) / 10
    y_train = y_noise.astype(np.float32)
    net = Net()
    grad_net = GradOfFirstInput(net, sens_param=False)
    epoch = 2
    momentum = 0.0
    learning_rate = 0.001
    optimizer = Momentum(filter(lambda x: x.requires_grad,
                                grad_net.get_parameters()), learning_rate, momentum)
    criterion = nn.loss.MSELoss()
    net_with_criterion = WithLossCellLocal(grad_net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    for i in range(epoch):
        train_network(Tensor([x_train[i]]), Tensor([y_train[i]]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_high_grad_environ_eliminate():
    """
    Feature: eliminate the environ node.
    Description: eliminate the environ node in high grad.
    Expectation: Null.
    """

    class AutoNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([1], dtype.float32), name='weight')

        def construct(self, x, y):
            if x <= 0:
                x = x - x
                y = y / y
            elif x > y:
                x = y / 3
            elif x > 5:
                y = x - x
            elif y > x:
                y = x + self.w
            else:
                x = x - x
            return x + y

    x = np.array([3], np.float32)
    y = np.array([4], np.float32)
    net = AutoNet()
    grad_net = F.grad(net, grad_position=(0, 1))
    sgrad_net = jit(F.grad(grad_net), backend="ms_backend")
    sgrad = sgrad_net(Tensor(x), Tensor(y))
    print('second grad: ', sgrad)


class HighGrad(nn.Cell):
    """
    get any order of grad
    """
    def __init__(self, network, grad_list, sens_param=False, real_inputs_count=None):
        super().__init__()
        self.grads = [network, ]
        for i in range(len(grad_list)-1):
            _grad = grad_list[i](self.grads[i], sens_param=False)
            self.grads.append(_grad)
        self.final_grad = grad_list[-1](self.grads[-1],
                sens_param=sens_param, real_inputs_count=real_inputs_count)

    def construct(self, *inputs):
        return self.final_grad(*inputs)

class LayerSin(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.Sin()

    def construct(self, input_x):
        return self.op(input_x)

class LayerCos(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.Cos()

    def construct(self, input_x):
        return self.op(input_x)

class LayerInputFinalNet(nn.Cell):
    def __init__(self, layer1, layer2):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.funcs = (self.layer1, self.layer2)

    def construct(self, input_x, i):
        output = self.funcs[i](input_x)
        return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_highgrad_switch_layer_grad():
    """
    Feature: High grad with switch layer.
    Description: High grad with switch layer.
    Expectation: No exception.
    """

    func1 = LayerSin()
    func2 = LayerCos()
    net = LayerInputFinalNet(func1, func2)
    input_np = np.array([0, 0]).astype(np.float32)
    index = Tensor(0, dtype.int32)
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput], sens_param=False)
    grad_net(Tensor(input_np), index)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.03)
def test_highgrad_grad_in_switch_layer():
    """
    Feature: High grad with switch layer.
    Description: High grad with switch layer.
    Expectation: No exception.
    """

    func1 = LayerSin()
    func2 = LayerCos()
    grad_func1 = HighGrad(func1, [GradOfFirstInput, GradOfFirstInput], sens_param=False)
    grad_func2 = HighGrad(func2, [GradOfFirstInput, GradOfFirstInput], sens_param=False)
    net = LayerInputFinalNet(grad_func1, grad_func2)
    input_np = np.array([0, 0]).astype(np.float32)
    index = Tensor(0, dtype.int32)
    net(Tensor(input_np), index)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.25)
def test_highgrad_grad_in_net():
    """
    Feature: High grad with grad in net.
    Description: High grad with grad in net.
    Expectation: No exception.
    """
    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = ops.Mul()

        def construct(self, input_x):
            return self.mul(input_x, input_x)

    net1 = Net1()

    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = ops.Mul()
            self.gradnet = GradOfFirstInput(net1, sens_param=False)

        def construct(self, input_x):
            x_square = self.mul(input_x, input_x)
            output = self.gradnet(x_square)
            return output

    net2 = Net2()
    grad_net = GradOfFirstInput(net2, sens_param=False)
    input_np = np.array([[1, 1], [1, 1]]).astype(np.float32)
    grad = grad_net(Tensor(input_np))
    assert (grad.asnumpy() == np.array([[4, 4], [4, 4]]).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.25)
def test_highgrad_two_input_sec_grad():
    """
    Feature: High grad with bprop two input.
    Description: High grad with bprop two input.
    Expectation: No exception.
    """
    class TwoInputBprop(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Mul()

        def construct(self, x, y):
            return self.op(x, y)

        def bprop(self, x, y, out, dout):
            return x * 5, y * 8

    net = TwoInputBprop()
    input_x = Tensor(np.array([1, 1]).astype(np.float32))
    input_y = Tensor(np.array([1, 1]).astype(np.float32))
    grad_net = HighGrad(net, [GradOfAllInputs, GradOfAllInputs],
                        sens_param=True,
                        real_inputs_count=2)
    sens_0 = Tensor(np.array([0, 0]).astype(np.float32))
    sens_1 = Tensor(np.array([1, 1]).astype(np.float32))
    dxdx, dxdy = grad_net(Tensor(input_x), Tensor(input_y), sens_1, sens_0)
    assert (dxdx.asnumpy() == np.array([5, 5]).astype(np.float32)).all()
    assert (dxdy.asnumpy() == np.array([0, 0]).astype(np.float32)).all()
    dydx, dydy = grad_net(Tensor(input_x), Tensor(input_y), sens_0, sens_1)
    assert (dydx.asnumpy() == np.array([0, 0]).astype(np.float32)).all()
    assert (dydy.asnumpy() == np.array([8, 8]).astype(np.float32)).all()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_highgrad_control_if_mul():
    """
    Feature: High grad with control flow.
    Description: High grad with control flow.
    Expectation: No exception.
    """
    class IfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = ops.Mul()
            self.scalar1 = Tensor(1, dtype.float32)
            self.scalar2 = Tensor(-1, dtype.float32)

        def construct(self, x):
            if x < 0:
                out = self.mul(x, self.scalar1)
            else:
                out = self.mul(x, self.scalar2)
            return out

    net = IfNet()
    input_x = Tensor(2, dtype.float32)
    grad_net = HighGrad(net, [GradOfFirstInput, GradOfFirstInput])
    grad = grad_net(input_x)
    assert (grad.asnumpy() == np.array([0]).astype(np.float32)).all()
