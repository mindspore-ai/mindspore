# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test_auto_grad """

import numpy as np
import mindspore
from mindspore.ops import composite as C
from mindspore import Tensor, Parameter
from mindspore import nn
from mindspore import ops


class MultiInputNet(nn.Cell):
    def construct(self, x, t):
        y = x * x
        z = y * t[0]
        return z


class SplitNet(nn.Cell):
    def __init__(self):
        super(SplitNet, self).__init__()
        self.split = ops.split

    def construct(self, x):
        output = self.split(x, 3)
        return output


class SplitAddNet(nn.Cell):
    def __init__(self):
        super(SplitAddNet, self).__init__()
        self.split = ops.split

    def construct(self, x):
        output = self.split(x, 3)
        y = output[0] * 3
        z = output[1] * 2
        return y + z


class ConcatNet(nn.Cell):
    def __init__(self):
        super(ConcatNet, self).__init__()
        self.concat = ops.concat

    def construct(self, x, y):
        output = self.concat((x, y), 0)
        return output


class StackNet(nn.Cell):
    def __init__(self):
        super(StackNet, self).__init__()
        self.stack = ops.stack

    def construct(self, x, y):
        output = self.stack((x, y), 0)
        return output


def print_gradient(dx):
    print("dx: ", dx)
    return dx


class InsertGradientOfNet(nn.Cell):
    def __init__(self):
        super(InsertGradientOfNet, self).__init__()
        self.insert_gradient_of = ops.InsertGradientOf(print_gradient)

    def construct(self, x):
        output = x * x
        y = self.insert_gradient_of(output)
        return y * y


class NormalNet(nn.Cell):
    def __init__(self):
        super(NormalNet, self).__init__()
        self.p1 = Parameter(Tensor([1], dtype=mindspore.float32))
        self.p2 = Parameter(Tensor([2], dtype=mindspore.float32))

    def construct(self, x):
        y = x + self.p1
        z = y * self.p2
        return z


class NoneTensorInputNet(nn.Cell):
    def construct(self, x):
        y = x[0] * x[0]
        z = y * x[0]
        return z


class ParamNet(nn.Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.p1 = Parameter(Tensor([2], dtype=mindspore.float32))
        self.p1.requires_grad = True

    def construct(self, x):
        return self.p1


class CustomBpropNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        return z

    def bprop(self, *args):
        return (args[0] * 4,)


class StopGradientNet(nn.Cell):
    def __init__(self):
        super(StopGradientNet, self).__init__()
        self.p1 = Parameter(Tensor([2], dtype=mindspore.float32))

    def construct(self, x):
        y = x * x
        y = ops.stop_gradient(y)
        z = y * self.p1
        return z


def test_auto_grad_multi_input():
    """
    Feature: Test auto grad multi input
    Description: Test multi input.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    net = MultiInputNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32),
                   (Tensor(shape=[None], dtype=mindspore.float32), Tensor(shape=[None], dtype=mindspore.float32)))
    grads = grad_net(net)(x, (y, z))
    assert np.allclose(grads[0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_multi_input_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = ConcatNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32), Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_stack_op():
    """
    Feature: Test auto grad multi input op
    Description: Test multi input op.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = StackNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32), Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


def test_ir_grad_multi_output():
    """
    Feature: Test auto grad multi output
    Description: Test multi output.
    Expectation: Success.
    """
    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitNet()
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_multi_output_add_gradient():
    """
    Feature: Test auto grad multi output add.
    Description: Test multi output add.
    Expectation: Success.
    """
    input1 = Tensor(np.arange(9).astype("float32"))
    net = SplitAddNet()
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([3, 3, 3, 2, 2, 2, 0, 0, 0], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([3, 3, 3, 2, 2, 2, 0, 0, 0], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_not_register_expander_op():
    """
    Feature: Test auto grad not expander
    Description: Test auto grad not expander.
    Expectation: Success.
    """
    input1 = Tensor([2], mindspore.float32)
    net = InsertGradientOfNet()
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([32], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grad = ops.grad(net)(input1)
    assert np.allclose(grad.asnumpy(), np.array([32], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_weights_grad():
    """
    Feature: Test auto grad weights grad.
    Description: Test auto grad weights grad.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_single_weight():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net, [net.p1])(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_with_sens():
    """
    Feature: Test auto grad single input.
    Description: Test auto grad single input.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    sens = Tensor([1], mindspore.float32)
    net = NormalNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True, sens_param=True)
    grads = grad_net(net, [net.p1])(x, sens)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net, [net.p1])(x, sens)
    assert np.allclose(grads[0][0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_none_inputs_and_weights():
    """
    Feature: Test auto grad none inputs and weights.
    Description: Test auto grad none inputs and weights.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    net = NoneTensorInputNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)((x, y))
    assert len(grads) == 2
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net)((x, y))
    assert len(grads) == 2
    assert not grads[0]
    assert not grads[1]


def test_auto_grad_by_position():
    """
    Feature: Test auto grad by position.
    Description: Test auto grad by position.
    Expectation: Success.
    """
    x = Tensor([1], mindspore.float32)
    net = NormalNet()
    _, grad = ops.value_and_grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    _, grad = ops.value_and_grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_return_param():
    """
    Feature: Test auto grad return param.
    Description: Test auto grad return param.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    net = ParamNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net)(x)
    assert np.allclose(grads[1][0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_stop_gradient():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    net = StopGradientNet()
    grad_net = C.GradOperation(get_all=True, get_by_list=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grads = grad_net(net)(x)
    assert np.allclose(grads[0][0].asnumpy(), np.array([0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


def test_auto_grad_bprop_net():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    x = Tensor([2], mindspore.float32)
    net = CustomBpropNet()
    grad = ops.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)
    net.set_inputs(Tensor(shape=[None], dtype=mindspore.float32))
    grad = ops.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)
