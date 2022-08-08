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
"""test function grad in graph mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore import ms_function
from mindspore.ops.functional import grad
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import Parameter, ParameterTuple

context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3

class SingleInputMultipleOutputsNet(nn.Cell):
    def construct(self, x):
        return x**3, 2*x

class MultipleInputsSingleOutputNet(nn.Cell):
    def construct(self, x, y, z):
        return x*y*z

class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x**2 + y**2 + z**2, x*y*z


class ParamNet(nn.Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.w = Parameter(Tensor([2., 2.]), name="w")
        self.z = Parameter(Tensor([3., 3.]), name="z")

    def construct(self, x):
        res = x * self.w * self.z
        return res


def function(x, y, z):
    return x**2 + y**2 + z**2, x*y*z


def iteration_grad_function(x, y, z):
    return x**2*y*z


@ms_function
def grad_warp_with_msfunction(x, y, z):
    output = grad(function)(x, y, z)
    return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_single_output_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with single input and single output net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_single_input_multiple_outputs_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with single input and multiple outputs net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsNet()
    expect_grad = Tensor(np.array([[5, 14], [29, 50]]).astype(np.float32))
    real_grad = grad(net)(x)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_single_output_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and single output net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    expect_grad1 = Tensor(np.array([[0, 6], [15, -4]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 6], [-3, 8]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_multiple_inputs_multiple_outputs_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and multiple outputs net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    expect_grad1 = Tensor(np.array([[-4, 12], [13, 0]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 12], [7, 6]]).astype(np.float32))
    real_grad = grad(net, grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_function_with_sens_graph():
    """
    Features: Function grad.
    Description: Test F.grad with function setting sens_param in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    v = Tensor(np.array([[-1, 3], [2, 1]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[4, 36], [26, 0]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[2, 36], [14, 6]]).astype(np.float32))
    real_grad = grad(function, grad_position=(1, 2), sens_param=True)(x, y, z, (v, v))
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_iteration_function_graph():
    """
    Features: Function grad.
    Description: Test calling F.grad iterative with function in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[0, 12], [30, -8]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-4, 12], [-6, 16]]).astype(np.float32))
    real_grad = grad(grad(iteration_grad_function), grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_warp_with_msfunction_graph():
    """
    Features: Function grad.
    Description: Test F.grad warpped with ms_function in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad = Tensor(np.array([[2, 13], [1, 6]]).astype(np.float32))
    real_grad = grad_warp_with_msfunction(x, y, z)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_grad_position_twice_graph():
    """
    Features: Function grad.
    Description: Test F.grad with function setting grad_position twice in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    z = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsSingleOutputNet()
    out1 = grad(net, grad_position=0)(x, y, z)
    out2 = grad(net, grad_position=(0, 1))(x, y, z)
    assert isinstance(out1, Tensor)
    assert isinstance(out2, tuple)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_with_weights_twice_graph():
    """
    Features: GradOperation and grad.
    Description: Test F.grad with different weights twice in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNet()
    grad_fn = C.GradOperation(get_by_list=True)
    weights1 = ParameterTuple(net.trainable_params()[:1])
    weights2 = ParameterTuple(net.trainable_params()[1:])
    expect1 = np.array([3, 6]).astype(np.float32)
    expect2 = np.array([2, 4]).astype(np.float32)
    out1 = grad_fn(net, weights1)(x)
    out2 = grad_fn(net, weights2)(x)
    assert np.allclose(out1[0].asnumpy(), expect1)
    assert np.allclose(out2[0].asnumpy(), expect2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_if_ith_train_one_step():
    """
    Features: Grad with multiple funcgraph at the same J level.
    Description: Grad a network with each output. A simplification for GAN network.
    Expectation: Compile success.
    """
    class IthOutputCell(nn.Cell):
        def __init__(self, network, output_index):
            super().__init__()
            self.network = network
            self.output_index = output_index

        def construct(self, x1, x2):
            loss = self.network(x1, x2)[self.output_index]
            return loss

    class SingleIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight_x = Parameter(Tensor(2, mstype.int32), name="weightx")
            self.weight_y = Parameter(Tensor(5, mstype.int32), name="weighty")

        def construct(self, x, y):
            if self.weight_x < self.weight_y:
                x = x + y
                y = y + x
            else:
                x = x - y
                y = y - x
            return x, y

    class MyTrainOneStepCell(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.network = network
            self.network.set_train()
            self.weights = ParameterTuple(network.trainable_params())
            self.grad = C.GradOperation(get_by_list=True)

            self.loss_net_g = IthOutputCell(network, output_index=0)
            self.loss_net_d = IthOutputCell(network, output_index=0)
            self.loss_net_g.set_grad()
            self.loss_net_d.set_grad()

        def construct(self, x, y):
            forward = self.network(x, y)
            weights = self.weights
            grads_g = self.grad(self.loss_net_g, weights)(x, y)
            grads_d = self.grad(self.loss_net_d, weights)(x, y)
            return (forward, grads_g, grads_d)

    x = Tensor(2, mstype.int32)
    y = Tensor(5, mstype.int32)
    if_net = SingleIfNet()
    train_one_if_net = MyTrainOneStepCell(if_net)
    train_one_if_net(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_net_d_net_g():
    """
    Features: Grad with multiple funcgraph at the same J level.
    Description: Grad two different network. A simplification for GAN network.
    Expectation: Compile success.
    """
    class NetD(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight_d = Parameter(Tensor(2, mstype.int32), name="weightd")

        def construct(self, x, y):
            if self.weight_d < x:
                x = x + y
            else:
                x = x - y
            return x, y

    class NetG(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight_g = Parameter(Tensor(2, mstype.int32), name="weightg")

        def construct(self, x, y):
            if self.weight_g < x:
                x = x - y
            else:
                x = x + y
            return x, y

    class Backbone(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net_d = NetD()
            self.net_g = NetG()
            self.trainable_params_d = self.net_d.trainable_params()
            self.trainable_params_g = self.net_g.trainable_params()

        def construct(self, x, y):
            m, n = self.net_d(x, y)
            p, q = self.net_g(x, y)
            return m + n + p + q

    class LossNetD(nn.Cell):
        def __init__(self, backbone):
            super().__init__()
            self.net_d = backbone.net_d
            self.net_g = backbone.net_g

        def construct(self, x, y):
            m, n = self.net_d(x, y)
            p, q = self.net_g(x, y)
            return m + n + p + q

    class LossNetG(nn.Cell):
        def __init__(self, backbone):
            super().__init__()
            self.net_d = backbone.net_d
            self.net_g = backbone.net_g

        def construct(self, x, y):
            m, n = self.net_d(x, y)
            p, q = self.net_g(x, y)
            return m + n + p + q

    class MyTrainOneStepCell(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.network = network
            self.weights_d = ParameterTuple(network.net_d.trainable_params())
            self.weights_g = ParameterTuple(network.net_g.trainable_params())
            self.grad = C.GradOperation(get_by_list=True)

            self.loss_net_d = LossNetD(network)
            self.loss_net_g = LossNetG(network)
            self.loss_net_g.set_grad()
            self.loss_net_d.set_grad()

        def construct(self, x, y):
            grads_d = self.grad(self.loss_net_d, self.weights_d)(x, y)
            grads_g = self.grad(self.loss_net_g, self.weights_g)(x, y)
            return (grads_g, grads_d)

    x = Tensor(2, mstype.int32)
    y = Tensor(5, mstype.int32)
    network = Backbone()
    train_one_net = MyTrainOneStepCell(network)
    train_one_net(x, y)
