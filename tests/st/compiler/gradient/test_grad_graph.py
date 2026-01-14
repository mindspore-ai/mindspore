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
"""test function grad in graph mode"""
import numpy as np
import pytest
import mindspore
from mindspore import nn, context
from mindspore import Tensor
from mindspore import jit, ops
from mindspore.ops.functional import grad, value_and_grad, get_grad
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import Parameter, ParameterTuple
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class SingleInputMultipleOutputsNet(nn.Cell):
    def construct(self, x):
        return x ** 3, 2 * x


class MultipleInputsSingleOutputNet(nn.Cell):
    def construct(self, x, y, z):
        return x * y * z


class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x ** 2 + y ** 2 + z ** 2, x * y * z


class ParamNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([2., 2.]), name="w")
        self.z = Parameter(Tensor([3., 3.]), name="z")

    def construct(self, x):
        res = x * self.w * self.z
        return res


def function(x, y, z):
    return x ** 2 + y ** 2 + z ** 2, x * y * z


def iteration_grad_function(x, y, z):
    return x ** 2 * y * z


@jit
def grad_wrap_with_msfunction(x, y, z):
    output = grad(function)(x, y, z)
    return output


@jit
def grad_wrap_with_msfunction_get_grad(x, y, z):
    out_with_id = grad(function, return_ids=True)(x, y, z)
    output = get_grad(out_with_id, 0)
    return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    real_grad = grad(grad(iteration_grad_function),
                     grad_position=(1, 2))(x, y, z)
    assert isinstance(real_grad, tuple)
    assert len(real_grad) == 2
    assert np.allclose(real_grad[0].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_grad[1].asnumpy(), expect_grad2.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_wrap_with_msfunction_graph():
    """
    Features: Function grad.
    Description: Test F.grad wrapped with @jit decorated function in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad = Tensor(np.array([[2, 13], [1, 6]]).astype(np.float32))
    real_grad = grad_wrap_with_msfunction(x, y, z)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_with_weights_has_aux_graph():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux in graph mode.
    Expectation: No exception.
    """

    class ParamNetAux(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            res = x * self.w * self.z
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetAux()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_function_grad_with_weights_has_aux_graph():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    net = ParamMultipleInputNet()
    weights = net.trainable_params()

    @jit
    def user_fn(x, y):
        res, aux = grad(net, 0, weights, True)(x, y)
        return res, aux

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    res, aux = user_fn(x, y)
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_grad_with_weights_has_aux_graph():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res, aux = grad(self.net, 0, self.weights, True)(x, y)
            return res, aux

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    res, aux = grad_net(x, y)
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_if_with_weights_has_aux_graph():
    """
    Features: Function grad.
    Description: Test F.grad with different weights and has_aux as well as if case in graph mode.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")

        def construct(self, x):
            if x[0] == 1:
                res = x * self.w * self.z
            else:
                res = x * x
            return res, x, self.w

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = Net()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_aux1 = np.array([1, 2]).astype(np.float32)
    expect_aux2 = np.array([2, 2]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux1)
    assert np.allclose(aux[1].asnumpy(), expect_aux2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_grad_nest_with_weights_has_aux_graph():
    """
    Features: Function value_and_grad.
    Description: Test F.grad with different weights and has_aux as well as nested nets in graph mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([18, 18]).astype(np.float32)
    expect_grad_weight1 = np.array([9, 18]).astype(np.float32)
    expect_grad_weight2 = np.array([6, 12]).astype(np.float32)
    expect_aux = np.array([6, 12]).astype(np.float32)
    res, aux = grad(net, 0, weights, True)(x)
    assert np.allclose(res[0].asnumpy(), expect_grad_input)
    assert np.allclose(res[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[1][1].asnumpy(), expect_grad_weight2)
    assert np.allclose(aux[0].asnumpy(), expect_aux)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_with_weights_has_aux_graph():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux in graph mode.
    Expectation: No exception.
    """

    class ParamNetMultipleOutputs(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor([2., 2.], mstype.float32), name="w1")
            self.w2 = Parameter(Tensor([3., 3.], mstype.float32), name="w2")

        def construct(self, x):
            res = x * self.w1 * self.w2
            return res, x, self.w1

    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetMultipleOutputs()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_value0 = np.array([6, 12]).astype(np.float32)
    expect_value1 = np.array([1, 2]).astype(np.float32)
    expect_value2 = np.array([2, 2]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, True)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(value[2].asnumpy(), expect_value2)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_value_and_grad_with_weights_has_aux_graph():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux in graph mode.
    Expectation: No exception.
    """

    class ParamNetMultipleInputsOutputs(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            res = x * y * self.w
            return res, x, self.w

    class GradNet2(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            value, gradient = value_and_grad(
                self.net, 0, self.weights, True)(x, y)
            return value, gradient

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamNetMultipleInputsOutputs()
    grad_net = GradNet2(inner_net)
    value, gradient = grad_net(x, y)
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_value0 = np.array([6, 12]).astype(np.float32)
    expect_value1 = np.array([1, 2]).astype(np.float32)
    expect_value2 = np.array([2, 2]).astype(np.float32)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(value[2].asnumpy(), expect_value2)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_nest_with_weights_graph():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux as well as nested nets in graph mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([24, 24]).astype(np.float32)
    expect_grad_weight1 = np.array([12, 24]).astype(np.float32)
    expect_grad_weight2 = np.array([8, 16]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, False)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_value_and_grad_nest_with_weights_has_aux_graph():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad with different weights and has_aux as well as nested nets in graph mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([18, 18]).astype(np.float32)
    expect_grad_weight1 = np.array([9, 18]).astype(np.float32)
    expect_grad_weight2 = np.array([6, 12]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    value, gradient = value_and_grad(net, 0, weights, True)(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_grad_single_position_with_return_ids():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, return_ids=True)(x, y)
            return res

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    res = grad_net(x, y)
    expect_grad_input = (0, np.array([7, 7]).astype(np.float32))
    assert np.allclose(res[1].asnumpy(), expect_grad_input[1])
    assert np.allclose(res[0], expect_grad_input[0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_grad_multiplt_positions_with_return_ids():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, (0, 1), return_ids=True)(x, y)
            return res

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    res = grad_net(x, y)
    expect_grad_input1 = (0, np.array([7, 7]).astype(np.float32))
    expect_grad_input2 = (1, np.array([2, 4]).astype(np.float32))
    assert np.allclose(res[0][1].asnumpy(), expect_grad_input1[1])
    assert np.allclose(res[1][1].asnumpy(), expect_grad_input2[1])
    assert np.allclose(res[0][0], expect_grad_input1[0])
    assert np.allclose(res[1][0], expect_grad_input2[0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_grad_with_weights_with_return_ids():
    """
    Features: Function grad_with_ids.
    Description: Test F.grad with different weights and output ids in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            return res

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    res = grad_net(x, y)
    expect_grad_input = (0, np.array([7, 7]).astype(np.float32))
    expect_grad_weight1 = np.array([4, 7]).astype(np.float32)
    assert np.allclose(res[0][1].asnumpy(), expect_grad_input[1])
    assert np.allclose(res[1][0][1].asnumpy(), expect_grad_weight1)
    assert np.allclose(res[0][0], expect_grad_input[0])
    assert res[1][0][0] == inner_net.w.name


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_get_grad_by_position():
    """
    Features: Function get_grad.
    Description: Test get_grad with position id and output gradient in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            grad_out = get_grad(res, 0)
            return grad_out

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    grad_out = grad_net(x, y)
    expect_grad_input = np.array([7, 7]).astype(np.float32)
    assert np.allclose(grad_out.asnumpy(), expect_grad_input)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_get_grad_by_weight():
    """
    Features: Function get_grad.
    Description: Test get_grad with parameter and output gradient in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            grad_out = get_grad(res, self.net.w)
            return grad_out

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    expect_grad_input = np.array([4, 7]).astype(np.float32)
    inner_net = ParamMultipleInputNet()
    grad_net = GradNet(inner_net)
    grad_out = grad_net(x, y)
    assert np.allclose(grad_out.asnumpy(), expect_grad_input)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_get_grad_not_found():
    """
    Features: Function get_grad.
    Description: Test get_grad with invalid id and raise error in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, 0, self.weights, return_ids=True)(x, y)
            grad_out = get_grad(res, 1)
            return grad_out

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 3]).astype(np.float32))
    inner_net = ParamMultipleInputNet()
    with pytest.raises(RuntimeError):
        grad_net = GradNet(inner_net)
        grad_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_construct_get_grad_not_found_from_empty_tuple():
    """
    Features: Function get_grad.
    Description: Test get_grad with invalid id and raise error in graph mode.
    Expectation: No exception.
    """

    class ParamMultipleInputNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")

        def construct(self, x, y):
            outputs = x * y * self.w
            return outputs, x, self.w

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x, y):
            res = grad(self.net, return_ids=True)(x, y)
            grad_out = get_grad(res, 1)
            return grad_out

    x = Tensor(1)
    y = Tensor(2)
    inner_net = ParamMultipleInputNet()
    with pytest.raises(RuntimeError):
        grad_net = GradNet(inner_net)
        grad_net(x, y)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_get_grad_wrap_with_msfunction_graph():
    """
    Features: Function get_grad.
    Description: Test get_grad wrapped with @jit decorated function in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    expect_grad = Tensor(np.array([[2, 13], [1, 6]]).astype(np.float32))
    real_grad = grad_wrap_with_msfunction_get_grad(x, y, z)
    assert np.allclose(real_grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_primal_graph_call_others():
    """
    Features: Auto grad.
    Description: Two graph need to take a derivative and one calls the other graph.
    Expectation: Get the correct gradient.
    """
    def f(x, y):
        return x + y

    def g(x, y):
        return f(x, y) * y

    @jit
    def net(x, y):
        a = grad(f)(x, y)
        b = grad(g)(x, y)
        return a + b

    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([3, 4]).astype(np.float32))
    expected = Tensor(np.array([4, 5]).astype(np.float32))
    output = net(x, y)
    assert np.allclose(output.asnumpy(), expected.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_get_grad_outer_list_weight():
    """
    Features: Function get_grad.
    Description: Test get_grad with a list of parameter as input of the network in graph mode.
    Expectation: No exception.
    """
    class InnerNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter([1, 2], name='w')
            self.b = Parameter([1, 2], name='b')

        def construct(self, x, y):
            out = self.w * x + self.b + y
            return out

    class GradNet(nn.Cell):
        def __init__(self, net, pos, param, get):
            super().__init__()
            self.net = net
            self.pos = pos
            self.param = param
            self.get = get

        def construct(self, x, y):
            grad_net = grad(self.net, self.pos, self.param, return_ids=True)
            out_grad = grad_net(x, y)
            out = []
            for i in self.get:
                out.append(get_grad(out_grad, i))
            return out

    net = InnerNet()
    grad_net = GradNet(net, (0, 1), (net.w, net.b), (0, net.w))
    x = Tensor([1, 2], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    out = grad_net(x, y)
    expect_value0 = Tensor([1, 2], mstype.float32)
    expect_value1 = Tensor([1, 2], mstype.int64)
    assert np.allclose(out[0].asnumpy(), expect_value0.asnumpy())
    assert np.allclose(out[1].asnumpy(), expect_value1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grad_of_pyexecute_with_more_than_three_inputs():
    """
    Features: Function grad.
    Description: Test F.grad of_pyexecute_with_more_than_three_inputs in graph mode.
    Expectation: No exception.
    """
    class GradNet3(nn.Cell):
        def __init__(self, net, grad_position=0):
            super().__init__()
            self.net = net
            self.grad_position = grad_position
            self.grad = grad

        def construct(self, *inputs):
            return self.grad(self.net, self.grad_position)(*inputs)

    class ModifyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = 4

        def construct(self, x):
            self.p = self.p + x
            return self.p

    out = ModifyNet()(Tensor([2]))
    assert out == Tensor([6])

    ms_grad = GradNet3(ModifyNet())(Tensor([2]))
    assert ms_grad == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_nest_with_weights_graph_return_ids():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad when setting return_ids to true in graph mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x):
            value, gradient = value_and_grad(
                net, 0, self.weights, False, True)(x)
            return value, gradient

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    grad_net = GradNet(net)
    value, gradient = grad_net(x)
    expect_grad_input = np.array([24, 24]).astype(np.float32)
    expect_grad_weight1 = np.array([12, 24]).astype(np.float32)
    expect_grad_weight2 = np.array([8, 16]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(gradient[0][1].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0][1].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1][1].asnumpy(), expect_grad_weight2)
    assert gradient[0][0] == 0
    assert gradient[1][0][0] == net.w.name
    assert gradient[1][1][0] == net.z.name


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_nest_with_weights_graph_get_grad():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad and getgrad from the result in graph mode.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            return x * 3, x

    class Net(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.w = Parameter(Tensor([2., 2.], mstype.float32), name="w")
            self.z = Parameter(Tensor([3., 3.], mstype.float32), name="z")
            self.net = net

        def construct(self, x):
            res1 = x * self.w * self.z
            res2 = self.net(res1)
            return res2

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = net.trainable_params()

        def construct(self, x):
            value, gradient = value_and_grad(
                net, 0, self.weights, False, True)(x)
            res1 = get_grad(gradient, 0)
            res2 = get_grad(gradient, self.net.w)
            res3 = get_grad(gradient, self.net.z)
            return value, res1, res2, res3

    x = Tensor(np.array([1, 2]).astype(np.float32))
    inner_net = InnerNet()
    net = Net(inner_net)
    grad_net = GradNet(net)
    value, res1, res2, res3 = grad_net(x)
    expect_grad_input = np.array([24, 24]).astype(np.float32)
    expect_grad_weight1 = np.array([12, 24]).astype(np.float32)
    expect_grad_weight2 = np.array([8, 16]).astype(np.float32)
    expect_value0 = np.array([18, 36]).astype(np.float32)
    expect_value1 = np.array([6, 12]).astype(np.float32)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(res1.asnumpy(), expect_grad_input)
    assert np.allclose(res2.asnumpy(), expect_grad_weight1)
    assert np.allclose(res3.asnumpy(), expect_grad_weight2)


class ComputeNet(nn.Cell):
    def __init__(self, w, b):
        super().__init__()
        mw = Tensor(w, mstype.float32)
        mb = Tensor(b, mstype.float32)
        self.w = Parameter(mw, name='w')
        self.b = Parameter(mb, name='b')

    def construct(self, x, y):
        out = self.w * x + self.b + y
        return out


class GetGradNet(nn.Cell):
    def __init__(self, net, pos, param, get):
        super().__init__()
        self.net = net
        self.pos = pos
        self.param = param
        self.get = get

    def construct(self, x, y):
        grad_net = ops.value_and_grad(
            self.net, self.pos, self.param, return_ids=True)
        value, grads = grad_net(x, y)
        out = []
        for i in self.get:
            grad_result = ops.get_grad(grads, i)
            out.append(grad_result)
        return out, value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_return_ids_pos2():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad and getgrad out of range from the result in graph mode.
    Expectation: No exception.
    """
    w = np.array([1, 2])
    b = np.array([1, 2])
    net = ComputeNet(w, b)
    grad_net = GetGradNet(net, 0, None, (2,))
    x = Tensor([1, 2], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    with pytest.raises((ValueError, RuntimeError)) as er:
        grad_net(x, y)
        _pynative_executor.sync()
    assert "Can not find the gradient" in str(er)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_value_and_grad_return_ids_weight_b():
    """
    Features: Function value_and_grad.
    Description: Test F.value_and_grad and getgrad out of range from the result in graph mode.
    Expectation: No exception.
    """
    w = np.array([1, 2])
    b = np.array([1, 2])
    net = ComputeNet(w, b)
    grad_net = GetGradNet(net, None, net.w, (net.b,))
    x = Tensor([1, 2], mstype.float32)
    y = Tensor([1, 2], mstype.float32)
    with pytest.raises(RuntimeError) as er:
        grad_net(x, y)
        _pynative_executor.sync()
    assert "Can not find the gradient" in str(er)


class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, *inputs):
        x = inputs[0]
        y = inputs[1]
        x = x * self.z
        out = self.matmul(x, y)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_varargs_single_call():
    """
    Features: Support varargs for forward graph.
    Description: The forward graph has varargs while the grad graph has no varargs.
    Expectation: The gradient result is correct.
    """

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [
               2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(MatMulNet())(x, y)
    expect = np.array([[1.41, 1.6, 6.6], [1.41, 1.6, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_varargs_single_call_need_unpack():
    """
    Features: Support varargs for forward graph.
    Description: Both the forward graph and the grad graph have varargs.
    Expectation: The gradient result is correct.
    """

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, *inputs):
            gradient_function = self.grad_op(self.net)
            return gradient_function(*inputs)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [
               2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(MatMulNet())(x, y)
    expect = np.array([[1.41, 1.6, 6.6], [1.41, 1.6, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_varargs_grad_inited():
    """
    Features: Support varargs for forward graph.
    Description: Both the forward and grad graph have varargs, and the GradOperation is initialized in __init__.
    Expectation: The gradient result is correct.
    """

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.grad = ops.GradOperation()(net)

        def construct(self, *inputs):
            return self.grad(*inputs)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [
               2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(MatMulNet())(x, y)
    expect = np.array([[1.41, 1.6, 6.6], [1.41, 1.6, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_varargs_double_call():
    """
    Features: Support varargs for forward graph.
    Description: Both the forward graph and the grad graph have varargs, and the grad graph is called twice.
    Expectation: The gradient result is correct.
    """

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, *inputs):
            x = inputs[0]
            y = inputs[1]
            gradient_function = self.grad_op(self.net)
            return gradient_function(*inputs), gradient_function(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [
               2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(MatMulNet())(x, y)
    expect = np.array([[1.41, 1.6, 6.6], [1.41, 1.6, 6.6]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect)
    assert np.allclose(output[1].asnumpy(), expect)


@pytest.mark.skip(reason="Func grad node not match GradOperation")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_func_grad_varargs_single_call():
    """
    Features: Support varargs for forward graph.
    Description: Use the func grad and both the forward graph and the grad graph have varargs.
    Expectation: The gradient result is correct.
    """

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, *inputs):
            gradient_function = ops.grad(self.net)
            return gradient_function(*inputs)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [
               2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(MatMulNet())(x, y)
    expect = np.array([[1.41, 1.6, 6.6], [1.41, 1.6, 6.6]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_real_inputs_complex_outputs():
    """
    Features: Support automatic differentiation for complex number.
    Description: Test the imag bprop with real inputs and complex outputs.
    Expectation: The gradient result is correct.
    """

    class ImagNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.imag = ops.Imag()

        def construct(self, x):
            return self.imag(x)

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    output = GradNetWrtX(ImagNet())(x)
    expect = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_complex_inputs_complex_outputs():
    """
    Features: Support automatic differentiation for complex number.
    Description: Test the imag bprop with complex inputs and complex outputs.
    Expectation: The gradient result is correct.
    """

    class ImagNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.imag = ops.Imag()

        def construct(self, x):
            return self.imag(x)

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation()

        def construct(self, x):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x)

    x = Tensor(np.asarray(complex(1.3 + 0.4j)), mstype.complex64)
    output = GradNetWrtX(ImagNet())(x)
    expect = np.asarray(complex(1j), dtype=np.complex64)
    assert np.allclose(output.asnumpy(), expect)


class GradNet21(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.add
        self.relu = ops.relu
        self.div = ops.div
        self.y1 = Parameter(Tensor([1.0, -2.0, 3.0]), name="b")

    def construct(self, x1, x2):
        out = self.add(x1, self.y1)
        out = self.relu(out)
        out = self.div(out, x2)
        return out


class GradNetWrtWeight(nn.Cell):
    def __init__(self, net, grad_position=0, weight=None):
        super().__init__()
        self.grad_func = grad(net, grad_position, weight)

    def construct(self, *args):
        return self.grad_func(*args)


class GradOperationNet(nn.Cell):
    def __init__(self, net, get_all=False, get_by_list=False):
        super().__init__()
        self.net = net
        self.grad_op = C.GradOperation(
            get_all=get_all, get_by_list=get_by_list)

    def construct(self, *args):
        gradient_function = self.grad_op(self.net)
        return gradient_function(*args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_grad_grad_attr_illegal():
    """
    Features: Test grad.
    Description: Test calling grad with illegal arguments in graph mode.
    Expectation: Raise exception.
    """
    class GradNet(nn.Cell):
        def __init__(self, net, grad_position=0):
            super().__init__()
            self.grad_func = grad(net, grad_position)

        def construct(self, *args):
            return self.grad_func(*args)

    net = GradNet21()
    with pytest.raises(ValueError) as e:
        GradNetWrtWeight(net, grad_position=None, weight=None)
        _pynative_executor.sync()
    assert "can not be None at the same time" in str(e.value)
    with pytest.raises(TypeError) as e:
        GradNet(net, grad_position="True")
        _pynative_executor.sync()
    assert "For 'F.grad', the 'grad_position' must be int or tuple" in str(
        e.value)
    x1 = np.random.randn(3, 3, 3).astype(np.float32)
    x2 = np.random.randn(3, 3, 3).astype(np.float32)
    with pytest.raises(ValueError):
        GradNetWrtWeight(net, grad_position=(
            0, 1), weight=0)(Tensor(x1), Tensor(x2))
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_grad_grad_operation_attr_illegal():
    """
    Features: Test GradOperation.
    Description: Test calling GradOperation with illegal arguments in graph mode.
    Expectation: Raise exception.
    """
    net = GradNet21()
    with pytest.raises(TypeError) as e:
        GradOperationNet(net, get_all=None, get_by_list=False)
        _pynative_executor.sync()
    assert "the 'get_all' should be bool, but got NoneType" in str(e.value)
    with pytest.raises(TypeError) as e:
        GradOperationNet(net, get_all=False, get_by_list="True")
        _pynative_executor.sync()
    assert "the 'get_by_list' should be bool, but got str" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_with_sense():
    """
    Features: grad_with_sense
    Description: Test grad_with_sense
    Expectation: The gradient result is correct.
    """
    class InputGrad(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.grad = ops.grad(network, (0, 1), sens_param=True)

        @jit
        def construct(self, x1, x2, sens):
            return self.grad(x1, x2, sens)

    class InputGradOperation(nn.Cell):
        def __init__(self, network):
            super().__init__()
            self.network = network
            self.grad = C.GradOperation(get_all=True, sens_param=True)

        @jit
        def construct(self, x1, x2, sens):
            return self.grad(self.network)(x1, x2, sens)

    class AddNet(nn.Cell):
        def construct(self, x, y):
            return x + y

    net_grad = InputGrad(AddNet())
    net_gradoperation = InputGradOperation(AddNet())
    x = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    sens = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    res_grad = net_grad(x, y, sens)
    res_gradoperation = net_gradoperation(x, y, sens)
    for grad_out, gradop_out in zip(res_grad, res_gradoperation):
        assert np.allclose(grad_out.asnumpy(), gradop_out.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_value_and_grad_with_sense():
    """
    Features: value_and_grad_with_sense
    Description: Test value_and_grad_with_sense
    Expectation: The gradient result is correct.
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            return x * y * z
    x = Tensor([1, 2], mindspore.float32)
    y = Tensor([-2, 3], mindspore.float32)
    z = Tensor([0, 3], mindspore.float32)
    sense = Tensor([1, 2], mindspore.float32)
    net = Net()
    grad_fn = value_and_grad(net, grad_position=1, sens_param=True)
    _, inputs_gradient = grad_fn(x, y, z, sense)
    expect = np.array([[0, 12]]).astype(np.float32)
    assert np.allclose(inputs_gradient.asnumpy(), expect)
