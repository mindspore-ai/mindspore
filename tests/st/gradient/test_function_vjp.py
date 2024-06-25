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
"""test vjp in graph mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore import jit, ops
from mindspore.ops.functional import vjp
from tests.mark_utils import arg_mark


class SingleInputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class MultipleInputsOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2 * x, y ** 3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_single_input_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad_fn = vjp(net, x)
    gradient = grad_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_multiple_inputs_default_v_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad_fn = vjp(net, x, y)
    gradient = grad_fn(v, v)
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(gradient, tuple)
    assert len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_jit_function_single_input_single_output_default_v_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with @jit decorated function, single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputNet()

    @jit
    def vjp_with_jit_function(inputs, vectors):
        output, grad_fn = vjp(net, inputs)
        vjp_grad = grad_fn(vectors)
        return output, vjp_grad

    primal, gradient = vjp_with_jit_function(x, v)
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_input_function_single_input_single_output_default_v_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with function, single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))

    def test_function(inputs):
        return inputs ** 3

    primal, grad_fn = vjp(test_function, x)
    gradient = grad_fn(v)
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_construct_single_input_single_output_default_v_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with function, single input, single output and default v in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))

    class Net(nn.Cell):
        def __init__(self, network):
            super(Net, self).__init__()
            self.net = network

        def construct(self, inputs, vectors):
            net_out, grad_fn = vjp(self.net, inputs)
            vjp_out = grad_fn(vectors)
            return net_out, vjp_out

    test_net_graph = Net(SingleInputNet())
    primal, gradient = test_net_graph(x, v)
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_multiple_outputs_with_has_aux_graph(mode):
    """
    Features: Function vjp
    Description: Test vjp with multiple inputs, multiple outputs with set_aux as True in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=mode)

    def fn(x, y):
        return 2 * x + y, y ** 3

    def fn2(*args):
        return fn(*args)

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_primal = Tensor(np.array([[3, 6], [9, 12]]).astype(np.float32))
    expect_aux = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    primal, grad_fn, aux = vjp(fn2, x, y, has_aux=True)
    gradient = grad_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(aux.asnumpy(), expect_aux.asnumpy())
    assert isinstance(gradient, tuple)
    assert len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_multiple_outputs_with_weight(mode):
    """
    Features: Function vjp
    Description: Test vjp with multiple outputs network and get gradients for weights.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    net = nn.Dense(10, 1)
    inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
    weights = net.trainable_params()

    grad_fn = ops.grad(net, grad_position=None, weights=weights)
    params_gradient_grad = grad_fn(inputs)

    forward_res, vjp_fn = vjp(net, inputs, weights=weights)
    _, params_gradient_vjp = vjp_fn(ops.ones_like(forward_res))
    assert np.allclose(params_gradient_vjp[0].asnumpy(), params_gradient_grad[0].asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_vjp_multiple_outputs_merge_forward(mode):
    """
    Features: Function vjp
    Description: Test vjp with multiple outputs network and get gradients with each output.
    Expectation: No exception.
    """
    context.set_context(mode=mode)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.scale = Parameter(Tensor(np.ones([2]).astype(np.float32)))
            self.bias = Parameter(Tensor(np.ones([2]).astype(np.float32)))
            self.mean = Parameter(Tensor(np.ones([2]).astype(np.float32)))
            self.variance = Parameter(Tensor(np.ones([2]).astype(np.float32)))
            self.batch_norm = ops.BatchNorm(is_training=True)

        def construct(self, x):
            # self.scale += Tensor(np.ones([2]).astype(np.float32))
            out1 = self.batch_norm(x, self.scale, self.bias, self.mean, self.variance)
            out2 = self.relu(out1[0])
            out3 = self.relu(out2)
            return out1[0], out2, out3

    class Grad(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, x):
            outputs, vjp_fn = ops.vjp(self.net, x)
            out1 = vjp_fn((ops.ones_like(outputs[0]), ops.zeros_like(outputs[1]), ops.zeros_like(outputs[2])))
            out2 = vjp_fn((ops.zeros_like(outputs[0]), ops.ones_like(outputs[1]), ops.zeros_like(outputs[2])))
            out3 = vjp_fn((ops.zeros_like(outputs[0]), ops.zeros_like(outputs[1]), ops.ones_like(outputs[2])))
            return outputs, out1, out2, out3

    net = Net()
    x = Tensor(np.ones([2, 2]).astype(np.float32))
    Grad(net)(x)
    assert np.allclose(net.variance.value().asnumpy(), np.array([0.9, 0.9]))
