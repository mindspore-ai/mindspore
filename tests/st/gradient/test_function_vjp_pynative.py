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
"""test vjp in pynative mode"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.functional import vjp

context.set_context(mode=context.PYNATIVE_MODE)


class SingleInputNet(nn.Cell):
    def construct(self, x):
        return x ** 3


class MultipleInputsOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2 * x, y ** 3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vjp_single_input_pynative():
    """
    Features: Function vjp
    Description: Test vjp with single input, single output and default v in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputNet()
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    primal, grad_fn = vjp(net, x)
    gradient = grad_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vjp_multiple_inputs_default_v_pynative():
    """
    Features: Function vjp
    Description: Test vjp with multiple inputs, multiple outputs and default v in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsOutputNet()
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    primal, grad_fn = vjp(net, x, y)
    gradient = grad_fn(v, v)
    assert isinstance(gradient, tuple)
    assert len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1.asnumpy())
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vjp_input_function_single_input_single_output_default_v_pynative():
    """
    Features: Function vjp
    Description: Test vjp with function, single input, single output and default v in pynative mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))

    def test_function(inputs):
        return inputs ** 3

    primal, grad_fn = vjp(test_function, x)
    gradient = grad_fn(v)
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vjp_construct_single_input_single_output_default_v_pynative():
    """
    Features: Function vjp
    Description: Test vjp with function, single input, single output and default v in pynative mode.
    Expectation: No exception.
    """
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

    test_net_pynative = Net(SingleInputNet())
    primal, gradient = test_net_pynative(x, v)
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(gradient[0].asnumpy(), expect_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vjp_multiple_outputs_with_has_aux_pynative():
    """
    Features: Function vjp
    Description: Test vjp with multiple inputs, multiple outputs with set_aux as True in pynative mode.
    Expectation: No exception.
    """

    def fn(x, y):
        return 2 * x + y, y ** 3

    def fn2(*args):
        return fn(*args)

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    expect_primal = Tensor(np.array([[3, 6], [9, 12]]).astype(np.float32))
    expect_aux = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    primal, grad_fn, aux = vjp(fn2, x, y, has_aux=True)
    gradient = grad_fn(v)
    assert isinstance(gradient, tuple)
    assert len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1.asnumpy())
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(aux.asnumpy(), expect_aux.asnumpy())
