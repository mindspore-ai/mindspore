# Copyright 2026 Huawei Technologies Co., Ltd
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
""" test_mindspore_vjp """

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import nn, ops, context, jit
from mindspore.ops.functional import vjp
from tests.mark_utils import arg_mark


class WrapFnNet(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class WeightedNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([2.0, 3.0], dtype=ms.float32), name='w')

    def construct(self, x):
        return x * self.w


class CounterNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.count = 0

    def construct(self, x):
        self.count += 1
        return x ** 2 + x


def test_vjp_single_input_single_output():
    """
    Feature: vjp
    Description: Test vjp with single input, single output.
    Expectation: Success.
    """
    def fn(x):
        return 2 * x ** 2 + 3 * x

    x = Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    v = Tensor([[2, 3], [0.5, 1]], dtype=ms.float32)
    expect_primal = np.array([[5, 14], [27, 44]], dtype=np.float32)
    expect_grad = np.array([[14, 33], [7.5, 19]], dtype=np.float32)

    # test fn
    primal, vjp_fn = vjp(fn, x)
    gradient = vjp_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), expect_grad, 0.00001, 0.00001)

    # test cell
    net = WrapFnNet(fn)
    primal, vjp_fn = vjp(net, x)
    gradient = vjp_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), expect_grad, 0.00001, 0.00001)


def test_vjp_multiple_inputs_single_output():
    """
    Feature: vjp
    Description: Test vjp with multiple inputs, single output.
    Expectation: Success.
    """
    def fn(x, y):
        return x * y

    x = Tensor([2.0, 3.0], dtype=ms.float32)
    y = Tensor([4.0, 5.0], dtype=ms.float32)
    v = Tensor([1.0, 1.0], dtype=ms.float32)
    expect_primal = np.array([8.0, 15.0], dtype=np.float32)
    expect_grad_x = np.array([4.0, 5.0], dtype=np.float32)
    expect_grad_y = np.array([2.0, 3.0], dtype=np.float32)

    # test fn
    primal, vjp_fn = vjp(fn, x, y)
    gradient = vjp_fn(v)

    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), expect_grad_x, 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), expect_grad_y, 0.00001, 0.00001)

    # test cell
    net = WrapFnNet(fn)
    primal, vjp_fn = vjp(net, x, y)
    gradient = vjp_fn(v)
    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_x, 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), expect_grad_y, 0.00001, 0.00001)


def test_vjp_multiple_inputs_multiple_outputs():
    """
    Feature: vjp
    Description: Test vjp with multiple inputs, multiple outputs.
    Expectation: Success.
    """
    def fn(x, y):
        return 2 * x, y ** 3

    x = Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    y = Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    v = Tensor([[1, 1], [1, 1]], dtype=ms.float32)
    expect_primal_0 = np.array([[2, 4], [6, 8]], dtype=np.float32)
    expect_primal_1 = np.array([[1, 8], [27, 64]], dtype=np.float32)
    expect_grad_0 = np.array([[2, 2], [2, 2]], dtype=np.float32)
    expect_grad_1 = np.array([[3, 12], [27, 48]], dtype=np.float32)

    # test fn
    primal, vjp_fn = vjp(fn, x, y)
    gradient = vjp_fn(v, v)
    assert np.allclose(primal[0].asnumpy(), expect_primal_0, 0.00001, 0.00001)
    assert np.allclose(primal[1].asnumpy(), expect_primal_1, 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0, 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1, 0.00001, 0.00001)

    # test cell
    net = WrapFnNet(fn)
    primal, vjp_fn = vjp(net, x, y)
    gradient = vjp_fn(v, v)
    assert np.allclose(primal[0].asnumpy(), expect_primal_0, 0.00001, 0.00001)
    assert np.allclose(primal[1].asnumpy(), expect_primal_1, 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_0, 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), expect_grad_1, 0.00001, 0.00001)


def test_vjp_with_weights_custom_param():
    """
    Feature: vjp
    Description: Test vjp with custom weighted Cell.
    Expectation: Success.
    """
    net = WeightedNet()
    x = Tensor(np.array([3.0, 4.0], dtype=np.float32))
    v = Tensor(np.array([1.0, 1.0], dtype=np.float32))
    weights = net.trainable_params()

    primal, vjp_fn = vjp(net, x, weights=weights)
    input_grad, param_grad = vjp_fn(v)

    expect_primal = np.array([6.0, 12.0], dtype=np.float32)
    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert np.allclose(input_grad[0].asnumpy(), np.array([2.0, 3.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(param_grad[0].asnumpy(), np.array([3.0, 4.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_with_weights_dense():
    """
    Feature: vjp
    Description: Test vjp with nn.Dense and compare with ops.grad.
    Expectation: vjp gradient matches ops.grad result.
    """
    net = nn.Dense(10, 1)
    inputs = Tensor(np.random.randn(16, 10), dtype=ms.float32)
    weights = net.trainable_params()

    # Reference: use ops.grad
    grad_fn = ms.grad(net, grad_position=None, weights=weights)
    params_gradient_grad = grad_fn(inputs)

    # vjp
    forward_res, vjp_fn = vjp(net, inputs, weights=weights)
    _, params_gradient_vjp = vjp_fn(ops.ones_like(forward_res))

    assert np.allclose(params_gradient_vjp[0].asnumpy(),
                       params_gradient_grad[0].asnumpy(), 0.00001, 0.00001)


def test_vjp_without_weights():
    """
    Feature: vjp
    Description: Test vjp without weights returns only input gradients.
    Expectation: Success.
    """
    net = WeightedNet()
    x = Tensor(np.array([3.0, 4.0], dtype=np.float32))
    v = Tensor(np.array([1.0, 1.0], dtype=np.float32))

    primal, vjp_fn = vjp(net, x)
    gradient = vjp_fn(v)
    assert np.allclose(primal.asnumpy(), np.array([6.0, 12.0], dtype=np.float32), 0.00001, 0.00001)
    assert isinstance(gradient, tuple) and len(gradient) == 1
    assert np.allclose(gradient[0].asnumpy(), np.array([2.0, 3.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_has_aux_two_outputs():
    """
    Feature: vjp
    Description: Test vjp with has_aux=True, fn returns 2 outputs.
    Expectation: Success.
    """

    def fn(x, y):
        return 2 * x + y, y ** 3

    x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]], dtype=np.float32))

    primal, vjp_fn, aux = vjp(fn, x, y, has_aux=True)
    gradient = vjp_fn(v)

    expect_primal = np.array([[3, 6], [9, 12]], dtype=np.float32)
    expect_aux = np.array([[1, 8], [27, 64]], dtype=np.float32)
    assert np.allclose(primal.asnumpy(), expect_primal, 0.00001, 0.00001)
    assert np.allclose(aux.asnumpy(), expect_aux, 0.00001, 0.00001)
    assert isinstance(gradient, tuple) and len(gradient) == 2
    assert np.allclose(gradient[0].asnumpy(), np.array([[2, 2], [2, 2]], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), np.array([[1, 1], [1, 1]], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_has_aux_three_outputs():
    """
    Feature: vjp
    Description: Test vjp with has_aux=True, fn returns 3 outputs.
    Expectation: Success.
    """

    def fn(x, y):
        return x + y, x * y, x - y

    x = Tensor(np.array([3.0, 4.0], dtype=np.float32))
    y = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    v = Tensor(np.array([1.0, 1.0], dtype=np.float32))

    primal, vjp_fn, aux = vjp(fn, x, y, has_aux=True)
    gradient = vjp_fn(v)

    assert np.allclose(primal.asnumpy(), np.array([4.0, 6.0], dtype=np.float32), 0.00001, 0.00001)
    assert isinstance(aux, tuple) and len(aux) == 2
    assert np.allclose(aux[0].asnumpy(), np.array([3.0, 8.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(aux[1].asnumpy(), np.array([2.0, 2.0], dtype=np.float32), 0.00001, 0.00001)

    assert np.allclose(gradient[0].asnumpy(), np.array([1.0, 1.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(gradient[1].asnumpy(), np.array([1.0, 1.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_has_aux_gradient_correctness():
    """
    Feature: vjp
    Description: Verify only the first output contributes to gradient.
    Expectation: Aux outputs do not affect gradient values.
    """

    def fn_with_aux(x):
        return x ** 2, x ** 3, x ** 4

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32))
    v = Tensor(np.array([1.0, 1.0], dtype=np.float32))

    primal, vjp_fn, aux = vjp(fn_with_aux, x, has_aux=True)
    gradient = vjp_fn(v)

    assert np.allclose(primal.asnumpy(), np.array([4.0, 9.0], dtype=np.float32), 0.00001, 0.00001)
    assert isinstance(aux, tuple) and len(aux) == 2
    assert np.allclose(aux[0].asnumpy(), np.array([8.0, 27.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(aux[1].asnumpy(), np.array([16.0, 81.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), np.array([4.0, 6.0]), 0.00001, 0.00001)


def test_vjp_has_aux_with_weights():
    """
    Feature: vjp
    Description: Test vjp with both has_aux=True and weights.
    Expectation: Returns (primal, vjp_fn, aux); vjp_fn returns (input_grad, param_grad).
    """

    class AuxWeightNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([2.0], dtype=ms.float32), name='w')

        def construct(self, x):
            return x * self.w, x ** 2

    net = AuxWeightNet()
    x = Tensor(np.array([3.0], dtype=np.float32))
    v = Tensor(np.array([1.0], dtype=np.float32))
    weights = net.trainable_params()

    primal, vjp_fn, aux = vjp(net, x, weights=weights, has_aux=True)
    input_grad, param_grad = vjp_fn(v)

    assert np.allclose(primal.asnumpy(), np.array([6.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(aux.asnumpy(), np.array([9.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(input_grad[0].asnumpy(), np.array([2.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(param_grad[0].asnumpy(), np.array([3.0], dtype=np.float32), 0.00001, 0.00001)



def test_vjp_fn_no_extra_forward_function():
    """
    Feature: vjp
    Description: Calling vjp_fn should NOT trigger additional forward execution.
    Expectation: Success.
    """
    call_count = [0]

    def fn(x):
        call_count[0] += 1
        return x ** 2 + x

    x = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    v = Tensor(np.array([1.0, 1.0], dtype=np.float32))

    _, vjp_fn = vjp(fn, x)
    assert call_count[0] == 1, f"Expected 1 forward call after vjp(), got {call_count[0]}"

    vjp_fn(v)
    assert call_count[0] == 1, f"Expected 1 forward call after vjp_fn(), got {call_count[0]}"


def test_vjp_fn_no_extra_forward_cell():
    """
    Feature: vjp
    Description: Calling vjp_fn on Cell should NOT trigger additional forward execution.
    Expectation: Success.
    """
    net = CounterNet()
    x = Tensor(np.array([1.0, 2.0]).astype(np.float32))
    v = Tensor(np.array([1.0, 1.0]).astype(np.float32))

    _, vjp_fn = vjp(net, x)
    assert net.count == 1, f"Expected 1 forward call after vjp(), got {net.count}"

    vjp_fn(v)
    assert net.count == 1, f"Expected 1 forward call after vjp_fn(), got {net.count}"


def test_vjp_fn_called_twice_with_extra_forward():
    """
    Feature: vjp
    Description: Calling vjp_fn twice will trigger extra forward and warning.
    Expectation: Success.
    """
    def fn(x):
        return 3 * x ** 2

    x = Tensor(np.array([1.0, 2.0]), dtype=ms.float32)
    v = Tensor(np.array([1.0, 1.0]), dtype=ms.float32)

    _, vjp_fn = vjp(fn, x)
    vjp_fn(v)
    with pytest.warns(RuntimeWarning, match="the forward pass will be recomputed"):
        vjp_fn(v)


def test_vjp_nested_second_order():
    """
    Feature: vjp
    Description: Nested vjp for second-order derivative.
    Expectation: Success.
    """

    def fn(x):
        return x ** 4 + x ** 2

    x = Tensor(np.array([2.0, 3.0]), dtype=ms.float32)
    v_inner = Tensor(np.array([1.0, 1.0]), dtype=ms.float32)
    v_outer = Tensor(np.array([2.0, 0.5]), dtype=ms.float32)

    def first_order_grad(x_inner):
        _, vjp_fn_inner = vjp(fn, x_inner)
        return vjp_fn_inner(v_inner)[0]

    primal2, vjp_fn2 = vjp(first_order_grad, x)
    grad2 = vjp_fn2(v_outer)

    # f'(x) = 4x^3 + 2x → [36, 114]
    assert np.allclose(primal2.asnumpy(), np.array([36.0, 114.0], dtype=np.float32), 0.00001, 0.00001)
    # f''(x) = 12x^2 + 2, grad2 = v_outer * f''(x) → [100, 55]
    assert np.allclose(grad2[0].asnumpy(), np.array([100.0, 55.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_nested_third_order():
    """
    Feature: vjp
    Description: Triple-nested vjp for third-order derivative.
    Expectation: Success.
    """

    def fn(x):
        return x ** 5

    x = Tensor(np.array([2.0, 3.0]).astype(np.float32))
    v = Tensor(np.array([1.0, 1.0]).astype(np.float32))

    def first_grad(x_in):
        _, vjp_fn1 = vjp(fn, x_in)
        return vjp_fn1(v)[0]

    def second_grad(x_in):
        _, vjp_fn2 = vjp(first_grad, x_in)
        return vjp_fn2(v)[0]

    primal3, vjp_fn3 = vjp(second_grad, x)
    grad3 = vjp_fn3(v)

    # f'=5x^4, f''=20x^3, f'''=60x^2
    assert np.allclose(primal3.asnumpy(), np.array([160.0, 540.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad3[0].asnumpy(), np.array([240.0, 540.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_nested_second_order_multi_input():
    """
    Feature:vjp
    Description: Nested vjp with multi-input function.
    Expectation: Success.
    """

    def fn(x, y):
        return x * y ** 2

    x = Tensor(np.array([2.0]).astype(np.float32))
    y = Tensor(np.array([3.0]).astype(np.float32))
    v = Tensor(np.array([1.0]).astype(np.float32))

    # First order w.r.t. x: d(x*y^2)/dx = y^2
    def grad_wrt_x(x_in):
        _, vjp_fn_inner = vjp(fn, x_in, y)
        return vjp_fn_inner(v)[0]  # gradient w.r.t. x

    # Second order: d(y^2)/dx = 0
    primal2, vjp_fn2 = vjp(grad_wrt_x, x)
    grad2 = vjp_fn2(v)

    assert np.allclose(primal2.asnumpy(), np.array([9.0]), 0.00001, 0.00001)
    assert np.allclose(grad2[0].asnumpy(), np.array([0.0]), 0.00001, 0.00001)


def test_grad_of_vjp():
    """
    Feature: vjp
    Description: Use ms.grad on a function that internally calls vjp.
    Expectation: Success.
    """

    def fn(x):
        return x ** 4

    v = Tensor(np.array([2.0, 0.5]), dtype=ms.float32)

    def vjp_based_grad(x_in):
        _, vjp_fn_inner = vjp(fn, x_in)
        return vjp_fn_inner(v)[0]

    x = Tensor(np.array([2.0, 3.0]), dtype=ms.float32)

    second_grad_fn = ms.grad(vjp_based_grad)
    result = second_grad_fn(x)

    # f'=4x^3, vjp_based_grad = v*4x^3, ms.grad sums then diff: v_i*12*x_i^2
    assert np.allclose(result.asnumpy(), np.array([96.0, 54.0], dtype=np.float32), 0.00001, 0.00001)


def test_vjp_of_grad():
    """
    Feature: vjp
    Description: Use vjp on a function built from ms.grad.
    Expectation: Success.
    """

    def fn(x):
        return x ** 3 + 2 * x ** 2

    def grad_fn(x_in):
        return ms.grad(fn)(x_in)

    x = Tensor(np.array([2.0, 3.0]), dtype=ms.float32)
    v = Tensor(np.array([0.5, 2.0]), dtype=ms.float32)

    # grad_fn(x) = 3x^2 + 4x → [20, 39]
    primal, vjp_fn2 = vjp(grad_fn, x)
    grad2 = vjp_fn2(v)

    assert np.allclose(primal.asnumpy(), np.array([20.0, 39.0]), 0.00001, 0.00001)
    # d(grad_fn)/dx = 6x + 4, grad2 = v * (6x+4) → [0.5*16, 2*22] = [8, 44]
    assert np.allclose(grad2[0].asnumpy(), np.array([8.0, 44.0]), 0.00001, 0.00001)


def test_value_and_grad_of_vjp():
    """
    Feature: vjp
    Description: Use ms.value_and_grad on a function that internally uses vjp.
    Expectation: Success.
    """

    def fn(x):
        return x ** 4 + x

    v = Tensor(np.array([2.0]).astype(np.float32))

    def vjp_based_grad(x_in):
        _, vjp_fn_inner = vjp(fn, x_in)
        return vjp_fn_inner(v)[0]

    x = Tensor(np.array([2.0]).astype(np.float32))
    value, grad_val = ms.value_and_grad(vjp_based_grad)(x)

    assert np.allclose(value.asnumpy(), np.array([66.0]), 0.00001, 0.00001)
    assert np.allclose(grad_val.asnumpy(), np.array([96.0]), 0.00001, 0.00001)


def test_grad_operation_of_vjp():
    """
    Feature: vjp
    Description: Use GradOperation on a Cell that calls vjp internally.
    Expectation: Success.
    """

    class VjpCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.v = Tensor(np.array([1.0]).astype(np.float32))

        def construct(self, x):
            def fn(x_in):
                return x_in ** 3

            _, vjp_fn_inner = vjp(fn, x)
            return vjp_fn_inner(self.v)[0]

    net = VjpCell()
    x = Tensor(np.array([2.0]).astype(np.float32))

    grad_op = ops.GradOperation()
    second_grad = grad_op(net)(x)

    assert np.allclose(second_grad.asnumpy(), np.array([12.0]), 0.00001, 0.00001)


def test_vjp_inside_jit():
    """
    Feature: vjp
    Description: Test vjp called inside a @jit decorated function.
    Expectation: Success.
    """
    def fn(x):
        return x ** 2 + 2 * x

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[2, 1], [0.5, 3]]).astype(np.float32))

    @jit
    def jit_vjp(inputs, vectors):
        output, grad_fn = vjp(fn, inputs)
        g = grad_fn(vectors)
        return output, g

    primal, gradient = jit_vjp(x, v)
    assert np.allclose(primal.asnumpy(), np.array([[3, 8], [15, 24]]), 0.00001, 0.00001)
    assert np.allclose(gradient[0].asnumpy(), np.array([[8, 6], [4, 30]]), 0.00001, 0.00001)


def test_vjp_inside_jit_with_has_aux():
    """
    Feature: vjp
    Description: Test vjp with has_aux=True inside a @jit decorated function.
    Expectation: Success.
    """

    def fn(x, y):
        return x ** 2 + y, x * y

    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[2, 1], [0.5, 3]]).astype(np.float32))

    @jit
    def jit_vjp_aux(x_in, y_in, v_in):
        primal, grad_fn, aux = vjp(fn, x_in, y_in, has_aux=True)
        g = grad_fn(v_in)
        return primal, g, aux

    primal, gradient, aux = jit_vjp_aux(x, y, v)
    # primal = x^2 + y → [[2, 6], [12, 20]]
    assert np.allclose(primal.asnumpy(), np.array([[2, 6], [12, 20]]), 0.00001, 0.00001)
    # aux = x*y → [[1, 4], [9, 16]]
    assert np.allclose(aux.asnumpy(), np.array([[1, 4], [9, 16]]), 0.00001, 0.00001)
    # d(x^2+y)/dx = 2x, gx = v*2x → [[4, 4], [3, 24]]
    assert np.allclose(gradient[0].asnumpy(), np.array([[4, 4], [3, 24]]), 0.00001, 0.00001)
    # d(x^2+y)/dy = 1, gy = v*1 → [[2, 1], [0.5, 3]]
    assert np.allclose(gradient[1].asnumpy(), np.array([[2, 1], [0.5, 3]]), 0.00001, 0.00001)


def test_vjp_inside_jit_with_weights():
    """
    Feature: vjp
    Description: Test vjp with weights inside a @jit decorated function.
    Expectation: Success.
    """
    net = WeightedNet()
    weights = net.trainable_params()

    x = Tensor(np.array([3.0, 4.0]).astype(np.float32))
    v = Tensor(np.array([1.0, 1.0]).astype(np.float32))

    @jit
    def jit_vjp_weights(x_in, v_in):
        primal, grad_fn = vjp(net, x_in, weights=weights)
        input_grad, param_grad = grad_fn(v_in)
        return primal, input_grad, param_grad

    primal, input_grad, param_grad = jit_vjp_weights(x, v)
    assert np.allclose(primal.asnumpy(), np.array([6.0, 12.0]), 0.00001, 0.00001)
    assert np.allclose(input_grad[0].asnumpy(), np.array([2.0, 3.0]), 0.00001, 0.00001)
    assert np.allclose(param_grad[0].asnumpy(), np.array([3.0, 4.0]), 0.00001, 0.00001)


def test_vjp_graph_mode_single_input_single_output():
    """
    Feature: vjp
    Description: Test vjp with single input in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        def fn(x):
            return 2 * x ** 2 + 3 * x

        x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        v = Tensor(np.array([[2, 3], [0.5, 1]]).astype(np.float32))

        primal, vjp_fn = vjp(fn, x)
        gradient = vjp_fn(v)

        assert np.allclose(primal.asnumpy(), np.array([[5, 14], [27, 44]]), 0.00001, 0.00001)
        # f'(x)=4x+3, grad=v*f'(x) → [[14, 33], [7.5, 19]]
        assert np.allclose(gradient[0].asnumpy(), np.array([[14, 33], [7.5, 19]]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_vjp_graph_mode():
    """
    Feature: vjp
    Description: Test vjp in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        def fn(x, y):
            return x * y + x ** 2

        x = Tensor(np.array([2.0, 3.0]).astype(np.float32))
        y = Tensor(np.array([4.0, 5.0]).astype(np.float32))
        v = Tensor(np.array([2.0, 0.5]).astype(np.float32))

        primal, vjp_fn = vjp(fn, x, y)
        gradient = vjp_fn(v)

        assert np.allclose(primal.asnumpy(), np.array([12.0, 24.0]), 0.00001, 0.00001)
        assert np.allclose(gradient[0].asnumpy(), np.array([16.0, 5.5]), 0.00001, 0.00001)
        assert np.allclose(gradient[1].asnumpy(), np.array([4.0, 1.5]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_vjp_graph_mode_has_aux():
    """
    Feature: vjp
    Description: Test vjp with has_aux=True in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        def fn(x, y):
            return 2 * x + y, y ** 2

        x = Tensor(np.array([3.0, 4.0]).astype(np.float32))
        y = Tensor(np.array([1.0, 2.0]).astype(np.float32))
        v = Tensor(np.array([2.0, 3.0]).astype(np.float32))

        primal, vjp_fn, aux = vjp(fn, x, y, has_aux=True)
        gradient = vjp_fn(v)

        assert np.allclose(primal.asnumpy(), np.array([7.0, 10.0]), 0.00001, 0.00001)
        assert np.allclose(aux.asnumpy(), np.array([1.0, 4.0]), 0.00001, 0.00001)
        assert np.allclose(gradient[0].asnumpy(), np.array([4.0, 6.0]), 0.00001, 0.00001)
        assert np.allclose(gradient[1].asnumpy(), np.array([2.0, 3.0]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_vjp_graph_mode_with_weights():
    """
    Feature: vjp
    Description: Test vjp with weights in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        net = WeightedNet()
        weights = net.trainable_params()

        x = Tensor(np.array([3.0, 4.0]).astype(np.float32))
        v = Tensor(np.array([2.0, 0.5]).astype(np.float32))

        primal, vjp_fn = vjp(net, x, weights=weights)
        input_grad, param_grad = vjp_fn(v)

        assert np.allclose(primal.asnumpy(), np.array([6.0, 12.0]), 0.00001, 0.00001)
        assert np.allclose(input_grad[0].asnumpy(), np.array([4.0, 1.5]), 0.00001, 0.00001)
        assert np.allclose(param_grad[0].asnumpy(), np.array([6.0, 2.0]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_vjp_graph_mode_nested_second_order():
    """
    Feature: vjp
    Description: Nested vjp for second-order derivative in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        def fn(x):
            return x ** 3

        x = Tensor(np.array([2.0, 3.0]).astype(np.float32))
        v = Tensor(np.array([1.0, 1.0]).astype(np.float32))

        def first_order_grad(x_inner):
            _, vjp_fn_inner = vjp(fn, x_inner)
            return vjp_fn_inner(v)[0]

        primal2, vjp_fn2 = vjp(first_order_grad, x)
        grad2 = vjp_fn2(v)

        assert np.allclose(primal2.asnumpy(), np.array([12.0, 27.0]), 0.00001, 0.00001)
        assert np.allclose(grad2[0].asnumpy(), np.array([12.0, 18.0]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_vjp_graph_mode_nested_third_order():
    """
    Feature: vjp
    Description: Triple-nested vjp for third-order derivative in graph mode.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    try:
        def fn(x):
            return x ** 5

        x = Tensor(np.array([2.0, 3.0]).astype(np.float32))
        v = Tensor(np.array([1.0, 1.0]).astype(np.float32))

        def first_grad(x_in):
            _, vjp_fn1 = vjp(fn, x_in)
            return vjp_fn1(v)[0]

        def second_grad(x_in):
            _, vjp_fn2 = vjp(first_grad, x_in)
            return vjp_fn2(v)[0]

        primal3, vjp_fn3 = vjp(second_grad, x)
        grad3 = vjp_fn3(v)

        assert np.allclose(primal3.asnumpy(), np.array([160.0, 540.0]), 0.00001, 0.00001)
        assert np.allclose(grad3[0].asnumpy(), np.array([240.0, 540.0]), 0.00001, 0.00001)
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_vjp():
    """
    Feature: vjp
    Description: Test suite for vjp.
    Expectation: Success.
    """
    test_vjp_single_input_single_output()
    test_vjp_multiple_inputs_single_output()
    test_vjp_multiple_inputs_multiple_outputs()
    test_vjp_with_weights_custom_param()
    test_vjp_with_weights_dense()
    test_vjp_without_weights()
    test_vjp_has_aux_two_outputs()
    test_vjp_has_aux_three_outputs()
    test_vjp_has_aux_gradient_correctness()
    test_vjp_has_aux_with_weights()
    test_vjp_fn_no_extra_forward_function()
    test_vjp_fn_no_extra_forward_cell()
    test_vjp_fn_called_twice_with_extra_forward()
    test_vjp_nested_second_order()
    test_vjp_nested_third_order()
    test_vjp_nested_second_order_multi_input()
    test_vjp_of_grad()
    test_vjp_inside_jit()
    test_vjp_inside_jit_with_has_aux()
    test_vjp_inside_jit_with_weights()
    test_vjp_graph_mode_single_input_single_output()
    test_vjp_graph_mode()
    test_vjp_graph_mode_has_aux()
    test_vjp_graph_mode_with_weights()
    test_vjp_graph_mode_nested_second_order()
    test_vjp_graph_mode_nested_third_order()

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_vjp_non_tensor_input_raises():
    """
    Feature: vjp
    Description: Test vjp raises TypeError when input is not Tensor.
    Expectation: TypeError is raised.
    """

    def fn(x):
        return x ** 2

    with pytest.raises(TypeError):
        vjp(fn, 1)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_vjp_non_tensor_v_raises():
    """
    Feature: vjp
    Description: Test vjp_fn raises TypeError when v is not Tensor.
    Expectation: TypeError is raised.
    """

    def fn(x):
        return x ** 2

    x = Tensor(np.array([1.0]).astype(np.float32))
    _, vjp_fn = vjp(fn, x)

    with pytest.raises(TypeError):
        vjp_fn(1)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_vjp_has_aux_single_output_raises():
    """
    Feature: vjp
    Description: Test vjp raises ValueError when has_aux=True but fn returns single output.
    Expectation: ValueError is raised.
    """

    def fn(x):
        return x ** 2

    x = Tensor(np.array([1.0]).astype(np.float32))

    with pytest.raises(ValueError):
        vjp(fn, x, has_aux=True)
        ms.runtime.synchronize()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_vjp_has_aux_type_error():
    """
    Feature: vjp
    Description: Test vjp raises TypeError when has_aux is not bool.
    Expectation: TypeError is raised.
    """

    def fn(x):
        return x ** 2

    x = Tensor(np.array([1.0]).astype(np.float32))

    with pytest.raises(TypeError):
        vjp(fn, x, has_aux="yes")
        ms.runtime.synchronize()
