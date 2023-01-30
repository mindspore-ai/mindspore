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
""" test_fn_bprop """
import numpy as np
import pytest

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import context
from mindspore.common.api import jit
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.functional import vjp
from mindspore.ops.function.grad.grad_func import custom_vjp

context.set_context(mode=context.GRAPH_MODE)

grad_all = C.GradOperation(get_all=True)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_mul_add():
    """
    Features: Custom function bprop
    Description: Get the custom vjp of mul_add function.
    Expectation: No exception.
    """

    @custom_vjp
    def fn(x, y):
        return 2 * x + y

    def bprop_fn(x, y, out, dout):
        return 2 * dout, 2 * y

    fn.defbwd(bprop_fn)

    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    v = Tensor(1, dtype=ms.int32)
    _, grad_fn = vjp(fn, x, y)
    grads = grad_fn(v)
    assert grads[0] == Tensor(2, dtype=ms.int32)
    assert grads[1] == Tensor(4, dtype=ms.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_inline_mul_add():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when mul_add function is inline with other function.
    Expectation: No exception.
    """

    @custom_vjp
    def mul_add(x, y):
        return 2 * x + y

    def bprop_mul_add(x, y, out, dout):
        return 2 * dout, 2 * y

    mul_add.defbwd(bprop_mul_add)

    @jit
    def inline_mul_add(x, y):
        param = 2
        return mul_add(x, y) + x + param * y

    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    v = Tensor(1, dtype=ms.int32)
    _, grad_fn = vjp(inline_mul_add, x, y)
    grads = grad_fn(v)
    assert grads[0] == Tensor(3, dtype=ms.int32)
    assert grads[1] == Tensor(6, dtype=ms.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_with_no_bprop():
    """
    Features: Custom function bprop
    Description: Get the vjp with no bprop.
    Expectation: No exception.
    """

    def with_no_bprop(x, y):
        return 2 * x + y

    x = Tensor(1, dtype=ms.int32)
    y = Tensor(2, dtype=ms.int32)
    v = Tensor(1, dtype=ms.int32)
    _, grad_fn = vjp(with_no_bprop, x, y)
    grads = grad_fn(v)
    assert grads[0] == Tensor(2, dtype=ms.int32)
    assert grads[1] == Tensor(1, dtype=ms.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_bprop_in_fn_2():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when bprop in fn_2.
    Expectation: No exception.
    """

    def fn_1(x, y):
        relu = P.ReLU()
        return relu(x)

    @custom_vjp
    def fn_2(x, y):
        grads = grad_all(fn_1)(x, y)
        return fn_1(x, y), grads[0], grads[1]

    def bprop_fn_2(x, y, out, dout):
        grads = grad_all(fn_1)(x, y)
        return out[1], grads[1]

    fn_2.defbwd(bprop_fn_2)

    @jit
    def fn_3(x, y):
        return fn_2(x, y)

    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    x = Tensor(np.ones([2, 2]).astype(np.float32))
    y = Tensor(np.ones([2, 2]).astype(np.float32))

    _, grad_fn = vjp(fn_3, x, y)
    grads = grad_fn(v, v, v)
    assert (grads[0].asnumpy() == np.ones([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.zeros([2, 2]).astype(np.float32)).all()


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_bprop_in_fn3():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when bprop in fn_3.
    Expectation: No exception.
    """

    def fn_1(x, y):
        relu = P.ReLU()
        return relu(x)

    @custom_vjp
    def fn_2(x, y):
        grads = grad_all(fn_1)(x, y)
        return fn_1(x, y), grads[0], grads[1]

    def bprop_fn_2(x, y, out, dout):
        grads = grad_all(fn_1)(x, y)
        return out[1], grads[1]

    fn_2.defbwd(bprop_fn_2)

    @custom_vjp
    def fn_3(x, y):
        return fn_2(x, y)

    def bprop_fn_3(x, y, out, dout):
        return x + y + y + out[0], x + x + y + y + dout[0]

    fn_3.defbwd(bprop_fn_3)

    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    x = Tensor(np.ones([2, 2]).astype(np.float32))
    y = Tensor(np.ones([2, 2]).astype(np.float32))
    _, grad_fn = vjp(fn_3, x, y)
    grads = grad_fn(v, v, v)
    assert (grads[0].asnumpy() == np.array([[4, 4], [4, 4]]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([[5, 5], [5, 5]]).astype(np.float32)).all()


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_one_input_bprop():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when the function has only one input.
    Expectation: No exception.
    """

    def bprop_fn(x, out, dout):
        return (5 * x,)

    @custom_vjp
    def fn(x):
        op = P.ReLU()
        return op(x)

    fn.defbwd(bprop_fn)
    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    _, grad_fn = vjp(fn, input1)
    grads = grad_fn(v)
    assert (grads[0].asnumpy() == np.array([5, 5]).astype(np.float32)).all()


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_inline_bprop_two_input():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when the function has two inputs.
    Expectation: No exception.
    """

    def fn_1(x, y):
        return x * y

    @custom_vjp
    def fn_2(x, y):
        grads = grad_all(fn_1)(x, y)
        return fn_1(x, y), grads[0], grads[1]

    def bprop_fn_2(x, y, out, dout):
        grads = grad_all(fn_1)(x, y)
        return grads[0] * 2, grads[1] * 2

    fn_2.defbwd(bprop_fn_2)

    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    input2 = Tensor(np.ones([2, 2]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    _, grad_fn = vjp(fn_2, input1, input2)
    grads = grad_fn(v, v, v)
    assert (grads[0].asnumpy() == np.array([2, 2]).astype(np.float32)).all()
    assert (grads[1].asnumpy() == np.array([2, 2]).astype(np.float32)).all()
    assert len(grads) == 2


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_custom_vjp_inline_bprop_multi_input():
    """
    Features: Custom function bprop
    Description: Get the custom vjp of hybrid bprop function.
    Expectation: No exception.
    """

    def tensor_mul(x, y):
        return x * y

    @custom_vjp
    def two_input(x, y):
        op = P.Mul()
        return op(x, y)

    def two_input_bprop(x, y, out, dout):
        return 5 * x, 8 * y

    two_input.defbwd(two_input_bprop)

    def two_input_1(x, y):
        op = P.Mul()
        x = 1 + x
        return op(x, y)

    @custom_vjp
    def two_input_2(x, y):
        op = P.Mul()
        return op(x, y)

    def two_input_2_bprop(x, y, out, dout):
        return 5 * x, 8 * y

    two_input_2.defbwd(two_input_2_bprop)

    def inline_mutil_two_input(x, y):
        output = (
            two_input(x, y) + tensor_mul(x, y) + two_input_1(x, y) + two_input_2(x, y)
        )
        return output

    input1 = Tensor(np.ones([2, 2]).astype(np.float32))
    input2 = Tensor(np.ones([2, 2]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    _, grad_fn = vjp(inline_mutil_two_input, input1, input2)
    grads = grad_fn(v)
    assert (
        grads[0].asnumpy() == np.array([[12, 12], [12, 12]]).astype(np.float32)
    ).all()
    assert (
        grads[1].asnumpy() == np.array([[19, 19], [19, 19]]).astype(np.float32)
    ).all()
    assert len(grads) == 2


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_custom_vjp_fn_with_net():
    """
    Features: Custom function bprop
    Description: Get the custom vjp when the function contains Cell.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name="z")

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    def fn_bprop(x, y, out, dout):
        dx = x + x
        dy = y + y
        return dx, dy

    @custom_vjp
    def fn(x, y):
        net = Net()
        return net(x, y)

    fn.defbwd(fn_bprop)

    def grad_net(x, y):
        grad_f = grad_all(fn)
        return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor(
        [[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32
    )
    out = grad_net(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8], [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2], [0.2, 0.4, 2.6], [4.2, 2.4, 6.6]]).astype(
        np.float32
    )
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_custom_vjp_forward_net_call_fn():
    """
    Feature: Custom function bprop
    Description: Get the custom vjp when the forward net call the function.
    Expectation: No exception.
    """

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name="z")

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    @custom_vjp
    def fn(x, y):
        net = Net1()
        return net(x, y)

    def fn_bprop(x, y, out, dout):
        dx = x + x
        dy = y + y
        return dx, dy

    fn.defbwd(fn_bprop)

    class Net(nn.Cell):
        def construct(self, x, y):
            return fn(x, y)

    def grad_net(x, y):
        grad_f = grad_all(Net())
        return grad_f(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor(
        [[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32
    )
    out = grad_net(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8], [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2], [0.2, 0.4, 2.6], [4.2, 2.4, 6.6]]).astype(
        np.float32
    )
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dy)
