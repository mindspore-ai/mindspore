# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_framstruct """
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.utils.check_gradient import (
    ms_function, check_jacobian, Tensor, NNGradChecker,
    OperationGradChecker, check_gradient)

context.set_context(mode=context.PYNATIVE_MODE)


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


grad_all = C.GradOperation(get_all=True)
grad_by_list = C.GradOperation(get_by_list=True)


@ms_function
def while_upper_bound(upper):
    rval = 2
    while rval < upper:
        rval = rval * rval
    return rval


def test_while_upper_bound():
    res = while_upper_bound(10)
    assert res == 16


@ms_function
def while_lower_bound(lower):
    """ t_while """
    rval = lower
    while rval < 100:
        rval = rval * rval
    return rval


def test_while_lower_bound():
    res = while_lower_bound(2)
    assert res == 256


@ms_function
def dynamic_make_tuple(x, lower, upper):
    out = ()
    i = lower
    while i < upper:
        out = out + (x,)
        i = i + 1
    return out


def test_dynamic_make_tuple():
    assert dynamic_make_tuple(2, 1, 5) == (2, 2, 2, 2)


def test_make_tuple():
    # Statically recursively creating static type is valid in mindspore.
    @ms_function
    def make_tuple(x):
        out = ()
        for i in range(3):
            out = out + (x,)
        return out

    res = make_tuple(5)
    assert res == (5, 5, 5)


@ms_function
def add(x, y):
    """ add """
    return x + y


def mul(x, y):
    """ mul """
    return x * y


def add_mul(x, y):
    """ add_mul """
    return (x + y) * y


def mainf(x, y):
    """ mainf """
    return grad_all(mul)(x, y)


def grad_add_mul(x, y):
    """ grad_add_mul """
    return grad_all(add_mul)(x, y)


@ms_function
def sub(x, y):
    """ sub """
    return x - y


# pylint: disable=using-constant-test
@ms_function
def if_always_true(x):
    """ if_always_true """
    if True:
        return x
    else:
        return 0


def test_add():
    """ test_add """
    res = add(2.5, 3)
    assert res == 5.5


def test_sub():
    """ test_sub """
    res = sub(3.5, 3)
    assert res == 0.5


@non_graph_engine
def test_if_always_true():
    """ test_if_always_true """
    res = if_always_true(1)
    assert res == 1


@non_graph_engine
def test_f():
    """ test_f """
    res = mainf(Tensor(3, dtype=ms.int32), Tensor(2, dtype=ms.int32))
    assert res == (2, 3)


@non_graph_engine
def test_grad_add_mul():
    """ test_grad_add_mul """
    res = grad_add_mul(Tensor(3, dtype=ms.int32), Tensor(2, dtype=ms.int32))
    assert res == (2, 7)


def f(x):
    if x > 0:
        return f(x - 1)
    return x


@ms_function
def list_subscript():
    """ list_subscript """
    x = [1, 2, 3]
    return x[0] * x[1]


def test_list_subscript():
    """ test_list_subscript """
    res = list_subscript()
    assert res == 2


@ms_function
def ms_infer_for(xs, y):
    """ ms_infer_for """
    rval = y
    for x in xs:
        rval = rval + x
    return rval


def test_infer_for():
    """ test_infer_for """
    t = (1, 2, 3)
    y = 4
    res = ms_infer_for(t, y)
    assert res == 10


@ms_function
def if_construct(a, b):
    z = a
    if a > b:
        z = a + b
    else:
        z = a * b
    if z > b:
        return z - a
    else:
        return a - b


def test_if_construct():
    """ test_if_construct """
    res = if_construct(3, 6)
    assert res == 15


@ms_function
def if_scalar(a, b):
    """ if_abstract """
    if a:
        return a
    return b


def test_if_scalar1():
    """ test_if_abstract """
    res = if_scalar(3, 6)
    assert res == 3


def test_if_scalar2():
    """ test_if_abstract """
    res = if_scalar(0, 6)
    assert res == 6


@ms_function
def if_tensor(a, b):
    c = a
    if a < b:
        c = a + a
        if c < b:
            c = a + c
        else:
            c = a + b
    else:
        c = b + b
    out = c + c
    return out


def test_if_tensor():
    res = if_tensor(Tensor(np.ones([1]).astype(np.int32)), Tensor(np.ones([1]).astype(np.int32)))
    assert res == Tensor(np.ones([1]).astype(np.int32) * 4)


def rec(x):
    """ rec """
    if x > 0:
        return rec(x - 1)
    return x


def test_me_rec():
    """ test_me_rec """
    res = rec(10)
    assert res == 0


def t2_while(x, y):
    out = y - x
    i = 0
    while i < 10:
        out = mul(x, y)
        i = i + 1
    return out


def test_while2():
    res = t2_while(2, 3)
    assert res == 6


def if_test(a, b):
    """ if_test """
    if a > b:
        return 3 * a
    return 2 * b


def grad_if(x, y):
    """ grad_if """
    return grad_all(if_test)(x, y)


def test_grad_if():
    """ test_grad_if """
    assert grad_if(Tensor(5, dtype=ms.int32), Tensor(4, dtype=ms.int32)) == (3, 0)


class ConvNet(nn.Cell):
    def __init__(self):
        super(ConvNet, self).__init__()
        out_channel = 16
        kernel_size = 3
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="pad",
                             pad=0,
                             stride=1,
                             dilation=2,
                             group=1)
        self.w = Parameter(Tensor(np.ones([16, 16, 3, 3]).astype(np.float32)), name='w')

    def construct(self, x):
        return self.conv(x, self.w)


conv = ConvNet()
c1 = Tensor([2], mstype.float32)
c2 = Tensor([10], mstype.float32)
c3 = Tensor([1], mstype.float32)


@ms_function
def t1_while(x, y, z):
    out = x
    i = c1
    while i < c2:
        out = out + conv(z)
        i = i + c3
    out = out + out
    return out


def test_while_net():
    y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
    x = Tensor(np.ones([1, 16, 12, 12]).astype(np.float32))
    z = Tensor(np.ones([1, 16, 16, 16]).astype(np.float32))
    res = t1_while(x, y, z)
    assert np.all(res.asnumpy() == np.ones([1, 16, 12, 12]).astype(np.float32) * 2306.0)


@ms_function
def if_while(a, b, x, z):
    c = a
    i = c1
    out = x
    if a < b:
        c = a + a
        while i < c2:
            out = out + conv(z)
            i = i + c3
    else:
        c = b + b
    out = c + c
    return out


def test_if_while():
    x = Tensor(np.random.randn(1, 16, 12, 12).astype(np.float32))
    z = Tensor(np.random.randn(1, 16, 16, 16).astype(np.float32))
    res = if_while(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)), x, z)
    assert np.all(res.asnumpy() == np.ones([64, 10]).astype(np.float32) * 4.0)


def _while(x):
    """ _while """
    ret = x * x
    i = 2
    while i <= 3:
        ret = ret * i
        i = i + 1
    return ret


def grad_while(x):
    """ grad_while """
    return grad_all(_while)(x)


def test_grad_while():
    """ test_grad_while """
    assert grad_while(Tensor(5, dtype=ms.int32)) == (60,)


@ms_function
def factorial(n):
    """ factorial """
    if n == 0:
        return 1
    return n * factorial(n - 1)


def test_factorial():
    res = factorial(3)
    assert res == 6


@ms_function
def factorial2(n):
    """ factorial """
    if n != 0:
        return n * factorial2(n - 1)
    elif n == 1:
        return 1 * factorial2(n - 1)
    else:
        return 1


def test_factorial2():
    res = factorial2(3)
    assert res == 6


@ms_function
def foo(n):
    if n <= 1:
        if n == 1:
            return foo(n - 1)
        else:
            return 1
    else:
        return foo(n - 1)


def test_foo():
    res = foo(5)
    assert res == 1


@ms_function
def double_nested_loop(x):
    i = 0
    s = 0
    while i < x:
        j = 0
        i = i + 1
        while j < 3:
            j = j + 1
            s = s + j
    return s


def test_nested_loop():
    res = double_nested_loop(3)
    assert res == 18


@ms_function
def double_nested_loop2(x):
    s = 0
    for i in range(x):
        for j in range(3):
            s = s + j
    return s


def test_nested_loop2():
    res = double_nested_loop(1)
    assert res == 6


def _for(x):
    """ _for """
    ret = x * x
    for i in (2, 3):
        ret = ret * i
    return ret


@ms_function
def grad_for(x):
    """ grad_for """
    return grad_all(_for)(x)


@ms_function
def try_tail(x):
    """ try_tail """
    return C.tail(x)


@non_graph_engine
def test_tail():
    """ test_tail """
    try_tail((0, 1, 2, 3))


@ms_function
def zero_like_tensor(x):
    """ zero_like_tensor """
    return C.zeros_like(x)


def test_zeros():
    """ test_zeros """
    x = Tensor(np.ones([2, 3]).astype(np.int32))
    res = zero_like_tensor(x)
    assert np.all(res.asnumpy() == np.zeros([2, 3]).astype(np.int32))


@ms_function
def arithmetic_simplify_01(x, y):
    """ arithmetic_simplify_01 """
    return C.zeros_like(x) * y


def test_arithmetic_simplify_01():
    """ test_arithmetic_simplify_01 """
    x = Tensor(np.ones([2, 3]).astype(np.int32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_01(x, y)
    expect = np.zeros([2, 3]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_02(x, y):
    """ arithmetic_simplify_02 """
    return C.ones_like(x) * y


def test_arithmetic_simplify_02():
    """ test_arithmetic_simplify_02 """
    x = Tensor(np.ones([2, 3]).astype(np.int32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_02(x, y)
    expect = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_03(x, y):
    """ arithmetic_simplify_03 """
    return x * C.ones_like(y)


def test_arithmetic_simplify_03():
    """ test_arithmetic_simplify_03 """
    x = Tensor(np.ones([2, 3]).astype(np.int32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_03(x, y)
    expect = np.ones([2, 3]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_04(x):
    """ arithmetic_simplify_04 """
    return x + 0


def test_arithmetic_simplify_04():
    """ test_arithmetic_simplify_04 """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_04(x)
    expect = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_05(x):
    """ arithmetic_simplify_05 """
    return x * 1


def test_arithmetic_simplify_05():
    """ test_arithmetic_simplify_05 """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_05(x)
    expect = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_06(x):
    """ arithmetic_simplify_06 """
    return x * 2 * 5


def test_arithmetic_simplify_06():
    """ test_arithmetic_simplify_06 """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_06(x)
    expect = np.array([[10, 20, 30], [40, 50, 60]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_07(x):
    """ arithmetic_simplify_07 """
    return (x + 1) * 2 * 5


def test_arithmetic_simplify_07():
    """ test_arithmetic_simplify_07 """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    res = arithmetic_simplify_07(x)
    expect = np.array([[20, 30, 40], [50, 60, 70]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


@ms_function
def arithmetic_simplify_08(x, y):
    """ arithmetic_simplify_08 """
    return 1 * x * 1 * 1 + 1 * 0 * 1 + 0 + y * 1


def test_arithmetic_simplify_08():
    """ test_arithmetic_simplify_08 """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
    y = Tensor(np.ones([2, 3]).astype(np.int32))
    res = arithmetic_simplify_08(x, y)
    expect = np.array([[2, 3, 4], [5, 6, 7]]).astype(np.int32)
    assert np.all(res.asnumpy() == expect)


def test_GradCheckerPrimitive():
    """ test_GradCheckerPrimitive """
    matmul = P.MatMul()

    def prim_f(x, y):
        return matmul(x, y)

    check_gradient(prim_f, Tensor(np.array([[0.65, 0.8, 0.8]], np.float32)),
                   Tensor(np.array([[0.1], [0.2], [-.1]], np.float32)),
                   grad_checker_class=OperationGradChecker, sampling_times=2)


def test_NNGradChecker():
    """ test_NNGradChecker """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.dense = nn.Dense(10, 10)

        def construct(self, x):
            out = self.dense(x)
            return out

    check_gradient(Net(), Tensor(np.random.rand(1, 10).astype(np.float32)),
                   delta=1e-3,
                   max_error=1e-3,
                   grad_checker_class=NNGradChecker, sampling_times=3)


def test_OperationGradChecker():
    """ test_OperationGradChecker """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    check_gradient(Net(), Tensor(np.array([[0.65, 0.8, 0.8]], np.float32)),
                   Tensor(np.array([[0.1], [0.2], [-.1]], np.float32)), grad_checker_class=OperationGradChecker,
                   input_selector=[1], sampling_times=2)


def test_OperationJacobianChecker():
    """ test_OperationJacobianChecker """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return x, out

    check_jacobian(Net(), Tensor(np.array([[0.65, 0.8, 0.8], [0.1, 0.2, 0.3]], np.float32)),
                   Tensor(np.array([[0.1, 0.3], [0.2, 0.2], [-.1, 0.4]], np.float32)),
                   grad_checker_class=OperationGradChecker, input_selector=[0],
                   output_selector=[0])


def test_NNJacobianChecker():
    """ test_NNJacobianChecker """

    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.dense = nn.Dense(10, 10)

        def construct(self, x):
            out = self.dense(x)
            return out, x

    check_jacobian(Net(), Tensor(np.random.rand(1, 10).astype(np.float32)),
                   delta=1e-3,
                   max_error=1e-7,
                   grad_checker_class=NNGradChecker,
                   input_selector=[1],
                   output_selector=[0])


def multi_outputs(x, y):
    z = x + y
    return 2 * z, 2 * z


@ms_function
def while_sp(x, y, z):
    out = x
    i = c3
    while i < c2:
        out = mul(x, out)
        i = i + c3
    return out


def test_while_sp():
    y = Tensor(np.ones([1, 3]).astype(np.float32))
    z = Tensor(np.ones([1, 3]).astype(np.float32))
    x = Tensor(np.ones([1, 3]).astype(np.float32) * 2.0)
    res = while_sp(x, y, z)
    assert np.all(res.asnumpy() == np.ones([1, 3]).astype(np.float32) * 1024.0)


def grad_refactor_simple_1(x, y):
    """ add """
    return x * x + 2 * y


def test_grad_refactor_simple_1():
    assert grad_all(grad_refactor_simple_1)(Tensor(2, dtype=ms.int32), Tensor(1, dtype=ms.int32)) == (4, 2)


def grad_refactor_simple_2(x, y, z):
    """ add """
    return x * y + z + x * y * z + x + x * y


def test_grad_refactor_simple_2():
    x = Tensor(2, dtype=ms.int32)
    y = Tensor(3, dtype=ms.int32)
    z = Tensor(0, dtype=ms.int32)
    assert grad_all(grad_refactor_simple_2)(x, y, z) == (7, 4, 7)


def grad_refactor_1(a, b):
    """ if_test """

    def inner(x, y):
        return x * y

    return inner(a, b)


def test_grad_refactor_1():
    assert grad_all(grad_refactor_1)(Tensor(2, dtype=ms.int32), Tensor(3, dtype=ms.int32)) == (3, 2)


def grad_refactor_2(a, b):
    """ if_test """

    def inner(x):
        return x * b

    return inner(b) * inner(a)


def test_grad_refactor_2():
    assert grad_all(grad_refactor_2)(Tensor(2, dtype=ms.int32), Tensor(3, dtype=ms.int32)) == (27, 54)


def grad_refactor_3(a):
    """ if_test """
    if a > 3:
        return 0
    return 3 * a


def grad_refactor_4(a):
    """ if_test """
    if a > 3:
        return 3 * a
    return 0


def test_grad_refactor_4():
    assert grad_all(grad_refactor_4)(Tensor(4, dtype=ms.int32)) == (3,)


def grad_refactor_5(a):
    """ if_test """
    if a > 3:
        return 1
    return a


def grad_refactor_6(a, b):
    """ if_test """
    if a > b:
        return 3 * a + b
    return 2 * b * a


def test_grad_refactor_6():
    assert grad_all(grad_refactor_6)(Tensor(3, dtype=ms.int32), Tensor(2, dtype=ms.int32)) == (3, 1)


def grad_refactor_while(x):
    """ grad_refactor_while """
    rval = x
    while rval < 4:
        rval = rval * rval
    return rval


def grad_refactor__while_1(x):
    """ _while """
    ret = x * x
    i = 2
    while i <= 3:
        ret = ret * i
        i = i + 1
    return ret


def test_grad_refactor_10():
    """ test_grad_while """
    assert grad_all(grad_refactor__while_1)(Tensor(5, dtype=ms.int32)) == (60,)


def test_grad_refactor_11():
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return x * y * y

    net = Net()
    grad_all(net)(Tensor(np.ones([2]).astype(np.float32)), Tensor(np.ones([2]).astype(np.float32)))


def test_grad_refactor_12():
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            return x * self.z * y

    net = Net()
    grad_all(net)(Tensor(np.ones([2]).astype(np.float32)), Tensor(np.zeros([2]).astype(np.float32)))


def test_grad_refactor_13():
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.z = Parameter(Tensor(np.ones([2]).astype(np.float32)), name='z')

        def construct(self, x, y):
            return x * self.z * y

    net = Net()
    weights = ParameterTuple(net.trainable_params())
    grad_by_list(net, weights)(Tensor(np.ones([2]).astype(np.float32)), Tensor(np.zeros([2]).astype(np.float32)))


def grad_refactor_14(a, b):
    """ if_test """

    def inner1(x):
        return x * b

    def inner2(x):
        return a * b

    def inner3(x):
        if x > 2:
            return a
        return b

    return inner1(b) + inner2(a) + inner3(a)


# pylint: disable=using-constant-test
class IfDeferInline(nn.Cell):
    def __init__(self, mul_size):
        super().__init__()
        self.mul_weight = Tensor(np.full(mul_size, 0.6, dtype=np.float32))
        self.mul = P.Mul()

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        if True:
            x = x
        return x


def test_grad_if_defer_inline():
    """ test_grad_if_defer_inline """
    network = IfDeferInline([128, 96])
    network.add_flags(defer_inline=False)
    inp = Tensor(np.ones([128, 96]).astype(np.float32))
    grads = grad_all(network)(inp)
    assert np.all(grads[0].asnumpy() == np.full([128, 96], 0.6, dtype=np.float32))


def test_dict_const():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.res = {'1': 10}

        def construct(self):
            return self.res

    Net()()
