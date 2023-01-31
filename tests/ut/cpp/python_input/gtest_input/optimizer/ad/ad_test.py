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
import numpy as np

import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import functional as F
from tests.ut.python.model.resnet import resnet50


scala_add = F.scalar_add


def scalar_add(x, y):
    """Implement `scalar_add`."""
    return x + y


def scalar_mul(x, y):
    """Implement `scalar_mul`."""
    return x * y


# Test:common function
def test_null(x, y):
    return scala_add(10.0, 28.0 / 43.0)


def test_grad_add(x, y):
    return scala_add(x, y)


def test_grad_expr(x, y):
    return x ** 3.0 * y ** 4.0


def test_constant(x):
    return 18.0 * x


def test_dup_args_in_call(x):
    """The naive gradient update rule fails when a function's arguments
    contain the same variable more than once."""
    return x * x


def test_quadruple_args_in_call(x):
    """Test that duplicated arguments still cause no problem even if
    there are four of them."""

    def g(a, b, c, d):
        return a * b * c * d

    return g(x, x, x, x)


def test_tuples(x, y):
    tup = scala_add(x, y), x * y
    z = scala_add(tup[0], tup[1])
    return z


def test_hof(a, b):
    """Test higher order functions."""

    def f(g, x):
        return g(x) * g(scala_add(x, 10.0))

    def g(x):
        return x * b

    return scala_add(f(g, a), f(g, b))


def test_hof_tup(a, b):
    """Test higher order functions."""

    def f(gh, x, y):
        g, h = gh
        return scalar_mul(g(x, y), h(x, y))

    return f((scalar_add, scalar_mul), a, b)


def test_simple_closure(a, b):
    """Test some trivial closures."""

    def f():
        return a + 1.0

    def g():
        return b + 2.0

    return f() * g()


def test_closure(a):
    """This is the closure test in the paper."""

    def x1(b):
        def x4(c):
            return c * b

        return x4

    x2 = x1(a)
    x3 = x2(1.0)
    return x3


def test_if(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a
    return b


def test_if2(a, b):
    if a > b:
        return a * a
    return b + b


def test_fact(x):
    def fact(n):
        if n <= 1:
            return 1
        return n * fact(n - 1)

    return fact(x)


def test_while(x):
    rval = x
    while rval < 100:
        rval = rval * rval
    return rval


def test_while_2(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval = rval + y
        x = x - z
    return rval


def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v


def test_nested_closure(x):
    a = x * x
    b = x + 5

    def f():
        def g():
            return a + b

        def h():
            return a * b

        return g if x < 0 else h

    return f()()


def test_functions_in_tuples(x, y):
    tup = scalar_add, scalar_mul
    f, g = tup
    return f(x, y) + g(x, y)


def test_closures_in_tuples(x, y):
    def f():
        return x * y

    def g():
        return scala_add(x, y)

    tup = f, g
    ff, gg = tup
    return scala_add(ff(), gg())


# tensor test
def test_tensor_add(x, y):
    t1 = Tensor(np.ones(x))
    t2 = Tensor(np.zeros(y), ms.float32)
    return t1 + t2


def test_tensor_set_type(x):
    t = Tensor(x)
    t.set_dtype(ms.float32)
    return t


def test_tensor_mul(x, y):
    x = Tensor(x)
    y = Tensor(y)
    z = x * y

    return z


def test_tensor_sub(x, y):
    x = Tensor(x)
    y = Tensor(y)
    z = x - y
    return z


relu = Primitive('relu')


# Extension test
def test_ops_fn(x):
    foo = relu(x)
    return foo


def test_clone_simple(x, y):
    a = x * x
    b = y * y
    c = a + b
    return c


def test_more_closure(a, b):
    """Test some trivial closures."""
    z = 1

    def f():
        return a + z

    def g():
        return b + 2.0

    return f() * g()


def test_more_hof(a, b):
    """Test higher order functions."""

    def f(g, h, x):
        return g(x) * h(x) * g(x + 10.0)

    def g(x):
        return x * b

    def h(x):
        return x * a

    return scala_add(f(g, h, a), f(g, h, b))


def test_constant_output(x, y):
    return 1


# test resnet
def test_resnet_construct(x):
    # not right model to import
    network = resnet50()
    return network.construct(x)
