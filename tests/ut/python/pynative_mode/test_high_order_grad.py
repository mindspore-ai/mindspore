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
""" test_high_order_grad """
from mindspore import context
from mindspore.common.api import ms_function
from mindspore.ops.composite import grad, grad_all, grad_all_with_sens


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


def single(x):
    """ single """
    ret = 3 * x * x * x
    return ret


def first_derivative(x):
    """ first_derivative """
    return grad(single)(x)


def second_derivative(x):
    """ second_derivative """
    return grad(first_derivative)(x)


@ms_function
def third_derivative(x):
    """ third_derivative """
    return grad(second_derivative)(x)


def dual(x, y):
    """ dual """
    ret = 3 * x * x * x * y * y * y
    return ret


def first_derivative_all(x):
    """ first_derivative_all """
    return grad_all(single)(x)[0]


@ms_function
def second_derivative_all(x):
    """ second_derivative_all """
    return grad_all(first_derivative_all)(x)[0]


def third_derivative_all(x):
    """ third_derivative_all """
    return grad_all(second_derivative_all)(x)[0]


# will return a tuple (d(dual)/dx, d(dual)/dy)
def first_derivative_dual(x, y):
    """ first_derivative_dual """
    return grad_all_with_sens(dual)(x, y, 1)


def second_derivative_dual(x, y):
    """ second_derivative_dual """
    grad_fn = grad_all_with_sens(first_derivative_dual)
    dfdx = grad_fn(x, y, (1, 0))[0]
    dfdy = grad_fn(x, y, (0, 1))[1]
    return dfdx, dfdy


@ms_function
def third_derivative_dual(x, y):
    """ third_derivative_dual """
    grad_fn = grad_all_with_sens(second_derivative_dual)
    dfdx = grad_fn(x, y, (1, 0))[0]
    dfdy = grad_fn(x, y, (0, 1))[1]
    return dfdx, dfdy


def if_test(x):
    """ if_test """
    if x > 10:
        return x * x
    return x * x * x


def first_derivative_if(x):
    """ first_derivative_if """
    return grad(if_test)(x)


@ms_function
def second_derivative_if(x):
    """ second_derivative_if """
    return grad(first_derivative_if)(x)


def test_high_order_grad_1():
    """ test_high_order_grad_1 """
    # 18
    assert third_derivative(2) == 18
    # 18 * y * y * y, 18 * x * x * x
    assert third_derivative_dual(4, 5) == (2250, 1152)
    # 18 * x
    assert second_derivative_all(3) == 54


def test_high_order_grad_2():
    """ test_high_order_grad_2 """
    # 2
    assert second_derivative_if(12) == 2


def test_high_order_grad_3():
    """ test_high_order_grad_2 """
    # 6 * x
    assert second_derivative_if(4) == 24
