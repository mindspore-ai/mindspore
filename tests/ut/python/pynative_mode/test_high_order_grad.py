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
import mindspore.ops.composite as C


grad = C.GradOperation()
grad_all = C.GradOperation(get_all=True)
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE, check_bprop=False)


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
