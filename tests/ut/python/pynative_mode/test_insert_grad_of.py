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
""" test_insert_grad_of """
import numpy as np
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.api import ms_function
from ....mindspore_test_framework.utils.bprop_util import bprop
from ....mindspore_test_framework.utils.debug_util import PrintShapeTypeCell, PrintGradShapeTypeCell
from mindspore import Tensor

from mindspore import context


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


def stop_gradient(dx):
    """ stop_gradient """
    return C.zeros_like(dx)

stop = P.InsertGradientOf(stop_gradient)
def test_InsertGradientOf_1():
    """ test_InsertGradientOf_1 """
    def stop_test(x, y):
        x = stop(x)
        c = x * y
        return c

    def f(x, y):
        return C.grad_all(stop_test)(x, y)
    print("stop_gradient:", f(1, 2))

def clip_gradient(dx):
    """ clip_gradient """
    ret = dx
    if ret > 1.0:
        ret = 1.0

    if ret < 0.2:
        ret = 0.2

    return ret

clip = P.InsertGradientOf(clip_gradient)
def test_InsertGradientOf_2():
    """ test_InsertGradientOf_2 """
    def clip_test(x, y):
        x = clip(x)
        y = clip(y)
        c = x * y
        return c

    @ms_function
    def f(x, y):
        return clip_test(x, y)

    def fd(x, y):
        return C.grad_all(clip_test)(x, y)

    print("forward: ", f(1.1, 0.1))
    print("clip_gradient:", fd(1.1, 0.1))

summary = P.ScalarSummary()
def debug_gradient(dx):
    """ debug_gradient """
    dx = summary("dx: ", dx)
    return dx

debug = P.InsertGradientOf(debug_gradient)
def test_InsertGradientOf_3():
    """ test_InsertGradientOf_3 """
    def debug_test(x, y):
        x = debug(x)
        y = debug(y)
        c = x * y
        return c

    def f(x, y):
        return C.grad_all(debug_test)(x, y)
    print("debug_gradient:", f(1, 2))

def test_print_shape_type():
    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()
            self.print_shape_type = PrintShapeTypeCell()
            self.print_shape_type_gradient = PrintGradShapeTypeCell("Gradients")
        def construct(self, x, y):
            z = x * y
            self.print_shape_type("Forward", z)
            self.print_shape_type_gradient(z)
            return z
    bprop(Mul(), Tensor(np.ones([2, 2]).astype(np.float32)),
          Tensor(np.ones([2, 2]).astype(np.float32)))
