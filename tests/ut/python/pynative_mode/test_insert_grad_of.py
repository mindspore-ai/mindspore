# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import ms_function
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from ....mindspore_test_framework.utils.bprop_util import bprop
from ....mindspore_test_framework.utils.debug_util import PrintShapeTypeCell, PrintGradShapeTypeCell


grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)


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

    @ms_function
    def f(x, y):
        return grad_all(stop_test)(x, y)

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

    @ms_function
    def fd(x, y):
        return grad_all(clip_test)(x, y)

    print("forward: ", f(1.1, 0.1))
    print("clip_gradient:", fd(1.1, 0.1))


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


def test_cell_assign():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)

    class GradNetWrap(nn.Cell):
        """ GradNetWrap definition """

        def __init__(self, net):
            super(GradNetWrap, self).__init__()
            self.net = net
            self.weights = mindspore.ParameterTuple(net.get_parameters())

        def construct(self, x, y):
            return grad_by_list(self.net, self.weights)(x, y)

    class Mul(nn.Cell):
        def __init__(self):
            super(Mul, self).__init__()
            self.matrix_w = mindspore.Parameter(Tensor(np.ones([2, 2], np.float32)), name="matrix_w")
            self.matrix_g = mindspore.Parameter(Tensor(np.ones([2, 2], np.float32)), name="matrix_g")
            self.get_g = P.InsertGradientOf(self.save_gradient)

        def save_gradient(self, dout):
            self.matrix_g = dout + self.matrix_g
            return dout

        def construct(self, x, y):
            z = x * self.matrix_w
            z = self.get_g(z)
            z = z * y
            return z

    input_x = Tensor(np.ones([2, 2], np.float32))
    input_y = Tensor(np.ones([2, 2], np.float32))
    GradNetWrap(Mul())(input_x, input_y)
