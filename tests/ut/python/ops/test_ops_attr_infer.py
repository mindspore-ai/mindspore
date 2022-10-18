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
""" test nn ops """
import numpy as np
from numpy.random import normal
import pytest

import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops.composite import core
from mindspore.common.api import jit

from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import prim_attr_register, PrimitiveWithInfer

context.set_context(mode=context.GRAPH_MODE)


class FakeOp(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """"""

    def infer_shape(self, x, y):
        self.second_shape = y
        self.add_prim_attr("second_shape", y)
        return x

    def infer_dtype(self, x, y):
        return x


# test the normal case that should generate independent primitive because of different
# generated attributes after inference
def test_conv2d_same_primitive():
    class Conv2DSameNet(nn.Cell):
        def __init__(self):
            super(Conv2DSameNet, self).__init__()
            self.conv1 = nn.Conv2d(16, 64, (1, 41), (1, 4), "same", 0, 1, has_bias=True)
            self.conv2 = nn.Conv2d(16, 64, (1, 41), (1, 4), "same", 0, 1, has_bias=True)

        def construct(self, x, y):
            r1 = self.conv1(x)
            r2 = self.conv2(y)
            return (r1, r2)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    net = Conv2DSameNet()
    net(t1, t2)


# test free variable function list as parameter
def test_remove_and_fv_2():
    @core(loop_can_uroll=True)
    def inner_loop(x, input_data, fv_func_list):
        ret = ()
        for fv_fn in fv_func_list:
            ele = fv_fn(input_data)
            ret += (ele,)
        return ret

    @jit
    def out_loop(input1, input_data0, input_data1):
        ret = ()

        def fv_func1(y):
            return input1 * y
        def fv_func2(y):
            return input1 - y
        fv_func_list = [fv_func1, fv_func2]
        ele0 = inner_loop(input1, input_data0, fv_func_list)
        ele1 = inner_loop(input1, input_data1, fv_func_list)
        ret = (ele0, ele1)
        return ret

    input_data0 = Tensor(normal(0, 0.1, (3, 3)))
    input_data1 = Tensor(normal(0, 0.1, (3, 1)))
    input1 = Tensor(normal(0, 0.1, (3, 3)))
    out_loop(input1, input_data0, input_data1)


# test cell as high order argument
# The graph with free variables used as argument is not supported yet
# because of the limit of inference specialize system
def test_conv2d_op_with_argi_1():
    class Conv2dNet(nn.Cell):
        def __init__(self):
            super(Conv2dNet, self).__init__()

        def construct(self, op, x):
            return op(x)

    class OpsNet(nn.Cell):
        def __init__(self, net):
            super(OpsNet, self).__init__()
            self.opnet = net
            self.conv2 = nn.Conv2d(16, 64, (1, 41), (1, 4), "same", 0, 1, has_bias=True)

        def construct(self, x, y):
            conv_op = self.conv2
            a = self.opnet(conv_op, x)
            b = self.opnet(conv_op, y)
            return (a, b)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    net = OpsNet(Conv2dNet())
    net(t1, t2)


def test_conv2d_op_with_arg():
    class FackOpNet(nn.Cell):
        def __init__(self):
            super(FackOpNet, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y):
            return self.op(x, y)

    class OpNet(nn.Cell):
        def __init__(self):
            super(OpNet, self).__init__()

        def construct(self, op, x, y):
            return op(x, y)

    class OpsNet(nn.Cell):
        def __init__(self, net):
            super(OpsNet, self).__init__()
            self.opnet = net
            self.op = FackOpNet()

        def construct(self, x, y):
            op = self.op
            a = self.opnet(op, x, y)
            b = self.opnet(op, y, x)
            return (a, b)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    net = OpsNet(OpNet())
    net(t1, t2)


def test_conv2d_op_with_arg_same_input():
    class FackOpNet(nn.Cell):
        def __init__(self):
            super(FackOpNet, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y):
            return self.op(x, y)

    class OpNet(nn.Cell):
        def __init__(self):
            super(OpNet, self).__init__()

        def construct(self, op, x, y):
            return op(x, y)

    class OpsNet(nn.Cell):
        def __init__(self, net):
            super(OpsNet, self).__init__()
            self.opnet = net
            self.op = FackOpNet()

        def construct(self, x, y):
            op = self.op
            a = self.opnet(op, x, x)
            b = self.opnet(op, y, x)
            return (a, b)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    net = OpsNet(OpNet())
    net(t1, t2)


# test op with partial
def test_op_as_partial():
    class OpAsPartial(nn.Cell):
        def __init__(self):
            super(OpAsPartial, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y, z):
            partial_op = F.partial(self.op, x)
            a = partial_op(y)
            b = partial_op(z)
            return a, b

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = OpAsPartial()
    net(t1, t2, t3)


# test op with partial
def test_op_as_partial_inside():
    class OpAsPartial(nn.Cell):
        def __init__(self):
            super(OpAsPartial, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y, z):
            partial_op = F.partial(self.op, x)
            a = partial_op(y)
            b = partial_op(z)
            return a, b

    class OuterNet(nn.Cell):
        def __init__(self):
            super(OuterNet, self).__init__()
            self.net = OpAsPartial()

        def construct(self, x, y, z):
            a, b = self.net(x, y, z)
            return a, b

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = OuterNet()
    net(t1, t2, t3)


# test op with partial case 2
def test_op_as_partial_independent():
    class OpAsPartial(nn.Cell):
        def __init__(self):
            super(OpAsPartial, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y, z):
            partial_op1 = F.partial(self.op, x)
            a = partial_op1(y)
            partial_op2 = F.partial(self.op, x)
            b = partial_op2(z)
            return a, b

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = OpAsPartial()
    net(t1, t2, t3)


def test_nest_partial():
    class NestPartial(nn.Cell):
        def __init__(self):
            super(NestPartial, self).__init__()
            self.op = FakeOp()

        def construct(self, x, y, z):
            partial_op1 = F.partial(self.op)
            partial_op2 = F.partial(partial_op1, x)
            a = partial_op2(y)
            partial_op3 = F.partial(self.op)
            partial_op4 = F.partial(partial_op3, x)
            b = partial_op4(z)
            return a, b

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = NestPartial()
    net(t1, t2, t3)


# high order argument
# op and op args as network arguments
def test_op_with_arg_as_input():
    class WithOpArgNet(nn.Cell):
        def __init__(self):
            super(WithOpArgNet, self).__init__()

        def construct(self, op, x, y):
            return op(x, y)

    class OpsNet(nn.Cell):
        def __init__(self, net):
            super(OpsNet, self).__init__()
            self.opnet = net
            self.op = FakeOp()

        def construct(self, x, y, z):
            op = self.op
            a = self.opnet(op, x, z)
            b = self.opnet(op, x, y)
            return (a, b)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = OpsNet(WithOpArgNet())
    net(t1, t2, t3)


# The partial application used as argument is not supported yet
# because of the limit of inference specialize system
@pytest.mark.skip("poly in infer")
def test_partial_as_arg():
    class PartialArgNet(nn.Cell):
        def __init__(self):
            super(PartialArgNet, self).__init__()

        def construct(self, partial_op, y):
            return partial_op(y)

    class OpsNet(nn.Cell):
        def __init__(self, net):
            super(OpsNet, self).__init__()
            self.partial_net = net
            self.op = FakeOp()

        def construct(self, x, y, z):
            partial_op = F.partial(self.op, x)
            a = self.partial_net(partial_op, z)
            b = self.partial_net(partial_op, y)
            return (a, b)

    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1234]).astype(np.float32))
    net = OpsNet(PartialArgNet())
    net(t1, t2, t3)
