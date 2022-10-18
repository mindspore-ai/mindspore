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
"""
@File  : test_create_obj.py
@Author:
@Date  : 2019-06-26
@Desc  : test create object instance on parse function, eg: 'construct'
         Support class : nn.Cell ops.Primitive
         Support parameter: type is define on function 'ValuePtrToPyData'
                            (int,float,string,bool,tensor)
"""
import logging
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context, ops, dtype
from mindspore.common.api import jit
from mindspore.common import Tensor, Parameter
from mindspore.ops import operations as P
from ...ut_filter import non_graph_engine

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.softmax = nn.Softmax(0)
        self.axis = 0

    def construct(self, x):
        x = nn.Softmax(self.axis)(x)
        return x


# Test: Create Cell OR Primitive instance on construct
@non_graph_engine
def test_create_cell_object_on_construct():
    """ test_create_cell_object_on_construct """
    log.debug("begin test_create_object_on_construct")
    context.set_context(mode=context.GRAPH_MODE)
    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(np1)

    net = Net()
    output = net(input_me)
    out_me1 = output.asnumpy()
    print(np1)
    print(out_me1)
    log.debug("finished test_create_object_on_construct")


# Test: Create Cell OR Primitive instance on construct
class Net1(nn.Cell):
    """ Net1 definition """

    def __init__(self):
        super(Net1, self).__init__()
        self.add = P.Add()

    @jit
    def construct(self, x, y):
        add = P.Add()
        result = add(x, y)
        return result


@non_graph_engine
def test_create_primitive_object_on_construct():
    """ test_create_primitive_object_on_construct """
    log.debug("begin test_create_object_on_construct")
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]], np.float32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]], np.float32))

    net = Net1()
    net.construct(x, y)
    log.debug("finished test_create_object_on_construct")


# Test: Create Cell OR Primitive instance on construct use many parameter
class NetM(nn.Cell):
    """ NetM definition """

    def __init__(self, name, axis):
        super(NetM, self).__init__()
        self.name = name
        self.axis = axis
        self.softmax = nn.Softmax(self.axis)

    def construct(self, x):
        x = self.softmax(x)
        return x


class NetC(nn.Cell):
    """ NetC definition """

    def __init__(self, tensor):
        super(NetC, self).__init__()
        self.tensor = tensor

    def construct(self, x):
        x = NetM("test", 1)(x)
        return x


# Test: Create Cell OR Primitive instance on construct
@non_graph_engine
def test_create_cell_object_on_construct_use_many_parameter():
    """ test_create_cell_object_on_construct_use_many_parameter """
    log.debug("begin test_create_object_on_construct")
    context.set_context(mode=context.GRAPH_MODE)
    np1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(np1)

    net = NetC(input_me)
    output = net(input_me)
    out_me1 = output.asnumpy()
    print(np1)
    print(out_me1)
    log.debug("finished test_create_object_on_construct")


class NetD(nn.Cell):
    """ NetD definition """

    def construct(self, x, y):
        concat = P.Concat(axis=1)
        return concat((x, y))


# Test: Create Cell OR Primitive instance on construct
@non_graph_engine
def test_create_primitive_object_on_construct_use_kwargs():
    """ test_create_primitive_object_on_construct_use_kwargs """
    log.debug("begin test_create_primitive_object_on_construct_use_kwargs")
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    y = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    net = NetD()
    net(x, y)
    log.debug("finished test_create_primitive_object_on_construct_use_kwargs")


class NetE(nn.Cell):
    """ NetE definition """

    def __init__(self):
        super(NetE, self).__init__()
        self.w = Parameter(Tensor(np.ones([16, 16, 3, 3]).astype(np.float32)), name='w')

    def construct(self, x):
        out_channel = 16
        kernel_size = 3
        conv2d = P.Conv2D(out_channel,
                          kernel_size,
                          1,
                          pad_mode='valid',
                          pad=0,
                          stride=1,
                          dilation=1,
                          group=1)
        return conv2d(x, self.w)


# Test: Create Cell OR Primitive instance on construct
@non_graph_engine
def test_create_primitive_object_on_construct_use_args_and_kwargs():
    """ test_create_primitive_object_on_construct_use_args_and_kwargs """
    log.debug("begin test_create_primitive_object_on_construct_use_args_and_kwargs")
    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([1, 16, 16, 16]).astype(np.float32))
    net = NetE()
    net(inputs)
    log.debug("finished test_create_primitive_object_on_construct_use_args_and_kwargs")


# Test: Create Cell instance in construct
class SubCell(nn.Cell):
    def __init__(self, t):
        super(SubCell, self).__init__()
        self.t = t

    def construct(self):
        return ops.typeof(self.t)


class WrapCell(nn.Cell):
    def construct(self, t):
        type_0 = ops.typeof(t)
        type_1 = SubCell(t)()
        return type_0, type_1


def test_create_cell_with_tensor():
    """
    Feature: Raise exception while create Cell(that init use tensor input) in construct.
    Description: None
    Expectation: TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    t = Tensor(np.zeros((2, 2), np.float), dtype.float32)
    with pytest.raises(TypeError):
        print(WrapCell()(t))


def test_create_grad_operation():
    """
    Feature: Create instance for GradOperation.
    Description: Create instance for GradOperation, and use it.
    Expectation: No exception.
    """
    class NetInner(nn.Cell):
        def __init__(self):
            super(NetInner, self).__init__()
            self.mul = P.Mul()
            self.p = Parameter(Tensor(np.array([1.0], np.float32)), name='p')

        def construct(self, x, y):
            z = x * self.p
            out = self.mul(y, z)
            return out

    class GradNetWrtInputsAndParams(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtInputsAndParams, self).__init__()
            self.net = net
            self.params = self.net.trainable_params()

        def construct(self, x, y):
            gradient_function = ops.GradOperation(get_all=True, get_by_list=True)(self.net, self.params)
            return gradient_function(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    a = Tensor([[1, 2, 3]], dtype.float32)
    b = Tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype.float32)
    GradNetWrtInputsAndParams(NetInner())(a, b)
