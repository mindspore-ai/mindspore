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

import mindspore.nn as nn
from mindspore import context
from mindspore.common.api import ms_function
from mindspore.common.tensor import Tensor
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


# Test: creat CELL OR Primitive instance on construct
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


# Test: creat CELL OR Primitive instance on construct
class Net1(nn.Cell):
    """ Net1 definition """

    def __init__(self):
        super(Net1, self).__init__()
        self.add = P.Add()

    @ms_function
    def construct(self, x, y):
        add = P.Add()
        result = add(x, y)
        return result


@non_graph_engine
def test_create_primitive_object_on_construct():
    """ test_create_primitive_object_on_construct """
    log.debug("begin test_create_object_on_construct")
    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]], np.float32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]], np.float32))

    net = Net1()
    net.construct(x, y)
    log.debug("finished test_create_object_on_construct")


# Test: creat CELL OR Primitive instance on construct use many parameter
class NetM(nn.Cell):
    """ NetM definition """

    def __init__(self, name, axis):
        super(NetM, self).__init__()
        # self.relu = nn.ReLU()
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


# Test: creat CELL OR Primitive instance on construct
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
