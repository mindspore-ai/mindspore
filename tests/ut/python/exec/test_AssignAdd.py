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
test assign add
"""
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.AssignAdd = P.AssignAdd()
        self.inputdata = Parameter(initializer(1, [1], ms.int64), name="global_step")
        print("inputdata: ", self.inputdata)

    def construct(self, x):
        out = self.AssignAdd(self.inputdata, x)
        return out


@non_graph_engine
def test_AssignAdd_1():
    """test AssignAdd 1"""
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    x = Tensor(np.ones([1]).astype(np.int64) * 100)

    print("MyPrintResult dataX:", x)
    result = net(x)
    print("MyPrintResult data::", result)
    expect = np.ones([1]).astype(np.int64) * 101
    diff = result.asnumpy() - expect

    print("MyPrintExpect:", expect)
    print("MyPrintDiff:", diff)
    error = np.ones(shape=[1]) * 1.0e-3
    assert np.all(diff < error)


@non_graph_engine
def test_AssignAdd_2():
    """test AssignAdd 2"""
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    x = Tensor(np.ones([1]).astype(np.int64) * 102)

    print("MyPrintResult dataX:", x)
    result = net(x)
    print("MyPrintResult data::", result.asnumpy())
    expect = np.ones([1]).astype(np.int64) * 103
    diff = result.asnumpy() - expect

    print("MyPrintExpect:", expect)
    print("MyPrintDiff:", diff)
    error = np.ones(shape=[1]) * 1.0e-3
    assert np.all(diff < error)


class AssignAddNet(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(AssignAddNet, self).__init__()
        self.AssignAdd = P.AssignAdd()
        self.inputdata = Parameter(initializer(1, [1], ms.float16), name="KIND_AUTOCAST_SCALAR_TO_TENSOR")
        self.one = 1

    def construct(self, ixt):
        z1 = self.AssignAdd(self.inputdata, self.one)
        return z1


@non_graph_engine
def test_assignadd_scalar_cast():
    net = AssignAddNet()
    x = Tensor(np.ones([1]).astype(np.int64) * 102)
    # _executor.compile(net, 1)
    result = net(x)
