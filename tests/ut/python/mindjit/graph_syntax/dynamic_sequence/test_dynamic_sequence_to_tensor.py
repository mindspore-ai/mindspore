# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test mul operation for dynamic sequence and variable integer in graph mode"""
from mindspore.common import mutable
import mindspore.common.dtype as mstype
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import Tensor
from mindspore import jit
from mindspore import context
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE)


class SeqNet(nn.Cell):
    def __init__(self):
        super(SeqNet, self).__init__()
        self.list_to_tensor = seq.ListToTensor()
        self.tuple_to_tensor = seq.TupleToTensor()

    def construct(self, x, y):
        return self.list_to_tensor(x, mstype.int32), self.tuple_to_tensor(y, mstype.int32)


def test_dynamic_length_sequence_to_tensor():
    """
    Feature: Dynamic length sequence_to_tensor operation.
    Description: Dynamic length sequence_to_tensor should return a variable tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = mutable([1, 2, 3, 4], True)
        b = mutable((1, 2, 3, 4), True)
        net = SeqNet()
        ret = net(a, b)
        return isinstance(ret[0], Tensor), isinstance(ret[1], Tensor)

    ret1, ret2 = foo()
    assert ret1
    assert ret2


def test_constant_length_sequence_to_tensor():
    """
    Feature: Constant length sequence_to_tensor operation.
    Description: Dynamic length sequence_to_tensor should return a constant tensor.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [1, 1, 2]
        b = (1, 1, 2)
        net = SeqNet()
        ret = net(a, b)
        return isinstance(ret[0], Tensor), isinstance(ret[1], Tensor)

    ret1, ret2 = foo()
    assert ret1
    assert ret2
