# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest
from mindspore import context
from mindspore.nn import Cell
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class NetRange3(Cell):
    def construct(self, x, y, z):
        return range(x, y, z)


class NetRange2(Cell):
    def construct(self, x, y):
        return range(x, y)


class NetRange1(Cell):
    def construct(self, x):
        return range(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seqence_make_range():
    """
    Feature: test sequence makerange op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """

    def func3(x, y, z):
        return tuple(range(x, y, z))

    def func2(x, y):
        return tuple(range(x, y))

    def func1(x):
        return tuple(range(x))

    input_x = 10
    input_y = 1000
    input_z = 31
    net_ms = NetRange3()
    fact = TupleFactory(net_ms, func3, (input_x, input_y, input_z))
    fact.forward_cmp()

    net_ms = NetRange2()
    fact = TupleFactory(net_ms, func2, (input_x, input_y))
    fact.forward_cmp()

    net_ms = NetRange1()
    fact = TupleFactory(net_ms, func1, (input_x,))
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seqence_make_range_grad():
    """
    Feature: test sequence makerange grad
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    input_x = mutable(10)
    input_y = mutable(100)
    input_z = mutable(3)
    dout = (1, 1)

    net_ms = NetRange3()
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    grad_out = grad_func(input_x, input_y, input_z, dout)
    assert grad_out == (0, 0, 0)

    net_ms = NetRange2()
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    grad_out = grad_func(input_x, input_y, dout)
    assert grad_out == (0, 0)

    net_ms = NetRange1()
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    grad_out = grad_func(input_x, dout)
    assert grad_out == (0,)
