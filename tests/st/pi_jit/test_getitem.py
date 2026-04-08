# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test BINARY_SUBSCR"""

from dataclasses import dataclass
import pytest

from mindspore import Tensor, context, jit, nn
from mindspore._c_expression import get_code_extra

from .share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config

jit_cfg = {"compile_with_try": False}


def assert_no_graph_break(func):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Layer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 0

    def construct(self, x: Tensor):
        return x + 1


class SideEffectLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 0

    def construct(self, x: Tensor):
        self.a += 1  # getitem primitive infer will fail, if Cell has store_attr side-effect.
        return x + self.a


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('cell', [Layer, SideEffectLayer])
def test_CellList_getitem(cell):
    """
    Feature: Test getitem.
    Description: Test CellList getitem.
    Expectation: No exception, no graph break.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layers = nn.CellList([cell()])

        def construct(self, x: Tensor):
            return self.layers[0](x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    x = Tensor([1, 2, 3])
    o1 = net(x)

    net.layers[0].a = 0
    net.construct = pi_jit_with_config(net.construct, jit_config=jit_cfg)
    o2 = net(x)

    match_array(o1, o2)
    assert_no_graph_break(net.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('cell', [Layer, SideEffectLayer])
def test_list_of_cell_getitem(cell):
    """
    Feature: Test getitem.
    Description: Test getitem on list of Cell.
    Expectation: No exception, no graph break.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layers = [cell()]

        def construct(self, x: Tensor):
            return self.layers[0](x)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    x = Tensor([1, 2, 3])
    o1 = net(x)

    net.layers[0].a = 0
    net.construct = pi_jit_with_config(net.construct, jit_config=jit_cfg)
    o2 = net(x)

    match_array(o1, o2)
    assert_no_graph_break(net.construct)


@dataclass
class MyData:
    x: int = 1
    y: str = 'b'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_list_of_custom_class_getitem():
    """
    Feature: Test getitem.
    Description: Test getitem on list of custom class.
    Expectation: No exception, no graph break.
    """

    def fn(data_list: list, x: Tensor):
        return x + data_list[0].x

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, 2, 3])
    lst = [MyData()]
    o1 = fn(lst, x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(lst, x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


class MyClass:
    def __init__(self):
        self._data = {}

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_custom_class_getitem():
    """
    Feature: Test getitem.
    Description: Test getitem on custom class.
    Expectation: No exception, no graph break.
    """

    def fn(data: MyClass, x: Tensor):
        return x + data[0]

    data = MyClass()
    data[0] = 1
    data[1] = -1
    x = Tensor([1, 2, 3])
    o1 = fn(data, x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(data, x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_list_getitem_index_out_of_bound():
    """
    Feature: Test getitem.
    Description: Test list getitem, but index is out of bound.
    Expectation: Graph break, and exception.
    """

    @pi_jit_with_config(jit_config=jit_cfg)
    def fn(x: Tensor, lst: list):
        return x + lst[1]

    x = Tensor([1, 2, 3])
    lst = [1]
    with pytest.raises(IndexError):
        o = fn(x, lst)

    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 1
