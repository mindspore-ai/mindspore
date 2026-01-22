# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test tuple getitem by slice"""

import pytest
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
from mindspore.graph.api import _pynative_executor
from mindspore import jit
from tests.mark_utils import arg_mark


class NetWorka(Cell):
    def __init__(self, tuple_input, start):
        super().__init__()
        self.tensor_tuple = tuple_input
        self.start = start

    @jit
    def construct(self):
        tensor_tuple_slice = self.tensor_tuple[self.start]
        return tensor_tuple_slice


class NetWorkb(Cell):
    def __init__(self, tuple_input, start):
        super().__init__()
        self.tensor_tuple = tuple_input
        self.start = start

    @jit
    def construct(self):
        tensor_tuple_slice = self.tensor_tuple[self.start :]
        return tensor_tuple_slice


class NetWorkc(Cell):
    def __init__(self, tuple_input, stop):
        super().__init__()
        self.tensor_tuple = tuple_input
        self.stop = stop

    @jit
    def construct(self):
        tensor_tuple_slice = self.tensor_tuple[: self.stop]
        return tensor_tuple_slice


class NetWorkd(Cell):
    def __init__(self, tuple_input=(1, 3, 4, 5, 6, 7), start=None, stop=None, step=None, flag=0):
        super().__init__()
        self.tensor_tuple = tuple_input
        self.start = start
        self.stop = stop
        self.step = step
        """ flag: 1->tuple[start::], 2->tuple[:stop:], 3->tuple[::step] """
        self.flag = flag

    @jit
    def construct(self):
        if self.flag == 1:
            tensor_tuple_slice = self.tensor_tuple[self.start : :]
        elif self.flag == 2:
            tensor_tuple_slice = self.tensor_tuple[: self.stop :]
        elif self.flag == 3:
            tensor_tuple_slice = self.tensor_tuple[:: self.step]
        else:
            tensor_tuple_slice = self.tensor_tuple[self.start : self.stop : self.step]
        return tensor_tuple_slice


def net_train(net):
    net.set_train()
    ret = net()
    return ret


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_only_start_input_negative4():
    """
    Feature: Tuple slice with single index (no colon).
    Description: Test tuple slice with negative start index -4.
    Expectation: Returns the element at index -4.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_only_start_input_negative4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorka(tensor_tuple, -4)
    ret = net_train(net)
    assert ret == tensor_tuple[-4]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_only_start_input_0():
    """
    Feature: Tuple slice with single index (no colon).
    Description: Test tuple slice with start index 0.
    Expectation: Returns the element at index 0.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_only_start_input_0
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorka(tensor_tuple, 0)
    ret = net_train(net)
    assert ret == tensor_tuple[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_only_start_input_6_exception():
    """
    Feature: Tuple slice with single index (no colon) exception case.
    Description: Test tuple slice with start index 6 which is out of range.
    Expectation: Raises IndexError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_only_start_input_6_exception
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorka(tensor_tuple, 6)
    with pytest.raises(IndexError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_default():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with default slice [:] (all elements).
    Expectation: Returns all elements of the tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_default
    """

    class NetWork(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = tensor_tuple[:]
            return tensor_tuple_slice

    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWork(tensor_tuple)
    ret = net_train(net)
    assert ret == tensor_tuple[:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_none():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with start input None.
    Expectation: Returns tuple from start to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_none
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, None)
    ret = net_train(net)
    assert ret == tensor_tuple[None:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_negative4():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with start input -4.
    Expectation: Returns tuple from index -4 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_negative4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, -4)
    ret = net_train(net)
    assert ret == tensor_tuple[-4:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_0():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with start input 0.
    Expectation: Returns tuple from index 0 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_0
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, 0)
    ret = net_train(net)
    assert ret == tensor_tuple[0:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_4():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with start input 4.
    Expectation: Returns tuple from index 4 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, 4)
    ret = net_train(net)
    assert ret == tensor_tuple[4:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_negative10_exception_01():
    """
    Feature: Tuple slice with one colon exception case.
    Description: Test tuple slice with start input -10 (out of range).
    Expectation: Returns empty tuple or tuple from index -10 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_negative10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, -10)
    ret = net_train(net)
    assert ret == tensor_tuple[-10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_start_input_10_exception_01():
    """
    Feature: Tuple slice with one colon exception case.
    Description: Test tuple slice with start input 10 (out of range).
    Expectation: Returns empty tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_start_input_10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkb(tensor_tuple, 10)
    ret = net_train(net)
    assert ret == tensor_tuple[10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_stop_input_negative4():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with stop input -4.
    Expectation: Returns tuple from start to index -4.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_stop_input_negative4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkc(tensor_tuple, -4)
    ret = net_train(net)
    assert ret == tensor_tuple[:-4]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_stop_input_0():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with stop input 0.
    Expectation: Returns empty tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_stop_input_0
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkc(tensor_tuple, 0)
    ret = net_train(net)
    assert ret == tensor_tuple[:0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_stop_input_4():
    """
    Feature: Tuple slice with one colon.
    Description: Test tuple slice with stop input 4.
    Expectation: Returns tuple from start to index 4.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_stop_input_4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkc(tensor_tuple, 4)
    ret = net_train(net)
    assert ret == tensor_tuple[:4]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_stop_input_negative10_exception_01():
    """
    Feature: Tuple slice with one colon exception case.
    Description: Test tuple slice with stop input -10 (out of range).
    Expectation: Returns tuple from start to index -10.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_stop_input_negative10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkc(tensor_tuple, -10)
    ret = net_train(net)
    assert ret == tensor_tuple[:-10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_1_colon_stop_input_10_exception_01():
    """
    Feature: Tuple slice with one colon exception case.
    Description: Test tuple slice with stop input 10 (out of range).
    Expectation: Returns all elements of the tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_1_colon_stop_input_10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkc(tensor_tuple, 10)
    ret = net_train(net)
    assert ret == tensor_tuple[:10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_startnone_stopnone_stepdefault():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with start=None, stop=None, step=default.
    Expectation: Returns all elements of the tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_startnone_stopnone_stepdefault
    """

    class NetWork(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = tensor_tuple[None:None:]
            return tensor_tuple_slice

    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWork(tensor_tuple)
    ret = net_train(net)
    assert ret == tensor_tuple[None:None:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_start_input_negative4():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with start input -4 and two colons.
    Expectation: Returns tuple from index -4 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_start_input_negative4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=-4, flag=1)
    ret = net_train(net)
    assert ret == tensor_tuple[-4::]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_start_input_0():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with start input 0 and two colons.
    Expectation: Returns tuple from index 0 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_start_input_0
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=0, flag=1)
    ret = net_train(net)
    assert ret == tensor_tuple[0::]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_start_input_4():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with start input 4 and two colons.
    Expectation: Returns tuple from index 4 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_start_input_4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=4, flag=1)
    ret = net_train(net)
    assert ret == tensor_tuple[4::]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_start_input_negative10_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with start input -10 (out of range) and two colons.
    Expectation: Returns tuple from index -10 to end.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_start_input_negative10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=-10, flag=1)
    ret = net_train(net)
    assert ret == tensor_tuple[-10::]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_start_input_10_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with start input 10 (out of range) and two colons.
    Expectation: Returns empty tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_start_input_10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=10, flag=1)
    ret = net_train(net)
    assert ret == tensor_tuple[10::]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_stop_input_negative4():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with stop input -4 and two colons.
    Expectation: Returns tuple from start to index -4.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_stop_input_negative4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, stop=-4, flag=2)
    ret = net_train(net)
    assert ret == tensor_tuple[:-4:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_stop_input_0():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with stop input 0 and two colons.
    Expectation: Returns empty tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_stop_input_0
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, stop=0, flag=2)
    ret = net_train(net)
    assert ret == tensor_tuple[:0:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_stop_input_4():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with stop input 4 and two colons.
    Expectation: Returns tuple from start to index 4.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_stop_input_4
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, stop=4, flag=2)
    ret = net_train(net)
    assert ret == tensor_tuple[:4:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_stop_input_negative10_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with stop input -10 (out of range) and two colons.
    Expectation: Returns tuple from start to index -10.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_stop_input_negative10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, stop=-10, flag=2)
    ret = net_train(net)
    assert ret == tensor_tuple[:-10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_stop_input_10_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with stop input 10 (out of range) and two colons.
    Expectation: Returns all elements of the tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_stop_input_10_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, stop=10, flag=2)
    ret = net_train(net)
    assert ret == tensor_tuple[:10:]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_step_input_negative1():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with step input -1 and two colons.
    Expectation: Returns reversed tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_step_input_negative1
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, step=-1, flag=3)
    ret = net_train(net)
    assert ret == tensor_tuple[::-1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_step_input_2():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with step input 2 and two colons.
    Expectation: Returns tuple with step 2.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_step_input_2
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, step=2, flag=3)
    ret = net_train(net)
    assert ret == tensor_tuple[::2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_step_input_negative10():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with step input -10 and two colons.
    Expectation: Returns tuple with step -10.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_step_input_negative10
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, step=-10, flag=3)
    ret = net_train(net)
    assert ret == tensor_tuple[::-10]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_2_colon_only_step_input_10():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with step input 10 and two colons.
    Expectation: Returns tuple with step 10.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_2_colon_only_step_input_10
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, step=10, flag=3)
    ret = net_train(net)
    assert ret == tensor_tuple[::10]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_start5_stop3_step1():
    """
    Feature: Tuple slice with two colons.
    Description: Test tuple slice with start=5, stop=3, step=1.
    Expectation: Returns empty tuple (start > stop with positive step).
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_start5_stop3_step1
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=5, stop=3, step=1)
    ret = net_train(net)
    assert ret == tensor_tuple[5:3:1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_start1_stop10_step2_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with start=1, stop=10 (out of range), step=2.
    Expectation: Returns tuple from index 1 to end with step 2.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_start1_stop10_step2_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=1, stop=10, step=2)
    ret = net_train(net)
    assert ret == tensor_tuple[1:10:2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_start10_stop5_step2_exception_01():
    """
    Feature: Tuple slice with two colons exception case.
    Description: Test tuple slice with start=10 (out of range), stop=5, step=2.
    Expectation: Returns empty tuple (start > stop with positive step).
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_start10_stop5_step2_exception_01
    """
    tensor_tuple = (1, 3, 4, 5, 6, 7)
    net = NetWorkd(tensor_tuple, start=10, stop=5, step=2)
    ret = net_train(net)
    assert ret == tensor_tuple[10:5:2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_data_range_exception_01():
    """
    Feature: Tuple slice with range input data.
    Description: Test tuple slice with range(6) as input data, start=1, stop=3, step=1.
    Expectation: Returns range(1, 3).
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_data_range_exception_01
    """
    input_data = range(6)
    net = NetWorkd(input_data, start=1, stop=3, step=1)
    ret = net_train(net)
    assert ret == range(1, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_type_string_exception_01():
    """
    Feature: Tuple slice with string input data.
    Description: Test tuple slice with string as input data, start=1, stop=3, step=1.
    Expectation: Returns sliced string "tr".
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_type_string_exception_01
    """
    input_data = "string"
    net = NetWorkd(input_data, start=1, stop=3, step=1)

    ret = net_train(net)
    assert ret == "tr"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_type_user_defined_exception_01():
    """
    Feature: Tuple slice with user-defined type input data.
    Description: Test tuple slice with user-defined type as input data, start=1, stop=3, step=1.
    Expectation: Returns None (user-defined __getitem__ returns None).
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_type_user_defined_exception_01
    """

    class NewType:
        def __getitem__(self, key):
            pass

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            pass

    input_x = NewType()
    input_x[0] = [1, 2, 3, 4, 5, 6]
    net = NetWorkd(input_x, start=1, stop=3, step=1)
    ret = net_train(net)
    assert ret is None


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_input_type_list_exception_01():
    """
    Feature: Tuple slice with list input data.
    Description: Test tuple slice with list as input data, start=1, stop=3, step=1.
    Expectation: Executes without exception.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_input_type_list_exception_01
    """
    input_data = [1, 2, 3, 4, 5, 6]
    net = NetWorkd(input_data, start=1, stop=3, step=1)
    net_train(net)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_start_input_float_exception():
    """
    Feature: Tuple slice exception case with float start input.
    Description: Test tuple slice with float start input 1.1.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_start_input_float_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    start = 1.1
    net = NetWorkd(tensor_tuple, start=start, stop=6, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_start_input_string_exception():
    """
    Feature: Tuple slice exception case with string start input.
    Description: Test tuple slice with string start input "1".
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_start_input_string_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    start = "1"
    net = NetWorkd(tensor_tuple, start=start, stop=6, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_start_input_nan_exception():
    """
    Feature: Tuple slice exception case with NaN start input.
    Description: Test tuple slice with NaN start input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_start_input_nan_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    start = float("nan")
    net = NetWorkd(tensor_tuple, start=start, stop=6, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_start_input_inf_exception():
    """
    Feature: Tuple slice exception case with infinity start input.
    Description: Test tuple slice with infinity start input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_start_input_inf_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    start = float("inf")
    net = NetWorkd(tensor_tuple, start=start, stop=6, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_start_input_tensor():
    """
    Feature: Tuple slice with Tensor start input.
    Description: Test tuple slice with Tensor as start input.
    Expectation: Returns correct sliced tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_start_input_tensor
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    start = Tensor(1)
    net = NetWorkd(tensor_tuple, start=start, stop=6, step=1)
    ret = net_train(net)
    assert ret == tensor_tuple[1:6:1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_stop_input_float_exception():
    """
    Feature: Tuple slice exception case with float stop input.
    Description: Test tuple slice with float stop input 1.1.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_stop_input_float_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    stop = 1.1
    net = NetWorkd(tensor_tuple, start=1, stop=stop, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_stop_input_string_exception():
    """
    Feature: Tuple slice exception case with string stop input.
    Description: Test tuple slice with string stop input "1".
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_stop_input_string_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    stop = "1"
    net = NetWorkd(tensor_tuple, start=1, stop=stop, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_stop_input_nan_exception():
    """
    Feature: Tuple slice exception case with NaN stop input.
    Description: Test tuple slice with NaN stop input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_stop_input_nan_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    stop = float("nan")
    net = NetWorkd(tensor_tuple, start=1, stop=stop, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_stop_input_inf_exception():
    """
    Feature: Tuple slice exception case with infinity stop input.
    Description: Test tuple slice with infinity stop input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_stop_input_inf_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    stop = float("inf")
    net = NetWorkd(tensor_tuple, start=1, stop=stop, step=1)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_stop_input_tensor():
    """
    Feature: Tuple slice with Tensor stop input.
    Description: Test tuple slice with Tensor as stop input.
    Expectation: Returns correct sliced tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_stop_input_tensor
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    stop = Tensor(1)
    net = NetWorkd(tensor_tuple, start=1, stop=stop, step=1)
    ret = net_train(net)
    assert ret == tensor_tuple[1:1:1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_float_exception():
    """
    Feature: Tuple slice exception case with float step input.
    Description: Test tuple slice with float step input 1.1.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_float_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = 1.1
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_string_exception():
    """
    Feature: Tuple slice exception case with string step input.
    Description: Test tuple slice with string step input "1".
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_string_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = "1"
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_nan_exception():
    """
    Feature: Tuple slice exception case with NaN step input.
    Description: Test tuple slice with NaN step input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_nan_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = float("nan")
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_inf_exception():
    """
    Feature: Tuple slice exception case with infinity step input.
    Description: Test tuple slice with infinity step input.
    Expectation: Raises TypeError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_inf_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = float("inf")
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    with pytest.raises(TypeError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_tensor():
    """
    Feature: Tuple slice with Tensor step input.
    Description: Test tuple slice with Tensor as step input.
    Expectation: Returns correct sliced tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_tensor
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = Tensor(1)
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    ret = net_train(net)
    assert ret == tensor_tuple[1:6:1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_0_exception():
    """
    Feature: Tuple slice exception case with step input 0.
    Description: Test tuple slice with step input 0.
    Expectation: Raises ValueError.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_0_exception
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = 0
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    with pytest.raises(ValueError):
        net_train(net)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_step_input_none():
    """
    Feature: Tuple slice with None step input.
    Description: Test tuple slice with None as step input.
    Expectation: Returns correct sliced tuple.
    Migrated from: test_parser_tuple_slice.py::test_parser_slice_step_input_none
    """
    tensor_tuple = (1, 2, 3, 4, 5, 6, 7)
    step = None
    net = NetWorkd(tensor_tuple, start=1, stop=6, step=step)
    ret = net_train(net)
    assert ret == tensor_tuple[1:6:step]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_slice_0001():
    """
    Feature: Nested tuple slice.
    Description: Test nested tuple slice operation.
    Expectation: Executes successfully.
    Migrated from: test_parser_tuple_slice.py::test_parser_tuple_slice_0001
    """

    class Net(Cell):
        def __init__(self, _tuple):
            super().__init__()
            self.tuple = _tuple

        @jit
        def construct(self):
            return self.tuple[0][1:6:2]

    _tuple = ((1, 2, 3, 4, 5, 6, 7), 8, 9, 10)
    net = Net(_tuple)
    net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_st_ms_slice_input_data_range_abnormal_0001():
    """
    Feature: Tuple slice with range input data.
    Description: Test tuple slice with tuple(range(6)) as input data, start=1, stop=3, step=1.
    Expectation: Returns correct sliced tuple based on mode.
    Migrated from: test_parser_tuple_slice.py::test_st_ms_parser_slice_input_data_range_abnormal_0001
    """

    def net_train1(net):
        ret = net()
        return ret

    input_data = tuple(range(6))
    net = NetWorkd(input_data, start=1, stop=3, step=1)

    ret = net_train1(net)
    assert ret == (1, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_st_ms_slice_input_type_string_normal_0001():
    """
    Feature: Tuple slice with string input data.
    Description: Test tuple slice with string as input data, start=1, stop=3, step=1.
    Expectation: Returns sliced string.
    Migrated from: test_parser_tuple_slice.py::test_st_ms_parser_slice_input_type_string_normal_0001
    """

    def net_train2(net):
        ret = net()
        return ret

    input_data = "string"
    net = NetWorkd(input_data, start=1, stop=3, step=1)

    ret = net_train2(net)
    assert ret == input_data[1:3:1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_st_ms_slice_input_type_user_defined_normal_0001():
    """
    Feature: Tuple slice with user-defined type input data.
    Description: Test tuple slice with user-defined type as input data, start=1, stop=3, step=1.
    Expectation: Returns result from user-defined __getitem__.
    Migrated from: test_parser_tuple_slice.py::test_st_ms_parser_slice_input_type_user_defined_normal_0001
    """

    class NewType:
        def __getitem__(self, key):
            pass

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            pass

    def net_train1(net):
        ret = net()
        return ret

    input_data = NewType()
    input_data[0] = [1, 2, 3, 4, 5, 6]
    net = NetWorkd(input_data, start=1, stop=3, step=1)

    ret = net_train1(net)
    assert ret == input_data[1:3:1]
