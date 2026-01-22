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
"""Test list getitem by slice"""

import pytest
import numpy as np
from mindspore.nn import Cell
from mindspore.graph.api import _pynative_executor
from mindspore import jit, Tensor
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import allclose_nparray


class NetWork(Cell):
    def __init__(self, tuple_input=(1, 3, 4, 5, 6, 7), start=None, stop=None, step=None, flag=0):
        super().__init__()
        self.tensor_tuple = tuple_input
        self.start = start
        self.stop = stop
        self.step = step
        self.flag = flag

    @jit
    def construct(self):
        if self.flag == 1:
            tensor_tuple_slice = self.tensor_tuple[self.start : :]
        elif self.flag == 2:
            tensor_tuple_slice = self.tensor_tuple[: self.stop :]
        elif self.flag == 3:
            tensor_tuple_slice = self.tensor_tuple[:: self.step]
        elif self.flag == 4:
            tensor_tuple_slice = self.tensor_tuple[self.start : self.stop]
        elif self.flag == 5:
            tensor_tuple_slice = self.tensor_tuple[self.start : self.stop :]
        elif self.flag == 6:
            tensor_tuple_slice = self.tensor_tuple[self.start :: self.step]
        elif self.flag == 7:
            tensor_tuple_slice = self.tensor_tuple[: self.stop : self.step]
        elif self.flag == 8:
            tensor_tuple_slice = self.tensor_tuple[self.start]
        elif self.flag == 9:
            tensor_tuple_slice = self.tensor_tuple[self.start :]
        elif self.flag == 10:
            tensor_tuple_slice = self.tensor_tuple[: self.stop]
        elif self.flag == 11:
            tensor_tuple_slice = self.tensor_tuple[:]
        else:
            tensor_tuple_slice = self.tensor_tuple[self.start : self.stop : self.step]
        return tensor_tuple_slice


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_start_0():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, start=0.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_start_0
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=0, flag=1)
    net.set_train()
    ret = net()
    exp = tensor_tuple[0::]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_stop_5():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, stop=5.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_stop_5
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, stop=5, flag=2)
    net.set_train()
    ret = net()
    exp = tensor_tuple[:5:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_step_5():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, step=5.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_step_5
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, step=5, flag=3)
    net.set_train()
    ret = net()
    exp = tensor_tuple[::5]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_1_colon_start_0_stop_5():
    """
    Feature: List slice getitem.
    Description: Test list slice with 1 colon, start=0, stop=5.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_1_colon_start_0_stop_5
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=0, stop=5, flag=4)
    net.set_train()
    ret = net()
    exp = tensor_tuple[0:5]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_start_none_stop_none():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, start=None, stop=None.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_start_none_stop_none
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=None, stop=None, flag=5)
    net.set_train()
    ret = net()
    exp = tensor_tuple[None:None:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_start_1_step_3():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, start=1, step=3.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_start_1_step_3
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=1, step=3, flag=6)
    net.set_train()
    ret = net()
    exp = tensor_tuple[1::3]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_stop_5_step_3():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, stop=-5, step=3.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_stop_5_step_3
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, stop=-5, step=3, flag=7)
    net.set_train()
    ret = net()
    exp = tensor_tuple[:-5:3]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_start_0():
    """
    Feature: List slice getitem.
    Description: Test list slice without colon, start=0.
    Expectation: Result matches expected element.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_start_0
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=0, flag=8)
    net.set_train()
    ret = net()
    assert ret == tensor_tuple[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_1_colon_start_none():
    """
    Feature: List slice getitem.
    Description: Test list slice with 1 colon, start=None.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_1_colon_start_none
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=None, flag=9)
    net.set_train()
    ret = net()
    exp = tensor_tuple[None:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_1_colon_stop_5():
    """
    Feature: List slice getitem.
    Description: Test list slice with 1 colon, stop=-5.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_1_colon_stop_5
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, stop=-5, flag=10)
    net.set_train()
    ret = net()
    exp = tensor_tuple[:-5]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_1_colon():
    """
    Feature: List slice getitem.
    Description: Test list slice with 1 colon only.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_1_colon
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, flag=11)
    net.set_train()
    ret = net()
    exp = tensor_tuple[:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_2_colon_start_1_stop_10_step_2():
    """
    Feature: List slice getitem.
    Description: Test list slice with 2 colons, start=Tensor(1), stop=10, step=2.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_2_colon_start_1_stop_10_step_2
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    net = NetWork(tensor_tuple, start=Tensor(1), stop=10, step=2)
    net.set_train()
    ret = net()
    exp = tensor_tuple[1:10:2]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_3_times():
    """
    Feature: List slice getitem.
    Description: Test list slice 3 times consecutively.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_3
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[:-3][2:5][-1:]
            return tensor_tuple_slice

    tensor_tuple = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    exp = tensor_tuple[:-3][2:5][-1:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_dict():
    """
    Feature: List slice getitem.
    Description: Test list slice on list containing dict, int, tuple.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_dict
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[0:1][0]['ops'][-1:]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, (7, 8, 9)]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    exp = tensor_tuple[0:1][0]['ops'][-1:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_list():
    """
    Feature: List slice getitem.
    Description: Test list slice on list containing list.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_list
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[0:1][0][1:3][-1:]
            return tensor_tuple_slice

    tensor_tuple = [[1, 2, 3, 4, 5], 6, [7, 8, 9]]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    exp = tensor_tuple[0:1][0][1:3][-1:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_tuple():
    """
    Feature: List slice getitem.
    Description: Test list slice on list containing tuple.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_tuple
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[0:1][0][-1:]
            return tensor_tuple_slice

    tensor_tuple = [([1, 2, 3], 4, 5), 6, [7, 8, 9]]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    exp = tensor_tuple[0:1][Tensor(0)][-1:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(ret[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_numpy_1():
    """
    Feature: List slice getitem.
    Description: Test list slice with numpy tensor, using [..., 1].
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_numpy_1
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2][..., 1]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(64).reshape(4, 4, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2][..., 1].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_numpy_1_colon_1():
    """
    Feature: List slice getitem.
    Description: Test list slice with numpy tensor, using [..., :1].
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_numpy_1_colon_1
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2][..., :1]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(64).reshape(4, 4, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2][..., :1].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_numpy_2_colon_1():
    """
    Feature: List slice getitem.
    Description: Test list slice with numpy tensor, using [..., ::1].
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_numpy_2_colon_1
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2][..., ::1]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(64).reshape(4, 4, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2][..., ::1].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_numpy_none():
    """
    Feature: List slice getitem.
    Description: Test list slice with numpy tensor, using [..., None].
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_numpy_none
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2][..., None]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(64).reshape(4, 4, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2][..., None].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_fancy_index_list():
    """
    Feature: List slice getitem.
    Description: Test list slice with fancy index using list.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_fancy_index_list
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2:3][0][0:2, [-3, 0, 2, -1, -1]]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(12).reshape(3, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2:3][0][0:2, [-3, 0, 2, -1, -1]].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_fancy_index_tuple():
    """
    Feature: List slice getitem.
    Description: Test list slice with fancy index using tuple.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_fancy_index_tuple
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2:3][0][(1, 2), (3, 3)]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(12).reshape(3, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2:3][0][(1, 2), (3, 3)].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_fancy_index_bool():
    """
    Feature: List slice getitem.
    Description: Test list slice with fancy index using bool list.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_fancy_index_bool
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2:3][0][[True, False, True], [0, 3]]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(12).reshape(3, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2:3][0][[True, False, True], [0, 3]].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_fancy_index_list_tuple():
    """
    Feature: List slice getitem.
    Description: Test list slice with fancy index using list and tuple.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_fancy_index_list_tuple
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self):
            tensor_tuple_slice = self.tensor_tuple[2:3][0][(True, False, True), [0, 3]]
            return tensor_tuple_slice

    tensor_tuple = [{'ops': [1, 2, 3, 4, 5]}, 6, Tensor(np.arange(12).reshape(3, 4))]
    net = Net(tensor_tuple)
    net.set_train()
    ret = net()
    assert ret.asnumpy().all() == tensor_tuple[2:3][0][(True, False, True), [0, 3]].asnumpy().all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_ms_function():
    """
    Feature: List slice getitem.
    Description: Test list slice in jit function.
    Expectation: Result matches expected slice.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_ms_function
    """

    @jit
    def slice_assignment():
        x = [([1, 2, 3], 4, 5), 6, [7, 8, 9]]
        return x[0:1][0][-1:]

    tensor_tuple = [([1, 2, 3], 4, 5), 6, [7, 8, 9]]
    exp = tensor_tuple[0:1][Tensor(0)][-1:]
    num = len(exp)
    for i in range(num):
        allclose_nparray(slice_assignment()[i], exp[i], 0, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_input_error_start():
    """
    Feature: List slice getitem.
    Description: Test error case with invalid start types (float, string, list, tuple, dict).
    Expectation: Raises TypeError or ValueError.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_input_error_start
    """
    tensor_tuple = [1, 3, 4, 5, 6, 7]
    start = [1.5, 'str', (1,), [5], {'key': 1}]
    for i in range(5):
        with pytest.raises(TypeError):
            net = NetWork(tensor_tuple, start=start[i], stop=10, step=2)
            net.set_train()
            net()
            _pynative_executor.sync()
    with pytest.raises(ValueError):
        net = NetWork(tensor_tuple, start=0, stop=10, step=0)
        net.set_train()
        net()
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_assignment():
    """
    Feature: List slice getitem.
    Description: Test error case where list slice is used for assignment.
    Expectation: No exception or raises expected exception.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_assignment
    """

    @jit
    def slice_assignment():
        x = [[1, 2], 2, 3, 4]
        x[1:3] = (1, 2, 3)
        return x

    slice_assignment()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_variational_index():
    """
    Feature: List slice getitem.
    Description: Test error case with variational index for list slice.
    Expectation: No exception or raises expected exception.
    Migrated from: test_parser_list_slice.py::test_parser_list_slice_variational_index
    """

    class Net(Cell):
        def __init__(self, tuple_input):
            super().__init__()
            self.tensor_tuple = tuple_input

        @jit
        def construct(self, x):
            tensor_tuple_slice = self.tensor_tuple[x:]
            return tensor_tuple_slice

    tensor_tuple = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    net = Net(tensor_tuple)
    net.set_train()
    net(Tensor(0))
