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
"""Test one stage scope"""

from mindspore import jit
from mindspore import Tensor
from mindspore.nn import Cell
from tests.mark_utils import arg_mark
from .test_utils import check_ir_info


def check_scope_info_no_break(func, inputs, expect_dict, dir):
    """Check whether func(inputs) create IR match expect dict"""
    check_ir_info(func, inputs, expect_dict, '_validate', 1, dir)


def check_scope_info_with_break(func, inputs, expect_dict, expect_num, dir):
    """Check whether func(inputs) create IR match expect dict"""
    check_ir_info(func, inputs, expect_dict, '_validate', expect_num, dir)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scope_with_single_net():
    """
    Feature: PIJit enable scope info in IR.
    Description: Test whether scope is in IR for PIJit.
    Expectation: No exception, the IR should have scope info.
    """

    class Net(Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            ret = x + y
            return ret

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    net = Net()
    expect_dict = {"Default/Add-op": 1, "Default/Return-op0": 1}
    check_scope_info_no_break(net, (input_x, input_y), expect_dict, "./test_scope_with_single_net")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scope_with_nested_net():
    """
    Feature: PIJit enable scope info in IR.
    Description: Test whether scope is in IR for PIJit.
    Expectation: No exception, the IR should have scope info.
    """
    class InnerNet(Cell):
        def construct(self, x, y):
            ret = x - y
            return ret

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            ret = x + y
            ret = ret + self.inner_net(x, y)
            return ret

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    net = Net()
    expect_dict = {"Default/Add-op": 2, "Default/inner_net-InnerNet/Sub-op0": 1}
    check_scope_info_no_break(net, (input_x, input_y), expect_dict, "./test_scope_with_nested_net")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scope_with_single_net_with_break():
    """
    Feature: PIJit enable scope info in IR.
    Description: Test whether scope is in IR for PIJit.
    Expectation: No exception, the IR should have scope info.
    """

    class Net(Cell):
        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            ret = x + y
            print("aaaaa", flush=True)  # break here
            ret = ret - 1
            return ret

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    net = Net()
    expect_dict = {"Default/Add-op": 1, "Default/Sub-op0": 1}
    check_scope_info_with_break(net, (input_x, input_y), expect_dict, 2, "./test_scope_with_single_net_with_break")


# will fix it later.
@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_scope_with_nest_net_with_break():
    """
    Feature: PIJit enable scope info in IR.
    Description: Test whether scope is in IR for PIJit.
    Expectation: No exception, the IR should have scope info.
    """

    class InnerNet(Cell):
        def construct(self, x, y):
            ret = x - y
            return ret

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.inner_net = InnerNet()

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            ret = x + y
            print("aaaaa", flush=True)  # break here
            ret = ret + self.inner_net(x, y)
            return ret

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    net = Net()
    expect_dict = {"Default/Add-op": 2, "Default/inner_net-InnerNet/Sub-op0": 1}
    check_scope_info_with_break(net, (input_x, input_y), expect_dict, 2, "./test_scope_with_nest_net_with_break")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scope_info_cell_with_debug_info():
    """
    Feature: PIJit enable scope info and debug info in IR.
    Description: Test Cell generates debug info and scope info, IR contains "Default/Add-op0", "Default/Return-op0" scope info and "ret = x + y" debug info.
    Expectation: No exception, the IR should have both debug info and scope info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_003
    """
    class Net(Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x, y):
            ret = x + y
            return ret

    net = Net()
    input_np1 = Tensor([[1, 2], [3, 4]])
    input_np2 = Tensor([[6, 7], [8, 9]])
    expect_scope_info_dict = {"Default/Add-op0": 1, "Default/Return-op0": 1, "ret = x + y": 1}

    check_scope_info_no_break(net, (input_np1, input_np2), expect_scope_info_dict, "./test_scope_info_cell_with_debug_info")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scope_info_cell_with_graph_break():
    """
    Feature: PIJit enable scope info in IR.
    Description: Test Cell with graph break, IR generates scope info with "Default/Add-op0" and "Default/Return-op0" scope info.
    Expectation: No exception, the IR should have scope info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_004
    """
    class Net(Cell):
        @jit(capture_mode='bytecode')
        def construct(self, x, y):
            ret = x + Tensor(y.asnumpy())
            return ret

    net = Net()
    input_np1 = Tensor([[1, 2], [3, 4]])
    input_np2 = Tensor([[6, 7], [8, 9]])
    expect_scope_info_dict = {"Default/Add-op0": 1, "Default/Return-op0": 1}
    check_scope_info_no_break(net, (input_np1, input_np2), expect_scope_info_dict, "./test_scope_info_cell_with_graph_break")
