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
"""Test one stage debug info"""
from mindspore import jit
from mindspore import ops
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from .test_utils import check_ir_info

def check_debug_info_no_break(func, inputs, expect_dict, dir):
    """Check whether func(inputs) create IR match expect dict"""
    check_ir_info(func, inputs, expect_dict, 'graph_before_compile', 1, dir)


def check_debug_info_with_break(func, inputs, expect_ir_num, expect_dict, dir):
    """Check whether func(inputs) create IR match expect dict"""
    check_ir_info(func, inputs, expect_dict, 'graph_before_compile', expect_ir_num, dir)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_binary_op():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    check_debug_info_no_break(foo, (input_x, input_y), {"m = x + y": 1}, "./test_debug_info_for_binary_op")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_primitive_call():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = ops.add(x, y)
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    check_debug_info_no_break(foo, (input_x, input_y), {"m = ops.add(x, y)": 3}, "./test_debug_info_for_primitive_call")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_func_graph_call():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = inner(x, y)
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"ret = x + y": 1, "m = inner(x, y)": 3}
    check_debug_info_no_break(foo, (input_x, input_y), expect_dict, "./test_debug_info_for_func_graph_call")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_binary_op_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = x - y
        return m, n

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "n = x - y": 1, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 2, expect_dict, "./test_debug_info_for_binary_op_with_break")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_graph_call_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = inner(x, y)
        return m, n

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "ret = x + y": 1, "n = inner(x, y)": 3, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 2, expect_dict, "./test_debug_info_for_graph_call_with_break")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_ops_call_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = inner(x, y)
        print("bbbb", flush=True)  # break here
        z = ops.add(x, y)
        return m, n, z

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "ret = x + y": 1, "n = inner(x, y)": 3, "z = ops.add(x, y)": 3, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 3, expect_dict, "./test_debug_info_for_ops_call_with_break")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_debug_info_function_basic():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test function call generates debug info, IR should contain "out = x + y".
    Expectation: No exception, the IR should have debug info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_001
    """
    @jit(capture_mode='bytecode')
    def func_1(x, y):
        out = x + y
        return out

    input_np1 = Tensor([[1, 2], [3, 4]])
    input_np2 = Tensor([[6, 7], [8, 9]])
    expect_dict = {"out = x + y": 1}
    check_debug_info_no_break(func_1, (input_np1, input_np2), expect_dict, "./test_debug_info_function_basic")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_debug_info_function_with_graph_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test function with graph break, IR generates debug info with "out = x + y" and "return out + 1".
    Expectation: No exception, the IR should have debug info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_002
    """
    @jit(capture_mode='bytecode')
    def func_2(x, y):
        out = x + y
        out = Tensor(out.asnumpy())
        return out + 1

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"out = x + y": 1, "return out + 1": 1}
    check_debug_info_with_break(func_2, (input_x, input_y), 2, expect_dict, "./test_debug_info_function_with_graph_break")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_debug_info_function_with_subgraph_call():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test function with subgraph call, IR contains "out = f1(x) + y", "return x + x" and "return out + y" debug info.
    Expectation: No exception, the IR should have debug info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_005
    """
    @jit(capture_mode='bytecode')
    def func_3(x, y):
        out = f1(x) + y
        return out + y

    def f1(x):
        return x + x

    input_np1 = Tensor([[1, 2], [3, 4]])
    input_np2 = Tensor([[6, 7], [8, 9]])
    expect_scope_info_dict = {"out = f1(x) + y": 4, "return x + x": 1, "return out + y": 1}

    check_debug_info_no_break(func_3, (input_np1, input_np2), expect_scope_info_dict, "./test_debug_info_function_with_subgraph_call")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_debug_info_function_with_subgraph_and_graph_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test function with subgraph and graph break, IR contains "out = f1(x) + y" debug info.
    Expectation: No exception, the IR should have debug info generated.
    Migrated from: test_parse_pijit_debug_scope_info.py::test_pijit_debug_scope_info_006
    """
    @jit(capture_mode='bytecode')
    def func_4(x, y):
        out = f1(x) + y
        return out

    def f1(x):
        return x + Tensor(x.asnumpy())

    input_np1 = Tensor([[1, 2], [3, 4]])
    input_np2 = Tensor([[6, 7], [8, 9]])
    expect_scope_info_dict = {"out = f1(x) + y": 6}

    check_debug_info_no_break(func_4, (input_np1, input_np2), expect_scope_info_dict, "./test_debug_info_function_with_subgraph_and_graph_break")
