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
"""Test builtin function constant fold"""

import numpy as np

from mindspore import Tensor, ops, jit, context
from mindspore.nn import Cell

from tests.st.pi_jit.share.utils import match_array, assert_executed_by_graph_mode, pi_jit_with_config
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_abs():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    def fn(x: Tensor):
        return abs(x) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, -1, 2, -2])
    o1 = fn(x)

    fn = jit(fn, capture_mode='bytecode')
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_len():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    @jit(capture_mode='bytecode')
    def fn(x: Tensor):
        return len(x) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, 2, 3, 4])
    o = fn(x)

    assert o == 5


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    def fn(x: Tensor):
        return pow(x, 2) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, -1, 2, -2])
    o1 = fn(x)

    fn = jit(fn, capture_mode='bytecode')
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_builtin_function_type_v1():
    """
    Feature: python builtin type().
    Description: Test one stage basic operation.
    Expectation: No graph breaks.
    """

    def view(x: Tensor, *shape):
        # when x triggers dynamic shape, the shape argument may become a variable (contains kValueAny).
        if type(shape) is tuple:
            return ops.reshape(x, shape)
        else:
            return ops.flatten(x)

    def fn(x: Tensor, n: int, dim: int):
        B = x.shape[0]
        T = x.shape[1]  # may trigger dynamic shape
        return view(x, B, T, n, dim)

    compiled_fn = pi_jit_with_config(fn, jit_config={'_symbolic': 1}, fullgraph=True)

    # Currently, the 7th tensor shape change triggers dynamic shape compilation.
    for i in range(1, 10):
        x = ops.randn(1, i, 4)
        o1 = fn(x, 2, 2)
        o2 = compiled_fn(x, 2, 2)
        match_array(o1, o2)
        assert_executed_by_graph_mode(compiled_fn)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_builtin_all_control_flow():
    """
    Feature: Python builtin all in PIJit.
    Description: Use builtin all in control flow and ensure PIJit keeps consistent behavior with pynative mode.
    Expectation: JIT result matches pynative result and generates two graphs.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_buildin_func_no_op_correspond
    """

    class Net(Cell):
        def construct(self, x, y):
            if all(x > y):
                return x + y
            return x * y

    x = Tensor([3, 4, 5])
    y = Tensor([1, 2, 3])

    pynative_net = Net()
    pynative_out = pynative_net(x, y)

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(x, y)

    match_array(pynative_out, jit_out)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_builtin_abs_len_max_min():
    """
    Feature: Python builtin abs/len/max/min in PIJit.
    Description: Use abs, len, max, min and ensure PIJit matches pynative mode.
    Expectation: JIT result matches pynative result and generates one graph.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_buildin_func_op_correspond_001
    """

    class Net(Cell):
        def construct(self, a, b):
            m = max(a, b)
            a = abs(m)
            m = min(a, b)
            l_value = len((a, b))
            return m * l_value

    a = Tensor(4)
    b = Tensor(2)

    pynative_net = Net()
    pynative_out = pynative_net(a, b)

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode', fullgraph=True)
    jit_out = jit_net(a, b)

    match_array(pynative_out, jit_out)
    check_ir_num('graph_before_compile', 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_builtin_pow_round_sum():
    """
    Feature: Python builtin pow/round/sum in PIJit.
    Description: Use pow, round and sum to compute tensor values under PIJit.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_buildin_func_op_correspond_002
    """

    class Net(Cell):
        def construct(self, x, y):
            power = pow(x, 2)
            rounded = round(y)
            return sum(power, rounded)

    x = Tensor([4])
    y = Tensor([2.6375])

    pynative_net = Net()
    pynative_out = pynative_net(x, y)

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode', fullgraph=True)
    jit_out = jit_net(x, y)

    match_array(pynative_out, jit_out)


def func1(x):
    return x + x


def func2(x):
    return x * x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_builtin_eval_dispatch_function():
    """
    Feature: Python eval inside JIT compiled function.
    Description: Use eval to dispatch between different functions inside a jit(capture_mode='bytecode') function.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_catch.py::test_pijit_catch_func_eval
    """

    def call_with_eval(x):
        return eval(f"func{len(x)}(x)")

    input_np = np.random.rand(2, 3).astype(np.float32)
    tensor = Tensor(input_np)

    pynative_out = call_with_eval(tensor)
    jit_call = jit(call_with_eval, capture_mode='bytecode')
    jit_out = jit_call(tensor)

    match_array(pynative_out, jit_out)
