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
"""test graph break in call_function"""

import numpy as np
import sys
import pytest

import mindspore as ms
from mindspore import Tensor, jit, context, ops, nn
from mindspore.ops import operations as P
from mindspore._c_expression import get_code_extra

from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num
from tests.st.pi_jit.share.utils import match_array, assert_has_graph_break, assert_equal, pi_jit_with_config
from tests.mark_utils import arg_mark

SYS_VER = (sys.version_info.major, sys.version_info.minor)
if SYS_VER >= (3, 11):
    pytest.skip(reason="not implement for python" + str(SYS_VER), allow_module_level=True)

SKIP_PY37 = pytest.mark.skipif(sys.version_info[:2] == (3, 7), reason="Not support py37 setup loop bytecode")

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False, 'subgraph_break_opt': True}


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a * 2

    def f1(x: Tensor):
        y = f2(x)
        return y + 1

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v2():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a * 2

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v3():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        d = {'k': 2}  # alive local, unsupported output
        print('GRAPH BREAK', flush=True)  # break
        return a * d['k']

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v4():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        d = {'k': 2}  # alive local, unsupported output
        b = x - 1
        print('GRAPH BREAK', flush=True)  # break
        return a * d['k'] * b

    def f1(x: Tensor):
        y = x * 2  # alive local
        d = {'bias': 1}  # alive local
        z = f2(x)
        return y + z + d['bias']

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_two_layers_v5():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2, break at Tensor.asnumpy().
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        x = x - 1  # alive local
        x = Tensor(x.asnumpy())  # break
        return x * 2

    def f1(x: Tensor):
        x = x * 2
        y = f2(x)
        return x + y

    a = Tensor(np.random.randn(2, 3).astype(np.float32))
    o1 = f1(a)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(a)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_three_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f3.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f2(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f3(x)  # break
        return a + b + c

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_three_layers_v2():
    """
    Feature: test graph break in call_function.
    Description: two graph breaks in f3.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        a = x + 1  # alive local
        print('GRAPH BREAK', flush=True)  # break
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f2(x: Tensor):
        a = x + 1  # alive local
        c = f3(x)  # break
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b + c

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 4)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_four_layers_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f4.
    Expectation: The result of PIJit is same as pynative.
    """

    def f4(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        print('GRAPH BREAK', flush=True)  # break
        return a + b

    def f3(x: Tensor):
        a = f4(x)  # break
        return x - a

    def f2(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        return f3(a * b)  # break

    def f1(x: Tensor):
        a = x + 1  # alive local
        b = x * 2  # alive local
        c = f2(x)
        return a + b + c

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v1():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        for i in range(5):
            if i % 2 == 0:
                x += 1
            else:
                print('GRAPH BREAK', flush=True)  # break
        return x

    def f1(x: Tensor):
        a = f2(x)
        return a + x

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v2():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x, y):
        result = ops.zeros_like(x)
        for i in range(5):
            print('GRAPH BREAK', flush=True)  # graph break
            result += x * y
        return result

    def f2(x):
        y = ms.tensor([2.0])
        result = f3(x, y)
        return result * 2

    def f1(x):
        result = f2(x)
        return result + 1

    x = ops.randn(3, 3)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v3():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, num_loops: int) -> Tensor:
        result = ops.zeros_like(x)
        result += num_loops
        for i in range(num_loops):
            result += x * i
            print('GRAPH BREAK', end='\n\n')
            result = ops.relu(result)
            result = result / (i + 1)
        return result

    def f2(x: Tensor, num_loops: int) -> Tensor:
        x = x * 2
        x = ops.sin(x)
        return f3(x, num_loops)

    def f1(x: Tensor, num_loops: int) -> Tensor:
        x = x + 1
        x = ops.cos(x)
        return f2(x, num_loops)

    x = ops.randn(10)
    num_loops = 5
    o1 = f1(x, num_loops)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, num_loops)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v4():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(input_tensor, loop_count):
        result = ops.zeros_like(input_tensor)
        for i in range(loop_count):
            temp = input_tensor * i
            result = result + ops.sin(temp)
            if i % 2 == 0:
                print('GRAPH BREAK', flush=True)  # graph break
                result = result * 2
            else:
                result = result / 2
                print('GRAPH BREAK', flush=True)  # graph break
        return result

    def f2(input_tensor, loop_count, factor):
        a = input_tensor + factor
        b = a * 2
        c = ops.relu(b)
        result = f3(c, loop_count)
        d = result - factor
        e = ops.sum(d)
        return e

    def f1(input_tensor, loop_count=5):
        factor = ms.tensor(2.0)
        intermediate_result = f2(input_tensor, loop_count, factor)  # 调用中间层函数
        final_result = intermediate_result * input_tensor
        output = ops.mean(final_result)
        return output

    x = ops.randn(3, 4)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v5():
    """
    Feature: test graph break in call_function.
    Description: graph break in for loop.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, repeat_times: int) -> Tensor:
        result = x
        for i in range(repeat_times):
            result = result * 2 + ops.sin(result)

        count = 0
        while count < 3:
            result = ops.relu(result) + ops.mean(result)
            count += 1
            print('GRAPH BREAK', flush=True)  # graph break
        return result

    def f2(input_dict: dict) -> Tensor:
        tensor_data = input_dict['data']
        repeat_count = input_dict['count']
        processed = tensor_data * 3.0 + ops.ones_like(tensor_data)
        return f3(processed, repeat_count)

    def f1(input_tensor: Tensor) -> Tensor:
        input_dict = {'data': input_tensor + 1.0, 'count': 5}
        result = f2(input_dict)
        return ops.relu(result)

    x = ops.randn(3, 4)
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v6():
    """
    Feature: test graph break in call_function.
    Description: graph break at loop condition.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x: Tensor, y: int):
        a = ops.ones((5, 5))
        b = x + a
        c = ops.zeros((5, 5))

        while ops.sum(b) < 100:  # graph break
            b = b * 2.5
            c = c + b
            d = ops.matmul(b, c)

        e = ops.sin(d)
        return e.mean()

    def f1(t: Tensor, params: dict):
        x = t * 2
        y = params['factor']
        z = f2(x, y)
        w = ops.exp(z)
        v = w + ms.tensor(params['offset'])
        return ops.tanh(v)

    x = ops.ones((5, 5))
    params = {'factor': 3, 'offset': 1.5}
    o1 = f1(x, params)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, params)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v7():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at loop body; 2.f1 call f2 at last statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        count = 0
        while count < 3:
            print('GRAPH BREAK', end='\n\n')
            z = z * 2
            z = z - 1
            count += 1
        return z

    def f1(a, b):
        c = a**2
        d = b + 2
        return f2(c, d)

    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v8():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at loop body; 2.f1 call f2 at last statement, and f2 call f3 at last statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x, y):
        i = 0
        while i < y:
            i += 1
            print('GRAPH BREAK', end='\n\n')
            x = ops.sin(x)
        return x

    def f2(a, b):
        a = a + b
        a = ops.relu(a)
        return f3(a, b)

    def f1(x):
        x = x * 2.0
        x = ops.sqrt(x)
        return f2(x, 3)

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v9():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break in loop body, at break statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        i = 0
        while i < 10:
            z = z * 2
            if i % 2 != 0:
                print('GRAPH BREAK', end='\n\n')
                break
            z = z - 1
            i += 1
        return z

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d / a
        return e

    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v10():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at else statement (after loop body).
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, threshold):
        result = ops.zeros_like(x)
        i = 0
        while i < x.shape[0]:
            result += x * 2
            if i > threshold:
                break  # shouldn't reach here
            result += 1
            i += 1
        else:
            print('GRAPH BREAK', end='\n\n')
            result = result / 2
        return result

    def f1(x, y):
        z = x + y
        w = f2(z, 10)
        return w * 2

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v11():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at while-loop body; 2.the last statement of loop body is Return (which means this the loop will only be executed once)
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        i = 0
        while i < 3:
            z = z * 2
            print('GRAPH BREAK', end='\n\n')
            i += 1
            return z  # the last statement of loop body is Return

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d + c
        return e

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v12():
    """
    Feature: test graph break in call_function.
    Description: 1.graph break at for-in-range-loop body; 2.the last statement of loop body is Return (which means this the loop will only be executed once)
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x + y
        for i in range(3):
            z = z + i
            print('GRAPH BREAK', end='\n\n')
            return z  # the last statement of loop body is Return

    def f1(a, b):
        c = a * b
        d = f2(c, b)
        e = d + c
        return e

    x = ops.randn(2, 4)
    y = ops.randn(2, 4)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v13():
    """
    Feature: test graph break in call_function.
    Description: 1.f1 call f2 in loop; 2.f2 has graph break.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x, y):
        z = x - y
        z = z * 2
        print('GRAPH BREAK', end='\n\n')
        z = ops.sin(z)
        return z

    def f1(x, params):
        result = x
        for i in range(3):
            result = ops.relu(result)
            result = f2(result, params['y'])
            result = ops.exp(result)
        return result

    x = Tensor([1.0, 2.0, 3.0])
    params = {'y': Tensor([4.0, 5.0, 6.0])}
    o1 = f1(x, params)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, params)

    match_array(o1, o2, error=7)
    jcr = get_code_extra(f1)
    assert jcr['stat'] == 'NEVER_COMPILE'


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_loop_v14():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2; f2 call f3 in loop; f3 has graph break.
    Expectation: The result of PIJit is same as pynative.
    """

    def f1(x):
        x = f2(x)
        return P.ReLU()(x)

    def f2(x):
        for _ in range(3):
            x = x + f3(x)
        return P.ReLU()(x)

    def f3(x):
        x = x + 1
        x.asnumpy()  # graph break!
        return P.ReLU()(x)

    a = ops.randn(2, 3)
    o1 = f1(a)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(a)

    match_array(o1, o2, error=7)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_function_graph_break_in_recursion():
    """
    Feature: test graph break in call_function.
    Description: graph break in recursion.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor, n: int):
        if n == 0:
            return x
        x += 1
        print('GRAPH BREAK', flush=True)
        return f2(x, n - 1)

    def f1(x: Tensor):
        n = 3  # assert n >= 0
        a = f2(x, n)
        return a + x

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_break_at_first_statement():
    """
    Feature: test graph break in call_function.
    Description: graph break at first statement.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3():
        print('GRAPH BREAK', flush=True)  # break
        a = Tensor([2, 4, 6])
        return a + 1

    def f2():
        a = f3()
        return ops.mul(a, 2)

    def f1(x: Tensor, y: Tensor):
        z = f2()
        return x - y - z

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = f1(x, y)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x, y)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_is_dict_and_is_alive_local_v1():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    def f2(x: Tensor, d: dict):
        a = ops.add(d['a'], x)
        print('GRAPH BREAK', flush=True)  # break
        b = ops.sub(d['b'], x)
        return a, b

    def f1(x: Tensor, y: Tensor):
        d = {'a': x + y, 'b': x - y}
        a, b = f2(x, d)
        return a * b

    x = Tensor([1, 2, 3])
    y = Tensor([1, 1, 1])
    o1 = f1(x, y)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x, y)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_param_is_dict_and_is_alive_local_v2():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    d = {'a': Tensor([1.0, 2.0, 3.0]), 'b': ops.randn(3)}

    def f2(x: Tensor):
        d2 = d
        a = ops.add(d2['a'], x)
        print('GRAPH BREAK', flush=True)  # break
        b = ops.sub(d2['b'], x)
        return a, b

    def f1(x: Tensor):
        a, b = f2(x)
        return a * b

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v1():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        z = z + x
        z = z / 2
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        c = c - a
        f = f3
        print('GRAPH BREAK', flush=True)
        return f

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v2():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """
    xxx = 0

    def f3(x: Tensor, y: int) -> Tensor:
        nonlocal xxx  # unsupported syntax, cannot convert f3 to AnfNode
        xxx += 1
        z = x * y
        z = z + x
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v3():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        print('GRAPH BREAK', end='\n\n')
        z = z + x
        return z

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3, c

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f, c = f2(z, {'value': y[1]})
        print('GRAPH BREAK', end='\n\n')
        result = f(z, y[0])
        return result + c

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_function_output_type_unsupported_v4():
    """
    Feature: test graph break in call_function.
    Description: f1 call f2, and f2 will return f3. Then f1 will call f3.
    Expectation: The result of PIJit is same as pynative.
    """
    VAR = 0.25

    def f3(x: Tensor, y: int) -> Tensor:
        z = x * y
        a = VAR
        print('GRAPH BREAK', end='\n\n')
        z = z + x
        return z * a

    def f2(a: Tensor, b: dict) -> callable:
        c = a + b['value']
        c = c * 2
        return f3, c

    def f1(x: Tensor, y: tuple) -> Tensor:
        z = x + y[0]
        z = z * y[1]
        z = z - x
        f, c = f2(z, {'value': y[1]})
        result = f(z, y[0])
        return result + c

    x = Tensor([1.0, 2.0, 3.0])
    y = (2, 3)
    o1 = f1(x, y)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


GLOBAL_SCALE = 2.0
GLOBAL_TENSOR = ops.ones((4, 4))


def f2(input_tensor: Tensor, weights: Tensor, params: dict, sizes: tuple, count: int, tensors_list: list) -> tuple:
    outer_tensor = input_tensor * weights

    def f3(
        tensor1: Tensor, tensor2: Tensor, scale: float, config: dict, dims: tuple, values: list, flag: bool
    ) -> Tensor:
        scaled_tensor = tensor1 * GLOBAL_SCALE - count  # global var + free var
        print('GRAPH BREAK', end='')
        ret = ops.matmul(scaled_tensor, tensor2) + outer_tensor  # free var
        ret = ret + config['bias']

        if len(dims) > 2:
            ret = ret.reshape(dims)
            print('GRAPH BREAK', end='')
            ret += 1

        for val in values:
            ret = ret + ops.mean(val)
        return ret * scale

    new_config = {'bias': params['bias'] * 2, 'scale': params['scale'] * 1.5}

    new_dims = sizes + (1,)
    if count > 5:
        scale_factor = 2.0
    else:
        scale_factor = 1.0

    print('GRAPH BREAK', end='')
    result = f3(input_tensor, weights, scale_factor, new_config, new_dims, tensors_list, True)

    return result, new_config, new_dims


def f1(x: Tensor, y: Tensor, batch_size: int, shape: tuple, config: dict, tensor_array: list) -> Tensor:
    intermediate = x - ops.add(y, GLOBAL_TENSOR)
    print('GRAPH BREAK', end='')
    new_params = {'bias': 0.5, 'scale': config['learning_rate']}

    result, updated_config, dims = f2(intermediate, y, new_params, shape, batch_size, tensor_array)
    print('GRAPH BREAK', end='')
    return result * updated_config['scale']


@SKIP_PY37
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_many_params_and_many_alive_locals_and_free_vars():
    """
    Feature: test graph break in call_function.
    Description: param is dict, and it is alive local.
    Expectation: The result of PIJit is same as pynative.
    """

    x = ops.rand(4, 4)
    y = ops.rand(4, 4)
    batch_size = 8
    shape = (2, 8)
    config = {'learning_rate': 0.001}
    tensor_array = [ops.rand(2, 2) for _ in range(3)]

    o1 = f1(x, y, batch_size, shape, config, tensor_array)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, y, batch_size, shape, config, tensor_array)

    match_array(o1, o2, error=7)
    assert_has_graph_break(compiled_f1, break_count=1)


class SetattrNetV1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.scalar = 5
        self.tensor = Tensor([1.0, 2.0, 3.0])
        self.tuple_attr = (10, 20)
        self.list_attr = [Tensor([2.0, 4.0, 6.0]), Tensor([1.0, 3.0, 5.0])]

    def construct(self, x):
        return self._layer1(x)

    def _layer1(self, x):
        self.scalar += 1
        self.tensor -= x
        result = self._layer2(x)
        self.scalar += 2
        return result + self.tensor

    def _layer2(self, x):
        self.list_attr = [self.list_attr[0] * 2, self.list_attr[1] / 2]
        result = self._layer3(x, self.tuple_attr[0])
        self.tuple_attr = (self.tuple_attr[0] - self.tuple_attr[1],)
        return result + self.list_attr[0]

    def _layer3(self, x, val):
        temp = self.tensor + val
        result = temp + self._layer4(x, val)
        self.tensor = temp - x
        return result

    def _layer4(self, x, val):
        temp = self.scalar * x - val
        print('GRAPH BREAK', end='\n\n')
        self.scalar -= 1
        return temp


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setattr_side_effect_v1():
    """
    Feature: test graph break in call_function.
    Description: Has setattr side-effect operations.
    Expectation: The result of PIJit is same as pynative.
    """

    x = Tensor([0.5, 1.0, 1.5])
    net1 = SetattrNetV1()
    o1 = net1(x)

    net2 = SetattrNetV1()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    x = Tensor([0.5, 1.0, 1.5])
    o2 = net2(x)

    match_array(o1, o2, error=7)
    assert_has_graph_break(net2.construct, break_count=1)
    assert_equal(net1.scalar, net2.scalar)
    match_array(net1.tensor, net2.tensor, error=7)
    assert_equal(net1.list_attr, net2.list_attr)
    assert_equal(net1.tuple_attr, net2.tuple_attr)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_in_if_block_v1():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        if len(a.shape) >= 2:
            print('GRAPH BREAK', flush=True)  # break
            a = a * 2
        return a + 1

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_in_if_block_v2():
    """
    Feature: test graph break in call_function.
    Description: one graph break in f2.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        if len(a.shape) >= 2:
            if len(a.shape) < 3:
                print('GRAPH BREAK', flush=True)  # break
                return a
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_statement():
    """
    Feature: test graph break in call_function.
    Description: graph break at if statement. This situation is unsupported for now, cannot apply optimization.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        if a.sum() >= 100:  # break!
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_elif_else_statement():
    """
    Feature: test graph break in call_function.
    Description: graph break at if-elif-else statement, each branch compares Tensor and scalar.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1  # alive local
        if a.sum() > 100:  # break!
            a = a * 2
        elif a.sum() > 50:  # break!
            a = a * 3
        else:
            a = a * 4
        return a + 1

    def f1(x: Tensor):
        y = x * 2  # alive local
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_nested_if_else_statement():
    """
    Feature: test graph break in call_function.
    Description: graph break at nested if-else statement, inner and outer both compare Tensor and scalar.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        if a.sum() > 100:  # outer break!
            if a.mean() > 50:
                a = a * 2
            else:
                a = a * 3
        else:
            if a.min() > 0:  # inner break!
                a = a * 4
            else:
                a = a * 5
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_or_condition():
    """
    Feature: test graph break in call_function.
    Description: graph break at if with 'or' condition (Tensor and scalar).
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        if a.sum() > 100 or a.mean() > 50:  # break!
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_and_condition():
    """
    Feature: test graph break in call_function.
    Description: graph break at if with 'and' condition (Tensor and scalar).
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        if a.sum() > 10 and a.mean() > 2:  # break!
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_call_function_return_scalar_tensor():
    """
    Feature: test graph break in call_function.
    Description: graph break at if condition where the condition is a function call returning a 1-element Tensor.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        # Returns a Tensor with a single element
        return ops.sum(x)

    def f2(x: Tensor):
        a = x + 1
        if f3(a) > 10:  # break!
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_if_condition_in_nested_call():
    """
    Feature: test graph break in call_function.
    Description: f1 calls f2, f2 calls f3, and f3 has a variable if-condition that triggers graph break.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f3(x: Tensor):
        a = x + 1
        if a.sum() > 10:  # break!
            a = a * 2
        else:
            a = a * 3
        return a

    def f2(x: Tensor):
        b = x * 2
        return f3(b)

    def f1(x: Tensor):
        y = x - 1
        z = f2(y)
        return z + x

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_return_and_condition():
    """
    Feature: test graph break in call_function.
    Description: f2 returns (condition-a and condition-b), both involving Tensor and scalar.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        # JUMP_IF_TRUE_OR_POP, break only once.
        return (a.sum() > 10) and (a.mean() > 2)

    def f1(x: Tensor):
        cond = f2(x)
        return cond

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    assert o1 == o2
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_at_return_and_condition_v2():
    """
    Feature: Subgraph break at return with logical and-condition.
    Description: f2 returns an and-expression on Tensor reductions; f1 returns f2 result.
    Expectation: JIT result matches pynative; generates 2 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_006
    """

    def f2(x: Tensor):
        a = x + 1
        return (a.sum() > 10) and (a.mean() > 2)

    def f1(x: Tensor):
        return f2(x)

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    assert o1 == o2
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_return_or_condition():
    """
    Feature: test graph break in call_function.
    Description: f2 returns (condition-a or condition-b), both involving Tensor and scalar.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        # JUMP_IF_TRUE_OR_POP, break only once.
        return (a.sum() > 10) or (a.mean() > 100)

    def f1(x: Tensor):
        cond = f2(x)
        return cond

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    assert o1 == o2
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_return_complex_and_or_condition():
    """
    Feature: test graph break in call_function.
    Description: f2 returns a complex and/or expression, f1 returns f2(...) and another expression.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        # The first and: POP_JUMP_IF_FALSE (break!)
        # or: JUMP_IF_TRUE_OR_POP (skipped! As a.sum() > 10 is False)
        # The second and: JUMP_IF_FALSE_OR_POP (break!).
        return ((a.sum() > 10) and (a.mean() > 2)) or ((a.amin() > 0) and (a.amax() < 100))

    def f1(x: Tensor):
        # JUMP_IF_FALSE_OR_POP (break!)
        return f2(x) and x.sum() > 0

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    assert o1 == o2
    assert_has_graph_break(f1, break_count=1)
    check_ir_num('graph_before_compile', 4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_while_condition():
    """
    Feature: test graph break in call_function.
    Description: while loop condition is a Tensor and scalar comparison, triggers graph break.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        count = 0
        while a.sum() < 20:  # POP_JUMP_IF_FALSE, break!
            a = a + 2
            count += 1
        return a, count

    def f1(x: Tensor):
        a, cnt = f2(x)
        return a * cnt

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_while_condition_with_if_continue():
    """
    Feature: test graph break in call_function.
    Description: while loop condition is Tensor < scalar (break), loop body has if (Tensor > scalar) + continue.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        count = 0
        while a.sum() < 20:  # break!
            if a.max() > 10:  # break!
                a = a - 1
                count += 1
                continue
            a = a + 2
            count += 1
        return a, count

    def f1(x: Tensor):
        a, cnt = f2(x)
        return a * cnt

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_while_condition_with_if_break():
    """
    Feature: test graph break in call_function.
    Description: while loop condition is Tensor < scalar (break), loop body has if (Tensor > scalar) + break.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f2(x: Tensor):
        a = x + 1
        count = 0
        while a.sum() < 20:  # break!
            if a.max() > 10:  # break!
                count += 1
                break
            a = a + 2
            count += 1
        return a, count

    def f1(x: Tensor):
        a, cnt = f2(x)
        return a * cnt

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    x = Tensor([1, 2, 3])
    o2 = f1(x)

    match_array(o1, o2)
    assert_has_graph_break(f1, break_count=1)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_at_calling_nested_Cell():
    """
    Feature: test graph break in call_function.
    Description: one graph break at callling nested nn.Cell.
    Expectation: The result of PIJit is same as pynative.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Layer3(nn.Cell):
        def construct(self, x):
            y = x + 1
            print('GRAPH BREAK', flush=True)  # break!
            return x + y

    class Layer2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layer = Layer3()

        def construct(self, x):
            x = x * 2
            y = self.layer(x)
            return x + y

    class Layer1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layer = Layer2()

        def construct(self, x):
            x = x * 2
            y = self.layer(x)
            return x + y

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layer = Layer1()

        def construct(self, x: Tensor):
            x = x * 2
            y = self.layer(x)
            return x + y

    model = Model()
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o1 = model(x)

    model.construct = pi_jit_with_config(model.construct, jit_config=jit_cfg)
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    o2 = model(x)

    match_array(o1, o2)
    assert_has_graph_break(model.construct, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_if_condition_asnumpy():
    """
    Feature: Subgraph break at if condition using Tensor.asnumpy().
    Description: f2 compares Tensor with scalar via asnumpy() in if condition; f1 calls f2.
    Expectation: JIT result matches pynative; generates 2 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_001
    """

    def f2(x: Tensor):
        a = x + 1
        if a.asnumpy() >= 3:  # graph break
            a = a * 2
        else:
            a = a * 3
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([1])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_if_elif_condition_asnumpy():
    """
    Feature: Subgraph breaks at if/elif conditions using Tensor.asnumpy().
    Description: f2 has asnumpy() in if and elif conditions; both conditions are evaluated sequentially.
    Expectation: JIT result matches pynative; generates 3 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_002
    """

    def f2(x: Tensor):
        a = x + 1
        if a.asnumpy() > 10:  # graph break
            a = a * 2
        elif (a + 1).asnumpy() > 8:  # graph break
            a = a * 3
        else:
            a = a * 4
        return a + 1

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([2])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_elif_condition_asnumpy():
    """
    Feature: Subgraph break at elif condition using Tensor.asnumpy().
    Description: f2 has break only at elif condition; f1 passes an extra scalar arg z.
    Expectation: JIT result matches pynative; generates 2 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_003
    """

    def f2(x: Tensor, z):
        a = x + 1
        if z < 2:
            a = a * 2
        elif a.asnumpy() > 8:  # graph break
            a = a * 3
        else:
            a = a * 4
        return a + 1

    def f1(x: Tensor, z):
        y = x * 2
        w = f2(x, z)
        return y + w

    x = Tensor([2])
    o1 = f1(x, 2)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x, 2)

    match_array(o1, o2)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_two_separate_if_conditions_asnumpy():
    """
    Feature: Subgraph breaks at two separate if conditions using Tensor.asnumpy().
    Description: f2 contains two independent if-statements, both conditions use asnumpy(); f1 calls f2.
    Expectation: JIT result matches pynative; generates 3 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_004
    """

    def f2(x: Tensor):
        a = x + 1
        b = x * 5
        if a.asnumpy() >= 3:  # graph break
            a = a * 2
        else:
            a = a * 3
        if b.asnumpy() >= 6:  # graph break
            b = b * 2
        else:
            b = b * 3
        return a + b

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([1])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_nested_if_multiple_breaks_asnumpy():
    """
    Feature: Subgraph breaks in nested-if structure (multiple potential breaks).
    Description: f2 has nested if/elif/else with conditions on Tensor reductions; f1 calls f2.
    Expectation: JIT result matches pynative; generates 4 graphs.
    Migrated from: test_parse_pijit_improve_if_split.py::test_parse_pijit_improve_if_split_005
    """

    def f2(x: Tensor):
        a = x + 1
        b = x * 2
        c = a + b
        d = b + c
        if a.sum() >= 2:  # graph break
            if b.sum() >= 60:  # graph break
                b = b * 3
            elif c.sum() >= 70:  # graph break
                c = c * 2
            else:
                c = c * 3
        else:
            if d.sum() >= 100:
                d = d * 4
        return a + b + c + d

    def f1(x: Tensor):
        y = x * 2
        z = f2(x)
        return y + z

    x = Tensor([1, 2, 3])
    o1 = f1(x)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    o2 = compiled_f1(x)

    match_array(o1, o2)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 4)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_two_layer_tensor_asnumpy():
    """
    Feature: Subgraph break triggered by Tensor.asnumpy conversion.
    Description: Two-layer nested functions convert tensor to numpy and back inside callee.
    Expectation: JIT result matches pynative; generates 2 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_001
    """

    def f2(x: Tensor):
        x = x - 1
        x = Tensor(x.asnumpy())
        return x * 2

    def f1(x: Tensor):
        x = x * 2
        y = f2(x)
        return x + y

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_cell_multiple_tensor_asnumpy():
    """
    Feature: Subgraph break in Cell method with multiple Tensor.asnumpy calls.
    Description: Cell method converts tensor to numpy twice and mixes arithmetic, creating chain of breaks.
    Expectation: JIT result matches pynative; generates 3 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_002
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 2

        def construct(self, x: Tensor):
            x = x * self.scale
            y = self.f2(x)
            return x + y

        def f2(self, x: Tensor):
            x = x - 1
            x = Tensor(x.asnumpy())
            x = x + 1
            x.asnumpy()
            return x * 2

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_net = Net()
    pynative_out = pynative_net(Tensor(input_data))

    jit_net = Net()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    jit_out = jit_net(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(jit_net.construct, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_three_layers_negative_chain():
    """
    Feature: Subgraph break in three-layer function chain.
    Description: Deeply nested helpers apply neg operations with a Tensor.asnumpy call in the innermost function.
    Expectation: JIT result matches pynative; generates 2 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_003
    """

    def f3(x: Tensor):
        x = x + 1
        x.asnumpy()
        return P.Neg()(x)

    def f2(x: Tensor):
        x = f3(x)
        return P.Neg()(x)

    def f1(x: Tensor):
        x = f2(x)
        return P.Neg()(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_three_layers_multiple_asnumpy():
    """
    Feature: Subgraph break in three-layer function chain with multiple asnumpy calls.
    Description: Middle and inner helpers both convert tensors via asnumpy, leading to additional graphs.
    Expectation: JIT result matches pynative; generates 3 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_004
    """

    def f3(x: Tensor):
        x = x + 1
        x.asnumpy()
        return P.ReLU()(x)

    def f2(x: Tensor):
        x = f3(x)
        x.asnumpy()
        return P.ReLU()(x)

    def f1(x: Tensor):
        x = f2(x)
        return P.ReLU()(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_chained_calls_three_graphs():
    """
    Feature: Subgraph break across sequential helper calls.
    Description: Caller invokes two helpers with Tensor.asnumpy conversions, accumulating three graph breaks.
    Expectation: JIT result matches pynative; generates 4 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_005
    """

    def f3(x: Tensor):
        x = x + 1
        x.asnumpy()
        return P.ReLU()(x)

    def f2(x: Tensor):
        x = f3(x)
        x.asnumpy()
        return P.ReLU()(x)

    def f1(x: Tensor):
        x = f2(x)
        x = f3(x)
        return P.ReLU()(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 4)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_four_layers_tensor_merge():
    """
    Feature: Subgraph break in four-layer helper chain.
    Description: Nested functions combine Tensor.asnumpy conversion with additional helper returning numpy-backed tensor.
    Expectation: JIT result matches pynative; generates 3 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_006
    """

    def f4(x: Tensor):
        return x + Tensor(x.asnumpy())

    def f3(x: Tensor):
        x = f4(x) + 1
        x.asnumpy()
        return P.ReLU()(x)

    def f2(x: Tensor):
        x = f3(x)
        x.asnumpy()
        return P.ReLU()(x)

    def f1(x: Tensor):
        x = f2(x)
        return P.ReLU()(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 3)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_recursion_tensor_conversion():
    """
    Feature: Subgraph break inside recursive function.
    Description: Recursive helper converts tensor via asnumpy in base case while recursing on Tensor subtraction.
    Expectation: JIT result matches pynative; generates 2 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_007
    """

    def recursive_fn(x: Tensor):
        if x.all() > 1:
            return recursive_fn(x - 1) + x
        return x + Tensor(x.asnumpy())

    def recursive_baseline(x: Tensor):
        if x.all() > 1:
            return recursive_baseline(x - 1) + x
        return x + Tensor(x.asnumpy())

    input_tensor = Tensor(np.ones([2, 3]).astype(np.float32))
    pynative_out = recursive_baseline(input_tensor)

    compiled_recursive = pi_jit_with_config(recursive_fn, jit_config=jit_cfg)
    jit_out = compiled_recursive(Tensor(np.ones([2, 3]).astype(np.float32)))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_recursive, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_entry_tensor_conversion():
    """
    Feature: Subgraph break at callee entry.
    Description: Helper converts tensor to numpy at first statement before additional operations.
    Expectation: JIT result matches pynative; generates 2 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_008
    """

    def f3(x: Tensor):
        x = x + 1
        x.asnumpy()
        return P.ReLU()(x)

    def f2(x: Tensor):
        x = Tensor(x.asnumpy())
        x = f3(x)
        return P.ReLU()(x)

    def f1(x: Tensor):
        x = f2(x)
        return P.ReLU()(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_out = f1(Tensor(input_data))

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
    check_ir_num('graph_before_compile', 2)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_if_branch_with_prints():
    """
    Feature: Subgraph break inside conditional branches with side effects.
    Description: Cell helper prints in if-branch and nested call, causing graph breaks within branch logic.
    Expectation: JIT result matches pynative; generates 3 graphs and reports 1 break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_009
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.ReLU()

        def construct(self, x: Tensor):
            x = self.f2(x)
            return self.op(x)

        def f2(self, x: Tensor):
            if len(x.shape) > 1:
                x = x + 1
                print('GRAPH BREAK', flush=True)
                x = x + self.f3(x)
            return self.op(x)

        def f3(self, x: Tensor):
            x = x + 1
            print('GRAPH BREAK', flush=True)
            return self.op(x)

    input_data = np.random.randn(2, 3).astype(np.float32)
    pynative_net = Net()
    pynative_out = pynative_net(Tensor(input_data))

    jit_net = Net()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_cfg)
    jit_out = jit_net(Tensor(input_data))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(jit_net.construct, break_count=1)
    check_ir_num('graph_before_compile', 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_subgraph_break_if_condition_tensor_comparison():
    """
    Feature: Subgraph break in if condition using Tensor conversion.
    Description: Condition compares Tensor.asnumpy() result with scalar before invoking helper that also breaks.
    Expectation: JIT result matches pynative and records a single graph break.
    Migrated from: test_parse_pijit_subgraph_split.py::test_pijit_subgraph_010
    """

    def f3(x: Tensor):
        x = x + 1
        x.asnumpy()
        return P.ReLU()(x)

    def f2(x: Tensor):
        if Tensor(x.asnumpy()) > 2:
            x = x + f3(x)
        return P.ReLU()(x)

    def f1(x: Tensor):
        x = f2(x)
        return P.ReLU()(x)

    input_tensor = Tensor([3])
    pynative_out = f1(input_tensor)

    compiled_f1 = pi_jit_with_config(f1, jit_config=jit_cfg)
    jit_out = compiled_f1(Tensor([3]))

    match_array(pynative_out, jit_out)
    assert_has_graph_break(compiled_f1, break_count=1)
