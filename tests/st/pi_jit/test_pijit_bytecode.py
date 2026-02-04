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
"""Test pijit bytecode"""

from math import pi

import numpy as np

import mindspore as ms
from mindspore import Tensor

from tests.mark_utils import arg_mark
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num
from tests.st.pi_jit.share.utils import match_array, match_value


def _clone_input(data):
    if isinstance(data, Tensor):
        return Tensor(data.asnumpy())
    if isinstance(data, (list, tuple)):
        cloned = [_clone_input(item) for item in data]
        return type(data)(cloned)
    if isinstance(data, dict):
        return {key: _clone_input(value) for key, value in data.items()}
    return data


def _clone_inputs(*inputs):
    return [_clone_input(item) for item in inputs]


def _run_cell(cell_factory, *base_inputs):
    pynative_net = cell_factory()
    pynative_inputs = _clone_inputs(*base_inputs)
    pynative_result = pynative_net(*pynative_inputs)

    jit_net = cell_factory()
    jit_net.construct = ms.jit(jit_net.construct, capture_mode="bytecode")
    jit_inputs = _clone_inputs(*base_inputs)
    jit_result = jit_net(*jit_inputs)
    return pynative_result, jit_result


def _run_function(function, *base_inputs):
    pynative_inputs = _clone_inputs(*base_inputs)
    pynative_result = function(*pynative_inputs)

    jit_function = ms.jit(function, capture_mode="bytecode")
    jit_inputs = _clone_inputs(*base_inputs)
    jit_result = jit_function(*jit_inputs)
    return pynative_result, jit_result



@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unary_ops():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify unary operators when compiling a Cell with bytecode capture.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_unary
    """

    class UnaryNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.flag = True

        def construct(self, x):
            if not self.flag:
                y = -x
            else:
                y = +x
            out = (~5) * y
            return out

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(UnaryNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([-6, -12], np.int32))



@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_inplace_ops():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify in-place arithmetic instructions in bytecode compiled Cell.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_inplace
    """

    class InplaceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.power = 2

        def construct(self, x):
            x **= self.power
            x *= 3
            x //= 2
            return x

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(InplaceNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([1, 6], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_async_function_call():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Compile Cells that call async functions inside construct.
    Expectation: JIT preserves the same behaviour as pynative.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_async
    """

    async def move_data(x):
        for item in x:
            return item.move_to("CPU")

    async def move_func(x, y):
        data1 = await move_data(x)
        data2 = await move_data(y)
        return data1, data2

    class AsyncNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 2

        def construct(self, x):
            y = x * x * self.scale
            move_func(x, y)
            return None

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(AsyncNet, base_input)
    assert pynative_result is None
    assert jit_result is None


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_comprehension_operations():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Execute list/set/dict comprehensions inside bytecode compiled Cell.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_comprehension
    """

    class ComprehensionNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.repeats = 4

        def construct(self, x):
            list_items = [x for _ in range(self.repeats)]
            set_items = {x for _ in range(3)}
            dict_items = {str(i): x for i in range(3)}
            summed = list_items[0] + dict_items.get('1')
            out = summed * len(set_items)
            return out

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(ComprehensionNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([2, 4], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_build_class():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Compile Cells that define inner classes at runtime.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_build_class_func
    """

    class BuildClassNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.coeff = 2

        def construct(self, x):
            class Inner:
                factor = self.coeff

            def inner():
                value = Inner()
                return value.factor * x

            out = x * inner()
            return out

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(BuildClassNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([2, 8], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_sequence_unpack():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify sequence unpacking bytecode instructions.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_unpack
    """

    class SequenceUnpackNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.factor = 5

        def construct(self, x):
            a = x * self.factor
            b = x * 2
            series = (x, b, x * x, a // b, a % b)
            i, *j, k = series
            m, n, _ = j
            return i + k + m + n

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(SequenceUnpackNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([5, 12], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_load_method():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Ensure bound method lookups work in bytecode compiled Cells.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_load_method
    """

    class LoadMethodNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.mask = 1

        def bitwise(self, value):
            return value ^ self.mask

        def construct(self, x):
            return self.bitwise(x)

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(LoadMethodNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([0, 3], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_function_call_unpack():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Test argument unpacking for positional and keyword parameters.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_func_call
    """

    class FunctionCallNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.key = 'primary'

        def construct(self, x):
            def tuple_add(left, right):
                return left + right

            def dict_add(**items):
                return items[self.key] + items['secondary']

            tuple_value = (x, x)
            dict_value = {self.key: x, 'secondary': x}
            positional = tuple_add(*tuple_value)
            keyword = dict_add(**dict_value)
            return positional * keyword

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(FunctionCallNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([4, 16], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_import_from_math():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify imported constants are usable inside bytecode compiled Cells.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_import
    """

    class ImportNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, x):
            return x * pi * self.scale

    base_input = Tensor(np.array([1, 2], np.float32))
    pynative_result, jit_result = _run_cell(ImportNet, base_input)
    match_array(pynative_result, jit_result, error=5)
    match_array(pynative_result, base_input.asnumpy() * pi, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_build_str():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Use f-string construction inside bytecode compiled Cells.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_build_str
    """

    class BuildStrNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, x):
            result = (x + x) * self.scale
            info = f"{result.shape},{result.dtype}"
            return result, info

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(BuildStrNet, base_input)
    match_array(pynative_result[0], jit_result[0])
    assert pynative_result[1] == jit_result[1] == "(2,),Int32"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_shift_operations():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Validate left and right shift bytecode operations.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_shift
    """

    class ShiftNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.value = 8

        def construct(self, x):
            shifted = self.value << 2
            result = shifted >> 3
            return x * result

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(ShiftNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([4, 8], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_load_closure():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify LOAD_CLOSURE/LOAD_DEREF behaviour when returning inner functions.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_load_closure
    """

    class LoadClosureNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, value):
            def outer(x):
                def inner(y):
                    return x + y

                return inner

            func = outer(value)
            out = func(value + value)
            return out * self.scale

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(LoadClosureNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([3, 6], np.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_load_deref():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Validate access to closed-over values inside nested functions.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_load_deref
    """

    class LoadDerefNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.scale = 1

        def construct(self, value):
            def outer():
                captured = value

                def inner():
                    return captured

                return inner

            func = outer()
            out = func()
            return out * self.scale

    base_input = Tensor(np.array([1, 2], np.int32))
    pynative_result, jit_result = _run_cell(LoadDerefNet, base_input)
    match_array(pynative_result, jit_result)
    match_array(pynative_result, np.array([1, 2], np.int32))


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_try_except_single_graph():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Compile try/except/finally blocks and ensure one graph is generated.
    Expectation: JIT result matches pynative result and saves a single graph.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_try
    """

    class TryNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            out = x + x
            try:
                raise ValueError
            except ValueError:
                out = x + out
            finally:
                pass
            return out

    base_input = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    pynative_net = TryNet()
    pynative_result = pynative_net(_clone_input(base_input))

    jit_net = TryNet()
    jit_net.construct = ms.jit(jit_net.construct, capture_mode="bytecode")
    jit_result = jit_net(_clone_input(base_input))
    match_array(pynative_result, jit_result, error=5)
    check_ir_num('graph_before_compile', 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_try_in_try_raise_outer():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Nested try/except blocks that re-raise exceptions to outer scope.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_try_in_try_001
    """

    class TryInTryNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            out = x
            try:
                out = x + x
                try:
                    raise ValueError
                except ValueError:
                    raise TypeError
            except TypeError:
                out = x * out
            return out

    base_input = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    pynative_result, jit_result = _run_cell(TryInTryNet, base_input)
    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_try_in_try_no_match_inner():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Nested try/except with mismatched inner exception handlers.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_try_in_try_002
    """

    class TryInTryNoMatchNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            out = x
            try:
                out = x + x
                try:
                    raise ValueError
                except TypeError:
                    out = out + out
            except ValueError:
                out = x * out
            return out

    base_input = Tensor(np.arange(4, dtype=np.float32).reshape(2, 2))
    pynative_result, jit_result = _run_cell(TryInTryNoMatchNet, base_input)
    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_try_in_try_raise_in_finally():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Nested try/except with finally raising to outer handler.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_try_in_try_003
    """

    class TryInTryFinallyNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            out = x
            try:
                out = x + x
                try:
                    out = x + out
                except TypeError:
                    out = out + out
                finally:
                    raise ValueError
            except ValueError:
                out = x * out
            return out

    base_input = Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    pynative_result, jit_result = _run_cell(TryInTryFinallyNet, base_input)
    match_array(pynative_result, jit_result, error=5)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_try_with_graph_break():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Handle graph break caused by asnumpy inside try/except.
    Expectation: JIT result matches pynative result and saves a single graph.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_try_split_graph
    """

    class TrySplitGraphNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            y = x + x
            try:
                raise ValueError
            except ValueError:
                numpy_out = y.asnumpy() + y.asnumpy()
                out = Tensor(numpy_out)
            finally:
                pass
            return out

    base_input = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    pynative_net = TrySplitGraphNet()
    pynative_result = pynative_net(_clone_input(base_input))

    jit_net = TrySplitGraphNet()
    jit_net.construct = ms.jit(jit_net.construct, capture_mode="bytecode")
    jit_result = jit_net(_clone_input(base_input))
    match_array(pynative_result, jit_result, error=5)
    check_ir_num('graph_before_compile', 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_with_context_manager_try():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Ensure with-statement and try/except interplay is preserved.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_with_try
    """

    class MyContext:
        def __init__(self):
            self.handle = True
            self.default = False

        def __enter__(self):
            return self.handle

        def __exit__(self, exc_type, value, trace):
            if exc_type == ValueError:
                return self.handle
            return self.default

    def body(x):
        context = MyContext()
        with context:
            try:
                raise ValueError
            except ValueError:
                out = x + x
            out = out + x
        return out

    base_input = Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    pynative_result, jit_result = _run_function(body, base_input)
    match_array(pynative_result, jit_result, error=5)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_list():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Verify list unpacking bytecode when expanding multiple inputs.
    Expectation: JIT result matches pynative result and saves a single graph.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_list_unpack
    """

    def func(x, y, z):
        return [*x, *y, *z]

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    base_inputs = (
        [base_tensor, base_tensor],
        [base_tensor, base_tensor],
        [base_tensor, base_tensor],
    )

    pynative_inputs = _clone_inputs(*base_inputs)
    pynative_result = func(*pynative_inputs)

    jit_inputs = _clone_inputs(*base_inputs)
    jit_func = ms.jit(func, capture_mode="bytecode")
    jit_result = jit_func(*jit_inputs)
    match_value(pynative_result, jit_result)
    check_ir_num('graph_before_compile', 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_tuple():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Validate tuple unpack expansion inside bytecode compiled function.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_tuple_unpack
    """

    def func(x, y, z):
        return (*x, *y, *z)

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    base_inputs = (
        (base_tensor, base_tensor),
        (base_tensor, base_tensor),
        (base_tensor, base_tensor),
    )
    pynative_result, jit_result = _run_function(func, *base_inputs)
    match_value(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_map():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Ensure dict unpack expansion works in bytecode compiled function.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_map_unpack
    """

    def func(x, y, z):
        return {**x, **y, **z}

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    base_inputs = (
        {'a': base_tensor},
        {'b': base_tensor},
        {'c': base_tensor},
    )
    pynative_result, jit_result = _run_function(func, *base_inputs)
    match_value(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_tuple_container():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Construct nested tuples to validate BUILD_TUPLE instructions.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_tuple
    """

    def func(a, b, c, d, e, f, g, h, i):
        return (a, b, c, d, e, f, g, h, i)

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    tuple_input = (base_tensor, base_tensor)
    base_inputs = (tuple_input,) * 9
    pynative_result, jit_result = _run_function(func, *base_inputs)
    match_value(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_map_with_call():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Expand dictionaries when forwarding arguments to helper function.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_map_unpack_with_call
    """

    def build_tuple(a, b, c, d, e, f, g, h, i):
        return (a, b, c, d, e, f, g, h, i)

    def func(x, y, z):
        return build_tuple(**x, **y, **z)

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    base_inputs = (
        {'a': base_tensor, 'b': base_tensor, 'c': base_tensor},
        {'d': base_tensor, 'e': base_tensor, 'f': base_tensor},
        {'g': base_tensor, 'h': base_tensor, 'i': base_tensor},
    )
    pynative_result, jit_result = _run_function(func, *base_inputs)
    match_value(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_unpack_build_tuple_with_call():
    """
    Feature: ms.jit capture_mode='bytecode'.
    Description: Expand tuples when forwarding arguments to helper function.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_bytecode.py::test_pijit_bytecode_optimize_unpack_build_tuple_unpack_with_call
    """

    def build_tuple(a, b, c, d, e, f, g, h, i):
        return (a, b, c, d, e, f, g, h, i)

    def func(x, y, z):
        return build_tuple(*x, *y, *z)

    base_tensor = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    base_inputs = (
        (base_tensor, base_tensor, base_tensor),
        (base_tensor, base_tensor, base_tensor),
        (base_tensor, base_tensor, base_tensor),
    )
    pynative_result, jit_result = _run_function(func, *base_inputs)
    match_value(pynative_result, jit_result)
