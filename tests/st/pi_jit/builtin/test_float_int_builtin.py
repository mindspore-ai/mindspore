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
import math
import pytest 
from mindspore import jit, context, Tensor
from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


def convert_numbers_to_float_and_int(val1, val2):
    return float(val1), float(val2), int(val1), int(val2)


@jit(capture_mode="bytecode")
def fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit(capture_mode="bytecode")
def fallback_float_and_int_empty():
    return float(), int()


@jit
def ms_fallback_float_and_int():
    return convert_numbers_to_float_and_int(5, 5.0)


@jit
def ms_fallback_float_and_int_empty():
    return float(), int()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_float_and_int])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int])
def test_int_float_conversion_with_args(func, ms_func):
    """
    Feature: Conversion of int and float
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_float_and_int_empty])
@pytest.mark.parametrize('ms_func', [ms_fallback_float_and_int_empty])
def test_int_float_conversion_no_args(func, ms_func):
    """
    Feature: Conversion of int and float without arguments
    Description: Test cases for argument support in PYNATIVE mode
    Expectation: Results match between GraphJit and JIT functions
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func()
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_builtin_function():
    """
    Feature: PIJit.
    Description: Support python built-in functions in pijit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        return float(x + 1), int(x + 1), bool(x + 1), str(x + 1)

    x = Tensor(1.0)
    out_float, out_int, out_bool, out_str = func(x)
    assert math.isclose(out_float, 2.0, abs_tol=1e-5)
    assert out_int == 2
    assert out_bool == True
    assert out_str == '2.0'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_builtin_executed_by_graph():
    """
    Feature: PIJit.
    Description: Support python built-in functions in pijit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        return float(x * 2), int(x * 2), bool(x * 2), str(x * 2)

    out_float, out_int, out_bool, out_str = func(1.0)
    assert math.isclose(out_float, 2.0, abs_tol=1e-5)
    assert out_int == 2
    assert out_bool == True
    assert out_str == '2.0'
