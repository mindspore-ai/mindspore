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
import pytest 
import numpy as np
from mindspore import Tensor, jit, context
from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark

@jit(capture_mode="bytecode")
def fallback_tuple_with_input_list(a):
    res = tuple(a)
    return res


@jit(capture_mode="bytecode")
def fallback_tuple_with_input_dict(a):
    res = tuple(a)
    return res


@jit(capture_mode="bytecode")
def fallback_tuple_with_input_numpy_array(a):
    res = tuple(a)
    return res


@jit(capture_mode="bytecode")
def fallback_tuple_with_input_numpy_tensor(a, b):
    res = tuple(a)
    res2 = tuple(b)
    res3 = tuple(())
    return res, res2, res3


@jit
def ms_fallback_tuple_with_input_list(a):
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_dict(a):
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_numpy_array():
    a = np.array([1, 2, 3])
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_numpy_tensor(a, b):
    res = tuple(a)
    res2 = tuple(b)
    res3 = tuple(())
    return res, res2, res3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_tuple_with_input_list])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_list])
@pytest.mark.parametrize('a', [[1, 2, 3]])
def test_list_with_input_tuple(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: list
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_tuple_with_input_dict])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_dict])
@pytest.mark.parametrize('a', [{'a': 1, 'b': 2, 'c': 3}])
def test_list_with_input_dict(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: dict
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_tuple_with_input_numpy_array])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_numpy_array])
@pytest.mark.parametrize('a', [np.array([1, 2, 3])])
def test_list_with_input_array(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: numpy array
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip(reason="pynative mode mix graph mode, results has an random error in pynative")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_tuple_with_input_numpy_tensor])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_numpy_tensor])
@pytest.mark.parametrize('a', [Tensor([1, 2])])
@pytest.mark.parametrize('b', [Tensor([2, 3])])
def test_list_with_input_tensor(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: tensor and (); output tuple
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))
    match_array(res[2], ms_res[2], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_executed_by_graph():
    """
    Feature: PIJit.
    Description: Support python built-in function tuple in pijit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        return tuple((x, x + 1, x + 2))

    x = Tensor([1, 2, 3, 4])
    out = func(x)
    assert isinstance(out, tuple)
    assert np.all(out[0].asnumpy() == x.asnumpy())
    assert np.all(out[1].asnumpy() == (x + 1).asnumpy())
    assert np.all(out[2].asnumpy() == (x + 2).asnumpy())
    assert_executed_by_graph_mode(func)
