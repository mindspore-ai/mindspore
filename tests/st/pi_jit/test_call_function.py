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
"""test bytecode CALL_FUNCTION*"""
import pytest
from mindspore import numpy as np
from mindspore import Tensor, jit, context, ops, nn
from mindspore._c_expression import get_code_extra
from .share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.st.pi_jit.conftest import run_in_subprocess


def func(x, k=1):
    return x + k

@jit(capture_mode="bytecode")
def jit_test1(x):
    return func(x)

@jit(capture_mode="bytecode")
def jit_test2(x):
    y = (x,)
    return func(*y)

@jit(capture_mode="bytecode")
def jit_test3(x):
    return func(x, k=10)

@jit(capture_mode="bytecode")
def jit_test4(x):
    d = {'k': 10}
    return func(x, **d)

@jit(capture_mode="bytecode")
def jit_test5(x):
    y = (x,)
    return func(*y, k=10)

@jit(capture_mode="bytecode")
def jit_test6(x):
    y = (x,)
    d = {'k': 10}
    return func(*y, **d)


def python_test1(x):
    return func(x)


def python_test2(x):
    y = (x,)
    return func(*y)


def python_test3(x):
    return func(x, k=10)


def python_test4(x):
    d = {'k': 10}
    return func(x, **d)


def python_test5(x):
    y = (x,)
    return func(*y, k=10)


def python_test6(x):
    y = (x,)
    d = {'k': 10}
    return func(*y, **d)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test1])
@pytest.mark.parametrize('jit_func', [jit_test1])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function1(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test2])
@pytest.mark.parametrize('jit_func', [jit_test2])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function2(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test3])
@pytest.mark.parametrize('jit_func', [jit_test3])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function3(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test4])
@pytest.mark.parametrize('jit_func', [jit_test4])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function4(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test5])
@pytest.mark.parametrize('jit_func', [jit_test5])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function5(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('python_func', [python_test6])
@pytest.mark.parametrize('jit_func', [jit_test6])
@pytest.mark.parametrize('x', [Tensor(np.ones((2, 3)).astype(np.float32))])
def test_call_function6(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = python_func(x)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@run_in_subprocess({'GLOG_v': '1', 'MS_SUBMODULE_LOG_v': '{PI:0}'})
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_return_a_freevar():
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: A subgraph returns a free variable.
    Expectation: no graph break.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def fn(x: Tensor):
        y = ops.add(x, 1)

        def inner():
            return x  # this subgraph returns a free variable

        z = inner()
        return ops.sub(y, z)

    a = Tensor([1, 2])
    o1 = fn(a)

    compiled_fn = pi_jit_with_config(fn, jit_config={'compile_with_try': False})
    o2 = compiled_fn(a)

    match_array(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_support_create_user_defined_class():
    """
    Feature: call function.
    Description: support creating user defined class.
    Expectation: no graph break.
    """
    var = 1

    class myclass():
        def __init__(self):
            self.var = var

        def __enter__(self):
            return None

        def __exit__(self):
            return False

        def clone(self):
            return self.__class__()

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        @jit(capture_mode='bytecode')
        def construct(self):
            a = myclass()
            b = a.clone()
            return b.var

    net = Net()
    res = net()
    jcr = get_code_extra(net.construct.__wrapped__)
    assert(res == var)
    assert(jcr["break_count_"] == 0)
