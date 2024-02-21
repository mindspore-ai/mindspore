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
from mindspore import Tensor, jit, context
from .share.utils import match_array


def func(x, k=1):
    return x + k

@jit(mode="PIJit")
def jit_test1(x):
    return func(x)

@jit(mode="PIJit")
def jit_test2(x):
    y = (x,)
    return func(*y)

@jit(mode="PIJit")
def jit_test3(x):
    return func(x, k=10)

@jit(mode="PIJit")
def jit_test4(x):
    d = {'k': 10}
    return func(x, **d)

@jit(mode="PIJit")
def jit_test5(x):
    y = (x,)
    return func(*y, k=10)

@jit(mode="PIJit")
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


@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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


@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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

@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
