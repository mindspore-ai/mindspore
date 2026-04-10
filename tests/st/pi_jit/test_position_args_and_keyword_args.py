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
"""Test position arguments and keyword arguments"""
import pytest     
from mindspore import jit, context
from .share.utils import match_array
from tests.mark_utils import arg_mark

def position_args_and_keyword_args_1(a, *b, **c):
    return a + b[0] + b[1] + c["e"] + c["f"]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [position_args_and_keyword_args_1])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [[10, 11]])
@pytest.mark.parametrize('c', [{"e" : 1, "f" : 2}])
def test_position_args_and_keyword_args_1(func, a, b, c):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, *b, **c)
    jit_res = jit(func, capture_mode="bytecode")(a, *b, **c)
    match_array(res, jit_res, error=0, err_msg=str(jit_res))

def position_args_and_keyword_args_2(a, *b, **c):
    return a + b[0] + b[1] + c["e"] + c["f"]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [position_args_and_keyword_args_2])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [[10, 11]])
def test_position_args_and_keyword_args_2(func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, *b, e=2, f=3)
    jit_res = jit(func, capture_mode="bytecode")(a, *b, e=2, f=3)
    match_array(res, jit_res, error=0, err_msg=str(jit_res))

def position_args_and_keyword_args_3(a, b, *c):
    return a + b + c[0] + c[1] + c[2]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [position_args_and_keyword_args_3])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [2])
@pytest.mark.parametrize('c', [[3, 4, 5]])
def test_position_args_and_keyword_args_3(func, a, b, c):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, *c)
    jit_res = jit(func, capture_mode="bytecode")(a, b, *c)
    match_array(res, jit_res, error=0, err_msg=str(jit_res))

def position_args_and_keyword_args_4(a, b, *c):
    return a + b + c[0] + c[1] + c[2]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [position_args_and_keyword_args_4])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [2])
@pytest.mark.parametrize('c', [3])
@pytest.mark.parametrize('d', [4])
@pytest.mark.parametrize('e', [5])
def test_position_args_and_keyword_args_4(func, a, b, c, d, e):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, c, d, e)
    jit_res = jit(func, capture_mode="bytecode")(a, b, c, d, e)
    match_array(res, jit_res, error=0, err_msg=str(jit_res))

def position_args_and_keyword_args_5(a, b, *, c):
    return a + b + c

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [position_args_and_keyword_args_5])
@pytest.mark.parametrize('a', [1])
@pytest.mark.parametrize('b', [2])
def test_position_args_and_keyword_args_5(func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b, c=3)
    jit_res = jit(func, capture_mode="bytecode")(a, b, c=3)
    match_array(res, jit_res, error=0, err_msg=str(jit_res))