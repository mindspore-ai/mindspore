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
"""test yield bytecode implement"""
import sys
import pytest
from tests.mark_utils import arg_mark
from mindspore import jit
from mindspore._c_expression import get_code_extra


@pytest.mark.skipif(sys.version_info[:2] == (3,7), reason="not support py37 setup loop bytecode")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_yield_case_1():
    """
    Feature: Test yield
    Description: Test yield process
    Expectation: break count == 0.
    """

    def func():
        yield 1
        yield 2
        yield 3

    def func2():
        iter = func()
        sum = 0
        for i in iter:
            sum = sum + i
        return sum

    fn = jit(function=func2, capture_mode="bytecode")
    got = fn()
    expected = func2()
    assert got == expected
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr["break_count_"] == 0


@pytest.mark.skipif(sys.version_info[:2] == (3,7), reason="not support py37 setup loop bytecode")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_yield_case_2():
    """
    Feature: Test yield from
    Description: Test yield from process
    Expectation: break count == 0.
    """

    def func():
        yield from [1, 2, 3]

    def func2():
        iter = func()
        sum = 0
        for i in iter:
            sum = sum + i
        return sum

    fn = jit(function=func2, capture_mode="bytecode")
    got = fn()
    expected = func2()
    assert got == expected
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr["break_count_"] == 0


@pytest.mark.skipif(sys.version_info[:2] == (3,7), reason="not support py37 setup loop bytecode")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_yield_case_3():
    """
    Feature: Test yield from
    Description: Test yield from process
    Expectation: break count == 0.
    """

    def func():
        yield from iter([1, 2, 3])

    def func2():
        iter = func()
        sum = 0
        for i in iter:
            sum = sum + i
        return sum

    fn = jit(function=func2, capture_mode="bytecode")
    got = fn()
    expected = func2()
    assert got == expected
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_yield_case_4():
    """
    Feature: Test yield from
    Description: not support yield iterator,only support yield container
    Expectation: break count == 1.
    """

    def func(x):
        yield from x

    def func2(x):
        iter = func(x)
        sum = 0
        for i in iter:
            sum = sum + i
        return {"sum": sum}

    fn = jit(function=func2, capture_mode="bytecode")
    got = fn(iter([1, 2, 3]))
    expected = func2(iter([1, 2, 3]))
    assert got == expected
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr["break_count_"] == 1
