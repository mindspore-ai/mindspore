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
"""Test string operations"""
from mindspore import Tensor
from mindspore import context
from mindspore.graph.api import jit
from mindspore._c_expression import get_code_extra

from tests.st.pi_jit.share.utils import match_array
from tests.mark_utils import arg_mark


cfg = {"compile_with_try": False}


def assert_graph_break_count(func, break_count: int):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == break_count


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_1():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str):
        return f'{s1}-xxx'

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello')

    fn = jit(fn, capture_mode='bytecode')
    s2 = fn('hello')

    assert s1 == s2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_2():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str, s2: str):
        return f'{s1}{s2}-xxx'

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello', ' world')

    fn = jit(fn, capture_mode='bytecode')
    s2 = fn('hello', ' world')

    assert s1 == s2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_3():
    """
    Feature: Test build f-string.
    Description: Use int variable to build f-string.
    Expectation: No exception, but has graph break.
    """
    def fn(x: int):
        return f'x={x}'

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn(1)

    fn = jit(fn, capture_mode='bytecode')
    s2 = fn(1)

    assert s1 == s2
    assert_graph_break_count(fn, 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_4():
    """
    Feature: Test build f-string.
    Description: Use int variable to build f-string, and use this string to do dict getitem.
    Expectation: No exception, but has graph break.
    """
    def fn(x: int, d: dict):
        i = x * 2
        k = f'a{i}'
        v = d[k]
        return v + x

    context.set_context(mode=context.PYNATIVE_MODE)
    x = 1
    d = {'a1': Tensor([1, 2, 3]), 'a2': Tensor([2, 4, 6])}
    o1 = fn(x, d)

    fn = jit(fn, capture_mode='bytecode')
    o2 = fn(x, d)

    match_array(o1, o2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_string_add_string():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str, s2: str):
        return s1 + s2

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello', ' world')

    fn = jit(fn, capture_mode='bytecode')
    s2 = fn('hello', ' world')

    assert s1 == s2
