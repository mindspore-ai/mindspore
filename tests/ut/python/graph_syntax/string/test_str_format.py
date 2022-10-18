# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import pytest
from mindspore import jit, Tensor, jit_class, context
from mindspore.ops import prim_attr_register, Primitive
from mindspore.nn import Cell


def test_str_format_single_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "string is {}".format("1")
        return ms_str

    assert foo() == "string is 1"


def test_str_format_mutiple_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "{} is {}".format("string", "1")
        return ms_str

    assert foo() == "string is 1"


def test_str_format_constant_tensor_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = Tensor([1])
        ms_str = "{} is {}".format("string", a)
        return ms_str

    assert foo() == "string is Tensor(shape=[1], dtype=Int64, value=[1])"


def test_str_format_variable_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(b, a):
        a = a + b
        ms_str = "{} is {}".format("String", a)
        return ms_str

    with pytest.raises(ValueError) as ex:
        foo(Tensor([1]), Tensor([1]))
    assert "str.format not support to input a variable." in str(ex.value)


def test_fallback_str_format_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = Tensor([1])
        ms_str = format(a)
        ms_str2 = format(ms_str, "4")
        return ms_str, ms_str2

    ms_str, ms_str2 = foo()
    assert ms_str == "[1]"
    assert ms_str2 == "[1] "


def test_format_with_number_placeholder_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "{1} {0} {1}"
        ms_format_str = ms_str.format("hello", "world")
        return ms_format_str

    ms_str = foo()
    assert ms_str == "world hello world"


def test_format_with_key_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "hello {name2},It's me, {name1}"
        ms_format_str = ms_str.format(name2="Mind", name1="Spore")
        return ms_format_str

    with pytest.raises(TypeError) as ex:
        result_st = foo()
        assert result_st == "hello Mind,It's me, Spore"
    assert "Unsupported parameter type for python primitive," \
           " the parameter value is KeywordArg[key : name2, value : Mind]" in str(ex.value)


def test_format_with_list_index():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "hello {0[1]},It's me {0[0]}"
        names = ["Mind", "Spore"]
        ms_format_str = ms_str.format(names)
        return ms_format_str

    result_st = foo()
    assert result_st == "hello Spore,It's me Mind"


def test_format_with_map():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ms_str = "hello {0[name2]},It's me {0[name1]}"
        names = {"name1": "Mind", "name2": "Spore"}
        ms_format_str = ms_str.format(names)
        return ms_format_str

    result_st = foo()
    assert result_st == "hello Spore,It's me Mind"



def test_format_as_function():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    @jit
    def foo():
        func = "hello {0[1]},It's me {0[0]}".format
        names = ["Mind", "Spore"]
        ms_format_str = func(names)
        return ms_format_str

    result_st = foo()
    assert result_st == "hello Spore,It's me Mind"


def test_format_number():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    @jit
    def foo():
        num1 = 3.1415926
        str1_format = "{:.2f}".format(num1)
        str2_format = "{:.0f}".format(num1)
        num2 = 1000000
        str3_format = "{:,}".format(num2)
        num3 = 0.25
        str4_format = "{:.2%}".format(num3)
        num4 = 1000000000
        str5_format = "{:.2e}".format(num4)
        num5 = 25
        str6_format = "{0:b}".format(num5)
        str7_format = "{0:d}".format(num5)
        str8_format = "{0:o}".format(num5)
        str9_format = "{0:x}".format(num5)
        result = (str1_format, str2_format, str3_format, str4_format, str5_format,
                  str6_format, str7_format, str8_format, str9_format)
        return result

    correct_str = ("3.14", "3", "1,000,000", "25.00%", "1.00e+09", "11001", "25", "31", "19")
    result_str = foo()
    assert result_str == correct_str


def test_format_padding():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    @jit
    def foo():
        num1 = 5
        str1_format = "{:0>2}".format(num1)
        str2_format = "{:x<4}".format(num1)
        num2 = 10
        str3_format = "{:x^4}".format(num2)
        num3 = 13
        str4_format = "{:10}".format(num3)
        str5_format = "{:<10}".format(num3)
        str6_format = "{:^10}".format(num3)

        result = (str1_format, str2_format, str3_format, str4_format, str5_format, str6_format)
        return result

    correct_str = ("05", "5xxx", "x10x", "        13", "13        ", "    13    ")
    result_str = foo()
    assert result_str == correct_str


def test_str_format_using_ms_class():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    @jit_class
    class TestClass:
        def __init__(self, value):
            self.value = value

    @jit
    def test_func():
        test_obj = TestClass(123)
        format_str = "value is {0.value}".format(test_obj)
        return format_str
    format_str = test_func()
    assert format_str == "value is 123"


def test_str_format_using_ms_class_in_init():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    context.set_context(mode=context.GRAPH_MODE)

    @jit_class
    class TestClass:
        def __init__(self, value):
            self.value = value

    class TestCell(Cell):
        def __init__(self):
            super(TestCell, self).__init__()
            self.obj = TestClass(123)
        def construct(self):
            format_str = "value is {0.value}".format(self.obj)
            return format_str

    test_cell = TestCell()
    format_str = test_cell()
    assert format_str == "value is 123"


def test_str_format_using_primitive():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    class TestPrim(Primitive):
        @prim_attr_register
        def __init__(self, x):
            self.x = x

    @jit
    def test_func():
        test_obj = TestPrim(123)
        format_str = "value is {0.x}".format(test_obj)
        return format_str
    format_str = test_func()
    assert format_str == "value is 123"


def test_str_format_using_primitive_in_init():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    class TestPrim(Primitive):
        @prim_attr_register
        def __init__(self, x):
            self.x = x

    class TestCell(Cell):
        def __init__(self):
            super(TestCell, self).__init__()
            self.prim = TestPrim(123)
        def construct(self):
            format_str = "value is {0.x}".format(self.prim)
            return format_str

    test_cell = TestCell()
    format_str = test_cell()
    assert format_str == "value is 123"


@pytest.mark.skip("Not support yet")
def test_str_format_using_cell():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    class TestSubCell(Cell):
        def __init__(self, x):
            super(TestSubCell, self).__init__()
            self.x = x

    class TestCell(Cell):
        def construct(self):
            test_obj = TestSubCell(123)
            format_str = "value is {0.x}".format(test_obj)
            return format_str

    test_obj = TestCell()
    format_str = test_obj()
    assert format_str == "value is 123"


@pytest.mark.skip("Not support yet")
def test_str_format_using_cell_in_init():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.git
    """

    class TestSubCell(Cell):
        def __init__(self, x):
            super(TestSubCell, self).__init__()
            self.x = x

    class TestCell(Cell):
        def __init__(self):
            super(TestCell, self).__init__()
            self.test_sub_cell = TestSubCell(123)
        def construct(self):
            format_str = "value is {0.x}".format(self.test_sub_cell)
            return format_str

    test_cell = TestCell()
    format_str = test_cell()
    assert format_str == "value is 123"
