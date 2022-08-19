# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test_string_ops """
from mindspore import context, ms_function

context.set_context(mode=context.GRAPH_MODE)


def test_string_add():
    """
    Feature: string operation
    Description: Test string addition in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "ab"
        y = "cd"
        return x + y

    assert foo() == "abcd"


def test_string_mul():
    """
    Feature: string operation
    Description: Test Multiplication of strings in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "abc" * 2
        y = 3 * "abc"
        z = "abc" * -1
        return x, y, z

    x, y, z = foo()
    assert x == "abcabc"
    assert y == "abcabcabc"
    assert z == ""


def test_string_getitem():
    """
    Feature: string operation
    Description: Test getting item of string by number index in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "string"
        return x[0], x[-1]

    x, y = foo()
    assert x == "s"
    assert y == "g"


def test_string_equal():
    """
    Feature: string operation
    Description: Test string equal to another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "str"
        y = "str"
        z = ""
        return x == y, x == z

    x, y = foo()
    assert x and (not y)


def test_string_not_equal():
    """
    Feature: string operation
    Description: Test string not equal to another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "str"
        y = "str"
        z = ""
        return x != y, x != z

    x, y = foo()
    assert (not x) and y


def test_string_less():
    """
    Feature: string operation
    Description: Test string less than another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "abd"
        y = "abc"
        z = "acd"
        return x < y, x < z

    x, y = foo()
    assert (not x) and y


def test_string_greater():
    """
    Feature: string operation
    Description: Test string greater than another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "abd"
        y = "abc"
        z = "acd"
        return x > y, x > z

    x, y = foo()
    assert x and (not y)


def test_string_less_equal():
    """
    Feature: string operation
    Description: Test string less than or equal to another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "abd"
        y = "abc"
        z = "acd"
        t = "abd"
        return x <= y, x <= z, x <= t

    x, y, z = foo()
    assert (not x) and y and z


def test_string_greater_equal():
    """
    Feature: string operation
    Description: Test string greater than or equal to another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "abd"
        y = "abc"
        z = "acd"
        t = "abd"
        return x >= y, x >= z, x >= t

    x, y, z = foo()
    assert x and (not y) and z


def test_string_logical_not():
    """
    Feature: string operation
    Description: Test logical_not of string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = ""
        y = "str"
        return (not x), (not y)

    x, y = foo()
    assert x and (not y)


def test_string_and():
    """
    Feature: string operation
    Description: Test string and string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "" and "abc"
        y = "abc" and ""
        return x, y

    x, y = foo()
    assert x == ""
    assert y == ""


def test_string_or():
    """
    Feature: string operation
    Description: Test string or string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "" or "abc"
        y = "abc" or ""
        return x, y

    x, y = foo()
    assert x == "abc"
    assert y == "abc"


def test_string_in_string():
    """
    Feature: string operation
    Description: Test whether string in another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "op"
        y = "ops"
        z = "operation"
        return x in z, y in z

    x, y = foo()
    assert x and (not y)


def test_string_not_in_string():
    """
    Feature: string operation
    Description: Test whether string not in another string in graph mode
    Expectation: No exception
    """
    @ms_function
    def foo():
        x = "op"
        y = "ops"
        z = "operation"
        return x not in z, y not in z

    x, y = foo()
    assert (not x) and y
