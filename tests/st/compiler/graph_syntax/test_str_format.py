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
"""test str.format()"""

import numpy as np
import pytest
from mindspore import Tensor, jit
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import Primitive
from mindspore.ops import prim_attr_register
from mindspore import Parameter
from mindspore import jit_class
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base():
    """
    Feature: str.format() method for string formatting.
    Description: Test basic str.format() with single input and f-string formatting.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_base
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            ms_str = "string is {}".format("1")
            x = "2"
            ms_f_str = f"string is {x}"
            return ms_str, ms_f_str

    ms_str, ms_f_str = Formatnet()()
    assert ms_str == "string is 1"
    assert ms_f_str == "string is 2"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base_add_sub():
    """
    Feature: str.format() method with subnetwork.
    Description: Test str.format() in a network that contains a subnetwork.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_base_add_sub
    """

    class Innernet(Cell):
        def __init__(self, weight):
            super().__init__()
            weight_np = np.random.randn(*weight).astype(np.float32)
            self.weight = Parameter(Tensor(weight_np), name="Mul_weight")
            self.mul = P.Mul()

        def construct(self, inputs):
            x = self.mul(inputs, self.weight)
            return x

    class Formatnet(Cell):
        def __init__(self, weight):
            super().__init__()
            self.net = Innernet(weight)

        @jit
        def construct(self, inputs):
            self.net(inputs)
            ms_str = "string is {}".format("son network")
            return ms_str

    net = Formatnet(weight=(128, 96))
    inputs = Tensor(np.random.randn(128, 96).astype(np.float32))
    ms_str = net(inputs)
    assert ms_str == "string is son network"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base_if_for():
    """
    Feature: str.format() method with if branch and for loop.
    Description: Test str.format() in if-else branch with for loop and while loop.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_base_if_for
    """

    class Formatnet(Cell):
        @jit
        def construct(self, x):
            ms_str = ""
            f_str = ""
            if x == 1:
                for i in range(0, x):
                    ms_str = "{} is {}".format("string", "if - for")
                    a, b = "string", "if - for"
                    f_str = f"{a} is {b}"
                    i += 1
            else:
                while x < 4:
                    ms_str = "{} is {}".format("string", "else - while")
                    a, b = "string", "else - while"
                    f_str = f"{a} is {b}"
                    x += 1
            return ms_str, f_str

    test_net = Formatnet()
    ms_str, f_str = test_net(1)
    assert ms_str == "string is if - for"
    assert f_str == "string is if - for"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base_else_while():
    """
    Feature: str.format() method with else branch and while loop.
    Description: Test str.format() in else branch with while loop.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_base_else_while
    """

    class Formatnet(Cell):
        @jit
        def construct(self, x):
            ms_str = ""
            if x == 1:
                for i in range(0, x):
                    ms_str = "{} is {}".format("string", "if - for")
                    i += 1
            else:
                while x < 4:
                    ms_str = "{} is {}".format("string", "else - while")
                    x += 1
            return ms_str

    test_net = Formatnet()
    ms_str = test_net(2)
    assert ms_str == "string is else - while"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base_more_value():
    """
    Feature: str.format() method with multiple values using index.
    Description: Test str.format() with multiple values passed by index.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_base_more_value
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            ms_str = "{0} {1}".format("hello", "world")
            ms_str1 = "{1} {0} {1}".format("hello", "world")
            return ms_str, ms_str1

    test_net = Formatnet()
    ms_str, ms_str1 = test_net()
    assert ms_str == "hello world"
    assert ms_str1 == "world hello world"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_base_init_int():
    """
    Feature: str.format() method with constant defined in __init__.
    Description: Test str.format() using attribute defined in __init__ method.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_base_init_int
    """

    class Formatnet(Cell):
        def __init__(self, x):
            super().__init__()
            self.x = x

        @jit
        def construct(self):
            ms_str = "string is {}".format(self.x)
            f_str = f"string is {self.x}"
            return ms_str, f_str

    test_net = Formatnet(100)
    ms_str, f_str = test_net()
    assert ms_str == "string is 100"
    assert f_str == "string is 100"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_list():
    """
    Feature: str.format() method with list input.
    Description: Test str.format() with list as input parameter.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_list
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            ms_str = "hello {0[1]},It's me {0[0]}"
            names = ["Mind", "Spore"]
            ms_format_str = ms_str.format(names)
            f_str = f"hello {names[1]},It's me {names[0]}"
            return ms_format_str, f_str

    test_net = Formatnet()
    ms_format_str, f_str = test_net()
    assert ms_format_str == "hello Spore,It's me Mind"
    assert f_str == "hello Spore,It's me Mind"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_tensor():
    """
    Feature: str.format() method with Tensor input.
    Description: Test str.format() with Tensor as input parameter.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_tensor
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            a = Tensor([1])
            ms_str = "{} is {}".format("string", a)
            return ms_str

    test_net = Formatnet()
    ms_str = test_net()
    assert ms_str == "string is Tensor(shape=[1], dtype=Int64, value=[1])" or ms_str == "string is [1]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_function():
    """
    Feature: str.format() method as function reference.
    Description: Test str.format() when format is assigned to a function variable.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_function
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            func = "{0[0]}{0[1]}".format
            names = ["Mind", "Spore"]
            ms_format_str = func(names)
            return ms_format_str

    test_net = Formatnet()
    ms_format_str = test_net()
    assert ms_format_str == "MindSpore"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_input():
    """
    Feature: format() builtin function.
    Description: Test format() builtin function with Tensor and string format specifier.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_input
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            a = Tensor([1])
            ms_str = format(a)
            ms_str2 = format(ms_str, "4")
            return ms_str, ms_str2

    test_net = Formatnet()
    ms_str, ms_str2 = test_net()
    assert ms_str == "[1]"
    assert ms_str2 == "[1] "


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_int_float():
    """
    Feature: str.format() method with numeric formatting.
    Description: Test str.format() with int and float number formatting
    including precision, percentage, and scientific notation.
    Expectation: The formatted string list matches expected values.
    Migrated from: test_str_format.py::test_str_format_int_float
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            num = 3.1415926
            num1 = 0.25
            num2 = 1000000000
            num3 = 5
            str1_format = "{:.2f}".format(num)
            str2_format = "{:.0f}".format(num)
            str3_format = "{:+.2f}".format(num)
            str4_format = "{:.2%}".format(num1)
            str5_format = "{:.2e}".format(num2)
            str6_format = "{:x<4d}".format(num3)
            str_format_list = [str1_format, str2_format, str3_format, str4_format, str5_format, str6_format]
            return str_format_list

    test_net = Formatnet()
    num_format_list = ["3.14", "3", "+3.14", "25.00%", "1.00e+09", "5xxx"]
    ms_format_list = test_net()
    assert ms_format_list == num_format_list


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_decimal_more():
    """
    Feature: str.format() method with base conversion.
    Description: Test str.format() with different base conversions (binary, decimal, octal, hexadecimal).
    Expectation: The formatted string list matches expected values.
    Migrated from: test_str_format.py::test_str_format_decimal_more
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            num = 11
            str1_format = "{:b}".format(num)
            str2_format = "{:d}".format(num)
            str3_format = "{:o}".format(num)
            str4_format = "{:x}".format(num)
            str5_format = "{:#x}".format(num)
            str6_format = "{:#X}".format(num)
            str_format_list = [str1_format, str2_format, str3_format, str4_format, str5_format, str6_format]
            return str_format_list

    test_net = Formatnet()
    num_format_list = ["1011", "11", "13", "b", "0xb", "0XB"]
    ms_format_list = test_net()
    assert ms_format_list == num_format_list


@pytest.mark.skip("Graph mode does not support f-string base conversion")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_f_format():
    """
    Feature: f-string formatting with base conversion.
    Description: Test f-string formatting with numeric formatting and base conversion.
    Expectation: The formatted string lists match expected values.
    Migrated from: test_str_format.py::test_str_format_f_format
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            num = 3.1415926
            num1 = 0.25
            num2 = 1000000000
            num3 = 5
            f_format1 = f"{num:.2f}"
            f_format2 = f"{num1:.2%}"
            f_format3 = f"{num2:.2e}"
            f_format4 = f"{num3:x<4d}"
            f_format_list1 = [f_format1, f_format2, f_format3, f_format4]
            num = 11
            f_format1 = f"{num:b}"
            f_format2 = f"{num:d}"
            f_format3 = f"{num:o}"
            f_format4 = f"{num:x}"
            f_format5 = f"{num:#x}"
            f_format6 = f"{num:#X}"
            f_format_list2 = [f_format1, f_format2, f_format3, f_format4, f_format5, f_format6]
            return f_format_list1, f_format_list2

    test_net = Formatnet()
    num_format_list1 = ["3.14", "3", "+3.14", "25.00%", "1.00e+09", "5xxx"]
    num_format_list2 = ["1011", "11", "13", "b", "0xb", "0XB"]
    f_format_list1, f_format_list2 = test_net()

    assert f_format_list1 == num_format_list1[:1] + num_format_list1[3:]
    assert f_format_list2 == num_format_list2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_space():
    """
    Feature: str.format() method with alignment and width.
    Description: Test str.format() with alignment (left, right, center) and width specification.
    Expectation: The formatted string list matches expected values.
    Migrated from: test_str_format.py::test_str_format_space
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            num = 5
            num1 = 10
            str1_format = "{:0>2}".format(num)
            str2_format = "{:x^4}".format(num1)
            str3_format = "{:10}".format(num1)
            str4_format = "{:<10}".format(num1)
            str5_format = "{:^10}".format(num1)
            str_format_list = [str1_format, str2_format, str3_format, str4_format, str5_format]
            return str_format_list

    test_net = Formatnet()
    num_format_tuple = ("05", "x10x", "        10", "10        ", "    10    ")
    num_format_list = ["05", "x10x", "        10", "10        ", "    10    "]
    ms_format_list = test_net()
    assert ms_format_list == num_format_list or ms_format_list == num_format_tuple


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_ms_class():
    """
    Feature: str.format() method with custom class object.
    Description: Test str.format() with jit_class decorated custom class object.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_ms_class
    """

    @jit_class
    class TestClass:
        def __init__(self, value):
            self.value = value

    class Formatnet(Cell):
        @jit
        def construct(self):
            test_obj = TestClass(123)
            format_str = "value is {0.value}".format(test_obj)
            f_str = f"value is {test_obj.value}"
            return format_str, f_str

    test_net = Formatnet()
    str_format, f_str = test_net()
    assert str_format == "value is 123"
    assert f_str == "value is 123"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_primitive():
    """
    Feature: str.format() method with Primitive object.
    Description: Test str.format() with Primitive class object.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_primitive
    """

    class TestPrim(Primitive):
        @prim_attr_register
        def __init__(self, value):
            self.value = value

    class Formatnet(Cell):
        @jit
        def construct(self):
            test_obj = TestPrim(123)
            format_str = "value is {0.value}".format(test_obj)
            return format_str

    test_net = Formatnet()
    str_format = test_net()
    assert str_format == "value is 123"


@pytest.mark.skip("Not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_f_primitive():
    """
    Feature: f-string formatting with Primitive object.
    Description: Test f-string formatting with Primitive class object.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_f_primitive
    """

    class TestPrim(Primitive):
        @prim_attr_register
        def __init__(self, value):
            self.value = value

    class Formatnet(Cell):
        @jit
        def construct(self):
            test_obj = TestPrim(123)

            f_str = f"value is {test_obj.value}"
            return f_str

    test_net = Formatnet()
    f_str = test_net()
    assert f_str == "value is 123"


@pytest.mark.skip("Not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_cell():
    """
    Feature: str.format() method with Cell object.
    Description: Test str.format() with Cell class object (currently not supported).
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_cell
    """

    class TestCell(Cell):
        def __init__(self, value):
            self.value = value

    class Formatnet(Cell):
        @jit
        def construct(self):
            test_obj = TestCell(Cell)
            format_str = "value is {0.value}".format(test_obj)
            f_str = f"value is {test_obj.value}"
            return format_str, f_str

    test_net = Formatnet()
    str_format, f_str = test_net()
    assert str_format == "value is 123"
    assert f_str == "value is 123"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_kwargs():
    """
    Feature: str.format() method with keyword arguments.
    Description: Test str.format() with keyword arguments.
    Expectation: The formatted string matches expected value.
    Migrated from: test_str_format.py::test_str_format_kwargs
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            name = "MindSpore"
            ms_str = "my name is {name}".format(name=name)
            return ms_str

    test_net = Formatnet()
    ret = test_net()
    assert ret == "my name is MindSpore"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_map():
    """
    Feature: str.format() method with dictionary input.
    Description: Test str.format() with dictionary as input parameter.
    Expectation: The formatted strings match expected values.
    Migrated from: test_str_format.py::test_str_format_map
    """

    class Formatnet(Cell):
        @jit
        def construct(self):
            map_string = {"name1": "Mind", "name2": "Spore"}
            ms_str = "my name is {0[name1]}".format(map_string)
            f_str = f'my name is {map_string["name2"]}'
            return ms_str, f_str

    test_net = Formatnet()
    str_format, f_str = test_net()
    assert str_format == "my name is Mind"
    assert f_str == "my name is Spore"
