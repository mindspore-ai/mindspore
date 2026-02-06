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
"""Test string format for variable."""
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, jit_class, mutable, jit
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_base():
    """
    Feature: String formatting with variable in jit.
    Description: Test basic str.format() and f-string with Tensor.
    Expectation: Returns correctly formatted strings matching Tensor values.
    """
    @jit
    def net(x):
        x = x + 1
        format_str = "str x is {}".format(x)
        f_str = f"str x is {x}"
        return format_str, f_str
    format_str, f_str = net(Tensor([0, 1, 2]))
    assert format_str == "str x is [1 2 3]"
    assert f_str == "str x is [1 2 3]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_setitem_init():
    """
    Feature: String formatting with variable in jit.
    Description: Assign formatted strings to Cell attributes in construct.
    Expectation: Correctly formatted output strings.
    """
    class FormatNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.str1 = None
            self.str2 = None

        def construct(self, x):
            self.str1 = "{} x is {}".format("str", x)
            self.str2 = f"str x is {x}"
            return self.str1, self.str2

    format_str, f_str = jit(FormatNet())(Tensor([1.0, 2.0, 3.0]))
    assert format_str == "str x is [1. 2. 3.]"
    assert f_str == "str x is [1. 2. 3.]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_subscript():
    """
    Feature: String formatting with variable in jit.
    Description: Use {0}, {1} style placeholders with multiple Tensors.
    Expectation: Formatted strings reflect correct Tensor values.
    """
    @jit
    def net(a, b):
        format_str = "{0} or {1} is in {1}".format(a, b)
        f_str = f"{a} or {b} is in {b}"
        return format_str, f_str

    a = Tensor([1, 2])
    b = Tensor([1, 2, 3])
    format_str, f_str = net(a, b)
    assert format_str == "[1 2] or [1 2 3] is in [1 2 3]"
    assert f_str == "[1 2] or [1 2 3] is in [1 2 3]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_list():
    """
    Feature: String formatting with variable in jit.
    Description: Access elements via {x[0]}, {x[1]} in format and f-string.
    Expectation: Correctly renders indexed Tensor elements.
    """
    @jit
    def net(x):
        format_str = "hello, it's mindspore {0[1]}.{0[0]}".format(x)
        f_str = f"hello, it's mindspore {x[1]}.{x[0]}"
        return format_str, f_str

    x = Tensor([0, 2])
    format_str, f_str = net(x)
    assert format_str == "hello, it's mindspore 2.0"
    assert f_str == "hello, it's mindspore 2.0"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_map():
    """
    Feature: String formatting with variable in jit.
    Description: Use f-string with mutable dict containing list/Tensor.
    Expectation: Correctly formats value from dict key.
    """
    @jit
    def net(x):
        format_str = ""
        f_str = f"str is {x['a']}"
        return format_str, f_str

    x = mutable({'a': [1, 2, 3], 'b': Tensor([4, 5, 6])})
    format_str, f_str = net(x)
    assert format_str == ""
    assert f_str == "str is [1, 2, 3]"


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_tensor_comb():
    """
    Feature: String formatting with variable in jit.
    Description: Attempt format with list/tuple/dict containing Tensors.
    Expectation: Skipped — not yet supported in jit.
    """
    @jit
    def format_net(x, y):
        format_str1 = "loss is {0[0]} + {0[1]}".format([x, y])
        format_str2 = "loss is {0[0]} + {0[1]}".format((x, y))
        return format_str1, format_str2

    x = Tensor(2.1)
    y = Tensor(1.03)
    format_str1, format_str2 = format_net(x, y)
    assert format_str1 == "loss is 2.1 + 1.03"
    assert format_str2 == "loss is 2.1 + 1.03"

    @jit
    def map_net(x, y):
        ms_str = "loss is {0[name1]} + {0[name2]}"
        names = {"name1": x, "name2": y}
        format_str3 = ms_str.format(names)
        return format_str3

    format_str3 = map_net(x, y)
    assert format_str3 == "loss is 2.1 + 1.03"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_ops():
    """
    Feature: String formatting with variable in jit.
    Description: Format strings using results of ops like add/mul.
    Expectation: Correctly displays computed Tensor values in strings.
    """

    @jit
    def net(x, y):
        format_str = "ret is {}".format(ops.mul(x, y))
        f_str = f"ret is {ops.add(x,y)}"
        return format_str, f_str

    x = Tensor([1+2j, 2+0j])
    y = Tensor([1-2j, 2-5j])
    format_str, f_str = net(x, y)
    assert format_str == "ret is [5. +0.j 4.-10.j]"
    assert f_str == "ret is [2.+0.j 4.-5.j]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_multi_lines():
    """
    Feature: String formatting with variable in jit.
    Description: Use line continuation in f-string across two lines.
    Expectation: Properly concatenated formatted string.
    """
    @jit
    def net(x):
        f_str = f"str[0] is {x[0]} and " \
                f"str[1] is {x[1]}"
        return f_str

    x = Tensor([1.0, 2.0])
    f_str = net(x)
    assert f_str == "str[0] is 1.0 and str[1] is 2.0"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_number_format():
    """
    Feature: String formatting with variable in jit.
    Description: Test format specifiers like .2f, %, e, b, x, etc.
    Expectation: Strings match expected formatted numeric outputs.
    """
    @jit
    def net(num, num1):
        str1_format = "{:.2f}".format(num)
        str2_format = "{:.2%}".format(num)
        str3_format = "{:.2e}".format(num)
        str4_format = "{:0<4}".format(num)
        str5_format = "{:0>2}".format(num)
        str6_format = "{:^10}".format(num)
        str7_format = "{:10}".format(num)
        num1_format = "{:b}".format(num1)
        num2_format = "{:d}".format(num1)
        num3_format = "{:o}".format(num1)
        num4_format = "{:x}".format(num1)
        num5_format = "{:#x}".format(num1)
        num6_format = "{:#X}".format(num1)

        str_format_list = [
            str1_format, str2_format, str3_format,
            str4_format, str5_format, str6_format, str7_format
        ]

        num_format_list = [
            num1_format, num2_format, num3_format,
            num4_format, num5_format, num6_format
        ]
        return str_format_list, num_format_list

    num = 3.14159
    num1 = mutable(15)
    expect_format_lis1 = ['3.14', '314.16%', '3.14e+00',
                          '3.14159', '3.14159', ' 3.14159  ', '   3.14159']
    expect_format_lis2 = ['1111', '15', '17', 'f', '0xf', '0XF']
    str_format_list, num_format_list = net(num, num1)
    assert str_format_list == expect_format_lis1
    assert num_format_list == expect_format_lis2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variablems_class():
    """
    Feature: String formatting with variable in jit.
    Description: Format string using attribute of jit_class object.
    Expectation: Correctly renders object attribute in string.
    """
    @jit_class
    class TestClass():
        def __init__(self, x):
            self.value = x

    @jit
    def net(x):
        test_obj = TestClass(x)
        format_str = "value is {0.value}".format(test_obj)
        f_str = f"value is {test_obj.value}"
        return format_str, f_str

    x = [1, 2, 3]
    format_str, f_str = net(x)
    assert format_str == "value is [1, 2, 3]"
    assert f_str == "value is [1, 2, 3]"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_cell():
    """
    Feature: String formatting with variable in jit.
    Description: Attempt to format string using Cell object attribute.
    Expectation: Raises TypeError — Cell not supported in formatting.
    """
    class TestCell(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.value = x

    @jit
    def net(x):
        test_obj = TestCell(x)
        format_str = "value is {0.value}".format(test_obj)
        f_str = f"value is {test_obj.value}"
        return format_str, f_str

    x = Tensor([1.1, 2.1, 3.1])

    with pytest.raises(TypeError):
        net(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_if():
    """
    Feature: String formatting with variable in jit.
    Description: Use format/f-string in if/for/while branches.
    Expectation: Correct formatted output based on control flow path.
    """
    @jit
    def net(x, y, z):
        format_str = ""
        f_str = ""
        if x == 1:
            for i in y:
                format_str = "ret  is {}".format(i*z)
                f_str = f"value is {i*z}"
        else:
            while x < 4:
                format_str = "ret  is {}".format(y+z)
                f_str = f"value is {y+z}"
                x += 1
        return format_str, f_str

    x = Tensor(1)
    y = Tensor([1, 2])
    z = Tensor([5, 8])
    format_str, f_str = net(x, y, z)
    assert format_str == "ret  is [10 16]"
    assert f_str == "value is [10 16]"

    x = Tensor(0)
    y = Tensor([1, 2])
    z = Tensor([5, 8])
    format_str, f_str = net(x, y, z)
    assert format_str == "ret  is [ 6 10]"
    assert f_str == "value is [ 6 10]"


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_kwargs():
    """
    Feature: String formatting with variable in jit.
    Description: Use .format(name1=..., name2=...) with Tensors.
    Expectation: Skipped — keyword args in format not yet supported.
    """
    @jit
    def net(x, y):
        ms_str = "hello {name2},It's me, {name1}"
        ms_format_str = ms_str.format(name2=x, name1=y)
        return ms_format_str

    x = Tensor([0])
    y = Tensor([1])
    result_st = net(x, y)
    assert result_st == "hello {name2},It's me, {name1}".format(
        name2=x, name1=y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_str_format_variable_dynamic_shape():
    """
    Feature: String formatting with variable in jit.
    Description: Format strings using shape results from ops like Unique/Gather.
    Expectation: Correctly displays dynamic and static shape info.
    """
    class FormatNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unique = ops.Unique()
            self.gather = ops.Gather()
            self.axis = 0
            self.shape = ops.Shape()

        def construct(self, x, indices, y):
            unique_indices, _ = self.unique(indices)
            x = self.gather(x, unique_indices, self.axis)
            x_shape = self.shape(x)
            y_shape = y.shape
            print(f"x.shape:{x_shape},y.shape:{y_shape}")
            format_str = "x.shape is {}, y.shape is {}".format(
                x_shape, y_shape)
            f_str = f"x.shape is {x_shape}, y.shape is {y_shape}"
            return format_str, f_str

    x = Tensor(np.random.randn(5, 4, 3), dtype=mindspore.float32)
    y = Tensor(np.random.randn(3, 3, 3), dtype=mindspore.float32)
    indices = Tensor(np.random.randint(0, 3, size=3))
    test_net = FormatNet()
    test_net.set_inputs(x, indices, Tensor(
        shape=None, dtype=mindspore.float32))
    format_str, f_str = jit(test_net)(x, indices, y)

    assert "x.shape is (" in format_str and "y.shape is [3 3 3]" in format_str
    assert format_str == f_str
