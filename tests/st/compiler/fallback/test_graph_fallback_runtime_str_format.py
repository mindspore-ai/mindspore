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
""" test graph fallback """
import pytest
import numpy as np
from mindspore.common import mutable
from mindspore.nn import Cell
import mindspore
from mindspore import jit, Tensor, context, ops, nn
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_single_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        ms_str = "string is {}".format(x)
        return ms_str

    x = Tensor([1])
    assert foo(x) == "string is {}".format(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_str_format_mutiple_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        ms_str = "{} is {}".format("string", x + x)
        return ms_str

    x = Tensor([1])
    assert foo(x) == "{} is {}".format("string", x + x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_str_format_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        ms_str = format(a)
        ms_str2 = format(ms_str, "4")
        return ms_str, ms_str2

    a = Tensor([1])
    ms_str, ms_str2 = foo(a)
    assert ms_str == "[1]"
    assert ms_str2 == "[1] "


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_with_number_placeholder_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        ms_str = "{1} {0} {1}"
        ms_format_str = ms_str.format(x, y)
        return ms_format_str

    x = Tensor([0])
    y = Tensor([1])
    ms_str = foo(x, y)
    assert ms_str == "{1} {0} {1}".format(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_with_key_input():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        ms_str = "hello {name2},It's me, {name1}"
        ms_format_str = ms_str.format(name2=x, name1=y)
        return ms_format_str

    x = Tensor([0])
    y = Tensor([1])
    with pytest.raises(KeyError):
        # TODO(LianLiguang): fix when the kwargs is move to after_opt_a_rewriter
        result_st = foo(x, y)
        assert result_st == "hello {name2},It's me, {name1}".format(name2=x, name1=y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_with_list_index():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        ms_str = "hello {0[1]},It's me {0[0]}"
        names = [x, y]
        ms_format_str = ms_str.format(names)
        return ms_format_str

    x = Tensor([1])
    y = Tensor([0])
    result_st = foo(x, y)
    assert result_st == "hello {0[1]},It's me {0[0]}".format([x, y])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_with_tuple_index():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        ms_str = "hello {0[1]},It's me {0[0]}"
        names = (x, y)
        ms_format_str = ms_str.format(names)
        return ms_format_str

    x = Tensor([1])
    y = Tensor([0])
    result_st = foo(x, y)
    assert result_st == "hello {0[1]},It's me {0[0]}".format((x, y))


@pytest.mark.skip("need make dict do not eliminate before opt A.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_with_map():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        ms_str = "hello {0[name2]},It's me {0[name1]}"
        names = {"name1": x, "name2": y}
        ms_format_str = ms_str.format(names)
        return ms_format_str

    x = Tensor([0])
    y = Tensor([1])

    with pytest.raises(TypeError) as err:
        # TODO(LianLiguang): fix when the dict convert is move to after opt a
        result_st = foo(x, y)
        names = {"name1": x, "name2": y}
        assert result_st == "hello {0[name2]},It's me {0[name1]}".format(names)
    assert "tuple indices must be integers or slices, not str." in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_as_function():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x, y):
        func = "hello {0[1]},It's me {0[0]}".format
        names = (x, y)
        ms_format_str = func(names)
        return ms_format_str

    x = Tensor([1])
    y = Tensor([0])
    result_st = foo(x, y)
    func = "hello {0[1]},It's me {0[0]}".format
    assert result_st == func([x, y])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_number():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(num1, num2, num3, num4, num5):
        str1_format = "{:.2f}".format(num1)
        str2_format = "{:.0f}".format(num1)
        str3_format = "{:,}".format(num2)
        str4_format = "{:.2%}".format(num3)
        str5_format = "{:.2e}".format(num4)
        str6_format = "{0:b}".format(num5)
        str7_format = "{0:d}".format(num5)
        str8_format = "{0:o}".format(num5)
        str9_format = "{0:x}".format(num5)
        result = (str1_format, str2_format, str3_format, str4_format, str5_format,
                  str6_format, str7_format, str8_format, str9_format)
        return result

    num1 = 3.1415926
    num2 = 1000000
    num3 = 0.25
    num4 = 1000000000
    num5 = 25
    correct_str = ("3.14", "3", "1,000,000", "25.00%", "1.00e+09", "11001", "25", "31", "19")
    result_str = foo(mutable(num1), mutable(num2), mutable(num3), mutable(num4), mutable(num5))
    assert result_str == correct_str


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_format_padding():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(num1, num2, num3):
        str1_format = "{:0>2}".format(num1)
        str2_format = "{:x<4}".format(num1)
        str3_format = "{:x^4}".format(num2)
        str4_format = "{:10}".format(num3)
        str5_format = "{:<10}".format(num3)
        str6_format = "{:^10}".format(num3)

        result = (str1_format, str2_format, str3_format, str4_format, str5_format, str6_format)
        return result

    num1 = 5
    num2 = 10
    num3 = 13
    correct_str = ("05", "5xxx", "x10x", "        13", "13        ", "    13    ")
    result_str = foo(mutable(num1), mutable(num2), mutable(num3))
    assert result_str == correct_str


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_str_format_using_cell():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
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
    with pytest.raises(TypeError) as err:
        format_str = test_obj()
        test_obj = TestSubCell(123)
        assert format_str == "value is {0.x}".format(test_obj)
    assert "Unsupported parameter type for python primitive, the parameter value is DeadNode" in str(err.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_str_format_using_cell_in_init():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
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
    with pytest.raises(TypeError) as err:
        format_str = test_cell()
        test_sub_cell = TestSubCell(123)
        assert format_str == "value is {0.x}".format(test_sub_cell)
    assert "Unsupported parameter type for python primitive, the parameter value is DeadNode" in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(paddings):
        if not isinstance(paddings, (tuple, list)):
            raise TypeError(f"For 'pad', the type of 'paddings' must be a tuple of int or list of int,"
                            f" but got {type(paddings)}.")
        return paddings

    with pytest.raises(TypeError) as err:
        x = Tensor([1])
        foo(x)
    assert "For 'pad', the type of 'paddings' must be a tuple of int or list of int, " \
           "but got <class 'mindspore.common.tensor.Tensor'>." in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation_2():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(paddings):
        if not isinstance(paddings, (tuple, list)):
            raise TypeError(f'The paddings must be a tuple of int or list of int,'
                            f" but got {type(paddings)}.")
        return paddings

    with pytest.raises(TypeError) as err:
        x = Tensor([1])
        foo(x)
    assert "The paddings must be a tuple of int or list of int, " \
           "but got <class 'mindspore.common.tensor.Tensor'>." in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation_3():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(paddings):
        if not isinstance(paddings, (tuple, list)):
            raise TypeError(f'The paddings must be a tuple of int or list of int,'
                            f' but got {type(paddings)}.')
        return paddings

    with pytest.raises(TypeError) as err:
        x = Tensor([1])
        foo(x)
    assert "The paddings must be a tuple of int or list of int, " \
           "but got <class 'mindspore.common.tensor.Tensor'>." in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation_4():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(paddings):
        if not isinstance(paddings, (tuple, list)):
            raise TypeError('The paddings must be a tuple of int or list of int,' + \
                            f' but got {type(paddings)}.')
        return paddings

    with pytest.raises(TypeError) as err:
        x = Tensor([1])
        foo(x)
    assert "The paddings must be a tuple of int or list of int, " \
           "but got <class 'mindspore.common.tensor.Tensor'>." in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation_5():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(paddings):
        if not isinstance(paddings, (tuple, list)):
            raise TypeError('The paddings must be a tuple of int or list of int,' + \
                            f' but got {type(paddings)}.' + \
                            'Please check it.')
        return paddings

    with pytest.raises(TypeError) as err:
        x = Tensor([1])
        foo(x)
    assert "The paddings must be a tuple of int or list of int, " \
           "but got <class 'mindspore.common.tensor.Tensor'>.Please check it." in str(err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_str_format_indentation_6():
    """
    Feature: JIT Fallback
    Description: Test str.format() in graph mode.
    Expectation: No exception.
    """
    @jit
    def test_fstring(x):
        output = ops.fmod(x, 2.5)
        return output
    input_x = Tensor(np.array([-4., -3.5, 0, 3.5, 4]), mindspore.float32)
    test_fstring(input_x)


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
        format_str = "x.shape is {}, y.shape is {}".format(x_shape, y_shape)
        f_str = f"x.shape is {x_shape}, y.shape is {y_shape}"
        return format_str, f_str


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_input_dynamic_len_tuple():
    """
    Feature: JIT Fallback
    Description: Dynamic len tuple and dynamic type in same graph.
    Expectation: No exception.
    """
    x = Tensor(np.random.randn(5, 4, 3), dtype=mindspore.float32)
    y = Tensor(np.random.randn(3, 3, 3), dtype=mindspore.float32)
    indices = Tensor(np.random.randint(0, 3, size=3))
    net = FormatNet()
    net.set_inputs(x, indices, Tensor(shape=None, dtype=mindspore.float32))
    a, b = net(x, indices, y)
    assert a
    assert b
