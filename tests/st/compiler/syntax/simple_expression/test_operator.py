# Copyright 2021-2023 Huawei Technologies Co., Ltd
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

import sys
import pytest

from mindspore import Tensor, context, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_inner_scalar_divisor():
    """
    Feature: Check whether the divisor of inner scalar is zero.
    Description: The divisor of inner scalar must not be zero.
    Expectation: The divisor of inner scalar must not be zero.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, ms.int32), name="param_a")
            self.param_b = Parameter(Tensor(5, ms.int32), name="param_b")

        def construct(self, x):
            return x + self.param_a + 5 / 0

    x = Tensor(2, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The divisor could not be zero."):
        ret = net(x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_inner_scalar_mod():
    """
    Feature: Check the input of inner scalar mod.
    Description: The input of inner scalar mod must not be zero.
    Expectation: The input of inner scalar mod must not be zero.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, ms.int32), name="param_a")

        def construct(self, x):
            return x + self.param_a + 5 % 0

    x = Tensor(2, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="Cannot perform modulo operation on zero."):
        ret = net(x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_inner_scalar_mod_args_length():
    """
    Feature: Check the length of input of inner scalar mod.
    Description: The length of input of inner scalar mod should not less than 2.
    Expectation: The length of input of inner scalar mod should not less than 2.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, ms.int32), name="param_a")
            self.mod = P.Mod()

        def construct(self, x):
            return x + self.param_a + self.mod(5)

    x = Tensor(2, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="For 'S_Prim_Mod', the size of input should be 2"):
        ret = net(x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_input_is_empty():
    """
    Feature: Check the length of inputs of make_range operator.
    Description: The inputs of make_range operator could not be empty.
    Expectation: The inputs of make_range operator could not be empty.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range():
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(RuntimeError) as err:
        ret = net(x, y)
        print("ret:", ret)
    assert "For 'make_range', the input size should within [1, 3] but got0" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_step_zero():
    """
    Feature: Check the length of inputs of make_range operator.
    Description: The step value of MakeRange operator could not be 0.
    Expectation: The step value of MakeRange operator could not be 0.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(1, 2, 0):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="For 'range', the argument 'step' could not be 0."):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_range_error_input_1():
    """
    Feature: Check the inputs of make_range operator.
    Description: If start > stop, the step need smaller than zero.
    Expectation: If start > stop, the step need smaller than zero.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(1, -1, 3):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    ret = net(x, y)
    assert ret == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_make_range_error_input_2():
    """
    Feature: Check the length of inputs of make_range operator.
    Description: If start < stop, the step need greater than zero.
    Expectation: If start < stop, the step need greater than zero.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(-1, 1, -3):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    ret = net(x, y)
    assert ret == 2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_input_type():
    """
    Feature: Check the type of inputs of make_range operator.
    Description: The type of inputs of make_range operator must be int64.
    Expectation: The type of inputs of make_range operator must be int64.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(0, 0.02):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(TypeError) as err:
        ret = net(x, y)
        print("ret:", ret)
    assert "For 'make_range', the 1th input should be a int scalar but got AbstractScalar" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_input_type_2():
    """
    Feature: Check the type of inputs of make_range operator.
    Description: The type of inputs of make_range operator must be int64.
    Expectation: The type of inputs of make_range operator must be int64.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(0, 1, 3.00):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(TypeError) as err:
        ret = net(x, y)
        print("ret:", ret)
    assert "For 'make_range', the 2th input should be a int scalar but got AbstractScalar" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_input_type_3():
    """
    Feature: Check the type of inputs of make_range operator.
    Description: The type of inputs of make_range operator must be int64.
    Expectation: The type of inputs of make_range operator must be int64.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(3.00):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(TypeError) as err:
        ret = net(x, y)
        print("ret:", ret)
    assert "For 'make_range', the 0th input should be a int scalar but got AbstractScalar" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_input_size():
    """
    Feature: Check the size of inputs of make_range operator.
    Description: The size of inputs of make_range operator could not exceed 3.
    Expectation: The size of inputs of make_range operator could not exceed 3.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in range(1, 2, 3, 4):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(RuntimeError) as err:
        ret = net(x, y)
        print("ret:", ret)
    assert "For 'make_range', the input size should within [1, 3] but got4" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_overflow():
    """
    Feature: Check the size of inputs of range operator.
    Description: The size of inputs of make_range operator could not exceed 3.
    Expectation: The size of inputs of make_range operator could not exceed 3.
    """
    class Net(Cell):
        def construct(self, x, y):
            max_index = sys.maxsize
            for _ in range(max_index - 1, max_index, 3):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="Integer overflow error occurred when traversing the range."):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_make_range_overflow_2():
    """
    Feature: Check the size of inputs of make_range operator.
    Description: The size of inputs of make_range operator could not exceed 3.
    Expectation: The size of inputs of make_range operator could not exceed 3.
    """
    class Net(Cell):
        def construct(self, x, y):
            min_index = -sys.maxsize
            for _ in range(min_index, min_index - 1, -3):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="Integer overflow error occurred when traversing the range."):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_typeof():
    """
    Feature: Check the size of inputs of typeof operator.
    Description: The size of inputs of typeof operator must be 1.
    Expectation: The size of inputs of typeof operator must be 1.
    """
    class Net(Cell):
        def construct(self, x):
            return F.typeof(x, x)

    x = Tensor([2, 3, 4, 5], dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The Typeof operator must requires 1 argument, "
                                        "but the size of arguments is 2."):
        ret = net(x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_div():
    """
    Feature: Check the size of inputs of tuple_div operator.
    Description: The size of inputs of tuple_div operator must be same.
    Expectation: The size of inputs of tuple_div operator must be same.
    """
    class Net(Cell):
        def construct(self, x, y):
            return F.tuple_div(x, y)

    x = (8, 14, 20)
    y = (2, 2)
    net = Net()
    with pytest.raises(Exception, match="The size of inputs of 'tuple_div' operator must be the same"):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_div_type():
    """
    Feature: Check the size of inputs of tuple_div operator.
    Description: The type of inputs of tuple_div operator must be int64 number.
    Expectation: The type of inputs of tuple_div operator must be int64 number.
    """
    class Net(Cell):
        def construct(self, x, y):
            return F.tuple_div(x, y)

    x = (8, 14, 20)
    y = (2, 2, 2.0)
    net = Net()
    with pytest.raises(Exception, match="The data type of inputs of 'tuple_div' operator should be an int64 number,"):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_div_zero():
    """
    Feature: Check the size of inputs of tuple_div operator.
    Description: The divisor value should not be 0.
    Expectation: The divisor value should not be 0.
    """
    class Net(Cell):
        def construct(self, x, y):
            return F.tuple_div(x, y)

    x = (8, 14, 20)
    y = (2, 2, 0)
    net = Net()
    with pytest.raises(Exception, match="The divisor value should not be 0"):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_div_input_is_not_divisible():
    """
    Feature: Check whether the inputs of tuple_div is divisible.
    Description: The inputs of tuple_div could be divisible.
    Expectation: The inputs of tuple_div could be divisible.
    """
    class Net(Cell):
        def construct(self, x, y):
            return F.tuple_div(x, y)

    x = (8, 14)
    y = (2, 3)
    net = Net()
    with pytest.raises(Exception, match="The inputs of 'tuple_div' operator should be divisible,"):
        ret = net(x, y)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_make_slice_scalar():
    """
    Feature: Check whether the scalar input of make_slice is int or bool.
    Description: The scalar input of make_slice is int or bool.
    Expectation: The scalar input of make_slice is int or bool.
    """
    class Net(Cell):
        def construct(self, data):
            return data[1.01:None:None]

    x = Tensor((8, 10, 12), dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="Slice indices must be integers or bool."):
        ret = net(x)
        print("ret:", ret)
