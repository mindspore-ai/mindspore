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

import sys
import pytest

from mindspore import Tensor, context, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
import mindspore as ms


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

    context.set_context(device_target="GPU")
    x = Tensor(2, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The divisor could not be zero."):
        ret = net(x)
        print("ret:", ret)


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
    with pytest.raises(Exception, match="Could not mod to zero."):
        ret = net(x)
        print("ret:", ret)


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
    with pytest.raises(Exception, match="Function S-Prim-Mod's input length is not equal to Signature length."):
        ret = net(x)
        print("ret:", ret)


def test_make_range_input_is_empty():
    """
    Feature: Check the length of inputs of make_range operator.
    Description: The inputs of make_range operator could not be empty.
    Expectation: The inputs of make_range operator could not be empty.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in F.make_range():
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The inputs of make_range operator could not be empty."):
        ret = net(x, y)
        print("ret:", ret)


def test_make_range_input_type():
    """
    Feature: Check the type of inputs of make_range operator.
    Description: The type of inputs of make_range operator must be int64.
    Expectation: The type of inputs of make_range operator must be int64.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in F.make_range(0, 0.02):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The type of inputs of make_range operator only support int64 number."):
        ret = net(x, y)
        print("ret:", ret)


def test_make_range_input_size():
    """
    Feature: Check the size of inputs of make_range operator.
    Description: The size of inputs of make_range operator could not exceed 3.
    Expectation: The size of inputs of make_range operator could not exceed 3.
    """
    class Net(Cell):
        def construct(self, x, y):
            for _ in F.make_range(1, 2, 3, 4):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The size of inputs of make_range operator could not exceed 3."):
        ret = net(x, y)
        print("ret:", ret)


def test_make_range_overflow():
    """
    Feature: Check the size of inputs of make_range operator.
    Description: The size of inputs of make_range operator could not exceed 3.
    Expectation: The size of inputs of make_range operator could not exceed 3.
    """
    class Net(Cell):
        def construct(self, x, y):
            max_index = sys.maxsize
            for _ in F.make_range(max_index - 1, max_index, 3):
                x += y
            return x

    x = Tensor(2, dtype=ms.int32)
    y = Tensor(4, dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="For make range, the required cycles number is greater than max cycles number"):
        ret = net(x, y)
        print("ret:", ret)


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
    with pytest.raises(Exception, match="Typeof evaluator requires 1 parameter, while the input size is 2."):
        ret = net(x)
        print("ret:", ret)


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
    with pytest.raises(Exception, match="The size of inputs of tuple_div operator must be same"):
        ret = net(x, y)
        print("ret:", ret)


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
    with pytest.raises(Exception, match="The inputs of tuple_div is not divisible"):
        ret = net(x, y)
        print("ret:", ret)


def test_make_slice_scalar():
    """
    Feature: Check whether the scalar input of make_slice is int or bool.
    Description: The scalar input of make_slice is int or bool.
    Expectation: The scalar input of make_slice is int or bool.
    """
    class Net(Cell):
        def construct(self, data):
            return data[F.make_slice(1.01, None, None)]

    x = Tensor((8, 10, 12), dtype=ms.int32)
    net = Net()
    with pytest.raises(Exception, match="The 0th input of scalar should be int or bool"):
        ret = net(x)
        print("ret:", ret)
