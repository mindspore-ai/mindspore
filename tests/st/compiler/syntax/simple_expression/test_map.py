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

import pytest
import numpy as np

from mindspore import Tensor, nn, Parameter, context
from mindspore.nn import Cell
import mindspore as ms
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_map_args_size():
    """
    Feature: Check the size of inputs of map.
    Description: The size of inputs of map must be greater than 1.
    Expectation: The size of inputs of map must be greater than 1.
    """
    class MapNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x):
            if map(self.mul) == 8:
                x = self.relu(x)
            return x
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)

    net = MapNet()
    with pytest.raises(Exception, match="The Map operator must have at least two arguments."):
        ret = net(input_me_x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_map_args_type():
    """
    Feature: Check the type of inputs of Map().
    Description: The type of inputs of Map() must be list, tuple.
    Expectation: The type of inputs of Map() must be list, tuple.
    """
    class MapNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x):
            if map(self.mul, 3, 4) == 8:
                x = self.relu(x)
            return x
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)

    net = MapNet()
    with pytest.raises(Exception, match="Map can only be applied to list, tuple"):
        ret = net(input_me_x)
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_map_args_full_make_list():
    """
    Feature: Check the types of all inputs in Map.
    Description: The types of all inputs in Map must be same.
    Expectation: The types of all inputs in Map must be same.
    """
    class MapNet(Cell):
        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x, y):
            if map(self.mul, x, y) == [8]:
                x = y
            return x

    input_me_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_me_y = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))

    net = MapNet()
    with pytest.raises(Exception, match="The types of arguments in Map must be consistent"):
        ret = net([input_me_x], (input_me_y))
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_map_args_full_make_list_same_length():
    """
    Feature: Check the length of list input Map.
    Description: The list in Map should have same length.
    Expectation: The list in Map should have same length.
    """
    class MapNet(Cell):
        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x, y):
            if map(self.mul, x, y) == [8]:
                x = y
            return x

    input_me_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_me_y = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))

    net = MapNet()
    with pytest.raises(Exception, match="For 'Map', the length of lists must be the same."):
        ret = net([input_me_x], [input_me_y, input_me_y])
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_map_args_full_make_tuple_same_length():
    """
    Feature: Check the length of tuple input Map.
    Description: The tuple in Map should have same length.
    Expectation: The tuple in Map should have same length.
    """
    class MapNet(Cell):
        def mul(self, x=2, y=4):
            return x * y

        def construct(self, x, y):
            if map(self.mul, x, y) == [8]:
                x = y
            return x

    input_me_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_me_y = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))

    net = MapNet()
    with pytest.raises(Exception, match="For 'Map', the length of tuples must be the same."):
        ret = net((input_me_x, input_me_x), (input_me_y, input_me_y, input_me_y))
        print("ret:", ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_map_param_cast():
    """
    Feature: Check the ref type when insert auto cast.
    Description: Check the ref type when insert auto cast.
    Expectation: Check the ref type when insert auto cast.
    """
    class MapNet(Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(5, ms.float32), name="param_b")

        def construct(self, x):
            self.param = x
            return self.param

    input_me_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float64))

    net = MapNet()
    with pytest.raises(Exception, match="Data type conversion of 'Parameter' is not supported"):
        ret = net(input_me_x)
        print("ret:", ret)
