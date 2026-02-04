# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test cell method attribute."""
# pylint: disable=E0213
# pylint: disable=C0115
# pylint: disable=W0238
# pylint: disable=W0212
import pytest

import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn, Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype

ms.set_context(mode=ms.GRAPH_MODE)

class InnerCellNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = ms.Tensor(3)

    def add(self, x, y):
        return x + y


inner_cell = InnerCellNet()


class CellNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = InnerCellNet()

    def construct(self, x, y):
        a = self.net.add(x, y)
        b = inner_cell.add(a, inner_cell.a)  # <== Use Cell object's method and attribute here.
        return b


def test_cell_call_cell_methods():
    """
    Feature: Support use Cell method and attribute.
    Description: Use Cell object's methods and attributes.
    Expectation: No exception.
    """
    net = CellNet()
    x = ms.Tensor(1)
    y = ms.Tensor(2)
    print(net(x, y))


def test_construct_require_self():
    """
    Feature: Support use Cell method and attribute.
    Description: Test function construct require self.
    Expectation: No exception.
    """
    x = ms.Tensor(1)
    class ConstructRequireSelf(nn.Cell):
        def construct(x):
            return x

    net = ConstructRequireSelf()
    with pytest.raises(TypeError) as info:
        net(x)
    assert "construct" in str(info.value)
    assert "self" in str(info.value)


def test_construct_exist():
    """
    Feature: Support use Cell method and attribute.
    Description: Test function construct not exist.
    Expectation: No exception.
    """
    class ConstructNotExist1(nn.Cell):
        def cnosrtuct(self):
            pass

    class ConstructNotExist2(nn.Cell):
        pass

    net1 = ConstructNotExist1()
    with pytest.raises(AttributeError):
        net1()

    net2 = ConstructNotExist2()
    with pytest.raises(AttributeError):
        net2()


def test_cell_private_attr():
    """
    Feature: Support use Cell private attribute.
    Description: Test private attribute startswith "__".
    Expectation: No exception.
    """
    class AttrNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.__x = 1

        def construct(self):
            return self.__x, ms.Tensor(self.__x)

    net = AttrNet()
    out = net()
    assert isinstance(out[0], int) and out[0] == 1
    assert isinstance(out[1], ms.Tensor) and out[1] == 1


class PrivateSubnet(Cell):
    def __init__(self):
        super().__init__()
        self.__tensor = Tensor([1, 2], dtype=mstype.int32)
        self.__mul = P.Mul()

    def __square(self, input_x):
        return self.__mul(input_x, input_x)

    def construct(self, input_x):
        return self.__mul(input_x, self.__tensor)


@pytest.mark.skip(reason="has not supported")
def test_parse_construct_subnet_private_tensor():
    """
    Feature: Support use Cell private attribute.
    Description: Test private attribute startswith "__" which is other net need raise error.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self, subnet):
            super().__init__()
            self.subnet = subnet
            self.__tensor = Tensor([2, 3], dtype=mstype.int32)
            self.mul = P.Mul()

        def construct(self, input_x):
            y = self.subnet.__tensor
            output = self.mul(input_x, y)
            return output

    subnet = PrivateSubnet()
    net = Net(subnet)
    input_x = Tensor([2, 3], dtype=mstype.int32)
    with pytest.raises(AttributeError):
        net(input_x)


@pytest.mark.skip(reason="has not supported")
def test_parse_construct_subnet_private_method():
    """
    Feature: Support use Cell private attribute.
    Description: Test private attribute startswith "__" which is other net need raise error.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self, subnet):
            super().__init__()
            self.subnet = subnet
            self.mul = P.Mul()

        def construct(self, input_x):
            output = self.subnet.__square(input_x, input_x)
            return output

    subnet = PrivateSubnet()
    net = Net(subnet)
    input_x = Tensor([2, 3], dtype=mstype.int32)
    with pytest.raises(AttributeError):
        net(input_x)


@pytest.mark.skip(reason="has not supported")
def test_parse_construct_subnet_private_op():
    """
    Feature: Support use Cell private attribute.
    Description: Test private attribute startswith "__" which is other net need raise error.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self, subnet):
            super().__init__()
            self.subnet = subnet
            self.mul = P.Mul()

        def construct(self, input_x):
            output = self.subnet.__mul(input_x, input_x)
            return output

    subnet = PrivateSubnet()
    net = Net(subnet)
    input_x = Tensor([2, 3], dtype=mstype.int32)
    with pytest.raises(AttributeError):
        net(input_x)
