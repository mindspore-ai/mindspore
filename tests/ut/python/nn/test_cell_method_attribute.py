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
import pytest

import mindspore as ms
import mindspore.nn as nn


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


def test_init_require_self():
    """
    Feature: Support use Cell method and attribute.
    Description: Test function __init__ require self.
    Expectation: No exception.
    """
    class InitRequireSelf(nn.Cell):
        def __init__():
            pass

    with pytest.raises(TypeError):
        InitRequireSelf()


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
