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

import mindspore as ms


class Net:
    @ms.ms_function
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_user_defined_class_with_ms_function():
    """
    Feature: User defined class with ms_function.
    Description: Test user defined class method with ms_function.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = Net()
    net.test(x, y)


@ms.ms_class
class MsClassNet:
    @ms.ms_function
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_ms_class_with_ms_function():
    """
    Feature: ms_class with ms_function.
    Description: Test ms_class method with ms_function.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = MsClassNet()
    net.test(x, y)


class CellNet(ms.nn.Cell):
    @ms.ms_function
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_cell_with_ms_function():
    """
    Feature: Cell with ms_function.
    Description: Test Cell method with ms_function.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = CellNet()
    net.test(x, y)
