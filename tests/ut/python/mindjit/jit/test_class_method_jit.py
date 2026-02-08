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
    @ms.jit
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_user_defined_class_with_jit_decorated_function():
    """
    Feature: User defined class in the function decorated with jit.
    Description: Test user defined class method in the function decorated with jit.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = Net()
    net.test(x, y)


def test_call_jit_decorated_class_funciton_without_an_instance():
    """
    Feature: User defined class in the function decorated with jit
    Description: Test call class function without class instance.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    Net.test(None, x, y)


@ms.jit_class
class MsClassNet:
    @ms.jit
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_jit_class_with_jit_decorated_function():
    """
    Feature: jit_class in the function decorated with jit.
    Description: Test jit_class method in the function decorated with jit.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = MsClassNet()
    net.test(x, y)


class CellNet(ms.nn.Cell):
    @ms.jit
    def test(self, x, y):
        return ms.ops.mul(x, y)


def test_cell_with_jit_decorated_function():
    """
    Feature: Cell in the function decorated with jit.
    Description: Test Cell method in the function decorated with jit.
    Expectation: No exception.
    """
    x = ms.Tensor([3])
    y = ms.Tensor([2])
    net = CellNet()
    net.test(x, y)
