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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, nn

ms.set_context(mode=ms.GRAPH_MODE)


class SwitchLayerNet(nn.Cell):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = funcs

    def construct(self, i, inputs):
        return self.funcs[i](inputs)


class TwoLayerReLU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = ops.ReLU()
        self.funcs2 = ops.Neg()

    def construct(self, inputs):
        x = self.funcs1(inputs)
        x = self.funcs2(x)
        return x


class TwoLayerSoftmax(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = ops.Softmax()
        self.funcs2 = ops.Neg()

    def construct(self, inputs):
        x = self.funcs1(inputs)
        x = self.funcs2(x)
        return x


class AddFuncNet(nn.Cell):
    def __init__(self, funcs, new_func):
        super().__init__()
        self.funcs = funcs
        self.new_func = new_func

    def construct(self, i, inputs):
        final_funcs = self.funcs + (self.new_func,)
        x = final_funcs[i](inputs)
        return x


def test_switch_layer_funcs_can_be_eliminated():
    """
    Feature: Switch layer.
    Description: test switch layer in construct.
    Expectation: No exception.
    """
    class Add(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.TensorAdd()

        def construct(self, x, y):
            return self.add(x, y)

    func1 = TwoLayerSoftmax()
    func2 = TwoLayerReLU()
    func3 = Add()
    funcs = (func1, func2)
    net = AddFuncNet(funcs, func3)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, ms.int32)
    with pytest.raises(TypeError) as ex:
        net(i, inputs)
    assert "The parameters number of the function is 2" in str(ex.value)


def test_switch_layer_1024func():
    """
    Feature: Switch layer.
    Description: test switch layer in construct.
    Expectation: No exception.
    """
    func1 = TwoLayerSoftmax()
    func2 = TwoLayerReLU()
    funcs = (func1, func2)
    i = 1022
    while i > 0:
        funcs = funcs + (func2,)
        i = i - 1

    net = SwitchLayerNet(funcs)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, ms.int32)
    with pytest.raises(ValueError) as ex:
        net(i, inputs)
    assert "switch_layer support at least 1 and at most 1000 but got 1024 branches" in str(ex.value)
