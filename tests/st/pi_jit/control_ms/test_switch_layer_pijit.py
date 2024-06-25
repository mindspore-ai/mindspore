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
"""test cell with switch layer in PIJit and pynative mode"""
import numpy as np

import mindspore.context as context
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore import jit
from tests.mark_utils import arg_mark


class CaseNet(nn.Cell):
    def __init__(self):
        super(CaseNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.layers1 = (self.relu, self.softmax)
        self.layers2 = (self.conv, self.relu1)

    @jit(mode="PIJit")
    def construct(self, x, index1, index2):
        x = self.layers1[index1](x)
        x = self.layers2[index2](x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_switch_layer_pi():
    """
    Feature: Switch layer.
    Description: test switch layer add function in construct.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = CaseNet()
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(1, mstype.int32)
    jit(CaseNet.construct, mode="PIJit")(net, data, idx, idx2)
    value = net(data, idx, idx2)
    relu = nn.ReLU()
    true_value = relu(data)
    ret = np.allclose(value.asnumpy(), true_value.asnumpy())
    assert ret


class TwoLayerRelU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = P.ReLU()
        self.funcs2 = P.Neg()

    @jit(mode="PIJit")
    def construct(self, inputs):
        x = self.funcs1(inputs)
        x = self.funcs2(x)
        return x


class TwoLayerSoftmax(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = P.Softmax()
        self.funcs2 = P.Neg()

    @jit(mode="PIJit")
    def construct(self, inputs):
        x = self.funcs1(inputs)
        x = self.funcs2(x)
        return x


class AddFuncNet(nn.Cell):
    def __init__(self, funcs, new_func):
        super().__init__()
        self.funcs = funcs
        self.new_func = new_func

    @jit(mode="PIJit")
    def construct(self, i, inputs):
        final_funcs = self.funcs + (self.new_func,)
        x = final_funcs[i](inputs)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_switch_layer_add_func_in_construct():
    """
    Feature: Switch layer.
    Description: test switch layer add function in construct.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func1 = TwoLayerSoftmax()
    func2 = TwoLayerRelU()
    func3 = TwoLayerSoftmax()
    funcs = (func1, func2)
    net = AddFuncNet(funcs, func3)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(2, mstype.int32)
    jit(AddFuncNet.construct, mode="PIJit")(net, i, inputs)
    ret = net(i, inputs)
    assert ret.shape == (2, 3, 4, 5)
