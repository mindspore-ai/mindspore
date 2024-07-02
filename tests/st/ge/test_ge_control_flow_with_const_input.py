# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore as ms
from mindspore import context, Tensor, ops, nn
from mindspore import SparseTensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_ge_switch_layer_with_const_input():
    """
    Feature: switch_layer with const input
    Description: test switch_layer with const input by jit_level='O2'
    Expectation: success
    """
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O2"})

    class SparseTensorNet(nn.Cell):
        def __init__(self, shape):
            super(SparseTensorNet, self).__init__()
            self.shape = shape

        def construct(self, indices, values):
            x = SparseTensor(indices, values, self.shape)
            return x.values, x.indices, x.shape

    class TwoInputFinalNet(nn.Cell):
        def __init__(self, funcs):
            super(TwoInputFinalNet, self).__init__()
            self.funcs = funcs

        def construct(self, i, a, b):
            x = self.funcs[i](a, b)
            return x

    func1 = SparseTensorNet((3, 4))
    func2 = SparseTensorNet((4, 4))
    net = TwoInputFinalNet((func1, func2))

    indices = Tensor(np.array([[0, 1], [1, 2]], dtype=np.int32))
    values = Tensor(np.array([100, 200], dtype=np.float32))
    output = net(Tensor(0, ms.int32), indices, values)
    expect = func1(indices, values)
    assert output[2] == expect[2]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_ge_control_flow_and_nested_list_with_const_input():
    """
    Feature: control flow and nested list with const input
    Description: test control flow and nested list with const input by jit_level='O2'
    Expectation: success
    """
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O2"})

    class NestedListNet(nn.Cell):
        def __init__(self, l, x):
            super(NestedListNet, self).__init__()
            self.l = l
            self.x = x
            self.relu = ops.ReLU()

        def construct(self):
            if self.x in self.l:
                x = self.relu(self.x)
            else:
                x = self.x
            return x

    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    l = [1, 'str', x, [2, 'ss']]
    output = NestedListNet(l, x)()
    expect = ops.relu(x)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_ge_control_flow_and_nested_tuple_with_const_input():
    """
    Feature: control flow and nested tuple with const input
    Description: test control flow and nested tuple with const input by jit_level='O2'
    Expectation: success
    """
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O2"})

    class NestedTupleNet(nn.Cell):
        def __init__(self, c):
            super(NestedTupleNet, self).__init__()
            self.c = c
            self.relu = ops.ReLU()

        def construct(self, a, b):
            t = (1, 'str', (2, 'ss', self.relu(self.c)), self.relu(a), self.relu(b))
            if self.c in t[2]:
                out = self.relu(self.c)
            else:
                out = self.c + 1
            return out

    a = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    b = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    c = Tensor(np.arange(120).reshape(2, 3, 4, 5), ms.float32)
    output = NestedTupleNet(c)(a, b)
    expect = ops.relu(c)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), 0.001, 0.001)
