# Copyright 2020-2025 Huawei Technologies Co., Ltd
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

"""test switch layer."""

import pytest
import numpy as np
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import context
from mindspore import Tensor, nn, jit
from mindspore.common import dtype as mstype
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P

context.set_context(jit_config={"jit_level": "O0"})

class Grad(nn.Cell):
    def __init__(self, net, get_all=False):
        super().__init__()
        self.grad = GradOperation(get_all=get_all)
        self.net = net

    def construct(self, x, y):
        grad_net = self.grad(self.net)
        grad = grad_net(x, y)
        return grad


class CaseNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.layers1 = (self.relu, self.softmax)
        self.layers2 = (self.conv, self.relu1)

    def construct(self, x, index1, index2):
        x = self.layers1[index1](x)
        x = self.layers2[index2](x)
        return x


class SimpleCell(nn.Cell):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def construct(self, x):
        return self.i * x


class CellInList(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(SimpleCell(4))
        self.cell_list.append(SimpleCell(5))
        self.cell_list.append(SimpleCell(6))

    def construct(self, t, x):
        out = t
        while x < 3:
            add = self.cell_list[x](t)
            out = out + add
            x += 1

        return out

class SwitchLayerNet(nn.Cell):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = funcs

    def construct(self, i, inputs):
        return self.funcs[i](inputs)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_switch_layer():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = CaseNet()
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(1, mstype.int32)
    value = net(data, idx, idx2)
    relu = nn.ReLU()
    true_value = relu(data)
    ret = np.allclose(value.asnumpy(), true_value.asnumpy())
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_cell_in_list():
    """
    Feature: Switch layer in while.
    Description: test recursive switch layer.
    Expectation: success if grad and output are correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = CellInList()
    t = Tensor(10, mstype.int32)
    x = Tensor(0, mstype.int32)
    out = net(t, x)

    t1 = Tensor(10, mstype.int32)
    x1 = Tensor(0, mstype.int32)
    grad_net = Grad(net)
    grad_out = grad_net(t1, x1)

    assert out == Tensor(160, mstype.int32)
    assert grad_out == Tensor(16, mstype.int32)


class TwoLayerReLU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = P.ReLU()
        self.funcs2 = P.Neg()

    def construct(self, inputs):
        x = self.funcs1(inputs)
        x = self.funcs2(x)
        return x


class TwoLayerSoftmax(nn.Cell):
    def __init__(self):
        super().__init__()
        self.funcs1 = P.Softmax()
        self.funcs2 = P.Neg()

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


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_switch_layer_add_func_in_construct():
    """
    Feature: Switch layer.
    Description: test switch layer add function in construct.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    func1 = TwoLayerSoftmax()
    func2 = TwoLayerReLU()
    func3 = TwoLayerSoftmax()
    funcs = (func1, func2)
    net = AddFuncNet(funcs, func3)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(2, mstype.int32)
    ret = net(i, inputs)
    assert ret.shape == (2, 3, 4, 5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_layer_add_diff_func():
    """
    Feature: Switch layer.
    Description: test switch layer add function in construct.
    Expectation: No exception.
    """
    class EliminatedA(nn.Cell):
        def __init__(self, flag):
            super().__init__()
            self.flag = flag

        def construct(self):
            if self.flag > 0:
                return 0
            return 1

    class EliminatedB(nn.Cell):
        def __init__(self, flag):
            super().__init__()
            self.flag = flag

        def construct(self):
            if self.flag > 0:
                return 1
            return 0

    class ZeroInputNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs

        @jit
        def construct(self, i, inputs):
            x = self.funcs[i]()
            if x != 0:
                return self.relu(inputs)
            return inputs

    func1 = EliminatedA(1)
    func2 = EliminatedB(-1)
    funcs = (func1, func2)
    net = ZeroInputNet(funcs)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(1, mstype.int32)
    out = net(i, inputs)
    assert np.allclose(out.asnumpy(), inputs.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_layer_outputs_diff_dtype():
    """
    Feature: Switch layer.
    Description: test switch layer in construct.
    Expectation: No exception.
    """
    class CastNet(nn.Cell):
        def __init__(self, dtype):
            super().__init__()
            self.op = P.Cast()
            self.dtype = dtype

        def construct(self, x):
            y = self.op(x, self.dtype)
            return y + y

    class SwitchNegNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs
            self.op = P.Neg()

        def construct(self, i, inputs):
            x = self.funcs[i](inputs)
            x = self.op(x)
            return x

    context.set_context(mode=context.GRAPH_MODE)
    func1 = TwoLayerSoftmax()
    func2 = CastNet(ms.int32)
    funcs = (func1, func2)
    net = SwitchNegNet(funcs)
    inputs = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    i = Tensor(0, mstype.int32)
    with pytest.raises(TypeError) as ex:
        net(i, inputs)
    assert "Cannot join the return values of different branches" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_layer_index():
    """
    Feature: Switch layer.
    Description: test switch layer.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    func1 = nn.ReLU()
    func2 = nn.Softmax()
    funcs = (func1, func2)

    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    i1 = Tensor(1, mstype.int32)
    net1 = SwitchLayerNet(funcs)
    assert np.allclose(net1(i1, x).asnumpy(), func2(x).asnumpy())

    i2 = Tensor(-2, mstype.int32)
    net2 = SwitchLayerNet(funcs)
    assert np.allclose(net2(i2, x).asnumpy(), func1(x).asnumpy())

    i3 = Tensor(2, mstype.int32)
    with pytest.raises(IndexError) as ex:
        SwitchLayerNet(funcs)(i3, x)
    assert "Given index 2 out of range" in str(ex.value)

    i4 = Tensor(-3, mstype.int32)
    with pytest.raises(IndexError) as ex:
        SwitchLayerNet(funcs)(i4, x)
    assert "Given index -3 out of range" in str(ex.value)

    i5 = Tensor(1.5, mstype.float32)
    with pytest.raises(ValueError) as ex:
        SwitchLayerNet(funcs)(i5, x)
    assert "switch_layer index must be an int32" in str(ex.value)

    i6 = Tensor(np.array([2, 2]), mstype.int32)
    with pytest.raises(ValueError) as ex:
        SwitchLayerNet(funcs)(i6, x)
    assert "switch_layer index must be a 0 dimension tensor" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_layer_bprop():
    """
    Feature: Switch layer.
    Description: test switch layer in construct.
    Expectation: No exception.
    """
    class BpropNet(nn.Cell):
        def __init__(self, funcs):
            super().__init__()
            self.funcs = funcs
            self.op = P.ReLU()

        @jit
        def construct(self, i, x):
            x = self.op(x)
            return x

        def bprop(self, i, x, out, dout):
            return i, self.funcs[i](x)

    context.set_context(mode=context.GRAPH_MODE)
    func1 = TwoLayerSoftmax()
    func2 = TwoLayerReLU()
    funcs = (func1, func2)
    net = BpropNet(funcs)
    x = Tensor(np.ones([2, 2]).astype(np.float32))
    i = Tensor(1, mstype.int32)
    grad_net = Grad(net, True)
    out_grad = grad_net(i, x)
    expect_grad = TwoLayerReLU()(x)
    np.allclose(out_grad[1].asnumpy(), expect_grad.asnumpy(), 0.001, 0.001)
