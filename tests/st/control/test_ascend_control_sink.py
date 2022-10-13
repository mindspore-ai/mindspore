# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test_ascend_control_sink """
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.ops import operations as op
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer


class ControlSimpleIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, z, input1, input2):
        addn1 = self.addn([input1, input1, input1])
        addn2 = self.addn([input2, input2, input2])
        addn11 = self.addn([addn1, addn1, addn1])
        addn22 = self.addn([addn2, addn2, addn2])
        cond1 = x > y
        cond2 = y > z
        # dodge pylint
        if cond1 and cond2:
            out = self.addn([addn11, addn11])
        else:
            out = self.addn([addn22, addn22])
        out_me = self.addn([out, input1])
        return out_me


class ControlSimpleIfWithAssign(nn.Cell):
    def __init__(self, input_shape):
        super().__init__()
        self.addn = op.AddN()
        self.assign = op.Assign()
        self.input_data = Parameter(initializer(1, input_shape, mstype.float32), name="var")

    def construct(self, x, y, input_data):
        if x > y:
            out = self.addn([input_data, input_data, input_data])
        else:
            self.assign(self.input_data, input_data)
            out = self.input_data
        return out


class ControlIfinIf(nn.Cell):
    """pass"""

    def construct(self, x, y):
        if x > y:
            x = x + 1
            if y < 0:
                y = y + 1
            else:
                y = y + 2
        else:
            x = x + 2
        x = x + y
        return x


class ControlIfbyIfbyIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, cond1, cond2, input_data):
        tri_in = self.addn([input_data, input_data, input_data])
        if x > y:
            addn_1 = self.addn([tri_in, tri_in])
        else:
            addn_1 = self.addn([tri_in, tri_in, tri_in])
        if cond1:
            addn_2 = self.addn([addn_1, addn_1])
        else:
            addn_2 = self.addn([addn_1, addn_1, addn_1])
        if cond2:
            out = self.addn([addn_2, addn_2, addn_2])
        else:
            out = self.addn([addn_2, addn_2])
        return out


class ControlSimpleWhile(nn.Cell):
    def __init__(self):
        super().__init__()
        self.addn = op.AddN()

    def construct(self, x, y, input_data):
        out = input_data
        while x:
            out = self.addn([input_data, input_data, input_data])
            x = y
        return out


class ControlMixedWhileIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.assign = op.Assign()
        self.var = Parameter(initializer(1, (1), mstype.float32), name="var")

    def construct(self, x, y, z, c2, c4):
        out = c4
        self.assign(self.var, c4)
        while x < c2:
            y = c4
            self.assign(self.var, c4)
            while y < c2 and x < c2:
                if 2 * y < c2:
                    y = y + 2
                else:
                    y = y + 1
            out = out + y
            z = c4
            self.assign(self.var, c4)
            while z < c2:
                z = z + 1
            out = out + z
            x = x + 1
        out = out + x
        while x < 2 * c2:
            y = c4
            self.assign(self.var, c4)
            x = x + 1
            while y < c2:
                z = c4
                self.assign(self.var, c4)
                while z < c2:
                    z = z + 1
                if x < c2:
                    y = y - 1
                else:
                    y = y + 1
                out = out + z
            out = out + y
        out = out + x
        return out


class AndOperation(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_sum = op.ReduceSum()

    def construct(self, x, y):
        x_sum = self.reduce_sum(x)
        y_sum = self.reduce_sum(y)
        out = x_sum and y_sum
        return out


class OrOperation(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_sum = op.ReduceSum()

    def construct(self, x, y):
        x_sum = self.reduce_sum(x)
        y_sum = self.reduce_sum(y)
        out = x_sum or y_sum
        return out


class NotOperation(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_sum = op.ReduceSum()

    def construct(self, x):
        x_sum = self.reduce_sum(x)
        return not x_sum


class SimpleCell(nn.Cell):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def construct(self, x):
        return self.i * x


class CellListInWhileByWhile(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(SimpleCell(4))
        self.cell_list.append(SimpleCell(5))
        self.cell_list.append(SimpleCell(6))

    def construct(self, t, x):
        out = t
        while x < 3:
            out += 4
            x += 1
        x = 0
        while x < 3:
            add = self.cell_list[x](t)
            out = out + add
            x += 1
        return out


def cell_list_in_while_by_while():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = CellListInWhileByWhile()
    t = Tensor(10, mstype.int32)
    x = Tensor(0, mstype.int32)
    out = net(t, x)
    return out


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_simple_if():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(3).astype(np.float32)
    y = np.array(2).astype(np.float32)
    z = np.array(3).astype(np.float32)
    input_shape = (127, 7, 53, 31)
    input1 = np.random.randn(*input_shape).astype(np.float32)
    input2 = np.random.randn(*input_shape).astype(np.float32)
    net = ControlSimpleIf()
    output = net(Tensor(x), Tensor(y), Tensor(z), Tensor(input1), Tensor(input2))
    expect = input2 * 3 * 3 * 2 + input1
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_simple_if_with_assign():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(0).astype(np.float32)
    y = np.array(1).astype(np.float32)
    input_shape = (127, 7, 53, 31)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    net = ControlSimpleIfWithAssign(input_shape)
    output = net(Tensor(x), Tensor(y), Tensor(input_data))
    expect = input_data
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(2.345678).astype(np.float32)
    y = np.array(1.234567).astype(np.float32)
    net = ControlIfinIf()
    output = net(Tensor(x), Tensor(y))
    expect = x + y + 3
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_by_if_by_if():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(2.345678).astype(np.float32)
    y = np.array(1.234567).astype(np.float32)
    cond1 = np.array(True).astype(np.bool)
    cond2 = np.array(False).astype(np.bool)
    input_shape = (127, 7, 53, 31)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    net = ControlIfbyIfbyIf()
    output = net(Tensor(x), Tensor(y), Tensor(cond1), Tensor(cond2), Tensor(input_data))
    expect = input_data * 3 * 2 * 2 * 2
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_simple_while():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(True).astype(np.bool)
    y = np.array(False).astype(np.bool)
    input_shape = (127, 7, 53, 31)
    input_data = np.random.randn(*input_shape).astype(np.float32)
    net = ControlSimpleWhile()
    output = net(Tensor(x), Tensor(y), Tensor(input_data))
    expect = input_data * 3
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mixed_while_if():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array(2).astype(np.int32)
    y = np.array(14).astype(np.int32)
    z = np.array(1).astype(np.int32)
    c2 = Tensor([14], mstype.int32)
    c4 = Tensor([0], mstype.int32)
    net = ControlMixedWhileIf()
    output = net(Tensor(x), Tensor(y), Tensor(z), c2, c4)
    expect = np.array(3318).astype(np.int32)
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_and_or_operation():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array([0, 1]).astype(np.float32)
    y = np.array([0, 0]).astype(np.float32)
    net = AndOperation()
    output = net(Tensor(x), Tensor(y))
    expect = np.sum(x) and np.sum(y)
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)

    net = OrOperation()
    output = net(Tensor(x), Tensor(y))
    expect = np.sum(x) or np.sum(y)
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)

    net = NotOperation()
    output = net(Tensor(x))
    expect = not np.sum(x)
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_control_flow_ref():
    """
    Feature: Control flow graph sinking scenarios.
    Description: If the return value of subgraph is Ref, should run graph mode with kernelbykernel.
    Expectation: No exception.
    """
    class IFFuncNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(1, ms.float32), name="a")
            self.param_b = Parameter(Tensor(2, ms.float32), name="b")
            self.one = Tensor(1, ms.float32)

        def subfunc(self, x):
            if x > 4:
                return self.param_a
            return self.param_b

        def construct(self, x):
            out = self.one
            F.assign(self.param_a, 3)
            out += self.subfunc(x)
            F.assign(self.param_b, 2)
            return out

    net = IFFuncNet()
    input_x = Tensor(6, ms.float32)
    out = net(input_x)
    assert out == 4
