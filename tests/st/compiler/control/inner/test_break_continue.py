# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
# pylint: disable=missing-module-docstring,super-with-arguments
import numpy as np
from tests.mark_utils import arg_mark
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import context
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})
grad_all = C.GradOperation(get_all=True)


class Grad(nn.Cell):
    def __init__(self, net):
        super(Grad, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


class ForBreakForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForBreakForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        out = self.zero
        for i in range(self.max_cycles):
            if i % 2 == 0:
                continue
            out = x * y + out
            if out == 20:
                return out
            if out > 20:
                break

        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_break_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForBreakForwardNet(max_cycles=3)
    graph_out = forward_net(x, y)
    assert graph_out == Tensor(np.array(3), mstype.int32)


class WhileBreakForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(WhileBreakForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles:
            if i % 2 == 0:
                i = i + 1
                continue
            out = x * y + out
            if out > 20:
                break
            if out == 20:
                return out
            i = i + 1
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_break_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = WhileBreakForwardNet(max_cycles=10)
    graph_mode_out = forward_net(x, y)
    assert graph_mode_out == Tensor(np.array(15))


class IfAfterIfInWhileBreakForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(IfAfterIfInWhileBreakForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)
        self.weight = Parameter(Tensor(np.array(0), mstype.int32))

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles:
            self.weight = i
            if self.weight % 2 == 0:
                i = i + 1
                continue
            if out <= 20:
                self.weight = i
                out = x * y + out
            else:
                break
            i = i + 1
        if out >= 30:
            self.weight = out
            out = out - 30
            return out
        out = out + 1
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_if_after_if_in_while_break_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = IfAfterIfInWhileBreakForwardNet(max_cycles=10)
    graph_mode_out = graph_forward_net(x, y)
    assert graph_mode_out == Tensor(np.array(16), mstype.int32)


class WhileAfterWhileInWhileBreakForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(WhileAfterWhileInWhileBreakForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)
        self.i = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        out = self.zero
        i = self.i
        while i < self.max_cycles:
            j = self.i
            while j < self.max_cycles + 3:
                out = x * y + out
                j = j + 1
                if j > 4:
                    break
            i = i + 1
            if i > 2:
                break
        i = self.i
        while i < self.max_cycles:
            out = x * y + out
            i = i + 1
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_after_while_in_while_break_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = WhileAfterWhileInWhileBreakForwardNet(max_cycles=3)
    graph_out = forward_net(x, y)

    assert graph_out == Tensor(np.array(54), mstype.int32)


class TwoBreakDeadForwardNet(nn.Cell):
    def __init__(self):
        super(TwoBreakDeadForwardNet, self).__init__()
        self.zero = Tensor(np.array(0), mstype.int32)

    def construct(self, x):
        while x < 5:
            if x > 3:
                x -= 2
            elif x == 3:
                break
            else:
                break
            x = x + 1
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_2break_dead_block():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    forward_net = TwoBreakDeadForwardNet()
    graph_out = forward_net(x)

    assert graph_out == Tensor(np.array(1), mstype.int32)


class ForInFor2BreakForwardNet(nn.Cell):
    def __init__(self):
        super(ForInFor2BreakForwardNet, self).__init__()
        self.relu = P.ReLU()
        self.add = P.TensorAdd()

    def construct(self, x, y, z):
        out = z
        for _ in range(2):
            for _ in range(3):
                if 2 * x < y:
                    out = self.add(out, out)
                    x = x + 1
                    if x + 6 == y:
                        break
        out = self.relu(out)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_for_in_for_break():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(7), mstype.float32)
    y = Tensor(np.array(20), mstype.float32)
    z = Tensor(np.array(2), mstype.float32)
    forward_net = ForInFor2BreakForwardNet()
    graph_out = forward_net(x, y, z)
    print("test_for_in_for_break graph out:", graph_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_partial_eliminate_while_for_if_break():
    """
    Feature: nest control flow.
    Description: nest control flow with while,for,if and break.
    Expectation: Null.
    """

    class NetWork(nn.Cell):
        def construct(self, x):
            while x < 3:
                for _ in range(2):
                    if x <= 4:
                        x = x + 1
                        break
                x = 1 + x
            return x

    x = np.array([0], np.float32)
    net = NetWork()
    grad_net = Grad(net)
    grad = grad_net(Tensor(x))
    print(grad)
