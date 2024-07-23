# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.st.compiler.control.cases_register import case_register
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import context
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)
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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_for_break_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForBreakForwardNet(max_cycles=3)
    backward_net = Grad(forward_net)
    graph_grads = backward_net(x, y)
    assert graph_grads == (Tensor(np.array(3), mstype.int32), Tensor(np.array(1), mstype.int32))


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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level0
@case_register.target_ascend
def test_while_break_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = WhileBreakForwardNet(max_cycles=10)
    backward_net = Grad(forward_net)
    graph_grads = backward_net(x, y)
    assert graph_grads == (Tensor(np.array(15), mstype.int32), Tensor(np.array(5), mstype.int32))


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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_if_in_while_break_backward():
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
    graph_backward_net = Grad(graph_forward_net)
    graph_mode_grads = graph_backward_net(x, y)

    assert graph_mode_grads == (Tensor(np.array(15), mstype.int32), Tensor(np.array(5), mstype.int32))


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_in_if_break():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class IfAfterForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

        def construct(self, x):
            out = x + self.param_a
            if self.param_a > self.param_b:
                for _ in range(4):
                    self.param_a += 1
                    if self.param_b < 0:
                        continue
                    self.param_b -= 3
                    if self.param_a > 6:
                        break

            self.param_b += 15
            if x < self.param_b:
                out -= self.param_b
                return out
            out = self.param_b + out
            return out

    x = Tensor(2, mstype.int32)

    # graph mode

    forward_net = IfAfterForInIfNet()
    graph_forward_res = forward_net(x)

    context.set_context(mode=context.GRAPH_MODE)
    if_after_for_in_if_net = IfAfterForInIfNet()
    net = Grad(if_after_for_in_if_net)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor(-6, mstype.int32)
    assert graph_backward_res == (Tensor(1, mstype.int32),)


@case_register.level1
@case_register.target_gpu
def test_if_after_for_in_for_break():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class IfAfterForInForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(2, mstype.int32), name='b')

        def construct(self, x):
            out = x + self.param_a
            for _ in range(0, 10):
                x *= 2
                if self.param_a % 2 == 0:
                    self.param_a += 1
                    continue
                for _ in range(0, 5):
                    self.param_a += 1
                    x += self.param_b
                    if x > 10:
                        break
                if x > 100:
                    return x
            if self.param_a > self.param_b:
                out += x
            return out

    x = Tensor(2, mstype.int32)

    # graph mode
    forward_net = IfAfterForInForNet()
    graph_forward_res = forward_net(x)

    if_after_for_in_for_net = IfAfterForInForNet()
    net = Grad(if_after_for_in_for_net)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor(106, mstype.int32)
    assert graph_backward_res == (Tensor(16, mstype.int32),)


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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_while_in_while_break_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = WhileAfterWhileInWhileBreakForwardNet(max_cycles=3)
    backward_net = Grad(forward_net)
    graph_grads = backward_net(x, y)

    assert graph_grads == (Tensor(np.array(54), mstype.int32), Tensor(np.array(18), mstype.int32))


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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_true_break():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class WhileTrueBreakNet(nn.Cell):
        def __init__(self, t):
            super(WhileTrueBreakNet, self).__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.para = Parameter(Tensor(t, mstype.int32), name="a")

        def construct(self, x, y):
            out = self.mul(y, self.para)
            while True:
                if x == 5:
                    x = x - 3
                    continue
                if x == 2:
                    break
                out = self.add(out, out)
            return out

    t = np.array([1]).astype(np.int32)
    y = Tensor([1], mstype.int32)
    x = Tensor([5], mstype.int32)
    net = WhileTrueBreakNet(t)
    grad_net = Grad(net)
    grad_out = grad_net(x, y)
    print(grad_out)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_continue_stuck_in_vm():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class NetWork(nn.Cell):
        def __init__(self, t):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.para = Parameter(Tensor(t, mstype.int32), name="a")

        def construct(self, x, y):
            out = self.mul(y, y)
            while x != 3:
                while self.para > 5:
                    # self.param -= 1 if set after if_switch, which is wrong
                    self.para -= 1
                    x += 1
                    if x > 3:
                        self.para -= x
                        return out
                    out = self.add(out, y)
                continue
            out = self.mul(out, y)
            return out

    x = Tensor(2, mstype.int32)
    t = 8
    y = Tensor(1, mstype.int32)
    net = NetWork(t)
    grad_net = Grad(net)
    grad = grad_net(x, y)
    print(grad)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
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
