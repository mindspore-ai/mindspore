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
"""Test control flow: for+break"""

import numpy as np
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore.common import Parameter
from mindspore.common import dtype as ms
from mindspore import nn
from mindspore import context, jit
from tests.mark_utils import arg_mark
from ..share.grad import GradOfFirstInput
from ..share.utils import match_array


class CtrlForBreakRange1(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(1, 10, 3):
            if i >= 7:
                break
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_1_10_3_break():
    """
    Feature: PIJit
    Description: create a net, with if break in for range(1, 10, 3)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBreakRange1()
    jit(function=CtrlForBreakRange1.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBreakRange1()
    jit(function=CtrlForBreakRange1.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForBreakRange2(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(4, -8, -4):
            if i < 0:
                break
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_4_n8_n4_break():
    """
    Feature: PIJit
    Description: create a net, with if break in for range(4, -8, -4)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBreakRange2()
    jit(function=CtrlForBreakRange2.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBreakRange2()
    jit(function=CtrlForBreakRange2.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForBreakRange3(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-5, 5, 2):
            if i == 3:
                break
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_n5_5_2_break():
    """
    Feature: PIJit
    Description: create a net, with if break in for range(-5, 5, 2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBreakRange3()
    jit(function=CtrlForBreakRange3.construct, capture_mode="ast")(ps_net, x) 
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBreakRange3()
    jit(function=CtrlForBreakRange3.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForBreakRange4(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x):
        out = x
        for i in range(-2, -8, -2):
            if i <= -4:
                break
            out = self.add(out, x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_range_n2_n8_n2_break():
    """
    Feature: PIJit
    Description: create a net, with if break in for range(-2, -8, -2)
    Expectation: No exception.
    """
    x = Tensor([2, 3, 4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBreakRange4()
    jit(function=CtrlForBreakRange4.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBreakRange4()
    jit(function=CtrlForBreakRange4.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForEnumerateIfBreak(Cell):
    def __init__(self, t1, t2, t3):
        super().__init__()
        self.p1 = Parameter(Tensor(t1, ms.float32), name="a")
        self.p2 = Parameter(Tensor(t2, ms.float32), name="b")
        self.p3 = Parameter(Tensor(t3, ms.float32), name="c")
        self.assignadd = P.AssignAdd()
        self.add = P.Add()

    def construct(self, x):
        plist = [self.p1, self.p2, self.p3]
        out = x
        for i, t in enumerate(plist):
            if t > 2:
                break
            out = self.add(out, i * x)
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_enumerate_if_break():
    """
    Feature: PIJit
    Description: create a net, with if break in for enumerate list
    Expectation: No exception.
    """
    t1 = 1
    t2 = 2
    t3 = 3
    x = Tensor([4], ms.int32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForEnumerateIfBreak(t1, t2, t3)
    jit(function=CtrlForBreakRange4.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForEnumerateIfBreak(t1, t2, t3)
    jit(function=CtrlForEnumerateIfBreak.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


class CtrlForBreakElifElse(Cell):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(nn.ReLU())
        self.cell_list.append(nn.Tanh())
        self.cell_list.append(nn.Sigmoid())

    def construct(self, x):
        out = x
        for activate in self.cell_list:
            add = activate(x)
            out = out + add
            if add > 1:
                out += x
            elif add < 1:
                break
            else:
                break
            x += add
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_for_break_in_elif_else():
    """
    Feature: PIJit
    Description: create a net, with if break in for in cell list
    Expectation: No exception.
    """
    x = Tensor([0.5], ms.float32)
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlForBreakElifElse()
    jit(function=CtrlForBreakElifElse.construct, capture_mode="ast")(ps_net, x)
    ps_out = ps_net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlForBreakElifElse()
    jit(function=CtrlForBreakElifElse.construct, capture_mode="bytecode")(pi_net, x)
    pi_out = pi_net(x)
    match_array(ps_out, pi_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_range_break_matches_pynative_and_jit_grad():
    """
    Feature: PIJit bytecode capture for for-loop break.
    Description: Verify bytecode JIT handles a for-range loop with break condition and produces gradients matching PyNative.
    Expectation: JIT forward result and gradient match PyNative execution.
    Migrated from: test_pijit_for_while.py::test_pijit_for_range
    """

    class ForRangeBreakNet(Cell):
        def __init__(self):
            super().__init__()
            self.a = 7

        def construct(self, x):
            out = x
            for i in range(1, 10, 3):
                if i >= self.a:
                    break
                out = out + x
            return out

    input_np = np.ones((2, 3, 4), np.float32)
    pynative_input = Tensor(input_np.copy())
    jit_input = Tensor(input_np.copy())

    net = ForRangeBreakNet()
    pynative_result = net(pynative_input)
    pynative_grad_net = GradOfFirstInput(net, sens_param=True)
    pynative_grad_net.set_train()
    sens_np = np.random.randn(*pynative_result.shape).astype(np.float32)
    pynative_grad = pynative_grad_net(pynative_input, Tensor(sens_np.copy()))

    jit_net = ForRangeBreakNet()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(jit_input)
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(jit_input, Tensor(sens_np.copy()))

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)
