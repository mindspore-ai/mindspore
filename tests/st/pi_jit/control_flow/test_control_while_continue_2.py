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
"""Test control flow: while+continue"""

import numpy as np
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore.common import dtype
from mindspore import context, jit, ops
from tests.mark_utils import arg_mark
from ..share.grad import GradOfFirstInput
from ..share.utils import allclose_nparray, match_array


class CtrlWhileContinueInElse(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, t, x, y):
        self.mul(t, t)
        while t > 2:
            t -= 1
            if (x and y) or not x:
                t -= 1
            elif x or y:
                x = not x
                t -= 2
            else:
                continue
        return t


class CtrlWhile2ElifContinueInIf(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()

    def construct(self, x):
        while x[2] < 4:
            x[2] -= 1
            if x[0] > 2:
                continue
            elif x[1] > 2:
                x[2] += 1
            elif x[2] > 2:
                x[1] += 1
            else:
                x = self.mul(x, x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_while_2elif_continue_in_if():
    '''
    Description: test control flow, 2elif in while, continue in if
    use tensor get_item, set_item as condition
    Expectation: no expectation
    '''
    x = [1, 2, 3]
    context.set_context(mode=context.GRAPH_MODE)
    ps_net = CtrlWhile2ElifContinueInIf()
    jit(function=CtrlWhile2ElifContinueInIf.construct, capture_mode="ast")(ps_net, Tensor(x, dtype.float32))
    ps_out = ps_net(Tensor(x, dtype.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    pi_net = CtrlWhile2ElifContinueInIf()
    jit(function=CtrlWhile2ElifContinueInIf.construct, capture_mode="bytecode")(pi_net, Tensor(x, dtype.float32))
    pi_out = pi_net(Tensor(x, dtype.float32))
    allclose_nparray(ps_out.asnumpy(), pi_out.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_continue_matches_pynative_and_jit_grad():
    """
    Feature: PIJit bytecode capture for while loop with continue.
    Description: Verify bytecode JIT handles a scalar while loop that uses continue and conditional square.
    Expectation: JIT forward result and gradient match PyNative execution.
    Migrated from: test_pijit_for_while.py::test_pijit_while_continue
    """

    class WhileContinueNet(Cell):
        def __init__(self):
            super().__init__()
            self.a = 2

        def construct(self, x):
            while x < self.a:
                x = x + 1
                if x >= 2:
                    continue
                if x == 1:
                    x = ops.mul(x, x)
            return x

    input_np = np.array(0.0, np.float32)
    pynative_input = Tensor(input_np.copy())
    jit_input = Tensor(input_np.copy())

    net = WhileContinueNet()
    pynative_result = net(pynative_input)
    pynative_grad_net = GradOfFirstInput(net, sens_param=True)
    pynative_grad_net.set_train()
    sens_np = np.array(1.0, np.float32)
    pynative_grad = pynative_grad_net(pynative_input, Tensor(sens_np.copy()))

    jit_net = WhileContinueNet()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(jit_input)
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(jit_input, Tensor(sens_np.copy()))

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)
