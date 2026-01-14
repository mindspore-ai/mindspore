# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
""" Test control flow: for in while """
import numpy as np
import torch
from tests.mark_utils import arg_mark
from mindspore import context, Tensor, nn, jit
from mindspore.nn import ForwardValueAndGrad
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(jit_config={"jit_level": "O0"})
grad_all = C.GradOperation(get_all=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_for_in_while_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForInWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            while self.param_a > self.param_b:
                x = self.mul(x, 2)
                for _ in range(0, 5):
                    x = self.add(x, x)
                    self.param_b = self.param_b + 1
            y = self.sub(x, self.param_b)
            self.assign(self.param_a, y)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_while_net = ForInWhileNet()
    backward_net = GradNet(for_in_while_net)

    forward_net = ForInWhileNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = backward_net(x)

    expect_forward_res = Tensor([128], mstype.int32)
    expect_backward_res = (Tensor([64], mstype.int32),)
    assert graph_forward_res == expect_forward_res
    assert graph_backward_res == expect_backward_res


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_for_in_while_02():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForInWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor([5], mstype.float32), name='a')
            self.param_b = Parameter(Tensor([7], mstype.float32), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            while self.param_a > self.param_b:
                for _ in range(0, 3):
                    x = self.add(x, self.param_a + self.param_b)
                    self.assign(self.param_b, self.param_b + 1)
            y = self.sub(x, self.param_b)
            self.assign(self.param_a, y)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.float32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_while_net = ForInWhileNet()
    net = GradNet(for_in_while_net)
    graph_forward_res = for_in_while_net(x)
    graph_backward_res = net(x)

    expect_forward_res = Tensor([2], mstype.float32)
    expect_backward_res = (Tensor([1], mstype.float32),)
    assert graph_forward_res == expect_forward_res
    assert graph_backward_res == expect_backward_res


@arg_mark(plat_marks=['cpu_linux',], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_if_by_for_in_while():
    """
    Feature: Control Flow
    Description: Test if-for in while.
    Expectation: No exception.
    """
    def func(x):
        out = x
        while x > 1:
            out = out + x
            x = x - 1
            if x < 5:
                break
            for _ in range(3):
                x = x - 1
                out = out + x
        return out

    x = Tensor(12)
    assert jit(func)(x) == 78


@arg_mark(plat_marks=['cpu_linux',], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_for_by_if_in_while():
    """
    Feature: Control Flow
    Description: Test for-if in while.
    Expectation: No exception.
    """
    def func(x):
        out = x
        while x > 1:
            x = x - 1
            out = out + x
            for _ in range(5):
                out = out + x
            if x < 3:
                return out
        return out

    x = Tensor(9)
    assert jit(func)(x) == 219


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_control_flow_for_in_while_break_continue():
    """
    Feature: Control Flow
    Description: Test for in while with break and continue
    Expectation: No exception.
    """
    class CtrlForInWhileBC(nn.Cell):
        def __init__(self, t):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()
            self.assignadd = P.AssignAdd()
            self.para = Parameter(t, name="a")

        def construct(self, x, y):
            out = self.add(y, y)
            while x > 3:
                x = x - 1
                if x < 5:
                    self.assignadd(self.para, y)
                    break
                for _ in range(1, 10):
                    x = x + 1
                    if x < 2:
                        out = self.add(out, y)
                    elif x < 5:
                        y = self.mul(y, y)
                        continue
            out = self.add(out, self.para)
            return out

    class TorchForInWhileBC:
        def __init__(self, t):
            self.para = torch.tensor(t, dtype=torch.float)

        def construct(self, x, y):
            out = y + y
            while x > 3:
                x = x - 1
                if x < 5:
                    self.para = self.para + y
                    break
                for _ in range(1, 10):
                    x = x + 1
                    if x < 2:
                        out = out + y
                    elif x < 5:
                        y = y * y
                        continue
            out = out + self.para
            return out

    context.set_context(mode=context.GRAPH_MODE, jit_level='O0')
    input_np = np.random.randn(3, 4, 5).astype(np.float32)
    ms_x = Tensor(3, mstype.float32)
    ms_y = Tensor(input_np, mstype.float32)
    tc_x = torch.tensor(3, dtype=torch.float)
    tc_y = torch.tensor(input_np, dtype=torch.float)
    tc_x.requires_grad = True
    tc_y.requires_grad = True

    ms_net = CtrlForInWhileBC(Tensor(input_np, mstype.float32))
    tc_net = TorchForInWhileBC(torch.tensor(input_np, dtype=torch.float))
    grad_net = ForwardValueAndGrad(ms_net, get_all=True)
    ms_out, ms_grad = grad_net(ms_x, ms_y)
    tc_out = tc_net.construct(tc_x, tc_y)
    tc_out.backward(torch.ones_like(tc_out))
    tc_grad = [t.grad for t in (tc_x, tc_y)]
    assert np.allclose(ms_out.asnumpy(), tc_out.detach().numpy(), 0.001, 0.001)
    for m, t in zip(ms_grad, tc_grad):
        if t is None:
            continue
        assert np.allclose(m.asnumpy(), t.detach().numpy(), 0.001, 0.001)
