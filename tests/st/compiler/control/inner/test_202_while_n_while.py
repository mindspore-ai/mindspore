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

import numpy as np
from tests.mark_utils import arg_mark
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor, Parameter, ops, jit
from mindspore.ops import composite as C
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


class ForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)
        self.i = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        out = self.zero
        i = self.i
        while i < self.max_cycles:
            out = x * y + out
            i = i + 1
        i = self.i
        while i < self.max_cycles:
            out = x * y + out
            i = i + 1
        return out


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation()

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    graph_out = forward_net(x, y)

    assert graph_out == Tensor(np.array(18), mstype.int32)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
def test_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=3)
    backward_net = BackwardNet(forward_net)
    graph_grads = backward_net(x, y)

    assert graph_grads == Tensor(np.array(18), mstype.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_if_by_while_in_while():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, t):
            super(Net, self).__init__()
            self.add = ops.Add()
            self.mul = ops.Mul()
            self.assginadd = ops.AssignAdd()
            self.param = Parameter(t, name='t')

        @jit
        def construct(self, x, y):
            out = ops.add(y, y)
            while x < 2:
                self.assginadd(self.param, y)
                x = x + 1
                if x < 4:
                    out = self.add(out, self.param)
                    break
                while x + 1 > 1:
                    x = x - 1
                    if x < 7:
                        out = self.mul(out, self.param)
                        continue
                    out = self.add(out, y)
            return out

    x = Tensor(1, mstype.int32)
    np_data = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int32)
    y = Tensor(np_data)
    t = Tensor(np_data)
    out = Net(t)(x, y)
    assert np.all(out.asnumpy() == np.array([[4, 8], [12, 16], [20, 24]]))
