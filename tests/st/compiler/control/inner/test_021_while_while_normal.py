# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.zero = Tensor(np.array(0), mstype.int32)
        self.i = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        out = self.zero
        j = self.i
        while j < self.max_cycles:
            i = self.i
            while i < self.max_cycles:
                out = x * y + out
                i = i + 1
            j = j + 1
        return out


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation()

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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

    assert graph_out == Tensor(np.array(27), mstype.int32)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
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

    assert graph_grads == Tensor(np.array(27), mstype.int32)
