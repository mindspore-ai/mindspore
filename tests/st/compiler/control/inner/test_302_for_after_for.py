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
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_for_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.div = P.Div()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            y = self.add(x, self.param_b)
            for _ in range(0, 2):
                x = self.sub(x, 3)
                y = x + self.param_b
            self.param_a = x + y
            for _ in range(0, 5):
                y = self.mul(x, self.param_a)

            z = self.relu(y + self.param_a)
            return z

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([7], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_for_net = ForAfterForNet()
    net = GradNet(for_after_for_net)

    forward_net = ForAfterForNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor([8], mstype.float32)
    assert graph_backward_res == (Tensor([4], mstype.int32),)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_for_02():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.int32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 11, dtype=np.int32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            y = self.add(x, self.param_a)
            for _ in range(0, 2):
                x = self.sub(x, 3)
                self.assign(self.param_b, x + self.param_b)
            self.param_a = x + y
            for _ in range(0, 5):
                y = self.mul(x, self.param_a)
                x = x + self.param_a
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([7], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_for_net = ForAfterForNet()
    net = GradNet(for_after_for_net)

    forward_net = ForAfterForNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    # pynative mode
    assert graph_forward_res == Tensor([1620], mstype.int32)
    assert graph_backward_res == (Tensor([20], mstype.int32),)
