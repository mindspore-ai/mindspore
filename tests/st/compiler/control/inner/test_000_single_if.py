# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from tests.st.compiler.control.cases_register import case_register
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)


class SingleIfNet(nn.Cell):
    def construct(self, x, y):
        x += 1
        if x < y:
            y += x
        else:
            y -= x
        y += 5
        return y


class SingleIfNet1(nn.Cell):
    def construct(self, x, y):
        x += 1
        out = self.func(x, y)
        out *= 2
        return out

    def func(self, x, y):
        if x < y:
            y += x
        else:
            y -= x
        y += 5
        return y


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net

    def construct(self, *inputs):
        return grad_all(self.net)(*inputs)


def control_flow_single_if(input_net, x, y, expect1, expect2):
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = input_net()
    grad_net = GradNet(net)

    forward_net = input_net()
    graph_forward_res = forward_net(x, y)
    assert graph_forward_res == expect1

    graph_backward_res = grad_net(x, y)
    assert graph_backward_res == expect2


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_if():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    y = Tensor(5, mstype.int32)
    expect1 = Tensor(13, mstype.int32)
    expect2 = (Tensor(1, mstype.int32), Tensor(1, mstype.int32))
    control_flow_single_if(SingleIfNet, x, y, expect1, expect2)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_if_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    y = Tensor(5, mstype.int32)
    expect1 = Tensor(26, mstype.int32)
    expect2 = (Tensor(2, mstype.int32), Tensor(2, mstype.int32))
    control_flow_single_if(SingleIfNet1, x, y, expect1, expect2)


@case_register.level1
@case_register.target_gpu
def test_single_if_any():
    """
    Feature: compile and run control flow with if statement
    Description: true-branch func graph refer a CNode in construct as free variable.
                  That CNode and the inputs will be specialized before ProcessCNode
                  of true-branch func graph, so it's no need to specialize the inputs
                  of that CNode again if it's a specialized func graph.
    Expectation: success
    """
    x = Tensor([True, True, False])
    y = Tensor([False])

    class Net(nn.Cell):
        def __init__(self, input1, input2):
            super().__init__()
            self.input1 = input1
            self.input2 = input2

        def construct(self):
            if self.input1.all() == self.input2:
                return self.input1.any()
            return self.input2

    context.set_context(mode=context.GRAPH_MODE)
    net = Net(x, y)
    output = net()
    assert output
