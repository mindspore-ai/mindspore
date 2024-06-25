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
from tests.st.compiler.control.cases_register import case_register
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter

grad_all = C.GradOperation(get_all=True)


class IfInIfNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

    def construct(self, x):
        if self.param_a > self.param_b:
            x += 10
            if x > self.param_a:
                self.param_b += 1
                x += self.param_a
        return x


class IfInIfNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

    def construct(self, x):
        if self.param_a > self.param_b:
            out = self.func(x)
        else:
            out = self.func(self.param_a)
        out += self.param_b
        return out

    def func(self, x):
        x += 10
        if x > self.param_a:
            self.param_b += 1
            x += self.param_a
        return x


class IfInIfNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')
        self.tensor_true = Tensor([True])
        self.tensor_false = Tensor([False])

    def construct(self, x):
        if self.check(self.param_a, self.param_b):
            out = self.func(x)
        else:
            out = x
        out += self.param_b
        return out

    def func(self, x):
        x += 10
        if x > self.param_a:
            self.param_b += 1
            x += self.param_a
        return x

    def check(self, x, y):
        if x < y:
            self.param_b += 1
            return self.tensor_true
        self.param_b -= 1
        return self.tensor_false


class IfInIfNet3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

    def construct(self, x):
        if self.func(x) > self.param_a:
            out = x
        else:
            out = self.param_a
        out += self.param_b
        return out

    def func(self, x):
        x += 10
        if x > self.param_a:
            self.param_b += 1
            x += self.param_a
        return x


# add a while to test if_in_if run with vm.Only should run in ascend.
class IfInIfNet4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
        self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

    def construct(self, x):
        while x < 1:
            x = x + 1
        if self.param_a > self.param_b:
            out = self.func(x)
        else:
            out = self.func(self.param_a)
        out += self.param_b
        return out

    def func(self, x):
        x += 10
        if x > self.param_a:
            self.param_b += 1
            x += self.param_a
        return x


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net

    def construct(self, *inputs):
        return grad_all(self.net)(*inputs)


def control_flow_if_in_if(input_net, x, expect1, expect2):
    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = input_net()
    grad_net = GradNet(net)

    forward_net = input_net()
    graph_forward_res = forward_net(x)
    graph_backward_res = grad_net(x)

    assert graph_forward_res == expect1
    assert graph_backward_res == expect2


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_if():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    expect1 = Tensor(17, mstype.int32)
    expect2 = (Tensor(1, mstype.int32),)
    control_flow_if_in_if(IfInIfNet, x, expect1, expect2)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_if_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    expect1 = Tensor(22, mstype.int32)
    expect2 = (Tensor(1, mstype.int32),)
    control_flow_if_in_if(IfInIfNet1, x, expect1, expect2)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_if_02():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    expect1 = Tensor(5, mstype.int32)
    expect2 = (Tensor(1, mstype.int32),)
    control_flow_if_in_if(IfInIfNet2, x, expect1, expect2)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_in_if_03():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    expect1 = Tensor(7, mstype.int32)
    expect2 = (Tensor(1, mstype.int32),)
    control_flow_if_in_if(IfInIfNet3, x, expect1, expect2)


@case_register.level1
@case_register.target_ascend
def test_if_in_if_04():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(2, mstype.int32)
    expect1 = Tensor(22, mstype.int32)
    expect2 = (Tensor(1, mstype.int32),)
    control_flow_if_in_if(IfInIfNet4, x, expect1, expect2)
