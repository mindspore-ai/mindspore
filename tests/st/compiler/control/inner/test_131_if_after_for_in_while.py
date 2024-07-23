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


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_in_while():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class IfAfterForInWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(2, mstype.int32), name='b')

        def construct(self, x):
            out = x + self.param_a
            while self.param_a > self.param_b:
                self.param_b += 1
                for _ in range(4):
                    self.param_a -= 3
            self.param_a -= 40
            if x > self.param_a:
                out += self.param_a * 10
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(2, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    if_after_for_in_while_net = IfAfterForInWhileNet()
    net = GradNet(if_after_for_in_while_net)

    forward_net = IfAfterForInWhileNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    expect_forward_res = Tensor([-463], mstype.int32)
    expect_backward_res = (Tensor([1], mstype.int32),)
    assert graph_forward_res == expect_forward_res
    assert graph_backward_res == expect_backward_res
