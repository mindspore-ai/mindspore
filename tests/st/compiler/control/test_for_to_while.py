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
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)


# Although we don't transform for to while any more, we keep this test case.
@case_register.level1
@case_register.target_ascend
def test_single_for_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()

        def construct(self, x, y, z):
            x = self.add(x, y)
            for _ in range(0, 3):
                z = self.add(z, x)
            y = self.mul(z, y)
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)
    z = Tensor([4], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    graph_forward_res = for_net(x, y, z)
    graph_backward_res = net(x, y, z)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    pynative_forward_res = for_net(x, y, z)
    pynative_backward_res = net(x, y, z)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
