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
import os
import pytest
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)
context.set_context(device_target="Ascend")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_01():
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

    os.environ['ENV_FOR_TO_WHILE_LOOP'] = '1'
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
    os.environ['ENV_FOR_TO_WHILE_LOOP'] = ''

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
