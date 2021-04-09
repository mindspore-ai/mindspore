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
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)
context.set_context(device_target="Ascend")

def test_signle_if():
    class SignleIfNet(nn.Cell):
        def construct(self, x, y):
            x += 1
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

    x = Tensor(2, mstype.int32)
    y = Tensor(5, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    if_net = SignleIfNet()
    net = GradNet(if_net)
    graph_forward_res = if_net(x, y)
    graph_backward_res = net(x, y)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    if_net = SignleIfNet()
    net = GradNet(if_net)
    pynative_forward_res = if_net(x, y)
    pynative_backward_res = net(x, y)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
