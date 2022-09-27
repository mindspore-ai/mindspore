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
import pytest
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore import context
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)
        self.weight = Parameter(Tensor(np.array(0), mstype.int32))

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles:
            if out <= 20:
                out = x * y + out
                # use F.Assign will throw NameSpace error.
                F.assign(self.weight, i)
                self.weight = i
            i = i + 1
        return out


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward():
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    graph_forward_net = ForwardNet(max_cycles=10)
    graph_mode_out = graph_forward_net(x, y)

    assert graph_mode_out == Tensor(21, mstype.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward():
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    graph_forward_net = ForwardNet(max_cycles=10)
    graph_backward_net = BackwardNet(graph_forward_net)
    graph_mode_grads = graph_backward_net(x, y)

    assert graph_mode_grads == (Tensor(21, mstype.int32), Tensor(7, mstype.int32))
