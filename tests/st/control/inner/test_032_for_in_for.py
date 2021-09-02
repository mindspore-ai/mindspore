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
import pytest
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_for_01():
    class ForInForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            for _ in range(0, 10):
                x = self.mul(x, 2)
                for _ in range(0, 5):
                    x = self.add(x, x)
                    self.param_b += 1
            y = self.sub(x, self.param_b)
            z = self.relu(x + y)
            return z

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_for_net = ForInForNet()
    net = GradNet(for_in_for_net)

    forward_net = ForInForNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor([0], mstype.float32)
    assert graph_backward_res == (Tensor([0], mstype.int32),)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_for_02():
    class ForInForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(11, mstype.int32), name='b')

        def construct(self, x):
            for _ in range(0, 3):
                x = x * 2
                self.assign(self.param_a, x + self.param_a)
                for _ in range(0, 2):
                    x = self.add(x, x)
                    self.param_b += 1
            y = self.sub(x, self.param_b + self.param_a)
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_for_net = ForInForNet()
    net = GradNet(for_in_for_net)

    forward_net = ForInForNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor([710], mstype.int32)
    assert graph_backward_res == (Tensor([512], mstype.int32),)
