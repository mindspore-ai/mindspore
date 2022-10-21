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
from mindspore import context, jit
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
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

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_net_foward = SingleForNet()
    graph_forward_res = for_net_foward(x, y, z)

    for_net = SingleForNet()
    net = GradNet(for_net)
    graph_backward_res = net(x, y, z)

    assert graph_forward_res == Tensor([125], mstype.int32)
    assert graph_backward_res == (Tensor([15], mstype.int32), Tensor([40], mstype.int32), Tensor([5], mstype.int32))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_02():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()

        def construct(self, x, y, z):
            x = self.add(x, y)
            for _ in range(10, -5, -3):
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

    for_net_forward = SingleForNet()
    graph_forward_res = for_net_forward(x, y, z)
    graph_backward_res = net(x, y, z)

    assert graph_forward_res == Tensor([195], mstype.int32)
    assert graph_backward_res == (Tensor([25], mstype.int32), Tensor([64], mstype.int32), Tensor([5], mstype.int32))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_03():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor([5], dtype=mstype.int32), name='a')
            self.param_b = Parameter(Tensor([2], dtype=mstype.int32), name='b')

        def func(self, x):
            x = self.mul(x, 2)
            for _ in range(0, 5):
                x = self.add(x, x)
                self.param_b = self.param_b + 1
            return x - self.param_b

        def construct(self, x, y):
            self.assign(self.param_a, x + self.param_a)
            z = self.func(x)
            x = self.param_a + y + z
            return x, self.param_b

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)

    for_net_forward = SingleForNet()
    graph_forward_res = for_net_forward(x, y)
    graph_backward_res = net(x, y)

    assert graph_forward_res == (Tensor([133], mstype.int32), Tensor([7], mstype.int32))
    assert graph_backward_res == (Tensor([64], mstype.int32), Tensor([1], mstype.int32))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_04():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor([5], dtype=mstype.int32), name='a')
            self.param_b = Parameter(Tensor([2], dtype=mstype.int32), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            for _ in range(1):
                F.assign(self.param_b, x - self.param_a)
            return self.param_b

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)

    for_net_forward = SingleForNet()
    graph_forward_res = for_net_forward(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor([-5], mstype.int32)
    assert graph_backward_res == (Tensor([0], mstype.int32),)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_05():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor(np.array(5), mstype.int32), name='a')
            self.param_b = Parameter(Tensor(np.array(2), mstype.int32), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            for _ in range(0, 3):
                self.assign(self.param_b, x - self.param_a)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([6], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)

    for_net_forward = SingleForNet()
    graph_forward_res = for_net_forward(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor([6], mstype.int32)
    assert graph_backward_res == (Tensor([1], mstype.int32),)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for():
    """
    Feature: The else branches of for loops aren't supported.
    Description: The else branches of for loops aren't supported.
    Expectation: No exception.
    """
    @jit
    def control_flow_for(x, y):
        for _ in range(3):
            y += x
            break
        else:
            y = x + 6
        return y

    with pytest.raises(RuntimeError, match="The 'for...else...' statement is not supported now."):
        input_x = Tensor([0], mstype.int32)
        input_y = Tensor([2], mstype.int32)
        res = control_flow_for(input_x, input_y)
        print("res:", res)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_with_not_iterable_object():
    """
    Feature: The else branches of for loops aren't supported.
    Description: The else branches of for loops aren't supported.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_with_not_iterable_object():
        ret = 0
        a = 1
        for i in a:
            ret = ret + i
        return ret

    with pytest.raises(TypeError, match="object is not iterable in graph mode"):
        control_flow_for_with_not_iterable_object()
