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

context.set_context(mode=context.GRAPH_MODE)


class ForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles:
            out = x * y + out
            if out > 20:
                break
            i = i + 1
        return out


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation()

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


class ForwardNetReplaceBreak(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNetReplaceBreak, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles and out <= 20:
            out = x * y + out
            i = i + 1
        return out


class BackwardNetReplaceBreak(nn.Cell):
    def __init__(self, net):
        super(BackwardNetReplaceBreak, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation()

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=10)
    graph_mode_out = forward_net(x, y)

    assert graph_mode_out == Tensor(np.array(21), mstype.int32)


# Problem: Exceed function call depth limit 1000.
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNet(max_cycles=10)
    backward_net = BackwardNet(forward_net)
    graph_grads = backward_net(x, y)

    assert graph_grads == Tensor(np.array(21), mstype.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_replace_break():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNetReplaceBreak(max_cycles=10)
    graph_out = forward_net(x, y)

    assert graph_out == Tensor(np.array(21), mstype.int32)


# Problem: Exceed function call depth limit 1000.
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_backward_replace_break():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    forward_net = ForwardNetReplaceBreak(max_cycles=10)
    backward_net = BackwardNetReplaceBreak(forward_net)
    graph_grads = backward_net(x, y)

    assert graph_grads == Tensor(np.array(21), mstype.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_if_elif_elif_else_break():
    """
    Feature: Parallel if transformation with break
    Description: break in else branch should stop the parallel if transformation of all outer if.
    Expectation: assertion success
    """
    class MyNet(nn.Cell):
        def __init__(self, max_cycles):
            super().__init__()
            self.max_cycles = max_cycles
            self.i = Tensor(np.array(0), mstype.int32)
            self.zero = Tensor(np.array(0), mstype.int32)
            self.seg1 = Tensor(np.array(10), mstype.int32)
            self.seg2 = Tensor(np.array(20), mstype.int32)
            self.seg3 = Tensor(np.array(30), mstype.int32)
            self.step = Tensor(np.array(10), mstype.int32)

        def construct(self):
            i = self.i
            out = self.zero
            while i < self.max_cycles:
                if out < self.seg1:
                    out += self.step
                elif out < self.seg2:
                    out += self.step
                elif out < self.seg3:
                    out += self.step
                else:
                    out += self.step
                    break
                i = i + 1
            return out + out

    context.set_context(mode=context.GRAPH_MODE)
    forward_net = MyNet(max_cycles=4)
    graph_mode_out = forward_net()

    assert graph_mode_out == Tensor(np.array(80), mstype.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_if_elif_elif_break_else():
    """
    Feature: Parallel if transformation with break
    Description: break in elif branch should stop the parallel if transformation of all outer if.
    Expectation: assertion success
    """
    class MyNet(nn.Cell):
        def __init__(self, max_cycles):
            super().__init__()
            self.max_cycles = max_cycles
            self.i = Tensor(np.array(0), mstype.int32)
            self.zero = Tensor(np.array(0), mstype.int32)
            self.seg1 = Tensor(np.array(10), mstype.int32)
            self.seg2 = Tensor(np.array(20), mstype.int32)
            self.seg3 = Tensor(np.array(30), mstype.int32)
            self.step = Tensor(np.array(10), mstype.int32)

        def construct(self):
            i = self.i
            out = self.zero
            while i < self.max_cycles:
                if out < self.seg1:
                    out += self.step
                elif out < self.seg2:
                    out += self.step
                elif out > self.seg3:
                    out += self.step
                    break
                else:
                    out += self.step
                i = i + 1
            return out + out

    context.set_context(mode=context.GRAPH_MODE)
    forward_net = MyNet(max_cycles=5)
    graph_mode_out = forward_net()

    assert graph_mode_out == Tensor(np.array(100), mstype.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_if_elif_elif_else_continue():
    """
    Feature: Parallel if transformation with continue
    Description: continue in else branch should stop the parallel if transformation of all outer if.
    Expectation: assertion success
    """
    class MyNet(nn.Cell):
        def __init__(self, max_cycles):
            super().__init__()
            self.max_cycles = max_cycles
            self.i = Tensor(np.array(0), mstype.int32)
            self.zero = Tensor(np.array(0), mstype.int32)
            self.seg1 = Tensor(np.array(10), mstype.int32)
            self.seg2 = Tensor(np.array(20), mstype.int32)
            self.seg3 = Tensor(np.array(30), mstype.int32)
            self.step = Tensor(np.array(10), mstype.int32)

        def construct(self):
            i = self.i
            out = self.zero
            while i < self.max_cycles:
                if out < self.seg1:
                    out += self.step
                elif out < self.seg2:
                    out += self.step
                elif out >= self.seg3:
                    out += self.step
                else:
                    out += self.step
                    continue
                i = i + 1
            return out + out

    context.set_context(mode=context.GRAPH_MODE)
    forward_net = MyNet(max_cycles=4)
    graph_mode_out = forward_net()

    assert graph_mode_out == Tensor(np.array(100), mstype.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_forward_if_elif_elif_continue_else():
    """
    Feature: Parallel if transformation with continue
    Description: continue in elif branch should stop the parallel if transformation of all outer if.
    Expectation: assertion success
    """
    class MyNet(nn.Cell):
        def __init__(self, max_cycles):
            super().__init__()
            self.max_cycles = max_cycles
            self.i = Tensor(np.array(0), mstype.int32)
            self.zero = Tensor(np.array(0), mstype.int32)
            self.seg1 = Tensor(np.array(10), mstype.int32)
            self.seg2 = Tensor(np.array(20), mstype.int32)
            self.seg3 = Tensor(np.array(30), mstype.int32)
            self.step = Tensor(np.array(10), mstype.int32)

        def construct(self):
            i = self.i
            out = self.zero
            while i < self.max_cycles:
                if out < self.seg1:
                    out += self.step
                elif out < self.seg2:
                    out += self.step
                elif out < self.seg3:
                    out += self.step
                    continue
                else:
                    out += self.step
                i = i + 1
            return out + out

    context.set_context(mode=context.GRAPH_MODE)
    forward_net = MyNet(max_cycles=4)
    graph_mode_out = forward_net()

    assert graph_mode_out == Tensor(np.array(100), mstype.int32)
