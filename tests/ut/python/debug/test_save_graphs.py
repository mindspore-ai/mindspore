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
import os
import shutil
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context, Parameter
from tests.security_utils import security_off_wrap

context.set_context(mode=context.GRAPH_MODE)


def count_ir_files(path):
    ir_files = 0
    dat_files = 0
    dot_files = 0
    for file in os.listdir(path):
        if file.endswith(".ir"):
            ir_files += 1
        if file.endswith(".dat"):
            dat_files += 1
        if file.endswith(".dot"):
            dot_files += 1
    return ir_files, dat_files, dot_files


def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@security_off_wrap
def test_save_graphs1():
    """
    Feature: test save_graphs.
    Description: test save_graphs.
    Expectation: success.
    """

    class ForwardNet(nn.Cell):
        def __init__(self, max_cycles=10):
            super(ForwardNet, self).__init__()
            self.max_cycles = max_cycles
            self.i = Tensor(np.array(0), mindspore.int32)
            self.zero = Tensor(np.array(0), mindspore.int32)
            self.weight = Parameter(Tensor(np.array(0), mindspore.int32))

        def construct(self, x, y):
            i = self.i
            out = self.zero
            while i < self.max_cycles:
                if out <= 19:
                    out = out + x * y
                    # use F.Assign will throw NameSpace error.
                    ops.assign(self.weight, i)
                    self.weight = i
                i = i + 2
            return out

    context.set_context(save_graphs=True, save_graphs_path="test_save_graphs1")
    a = Tensor(np.array(1), mindspore.int32)
    b = Tensor(np.array(3), mindspore.int32)
    graph_forward_net = ForwardNet(max_cycles=10)
    graph_backward_net = BackwardNet(graph_forward_net)
    graph_backward_net(a, b)

    ir, dat, dot = count_ir_files("test_save_graphs1")
    assert ir > 15
    assert dat == 0
    assert dot == 0
    remove_path("./test_save_graphs1")
    context.set_context(save_graphs=False)


@security_off_wrap
def test_save_graphs2():
    """
    Feature: test save_graphs.
    Description: test save_graphs.
    Expectation: success.
    """

    class ForwardNet(nn.Cell):
        def __init__(self, max_cycles=10):
            super(ForwardNet, self).__init__()
            self.max_cycles = max_cycles
            self.zero = Tensor(np.array(0), mindspore.int32)
            self.i = Tensor(np.array(0), mindspore.int32)

        def construct(self, x, y):
            out = self.zero
            j = self.i
            while j < self.max_cycles:
                i = self.i
                while i < self.max_cycles:
                    out = out + x * y
                    i = i + 1
                j = j + 2
            return out

    a = Tensor(np.array(1), mindspore.int32)
    b = Tensor(np.array(3), mindspore.int32)
    forward_net = ForwardNet(max_cycles=4)
    backward_net = BackwardNet(forward_net)

    context.set_context(save_graphs=True, save_graphs_path="./test_save_graphs2/tmp")
    backward_net(a, b)

    ir, dat, dot = count_ir_files("test_save_graphs2/tmp")
    assert ir > 15
    assert dat == 0
    assert dot == 0
    remove_path("./test_save_graphs2")
    context.set_context(save_graphs=False)


@security_off_wrap
def test_save_graphs3():
    """
    Feature: test save_graphs.
    Description: test save_graphs.
    Expectation: success.
    """

    class ForwardNetNoAssign(nn.Cell):
        def __init__(self, max_cycles=10):
            super(ForwardNetNoAssign, self).__init__()
            self.max_cycles = max_cycles
            self.zero = Tensor(np.array(0), mindspore.int32)
            self.i = Tensor(np.array(0), mindspore.int32)
            self.weight = Parameter(Tensor(np.array(0), mindspore.int32))

        def construct(self, x, y):
            out = self.zero
            i = self.i
            while x < y:
                while i < self.max_cycles:
                    out = out + x * y
                    i = i + 1
                x = x + 1
            if out < 19:
                out = out - 19
            return out

    a = Tensor(np.array(1), mindspore.int32)
    b = Tensor(np.array(3), mindspore.int32)
    graph_forward_net = ForwardNetNoAssign(max_cycles=4)
    graph_backward_net = BackwardNet(graph_forward_net)

    context.set_context(save_graphs=True, save_graphs_path="./test_save_graphs3/../test_save_graphs3")
    graph_backward_net(a, b)

    ir, dat, dot = count_ir_files("test_save_graphs3")
    assert ir > 15
    assert dat == 0
    assert dot == 0
    remove_path("./test_save_graphs3")
    context.set_context(save_graphs=False)
