# Copyright 2022 Huawei Technologies Co., Ltd
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
import psutil
import numpy as np
import mindspore as ms
from mindspore import ms_memory_recycle, context, nn, Tensor, Parameter
from mindspore.common.api import cells_compile_cache
from mindspore.ops import composite as C


def test_api_ms_memory_recycle():
    """
    Feature: Memory recycle.
    Description: Test api use.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms_memory_recycle()


def test_single_cell_memory_auto_recycle():
    """
    Feature: Memory recycle.
    Description: Test compiled graph cache in the cell mode.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x + y

    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([1], ms.float32)
    y = Tensor([2], ms.float32)
    net1 = Net()
    net1(x, y)
    assert len(net1.compile_cache) == 1, "net1.compile_cache's size should be 1"
    assert len(cells_compile_cache) == 1, "cells_compile_cache size should be 1"
    net2 = Net()
    net2(x, y)
    assert len(cells_compile_cache) == 2, "cells_compile_cache size should be 2"
    del net1, net2
    assert not cells_compile_cache, "cells_compile_cache size should be 0"


@pytest.mark.skip(reason="random failures")
def test_control_flow_cell_memory_auto_recycle():
    """
    Feature: Memory recycle.
    Description: Test memory auto recycle run in the cell mode.
    Expectation: No exception.
    """

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all = C.GradOperation(get_all=True)

        def construct(self, *inputs):
            return self.grad_all(self.net)(*inputs)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax()
            self.a = Parameter(Tensor(np.full((1,), 5, dtype=np.float32)), name='a')
            self.b = Parameter(Tensor(np.full((1,), 2, dtype=np.float32)), name='b')
            self.c = Parameter(Tensor(np.full((1,), 16, dtype=np.float32)), name='c')

        def construct(self, x, y):
            while self.c > x:
                self.b = self.c + self.b
                for _ in range(0, 5):
                    self.b = self.a + 2
                self.c = self.c - 1
                x = x + 2
                y = self.softmax(self.c) + self.a
                self.b = y - self.b
            x = self.b * self.a
            for _ in range(0, 4):
                y = y + self.b
                x = nn.ReLU()(self.c)
            self.a = x - y
            z = y + self.b
            return z

    def run():
        x = Tensor([11], ms.int32)
        y = Tensor([7], ms.int32)
        net = GradNet(Net())
        net.compile(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    init_mem = psutil.Process(os.getpid()).memory_info().rss / 1024
    first_mem = 0
    last_mem = 0
    for i in range(20):
        run()
        if i == 0:
            first_mem = psutil.Process(os.getpid()).memory_info().rss / 1024
        if i == 19:
            last_mem = psutil.Process(os.getpid()).memory_info().rss / 1024

    first_increase_mem = first_mem - init_mem
    all_increase_mem = last_mem - init_mem
    assert all_increase_mem < first_increase_mem * 1.1, f"first_increase_mem={first_increase_mem}, " \
                                                        f"all_increase_mem={all_increase_mem}, "
