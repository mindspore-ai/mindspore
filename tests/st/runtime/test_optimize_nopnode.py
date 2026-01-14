# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import re
import numpy as np
import mindspore
from mindspore import context, ops, nn, Tensor, jit
from tests.mark_utils import arg_mark

def grep(keyword, path):
    files = os.listdir(path)
    for file in files:
        filename = path + file
        with open(filename, 'r') as file:
            for line in file:
                if re.search(keyword, line):
                    return True
    return False

class SqueezeMiddle(nn.Cell):
    def __init__(self):
        super(SqueezeMiddle, self).__init__()
        self.add = ops.Add()
        self.squeeze = ops.Squeeze()
    @jit(backend="ms_backend")
    def construct(self, x, y):
        a = self.add(x, y)
        b = self.squeeze(a)
        return self.add(b, b)

@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_nopnode_launch_skip_success():
    """
    Feature: Nop node skip launch
    Description: Test nop node sequeeze when it locates at the middle of the graph.
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=True,
                        save_graphs_path="./test_nopnode_skip")
    input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
    input_y = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
    expect = np.array([[4, 4], [4, 4], [4, 4]], dtype=np.float32)
    net = SqueezeMiddle()
    output = net(input_x, input_y)
    assert np.allclose(output.asnumpy(), expect)
    keyword = r"Squeeze.*is_launch_skipped:1"
    path = "./test_nopnode_skip/actor_set/"
    assert grep(keyword, path), "nop node skip failed."


class SqueezeGraphInput(nn.Cell):
    def __init__(self):
        super(SqueezeGraphInput, self).__init__()
        self.add = ops.Add()
        self.squeeze = ops.Squeeze()
    @jit(backend="ms_backend")
    def construct(self, x):
        a = self.squeeze(x)
        b = self.add(a, a)
        return b

@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_nopnode_launch_skip_failed_when_graph_input():
    """
    Feature: Nop node skip launch
    Description: Test nop node sequeeze when it locates at the input of the graph.
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=True,
                        save_graphs_path="./test_nopnode_not_skip_when_graph_input")
    input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
    expect = np.array([[2, 2], [2, 2], [2, 2]], dtype=np.float32)
    net = SqueezeGraphInput()
    output = net(input_x)
    assert np.allclose(output.asnumpy(), expect)
    keyword = r"Squeeze.*is_launch_skipped:1"
    path = "./test_nopnode_not_skip_when_graph_input/actor_set/"
    assert not grep(keyword, path), "nop node skip success."

class SqueezeGraphOutput(nn.Cell):
    def __init__(self):
        super(SqueezeGraphOutput, self).__init__()
        self.add = ops.Add()
        self.squeeze = ops.Squeeze()
    @jit(backend="ms_backend")
    def construct(self, x):
        a = self.add(x, x)
        return self.squeeze(a)

@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential", mem_peak=1.00)
def test_nopnode_launch_skip_failed_when_graph_output():
    """
    Feature: Nop node skip launch
    Description: Test nop node sequeeze when it locates at the output of the graph.
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=True,
                        save_graphs_path="./test_nopnode_not_skip_when_graph_output")
    input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
    expect = np.array([[2, 2], [2, 2], [2, 2]], dtype=np.float32)
    net = SqueezeGraphOutput()
    output = net(input_x)
    assert np.allclose(output.asnumpy(), expect)
    keyword = r"Squeeze.*is_launch_skipped:1"
    path = "./test_nopnode_not_skip_when_graph_output/actor_set/"
    assert not grep(keyword, path), "nop node skip success."
