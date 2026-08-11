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
# pylint: disable=missing-module-docstring,super-with-arguments,unspecified-encoding
import os
import glob
import shutil
import mindspore
from mindspore import nn, Tensor, ops, context

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


class WhileSubGraphParam2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.update = mindspore.Parameter(Tensor(1, mindspore.float32), "update")

    def construct(self, x, y, z):
        out1 = z
        i = self.update
        while x < y:
            i = i + 1
            out1 = out1 + 1
            x = x + 1
        return out1, self.update


def test_save_graphs_with_fullname():
    """
    Feature: format of ir graph
    Description: Every cnode in ir of actions should contain fullname.
    Expectation: No exception.
    """
    x = Tensor(0, mindspore.float32)
    y = Tensor(10, mindspore.float32)
    z = Tensor(100, mindspore.float32)
    context.set_context(save_graphs=1, save_graphs_path="./test_save_graphs4")
    net = WhileSubGraphParam2()
    net(x, y, z)
    ir_files = glob.glob(os.path.join("test_save_graphs4", '*_validate*.ir'))
    node_counting_appear = False
    appear_count = 0
    with open(ir_files[0], 'r') as fp:
        for line in fp:
            if '# Fullname with scope' in line:
                appear_count += 1
            if 'Node counting information:' in line:
                node_counting_appear = True
    remove_path("./test_save_graphs4")
    assert appear_count == 29
    assert node_counting_appear
