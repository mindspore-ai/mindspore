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
# ============================================================================
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.common.initializer import initializer


def setup_function():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)


def compile_net(net, inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.AssignAdd = ops.AssignAdd()
        self.variable = ms.Parameter(initializer(1, [1], ms.float32), name="global_step")

    def construct(self, x):
        self.AssignAdd(self.variable, x)
        return self.variable


def test_remove_cast_before_assign_add_semi_auto_parallel():
    """
    Feature: test remove_cast_before_assign_add
    Description: semi_auto_parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    input_x = Tensor(np.ones([1]).astype(np.float16) * 100)
    inputs = [input_x]
    net = Net()
    compile_net(net, inputs)
