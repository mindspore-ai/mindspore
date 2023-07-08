# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, start, limit, delta):
        super().__init__()
        self.mul = P.Mul()
        self.limit = Tensor(limit, ms.float32)
        self.start = Tensor(start, ms.float32)
        self.delta = Tensor(delta, ms.float32)

        self.range = P.Range()
        self.relu = P.ReLU()
        self.add = P.Add()
        self.weight = Parameter(weight, "w")

    def construct(self, x):
        start = self.relu(self.start)
        limit = self.relu(self.limit)
        delta = self.relu(self.delta)
        r_out = self.range(start, limit, delta)
        out = self.mul(x, self.weight)
        out = self.add(out, r_out)
        return out


def test_range_dynamic():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    input_x = Tensor(np.ones([64, 8]), dtype=ms.int32)
    weight = Tensor(np.ones([8]), dtype=ms.float32)
    net = Net(weight, 0, 8, 1)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Mul-0'])
    assert validator.check_node_inputs_has('Add-0', ['Range-0'])
