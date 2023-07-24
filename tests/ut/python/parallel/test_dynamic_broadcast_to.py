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
from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, br_const_tensor=False):
        super().__init__()
        self.mul = P.Mul()
        self.br = DynamicBroadcastTo()
        self.concat = P.Concat()
        self.relu = P.ReLU()
        self.square = P.Square()
        self.weight = Parameter(weight, "w")
        self.shape_1 = Tensor(np.array([-1], dtype=np.int32))
        self.shape_2 = Tensor(np.array([64], dtype=np.int32))
        self.br_const_tensor = br_const_tensor

    def construct(self, x):
        out = self.mul(x, self.weight)
        shape = self.concat((self.shape_1, self.shape_2))
        if self.br_const_tensor:
            shape = self.relu(shape)
        out = self.br(out, shape)
        out = self.square(out)
        return out


def test_dynamic_broadcast_to_gen_strategy():
    """
    Feature:
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    input_x = Tensor(np.ones([64, 1]), dtype=ms.int32)
    weight = Tensor(np.ones([64, 1]), dtype=ms.float32)
    net = Net(weight, False)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Square-0', ['DynamicBroadcastTo-0'])


def test_dynamic_broadcast_to_gen_strategy_2():
    """
    Feature:
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    input_x = Tensor(np.ones([64, 1]), dtype=ms.int32)
    weight = Tensor(np.ones([64, 1]), dtype=ms.float32)
    net = Net(weight, True)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Square-0', ['DynamicBroadcastTo-0'])
