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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, strategy3=None, strategy4=None):
        super().__init__()
        self.slice = P.StridedSlice().shard(strategy1)
        self.gather = P.Gather().shard(strategy2)
        self.reshape = P.Reshape()
        self.begin = (0, 0)
        self.strides = (1, 1)
        self.gather_w = Parameter(Tensor(np.ones([8, 16]), dtype=ms.float32), "w1")
        self.matmul1 = P.MatMul().shard(strategy3)
        self.matmul2 = P.MatMul().shard(strategy4)
        self.matmul1_w = Parameter(Tensor(np.ones([16, 64]), dtype=ms.float32), "w2")
        self.matmul2_w = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), "w3")

    def construct(self, x):
        shape = P.Shape()(x)[-1]
        shape = shape - 1
        end = (1, shape)
        out = self.slice(x, self.begin, end, self.strides)
        out = self.gather(self.gather_w, out, 0)
        out = self.reshape(out, (-1, 16))
        out = self.matmul1(out, self.matmul1_w)
        out = self.matmul2(out, self.matmul2_w)
        return out


class PadV3Net(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.pad = P.PadV3().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.value = Tensor([0])
        self.shape = P.Shape()

    def construct(self, x):
        out = self.add(x, self.weight)
        shape = self.shape(out)[-1]
        shape = 1024 - shape
        out = self.pad(out, (0, shape), self.value)
        return out


def test_shape_sub():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1),)         # stridedslice
    strategy2 = ((1, 1), (1, 1))  # gather
    strategy3 = ((1, 1), (1, 8))  # matmul1
    strategy4 = ((1, 8), (8, 1))  # matmul2
    net = Net(strategy1, strategy2, strategy3, strategy4)
    input_x = Tensor(shape=[1, None], dtype=ms.int32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['MatMul-1'])
    assert validator.check_parameter_shape("w2", [16, 8])
    assert validator.check_parameter_shape("w3", [8, 128])


def test_padv3_dynamic():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1),)
    input_x = Tensor(shape=[32, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([32, 16, 1]), dtype=ms.float32)
    net = PadV3Net(weight, strategy1, strategy2)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])
