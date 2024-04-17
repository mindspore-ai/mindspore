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
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from parallel.utils.utils import ParallelValidator


class AttentionNet(Cell):
    def __init__(self, weight, bias, strategy1=None, strategy2=None, strategy3=None, strategy4=None, strategy5=None):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.add = P.Add().shard(strategy2)
        self.bias = Parameter(bias, "bias")
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose().shard(strategy3)
        self.realdiv = P.RealDiv().shard(strategy4)
        self.reduce_sum = P.ReduceSum().shard(strategy5)

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.add(out, self.bias)
        s = P.Shape()(out)[1]
        out = self.reshape(out, (-1, 1, s // 4, 4))
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.realdiv(out, 1.0)
        out = self.reduce_sum(out)  # now do not support split sens for dynamic shape, so using reduce_sum to avoid it
        return out


def compile_net(net, input_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(input_x)
    phase, _ = _cell_graph_executor.compile(train_net, input_x)
    context.reset_auto_parallel_context()
    return phase, train_net


def test_dynamic_bs_data_parallel():
    """
    Feature: the batch dimension is dynamic shape, dataset_strategy is default, and data parallel
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    context.set_context(save_graphs=True)
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((8, 1), (1,))
    strategy3 = ((8, 1, 1, 1),)
    strategy4 = ((8, 1, 1, 1), ())
    strategy5 = ((8, 1, 1, 1),)
    weight = Tensor(np.ones([32, 64]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    net = AttentionNet(weight, bias, strategy1, strategy2, strategy3, strategy4, strategy5)
    input_x = Tensor(shape=[None, 32], dtype=ms.float32)

    phase, train_net = compile_net(net, input_x)
    validator = ParallelValidator(train_net, phase)
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0'])
    assert validator.check_node_inputs_has('Transpose-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [32, 64])
    assert validator.check_parameter_shape('bias', [64])
