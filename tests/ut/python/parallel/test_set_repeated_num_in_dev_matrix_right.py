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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, attr_status=True):
        super().__init__()

        self.matmul = P.BatchMatMul(transpose_b=False).shard(strategy1)
        self.expand_dims = P.ExpandDims().shard(strategy2)
        self.expand_dims.add_prim_attr("repeated_num_in_dev_matrix_right_", attr_status)

    def construct(self, w, x):
        x = self.expand_dims(x, 0)
        y = self.matmul(w, x)
        return y

_w = Tensor(np.ones([1, 32, 16]), dtype=ms.float32)
_x = Tensor(np.ones([16, 16]), dtype=ms.float32)

def compile_net_train(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()

def test_set_repeated_num_in_dev_matrix_right_false():
    """
    Feature: test set repeated_num_in_dev_matrix_right_ attr success or not
    Description: under below parallel strategy, Net will be insert allreduce operation,
                 repeated_num_in_dev_matrix_right_ attr should eliminate them.
    Expectation: assert True
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 4), (1, 4, 1))
    strategy2 = ((4, 1),)
    net = Net(strategy1=strategy1, strategy2=strategy2, attr_status=False)
    phase = compile_net(net, _w, _x)
    validator = ParallelValidator(net, phase)

    assert validator.check_node_inputs_has('BatchMatMul-0', ['ExpandDims-0'])
