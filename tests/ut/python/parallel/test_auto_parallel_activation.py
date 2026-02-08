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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.sigmoid = P.Sigmoid().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.sigmoid(out)
        return out


_x = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_auto_parallel_activation():
    """
    Feature: test activation auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    from mindspore.parallel.strategy import get_current_strategy_metadata, enable_save_strategy_online
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=16,
                                      global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = None
    net = Net(_w1, strategy1, strategy2)
    enable_save_strategy_online()
    compile_net(net)

    # Sharding_propagation supports saving strategies
    local_info = get_current_strategy_metadata(network=net)
    print("local_info", local_info, flush=True)
    param_list = local_info[0]["w1"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == (1, 0)
