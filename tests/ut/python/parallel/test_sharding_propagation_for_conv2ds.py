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

import re
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum, BatchNorm2d
from mindspore.ops import operations as P
from mindspore.parallel import set_algo_parameters


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, conv2d_weight, conv2d_transpose_weight, out_channel, strategy=None):
        super().__init__()
        self.relu = P.ReLU().shard(strategy)
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=3,
                               pad_mode="same", stride=1)
        self.bn = BatchNorm2d(out_channel)
        self.conv2d_transpose = P.Conv2DTranspose(out_channel=out_channel, kernel_size=4,
                                                  pad_mode="same", stride=2)
        self.max_pool = P.MaxPool(kernel_size=2, strides=2)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.conv2d_transpose_weight = Parameter(conv2d_transpose_weight, "w2")


    def construct(self, x, b):
        out = self.relu(x)  # (32, 16, 64, 64)
        out = self.conv2d(out, self.conv2d_weight)  # (32, 8, 64, 64)
        out = self.bn(out)  # (32, 8, 64, 64)
        out = self.max_pool(out)  # (32, 8, 32, 32)
        out = self.conv2d_transpose(out, self.conv2d_transpose_weight, (32, 16, 64, 64))
        out = self.relu(out)  # (32, 16, 64, 64)
        return out


_x = Tensor(np.ones([32, 16, 64, 64]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 3, 3]), dtype=ms.float32)
_w2 = Tensor(np.ones([8, 16, 4, 4]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net, inputs=_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(train_net)
    return strategies


def test_sharding_propagation_8x1x1x1():
    """
    Features: test sharding propagation for conv2d/bn/maxpool/conv2d_transpose
    Description: the fixed strategy is 8x1x1x1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((8, 1, 1, 1),)
    net = Net(_w1, _w2, out_channel=8, strategy=strategy)
    strategies = compile_net(net)
    for (k, v) in strategies.items():
        if re.search("Conv2D", k) is not None:
            assert v == [[8, 1, 1, 1], [1, 1, 1, 1]]
        elif re.search("BatchNorm", k) is not None:
            assert v == [[8, 1, 1, 1], [1], [1], [1], [1]]
        elif re.search("MaxPool", k) is not None:
            assert v == [[8, 1, 1, 1],]
        elif re.search("Conv2DTranspose", k) is not None:
            assert v == [[8, 1, 1, 1], [1, 1, 1, 1]]
    context.reset_auto_parallel_context()


def test_sharding_propagation_1x1x1x8():
    """
    Features: test sharding propagation for conv2d/bn/maxpool/conv2d_transpose
    Description: the fixed strategy is 1x1x1x8
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 1, 1, 8),)
    net = Net(_w1, _w2, out_channel=8, strategy=strategy)
    strategies = compile_net(net)
    for (k, v) in strategies.items():
        if re.search("Conv2D", k) is not None:
            assert v == [[1, 1, 1, 8], [1, 1, 1, 1]]
        elif re.search("BatchNorm", k) is not None:
            assert v == [[1, 1, 1, 8], [1], [1], [1], [1]]
        elif re.search("MaxPool", k) is not None:
            assert v == [[1, 1, 1, 8],]
        elif re.search("Conv2DTranspose", k) is not None:
            assert v == [[1, 1, 1, 8], [1, 1, 1, 1]]
    context.reset_auto_parallel_context()


def test_dynamic_programming_1x1x1x8():
    """
    Features: test dynamic programming for conv2d/bn/maxpool/conv2d_transpose
    Description: the fixed strategy is 1x1x1x8
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="dynamic_programming")
    strategy = ((1, 1, 1, 8),)
    net = Net(_w1, _w2, out_channel=8, strategy=strategy)
    strategies = compile_net(net)
    for (k, v) in strategies.items():
        if re.search("Conv2D", k) is not None:
            assert v == [[8, 1, 1, 1], [1, 1, 1, 1]]
        elif re.search("BatchNorm", k) is not None:
            assert v == [[8, 1, 1, 1], [1], [1], [1], [1]]
        elif re.search("MaxPool", k) is not None:
            assert v == [[8, 1, 1, 1],]
        elif re.search("Conv2DTranspose", k) is not None:
            assert v == [[8, 1, 1, 1], [1, 1, 1, 1]]
    context.reset_auto_parallel_context()


def test_sharding_propagation_1x1x2x1():
    """
    Features: test dynamic programming for conv2d/bn/maxpool/conv2d_transpose
    Description: the fixed strategy is 1x1x2x1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    set_algo_parameters(fully_use_devices=False)
    strategy = ((1, 1, 2, 1),)
    net = Net(_w1, _w2, out_channel=8, strategy=strategy)
    strategies = compile_net(net)
    for (k, v) in strategies.items():
        if re.search("Conv2D", k) is not None:
            assert v == [[1, 1, 2, 1], [1, 1, 1, 1]]
        elif re.search("BatchNorm", k) is not None:
            assert v == [[1, 1, 2, 1], [1], [1], [1], [1]]
        elif re.search("MaxPool", k) is not None:
            assert v == [[1, 1, 2, 1]]
        elif re.search("Conv2DTranspose", k) is not None:
            assert v == [[1, 1, 2, 1], [1, 1, 1, 1]]
    context.reset_auto_parallel_context()
