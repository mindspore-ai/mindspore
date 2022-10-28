# Copyright 2021 Huawei Technologies Co., Ltd
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
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, pool_kernel_size, pool_strides,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.max_pool = P.MaxPool(kernel_size=pool_kernel_size, strides=pool_strides).shard(strategy2)

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.max_pool(out)
        return out


class Net2(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, pool_kernel_size, pool_strides,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.avg_pool = P.AvgPool(kernel_size=pool_kernel_size, strides=pool_strides).shard(strategy2)

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.avg_pool(out)
        return out


_x0 = Tensor(np.ones([32, 16, 10, 10]), dtype=ms.float32)
_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net, inputs=_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs, _b)
    context.reset_auto_parallel_context()


def test_maxpool_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_maxpool_model_parallel1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 2, 2),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_maxpool_model_parallel2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 2, 2),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=4,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_maxpool_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=4)
    compile_net(net)


def test_maxpool_output_is_not_divisible_by_strategy_w_dimension():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_maxpool_output_is_not_divisible_by_strategy_h_dimension():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 8, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_maxpool_shard_h_and_kernel_size_larger_than_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=3, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_maxpool_shard_w_and_kernel_size_larger_than_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 2),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=3, pool_strides=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_maxpool_shard_h_and_input_slice_is_not_divisible_by_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=1, pool_strides=3,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, inputs=_x0)


def test_maxpool_shard_w_and_input_slice_is_not_divisible_by_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=1, pool_strides=3,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, inputs=_x0)


def test_avgpool_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net2(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_avgpool_model_parallel1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 2, 2),)
    net = Net2(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_avgpool_model_parallel2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 2, 2),)
    net = Net2(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=4,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_avgpool_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, pool_kernel_size=2, pool_strides=4)
    compile_net(net)
