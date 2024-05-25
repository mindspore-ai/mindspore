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
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import Convolution
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = _get_cache_prim(Convolution)(stride, pad, dilation, False, (0, 0), group).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.neg(out)
        return out


_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_x2 = Tensor(np.ones([32, 16, 10, 10]), dtype=ms.float32)
_x3 = Tensor(np.ones([32, 16, 16, 16]), dtype=ms.float32)
_x4 = Tensor(np.ones([32, 4, 16, 24]), dtype=ms.float32)
_w0 = Tensor(np.ones([8, 16, 1, 1]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_w2 = Tensor(np.ones([8, 16, 3, 3]), dtype=ms.float32)
_w3 = Tensor(np.ones([8, 16, 5, 5]), dtype=ms.float32)
_w4 = Tensor(np.ones([8, 8, 2, 2]), dtype=ms.float32)
_w5 = Tensor(np.ones([8, 16, 4, 4]), dtype=ms.float32)
_w6 = Tensor(np.ones([10, 2, 6, 5]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net, input_x=_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    phase, _ = _cell_graph_executor.compile(train_net, input_x, _b)
    context.reset_auto_parallel_context()
    return phase


def test_convolution_data_parallel():
    """
    Feature: test convolution data parallel
    Description: shard n dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((4, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net)
    validator = ParallelValidator(net, phase)
    sub_graph = {
        '_VirtualDiv-0': ['Convolution-0'],
        'Neg-0': ['_VirtualDiv-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_convolution_not_support_split_w():
    """
    Feature: convolution do not support split w
    Description: split w dimension
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_convolution_not_support_split_h():
    """
    Feature: convolution do not support split h
    Description: split h dimension
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 8, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
