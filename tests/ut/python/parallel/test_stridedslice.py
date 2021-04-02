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
# ============================================================================
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, weight, w2, begin, end, strides, strategy1=None, strategy2=None, is_parameter=True, mask=0):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.strided_slice = P.StridedSlice(begin_mask=mask).shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(w2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x, b):
        out = self.strided_slice(self.weight, self.begin, self.end, self.strides)
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


class Net2(Cell):
    def __init__(self, weight2, begin, end, strides, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.strided_slice = P.StridedSlice().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.strided_slice(out, self.begin, self.end, self.strides)
        return out


_x = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_w1 = Tensor(np.ones([256, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_stridedslice_no_fully_fetch_split_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_stridedslice_strides_no_1_split_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 2), strategy1, strategy2, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_stridedslice_mask_no_0_split_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True, mask=1)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_stridedslice_begin_size_smaller():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0), (128, 64), (1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_stridedslice_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_stridedslice_tensor():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_stridedslice_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_stridedslice_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net)


def test_stridedslice_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 4, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net)


def test_stridedslice_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = None
    net = Net2(_w2, (0, 0, 0), (128, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net)


def test_stridedslice_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2, (0, 0, 0), (32, 64, 1), (1, 1, 1))
    compile_net(net)
