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


class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride, dilation=dilation, group=group).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.neg(out)
        return out


_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_x2 = Tensor(np.ones([32, 16, 10, 10]), dtype=ms.float32)
_w0 = Tensor(np.ones([8, 16, 1, 1]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_w2 = Tensor(np.ones([8, 16, 3, 3]), dtype=ms.float32)
_w3 = Tensor(np.ones([8, 16, 5, 5]), dtype=ms.float32)
_w4 = Tensor(np.ones([8, 8, 2, 2]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net, input_x=_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, input_x, _b)
    context.reset_auto_parallel_context()


def test_conv2d_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_data_parallel_invalid_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=(2, 2, 1, 1),
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_data_parallel_dilation():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, dilation=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_data_parallel_group():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w4, out_channel=8, kernel_size=2, pad_mode="same", stride=1, group=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel_dilation():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, dilation=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_model_parallel_group():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w4, out_channel=8, kernel_size=2, pad_mode="same", stride=1, group=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_model_parallel2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 2, 2), (2, 2, 1, 1))
    strategy2 = ((32, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel3():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1)
    compile_net(net)


def test_conv2d_model_parallel4():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 1, 4), (2, 2, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_left_and_right_no_need_to_send():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_kernel_size_larger_than_stride_and_split_h():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_valid_mode_kernel_size_larger_than_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="valid", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_output_can_not_divisible_by_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_output_can_not_divisible_by_strategy2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 8, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_split_kernel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 1), (1, 1, 2, 2))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_same_mode():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_valid_mode():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="valid", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_h_dimension_kernel_size_smaller_than_stride_and_slice_is_not_divisible_by_stride_same_mode():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_h_dimension_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_valid_mode():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="valid", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_split_h_dimension_and_pad_mode_is_pad():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="pad", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_kernel_size_larger_than_stride_and_input_can_not_divisible_by_stride():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w3, out_channel=8, kernel_size=5, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_kernel_size_larger_than_stride_and_slice_too_small():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w3, out_channel=8, kernel_size=5, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_same_mode_overlap_size_equal_to_slice_shape():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_kernel_size_larger_than_stride_and_left_pad_is_0():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
