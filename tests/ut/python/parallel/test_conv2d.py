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

import re
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
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                               stride=stride, dilation=dilation, group=group).shard(strategy1)
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
    _cell_graph_executor.compile(train_net, input_x, _b)
    context.reset_auto_parallel_context()


def test_conv2d_data_parallel():
    """
    Feature: test conv2d data parallel
    Description: shard n dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_pad_mode_overlap_is_negative():
    """
    Feature: test conv2d pad mode and overlap is negative
    Description: shard h/w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 4, 4), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    net = Net(_w5, out_channel=8, kernel_size=4, pad_mode="pad", stride=5, pad=(3, 0, 3, 0),
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x3)


def test_conv2d_pad_mode():
    """
    Feature: test conv2d pad mode and overlap is non-negative
    Description: shard h/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 4), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="pad", stride=1, pad=(3, 3, 3, 3),
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net, _x3)


def test_conv2d_pad_mode_2():
    """
    Feature: test conv2d pad mode and overlap is non-negative
    Description: shard w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    net = Net(_w6, out_channel=10, kernel_size=(6, 5), pad_mode="pad", stride=3, pad=(1, 1, 3, 3), dilation=2, group=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net, _x4)


def test_conv2d_valid_mode_output_shape_cannot_div_by_strategy():
    """
    Feature: test conv2d valid mode, and output shape can not div by strategy
    Description: shard w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="valid", stride=4,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x3)


def test_conv2d_data_parallel_invalid_stride():
    """
    Feature: test conv2d invalid stride
    Description: the first two elements of stride must be 1, but set 2
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=(2, 2, 1, 1),
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_data_parallel_dilation():
    """
    Feature: test conv2d data parallel and dilation is not 1
    Description: data parallel and dilation is not 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, dilation=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_data_parallel_group():
    """
    Feature: test conv2d data parallel and group is not 1
    Description: data parallel and group is not 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w4, out_channel=8, kernel_size=2, pad_mode="same", stride=1, group=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel1():
    """
    Feature: test conv2d model parallel
    Description: split n/c-in/c-out
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel_dilation():
    """
    Feature: test conv2d model parallel and dilation is not 1
    Description: model parallel and dilation is not 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, dilation=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel_group():
    """
    Feature: test conv2d model parallel and group is not 1
    Description: split cin and cout, and group is not 1
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w4, out_channel=8, kernel_size=2, pad_mode="same", stride=1, group=2,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_model_parallel_group2():
    """
    Feature: test conv2d model parallel and group is not 1
    Description: has not to split cin and cout, and group is not 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 2, 2), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w4, out_channel=8, kernel_size=2, pad_mode="same", stride=1, group=2,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel2():
    """
    Feature: same mode, stride = kernel_size, no need exchange
    Description: split n/c-in/c-out/h/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 2, 2), (2, 2, 1, 1))
    strategy2 = ((32, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_model_parallel3():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_auto_parallel():
    """
    Feature: same mode, auto parallel
    Description: generate data parallel strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1)
    compile_net(net)


def test_conv2d_model_parallel4():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 1, 4), (2, 2, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_left_and_right_no_need_to_send():
    """
    Feature: same mode, k - s = 1, left pad is 0, single direction exchange
    Description: support that the left no need to send
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_kernel_size_larger_than_stride_and_split_h():
    """
    Feature: same mode, stride < kernel_size, need exchange
    Description: split n/c-in/c-out/h
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_valid_mode_kernel_size_larger_than_stride():
    """
    Feature: valid mode, stride < kernel_size, need exchange
    Description: do not support to split w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="valid", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_output_can_not_divisible_by_strategy():
    """
    Feature: same mode, stride = kernel_size, but output shape can not be divided by strategy
    Description: split w dimension
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_output_can_not_divisible_by_strategy2():
    """
    Feature: same mode, stride = kernel_size, but output shape can not be divided by strategy
    Description: split h dimension
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 8, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_split_kernel():
    """
    Feature: split kernel size
    Description: do not support to split kernel size
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 1), (1, 1, 2, 2))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_same_mode():
    """
    Feature: same mode, slice shape can not be divided by stride
    Description: split w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_valid_mode():
    """
    Feature: valid mode, slice shape can not be divided by stride
    Description: split w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="valid", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_h_dimension_kernel_size_smaller_than_stride_and_slice_is_not_divisible_by_stride_same_mode():
    """
    Feature: same mode, slice shape can not be divided by stride
    Description: split h
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_h_dimension_kernel_size_smaller_than_stride_and_slice_can_not_divisible_by_stride_valid_mode():
    """
    Feature: valid mode, slice shape can not be divided by stride
    Description: split h
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="valid", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_split_h_dimension_and_pad_mode_is_pad():
    """
    Feature: pad mode
    Description: split h
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="pad", stride=2, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_kernel_size_larger_than_stride_and_input_can_not_divisible_by_stride():
    """
    Feature: same mode, input shape can not be divided by stride
    Description: split w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 2), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w3, out_channel=8, kernel_size=5, pad_mode="same", stride=3, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x2)


def test_kernel_size_larger_than_stride_and_slice_too_small():
    """
    Feature: same mode, slice shape is small than overlap shape
    Description: split w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w3, out_channel=8, kernel_size=5, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_dilation():
    """
    Feature: same mode, dilation is 2
    Description: split n/h/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 2, 2), (1, 1, 1, 1))
    strategy2 = ((2, 2, 1, 2),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, dilation=2, strategy1=strategy1,
              strategy2=strategy2)
    compile_net(net)


def test_conv2d_same_mode_overlap_size_equal_to_slice_shape():
    """
    Feature: same mode, slice shape is equal to overlap shape
    Description: split w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_kernel_size_larger_than_stride_and_left_pad_is_0():
    """
    Feature: same mode, kernel_size > stride and left pad is 0, single direction exchange
    Description: split w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_kernel_size_larger_than_stride_and_split_nchw():
    """
    Feature: same mode, stride < kernel_size, need to exchange
    Description: split n/c-in/c-out/h/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 2, 2), (2, 2, 1, 1))
    strategy2 = ((2, 2, 2, 2),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_kernel_size_smaller_than_stride_and_split_hw():
    """
    Feature: same mode, kernel_size < stride, no need to exchange
    Description: split h/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((1, 1, 4, 4), (1, 1, 1, 1))
    strategy2 = None
    net = Net(_w0, out_channel=8, kernel_size=1, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    _cell_graph_executor.compile(net, _x, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("ReLU", k) is not None:
            assert v == [[1, 1, 4, 4],]


def test_conv2d_dynamic_shape_constraint():
    """
    Feature: same mode, stride < kernel_size, need to exchange, input dynamic shape
    Description: split n/c-in/c-out/h/w
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0,
                                      full_batch=False)
    strategy1 = ((2, 2, 2, 2), (2, 2, 1, 1))
    strategy2 = ((2, 2, 2, 2),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    dynamic_x = Tensor(shape=[None, 16, 8, 8], dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, input_x=dynamic_x)
