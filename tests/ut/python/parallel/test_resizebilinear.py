# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
'''ResizeBilinear and ResizeNearestNeigbor ut'''
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
    '''
    create the test Net
    '''

    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride,
                 strategy1=None, strategy2=None):
        super(Net, self).__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.resize_bilinear = P.ResizeBilinearV2().shard(strategy2)

    def construct(self, x):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.resize_bilinear(out, (16, 16))
        return out


class Net2(Cell):
    '''
    create the test Net
    '''

    def __init__(self, conv2d_weight, mul_weight, out_channel, kernel_size, pad_mode, stride, align_corners=False,
                 strategy1=None, strategy2=None, out_strategy=None):
        super(Net2, self).__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.resize_neighbor = P.ResizeNearestNeighbor((16, 16), align_corners).shard(strategy2, out_strategy)
        self.mul = P.Mul()
        self.mul_weight = Parameter(mul_weight, "w2")

    def construct(self, x):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.resize_neighbor(out)
        out = self.mul(out, self.mul_weight)
        return out


class Net3(Cell):
    '''
    create the test Net
    '''

    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride,
                 strategy1=None):
        super(Net3, self).__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.resize_bilinear = P.ResizeBilinearV2()

    def construct(self, x):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.resize_bilinear(out, (16, 16))
        return out


_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_w2 = Tensor(np.ones([32, 8, 16, 16]), dtype=ms.float32)


def compile_net(net, inputs=_x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs)
    context.reset_auto_parallel_context()


def test_bililear_data_parallel():
    """
    Feature: test ResizeBilinear data parallel strategy
    Description: only shard batch dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_bilinear_model_parallel1():
    """
    Feature: test ResizeBilinear model parallel strategy
    Description: shard N/C
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((4, 2, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_bilinear_repeated_calc():
    """
    Feature: test ResizeBilinear repeated calculation parallel strategy
    Description: only shard batch dimension, but shard num smaller than device num
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_bilinear_auto_parallel():
    """
    Feature: test ResizeBilinear auto parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1)
    compile_net(net)


def test_bilinear_no_strategy():
    """
    Feature: test ResizeBilinear semi auto parallel, and has not set strategy for it
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net3(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1)
    compile_net(net)


def test_neighbor_data_parallel():
    """
    Feature: test ResizeNearestNeighbor data parallel strategy
    Description: only shard batch dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_neighbor_model_parallel_align_corners_shard_HW():
    """
    Feature: test ResizeNearestNeighbor model parallel strategy
    Description: the align_corners is True, and shard N/C/H/W
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 2, 2),)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1, align_corners=True,
               strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_neighbor_out_strategy():
    """
    Feature: test ResizeNearestNeighbor to set output parallel strategy
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 2, 2),)
    out_strategy = ((2, 2, 2, 2),)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
               strategy1=strategy1, strategy2=strategy2, out_strategy=out_strategy)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_neighbor_model_parallel_align_corners_shard_NC():
    """
    Feature: test ResizeNearestNeighbor model parallel strategy
    Description: the align_corners is True, and shard N/C
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((4, 4, 1, 1),)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1, align_corners=True,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_neighbor_model_parallel_align_corners_is_false():
    """
    Feature: test ResizeNearestNeighbor model parallel strategy
    Description: the align_corners is False, and shard N/C/H/W
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 2, 2),)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_neighbor_auto_parallel():
    """
    Feature: test ResizeNearestNeighbor auto parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net2(_w1, _w2, out_channel=8, kernel_size=2, pad_mode="same", stride=1)
    compile_net(net)


def test_bilinear_shard_n_c_w():
    """
    Feature: test ResizeBilinear shard n/c/w
    Description: shard n/c/w
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=3)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 1, 2),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_resizebilinear_shard_W_in_GPU():
    """
    Feature: test ResizeBilinear
    Description: the platform is GPU, and shard n/c/w
    Expectation: compile failed, can not shard h or w dimension in GPU
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=3)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 2, 1, 2),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_bilinear_dynamic_shape_constraint():
    """
    Feature: test ResizeBilinear W dimension dynamic shape
    Description: shard N/C
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0, full_batch=False)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((4, 2, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1,
              strategy1=strategy1, strategy2=strategy2)
    input_x = Tensor(shape=[32, 16, 8, None], dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, input_x)
