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
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, conv3d_weight, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv3d = P.Conv3D(out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                               stride=stride, dilation=dilation, group=group).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.conv3d_weight = Parameter(conv3d_weight, "w1")

    def construct(self, x, b):
        out = self.conv3d(x, self.conv3d_weight)
        out = self.neg(out)
        return out


_x = Tensor(np.ones([32, 16, 8, 8, 8]), dtype=ms.float32)
_x3 = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
_x4 = Tensor(np.ones([2, 16, 56, 56, 24]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2, 2]), dtype=ms.float32)
_w2 = Tensor(np.ones([8, 16, 3, 3, 3]), dtype=ms.float32)
_w5 = Tensor(np.ones([8, 16, 4, 4, 4]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8, 8]), dtype=ms.float32)


def test_conv3d_data_parallel():
    """
    Feature: test conv3d data parallel
    Description: shard n dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1, 1), (1, 1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Neg-0', ['Conv3D-0'])


def test_conv3d_pad_mode_overlap_is_negative():
    """
    Feature: test conv3d pad mode and overlap is negative
    Description: shard d/h
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 4, 4, 1), (1, 1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1, 1),)
    net = Net(_w5, out_channel=8, kernel_size=4, pad_mode="pad", stride=5, pad=(3, 0, 3, 0, 3, 0),
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x3, _b)


def test_conv3d_pad_mode_unet_3d_rank0():
    """
    Feature: test pad mode unet 3d
    Description: shard d/h
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 4, 1), (1, 1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="pad", stride=2, pad=1,
              strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x4, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 0, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 0, 0, 0]'})


def test_conv3d_pad_mode_unet_3d_rank1():
    """
    Feature: test pad mode unet 3d
    Description: shard d/h
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    strategy1 = ((1, 1, 2, 4, 1), (1, 1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="pad", stride=2, pad=1,
              strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x4, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 0, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 0, 1, 0]'})


def test_conv3d_pad_mode_unet_3d_rank7():
    """
    Feature: test pad mode unet 3d
    Description: shard d/h
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    strategy1 = ((1, 1, 2, 4, 1), (1, 1, 1, 1, 1))
    strategy2 = ((1, 1, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="pad", stride=2, pad=1,
              strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x4, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 0, 0, 0]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[1, 0, 1, 0]'})


def test_conv3d_valid_mode_output_shape_cannot_div_by_strategy():
    """
    Feature: test valid mode, and output shape can not div by strategy
    Description: shard d
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8, 1), (1, 1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="valid", stride=4,
              strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x3, _b)


def test_conv3d_pad_mode_unet_3d_auto_rank0():
    """
    Feature: test pad mode unet 3d
    Description: sharding propagation
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy2 = ((1, 1, 2, 4, 1),)
    net = Net(_w2, out_channel=8, kernel_size=3, pad_mode="pad", stride=2, pad=1, strategy2=strategy2)
    phase = compile_net(net, _x4, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 0, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 0, 0, 0]'})
