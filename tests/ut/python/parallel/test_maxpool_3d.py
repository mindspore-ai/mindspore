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


class MaxPool3DNet(Cell):
    def __init__(self, add_weight, kernel_size=1, strides=1, pad_mode="valid", pad_list=0, ceil_mode=None,
                 data_format="NCDHW", strategy0=None, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy0)
        self.max_pool = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode, pad_list=pad_list,
                                    ceil_mode=ceil_mode, data_format=data_format).shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)
        self.add_weight = Parameter(add_weight, "w1")

    def construct(self, x, b):
        out = self.add(x, self.add_weight)
        out = self.max_pool(out)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
_w = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)


def test_maxpool3d_data_parallel():
    """
    Feature: test maxpool3d data parallel
    Description: shard n dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((8, 1, 1, 1, 1), (8, 1, 1, 1, 1))
    strategy1 = ((8, 1, 1, 1, 1),)
    strategy2 = ((8, 1, 1, 1, 1),)
    net = MaxPool3DNet(_w, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('MaxPool3D-0', ['Add-0'])


def test_maxpool3d_kernel_size_equal_to_strides():
    """
    Feature: test maxpool3d kernel size = strides
    Description: shard d/h dimension, no need to exchange overlap
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy0 = ((1, 1, 2, 4, 1), (1, 1, 2, 4, 1))
    strategy1 = ((1, 1, 2, 4, 1),)
    strategy2 = ((1, 1, 2, 4, 1),)
    net = MaxPool3DNet(_w, strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('MaxPool3D-0', ['Add-0'])


def test_maxpool3d_pad_mode_overlap_is_negative():
    """
    Feature: test pad mode and overlap is negative
    Description: shard d/h
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 4, 4, 1),)
    strategy2 = ((1, 1, 1, 1, 1),)
    net = MaxPool3DNet(_w, kernel_size=4, strides=5, pad_mode="pad", pad_list=(3, 0, 3, 0, 3, 0),
                       strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


def test_maxpool3d_shard_w_dimension():
    """
    Feature: test shard w dimension
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 4, 2),)
    strategy2 = ((1, 1, 1, 4, 2),)
    net = MaxPool3DNet(_w, kernel_size=3, pad_mode="pad", strides=1, pad_list=1,
                       strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


def test_maxpool3d_pad_mode_rank0():
    """
    Feature: test pad mode, rank0
    Description: shard d/h, need exchange the overlap
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 4, 1),)
    strategy2 = ((1, 1, 2, 4, 1),)
    net = MaxPool3DNet(_w, kernel_size=3, pad_mode="pad", strides=1, pad_list=1,
                       strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 0, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 1, 0, 1]'})


def test_maxpool3d_pad_mode_rank1():
    """
    Feature: test pad mode, rank1
    Description: shard d/h, need exchange the overlap
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    strategy1 = ((1, 1, 2, 4, 1),)
    strategy2 = ((1, 1, 2, 4, 1),)
    net = MaxPool3DNet(_w, kernel_size=3, pad_mode="pad", strides=1, pad_list=1,
                       strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 1, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 1, 1, 1]'})


def test_maxpool3d_pad_mode_rank7():
    """
    Feature: test pad mode, rank7
    Description: shard d/h, need exchange the overlap
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    strategy1 = ((1, 1, 2, 4, 1),)
    strategy2 = ((1, 1, 2, 4, 1),)
    net = MaxPool3DNet(_w, kernel_size=3, pad_mode="pad", strides=1, pad_list=1,
                       strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[1, 0, 1, 0]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[1, 0, 1, 0]'})


def test_maxpool3d_same_mode_rank0():
    """
    Feature: test same mode, rank0
    Description: shard d/h, need exchange the overlap
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 2, 4, 1),)
    strategy2 = ((1, 1, 2, 4, 1),)
    net = MaxPool3DNet(_w, kernel_size=3, pad_mode="same", strides=1, pad_list=0,
                       strategy1=strategy1, strategy2=strategy2)
    phase = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'send_lens': '[0, 1, 0, 1]'})
    assert validator.check_node_attrs('NeighborExchangeV2-0', {'recv_lens': '[0, 1, 0, 1]'})


def test_maxpool3d_valid_mode_output_shape_cannot_div_by_strategy():
    """
    Feature: test valid mode, and output shape can not div by strategy
    Description: shard d
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8, 1),)
    strategy2 = ((1, 1, 1, 1, 1),)
    net = MaxPool3DNet(_w, kernel_size=2, pad_mode="valid", strides=4,
                       strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)
