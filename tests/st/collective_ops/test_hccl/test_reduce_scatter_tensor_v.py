# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test hccl ReduceScatter and reduce_scatter_tensor with 8p"""

import os
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations import ReduceScatterV
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context
from mindspore.ops.operations.comm_ops import ReduceOp

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
init()
this_rank = get_rank()
world_size = get_group_size()


class ReduceScatterVNet(nn.Cell):
    def __init__(self, op=ReduceOp.SUM):
        super(ReduceScatterVNet, self).__init__()
        self.reduce_scatterv = ReduceScatterV(op)

    def construct(self, x, input_split_sizes):
        return self.reduce_scatterv(x, input_split_sizes)


def test_hccl_reduce_scatter_tensor_v_list():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    data = [i for i in range(world_size)]
    input_split_sizes = [1 for _ in range(world_size)]
    net = ReduceScatterVNet()
    output = net(Tensor(data, dtype=ms.int32), input_split_sizes)
    expect_output = [this_rank * 2]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_v_op():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    data = [i for i in range(world_size)]
    input_split_sizes = [1 for _ in range(world_size)]
    net = ReduceScatterVNet(ReduceOp.MAX)
    output = net(Tensor(data, dtype=ms.int32), input_split_sizes)
    expect_output = [this_rank * 1]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)
    net = ReduceScatterVNet(ReduceOp.MIN)
    output = net(Tensor(data, dtype=ms.int32), input_split_sizes)
    expect_output = [this_rank * 1]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_v_tensor1():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    data = [i for i in range(world_size)]
    input_split_sizes = [1 for _ in range(world_size)]
    net = ReduceScatterVNet()
    output = net(Tensor(data, dtype=ms.int32), Tensor(input_split_sizes))
    expect_output = [this_rank * 2]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_v_tensor2():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        if this_rank == 0:
            expect_output = [2, 4, 6]
        if this_rank == 1:
            expect_output = [8, 10, 12, 14, 16]
        input_split_sizes = [3, 5]
        net = ReduceScatterVNet()
        output = net(Tensor(data, dtype=ms.int32), Tensor(input_split_sizes))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_v_tensor3():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        if this_rank == 0:
            expect_output = [2, 4, 6, 8, 10, 12, 14, 16]
        if this_rank == 1:
            expect_output = [0]
        input_split_sizes = [8, 0]
        net = ReduceScatterVNet()
        output = net(Tensor(data, dtype=ms.int32), Tensor(input_split_sizes))
        if isinstance(output, tuple):
            if output[0].shape != ():
                assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            if output.shape != ():
                assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_reduce_scatter_tensor_v_tensor4():
    """
    Feature: test 'ReduceScatterV' communication operator.
    Description: test 'ReduceScatterV' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [1]
        if this_rank == 1:
            expect_output = [2]
        if this_rank == 0:
            expect_output = [0]
        input_split_sizes = [0, 1]
        net = ReduceScatterVNet()
        output = net(Tensor(data, dtype=ms.int32), Tensor(input_split_sizes))
        if isinstance(output, tuple):
            if this_rank == 0:
                assert np.allclose(output[0].shape, ())
            if this_rank == 1:
                assert np.allclose(output[0].shape, (1,))
            if output[0].shape != ():
                assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            if this_rank == 0:
                assert np.allclose(output.shape, ())
            if this_rank == 1:
                assert np.allclose(output.shape, (1,))
            if output.shape != ():
                assert np.allclose(output.asnumpy(), expect_output)
