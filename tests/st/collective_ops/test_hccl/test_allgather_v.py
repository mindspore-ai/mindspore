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

"""test hccl AllGather and all_gather with 8p"""

import os
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations import AllGatherV
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
this_rank = get_rank()
world_size = get_group_size()

class AllGatherVNet(nn.Cell):
    def __init__(self):
        super(AllGatherVNet, self).__init__()
        self.all_gatherv = AllGatherV()

    def construct(self, x, output_split_sizes):
        return self.all_gatherv(x, output_split_sizes)


def test_hccl_all_gather_v_list():
    """
    Feature: test 'AllGatherV' communication operation.
    Description: test 'AllGatherV' communication operation.
    Expectation: expect correct result.
    """
    data = [this_rank]
    output_split_sizes = [1 for _ in range(world_size)]
    net = AllGatherVNet()
    output = net(Tensor(data, dtype=ms.int64), output_split_sizes)
    expect_output = [i for i in range(world_size)]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_v_tensor1():
    """
    Feature: test 'AllGatherV' communication operation.
    Description: test 'AllGatherV' communication operation.
    Expectation: expect correct result.
    """
    data = [this_rank]
    output_split_sizes = [1 for _ in range(world_size)]
    net = AllGatherVNet()
    output = net(Tensor(data, dtype=ms.int64), Tensor(output_split_sizes))
    expect_output = [i for i in range(world_size)]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_v_tensor2():
    """
    Feature: test 'AllGatherV' communication operation.
    Description: test 'AllGatherV' communication operation.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = []
        expect_output = [1, 2, 3, 4, 5, 6, 7, 8]
        if this_rank == 0:
            data = [1, 2, 3]
        if this_rank == 1:
            data = [4, 5, 6, 7, 8]
        output_split_sizes = [3, 5]
        net = AllGatherVNet()
        output = net(Tensor(data, dtype=ms.int64), Tensor(output_split_sizes))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_v_tensor3():
    """
    Feature: test 'AllGatherV' communication operation.
    Description: test 'AllGatherV' communication operation.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = []
        expect_output = [1, 2, 3, 4, 5, 6, 7, 8]
        if this_rank == 0:
            data = [1, 2, 3, 4, 5, 6, 7, 8]
        if this_rank == 1:
            data = []
        output_split_sizes = [8, 0]
        net = AllGatherVNet()
        output = net(Tensor(data, dtype=ms.int64), Tensor(output_split_sizes))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)
