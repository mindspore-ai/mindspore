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

"""test hccl AlltoAllVC with 8p"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations import AlltoAllVC
from mindspore.communication.management import init, get_rank, get_group_size

# 'AlltoAllV' operator only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
init()
this_rank = get_rank()
world_size = get_group_size()

class AlltoAllVCNet(nn.Cell):
    def __init__(self, group=None):
        super(AlltoAllVCNet, self).__init__()
        self.all_to_all = AlltoAllVC()
    def construct(self, x, send_count_matrix):
        return self.all_to_all(x, send_count_matrix)


def test_hccl_alltoallvc_tensor1():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    data = [this_rank for i in range(world_size)]
    send_count_matrix = [1 for _ in range(world_size * world_size)]
    net = AlltoAllVCNet()
    output = net(ms.Tensor(data, dtype=ms.float32), ms.Tensor(send_count_matrix))
    expect_output = [i for i in range(world_size)]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_tensor2():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    data = [this_rank for i in range(world_size * 3)]
    send_count_matrix = [3 for _ in range(world_size * world_size)]
    net = AlltoAllVCNet()
    output = net(ms.Tensor(data, dtype=ms.float32), ms.Tensor(send_count_matrix))
    expect_output = [i//3 for i in range(world_size * 3)]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_tensor3():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [1, 2, 3, 4, 5, 6]
        send_count_matrix = [2, 4, 4, 2]
        net = AlltoAllVCNet()
        output = net(ms.Tensor(data, dtype=ms.int64), ms.Tensor(send_count_matrix))
        expect_output = []
        if this_rank == 0:
            expect_output = [1, 2, 1, 2, 3, 4]
        if this_rank == 1:
            expect_output = [3, 4, 5, 6, 5, 6]
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_tensor4():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [this_rank * i for i in range(world_size * 3)]
        expect_output = []
        if this_rank == 0:
            expect_output = [0, 1, 2, 3, 4, 5]
        if this_rank == 1:
            expect_output = [0, 0, 0, 0, 0, 0]
        send_count_matrix = [0, 6, 6, 0]
        net = AlltoAllVCNet()
        output = net(ms.Tensor(data, dtype=ms.int64), ms.Tensor(send_count_matrix))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_tensor5():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [this_rank * i for i in range(world_size * 3)]
        expect_output = []
        if this_rank == 0:
            expect_output = [0, 0, 0, 0, 0, 0]
        if this_rank == 1:
            expect_output = [0, 1, 2, 3, 4, 5]
        send_count_matrix = [6, 0, 0, 6]
        net = AlltoAllVCNet()
        output = net(ms.Tensor(data, dtype=ms.int64), ms.Tensor(send_count_matrix))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_tensor6():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    if world_size == 2:
        data = [this_rank * i for i in range(world_size * 3)]
        expect_output = []
        if this_rank == 1:
            expect_output = [0]
        if this_rank == 0:
            expect_output = [0]
        send_count_matrix = [0, 0, 0, 0]
        net = AlltoAllVCNet()
        output = net(ms.Tensor(data, dtype=ms.int64), ms.Tensor(send_count_matrix))
        if isinstance(output, tuple):
            assert np.allclose(output[0].asnumpy(), expect_output)
        else:
            assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_alltoallvc_list1():
    """
    Feature: test 'AlltoAllVC' communication operator.
    Description: test 'AlltoAllVC' communication operator.
    Expectation: expect correct result.
    """
    data = [this_rank for i in range(world_size)]
    send_count_matrix = [1 for _ in range(world_size * world_size)]
    net = AlltoAllVCNet()
    output = net(ms.Tensor(data, dtype=ms.int64), send_count_matrix)
    expect_output = [i for i in range(world_size)]
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)
