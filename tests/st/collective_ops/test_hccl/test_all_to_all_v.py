# Copyright 2024 Huawei Technologies Co., Ltd
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

"""test hccl AlltoAllV with 8p"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations.comm_ops import AlltoAllV
from mindspore.communication.comm_func import all_to_all_with_output_shape, all_to_all_single_with_output_shape
from mindspore.communication.management import init, get_rank, get_group_size

# 'AlltoAllV' operator only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
init()
this_rank = get_rank()
world_size = get_group_size()

class AlltoAllVNet(nn.Cell):
    def __init__(self, group=None):
        super(AlltoAllVNet, self).__init__()
        self.all_to_all = AlltoAllV(group=group)
    def construct(self, x, send_numel_list, recv_numel_list):
        return self.all_to_all(x, send_numel_list, recv_numel_list)

class AllToAllFunNet(nn.Cell):
    def construct(self, output_tensor_list, input_tensor_list, group=None):
        return all_to_all_with_output_shape(output_tensor_list, input_tensor_list, group)

class AllToAllTensorFunNet(nn.Cell):
    def construct(self, output, input_x, output_split_sizes=None, input_split_sizes=None, group=None):
        return all_to_all_single_with_output_shape(output, input_x, output_split_sizes, input_split_sizes, group)

def test_hccl_alltoallv2_2p():
    """
    Feature: test 'AlltoAllV' communication operator.
    Description: test 'AlltoAllV' communication operator.
    Expectation: expect correct result.
    """
    data = [i + this_rank for i in range(world_size)]
    send_numel_list = [1 for _ in range(world_size)]
    recv_numel_list = [1 for _ in range(world_size)]
    net = AlltoAllVNet()
    output = net(ms.Tensor(data, dtype=ms.float32), send_numel_list, recv_numel_list)
    expect_output = np.array(data, dtype=np.float32)
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)

def test_hccl_alltoall_func_2p():
    """
    Feature: test 'AlltoAllV' communication operator.
    Description: test 'AlltoAllV' communication operator.
    Expectation: expect correct result.
    """
    data = [i + this_rank for i in range(world_size)]
    input_tensor_list = [ms.Tensor([data[i]], dtype=ms.float32) for i in range(world_size)]
    output_tensor_list = [(1,)] * world_size
    net = AllToAllFunNet()
    output = net(output_tensor_list, input_tensor_list)[0]
    for i, out in enumerate(output):
        expect_output = np.array([data[i]], dtype=np.float32)
        if isinstance(out, tuple):
            assert np.allclose(out[0].asnumpy(), expect_output)
        else:
            assert np.allclose(out.asnumpy(), expect_output)

def test_hccl_alltoall_tensor_2p():
    """
    Feature: test 'AlltoAllV' communication operator.
    Description: test 'AlltoAllV' communication operator.
    Expectation: expect correct result.
    """
    data = [i + this_rank for i in range(world_size)]
    input_x = ms.Tensor(data, dtype=ms.float32)
    output = (world_size,)
    net = AllToAllTensorFunNet()
    output = net(output, input_x)
    expect_output = np.array(data, dtype=np.float32)
    if isinstance(output, tuple):
        assert np.allclose(output[0].asnumpy(), expect_output)
    else:
        assert np.allclose(output.asnumpy(), expect_output)
