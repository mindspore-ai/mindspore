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
"""test hccl alltoall and alltoallv grad with 2p"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops.communication import init_process_group, set_comm_ops_inplace
from mindspore.ops.communication import all_to_all_single
from mindspore.ops.communication import get_rank, get_world_size
from mindspore import value_and_grad
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init_process_group()

set_comm_ops_inplace(False)
this_rank = get_rank()
size = get_world_size()


if size != 2:
    raise RuntimeError("Group size should be 2 exactly.")


class AllToAllSingleNet(nn.Cell):
    def construct(self, output_shape, data, data1, output_split_sizes=None, input_split_sizes=None, group=None):
        out, _ = all_to_all_single(output_shape, data, output_split_sizes, input_split_sizes, group)
        out = out * data1
        return out


def test_alltoall_grad():
    """
    Feature: test 'all_to_all_single' communication function in cell.
    Description: test 'all_to_all_single' communication function in cell.
    Expectation: expect correct result.
    """
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    data1 = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output_shape = None
    expect_output = None
    expect_output1 = None
    if this_rank == 0:
        expect_output = [[0, 1, 4, 9], [0, 5, 12, 21]]
        expect_output1 = [[0, 1, 2, 3], [0, 1, 2, 3]]
    if this_rank == 1:
        expect_output = [[0, 5, 12, 21], [16, 25, 36, 49]]
        expect_output1 = [[4, 5, 6, 7], [4, 5, 6, 7]]
    net = AllToAllSingleNet()
    grad_fn = value_and_grad(net, grad_position=1)
    out, inputs_gradient = grad_fn(output_shape, data, data1)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_alltoallv_grad():
    """
    Feature: test 'all_to_all_single' communication function in cell.
    Description: test 'all_to_all_single' communication function in cell.
    Expectation: expect correct result.
    """
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    data1 = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output_shape = None
    expect_output = None
    expect_output1 = None
    if this_rank == 0:
        expect_output = [[0, 1, 4, 9], [0, 5, 12, 21]]
        expect_output1 = [[0, 1, 2, 3], [0, 1, 2, 3]]
    if this_rank == 1:
        expect_output = [[0, 5, 12, 21], [16, 25, 36, 49]]
        expect_output1 = [[4, 5, 6, 7], [4, 5, 6, 7]]
    net = AllToAllSingleNet()
    grad_fn = value_and_grad(net, grad_position=1)
    out, inputs_gradient = grad_fn(output_shape, data, data1, [1, 1], [1, 1])
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)
