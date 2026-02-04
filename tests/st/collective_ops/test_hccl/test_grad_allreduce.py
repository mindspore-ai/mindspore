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
"""test hccl all_reduce grad with 2p"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops.communication import init_process_group, set_comm_ops_inplace
from mindspore.ops.communication import all_reduce
from mindspore.ops.communication import get_rank, get_world_size
from mindspore import value_and_grad
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init_process_group()
set_comm_ops_inplace(False)
this_rank = get_rank()
size = get_world_size()


if size != 2:
    raise RuntimeError("Group size should be 2 exactly.")


class AllReduceFuncNet(nn.Cell):
    def construct(self, x):
        out, _ = all_reduce(x)
        return out


def test_hccl_all_reduce_grad():
    """
    Feature: test 'all_reduce' communication function in cell.
    Description: test 'all_reduce' communication function in cell.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.ones([3, 4]).astype(np.float32))
    expect_output = np.ones([3, 4]).astype(np.float32) * 2
    expect_output1 = np.ones([3, 4]).astype(np.float32) * 2
    net = AllReduceFuncNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)
