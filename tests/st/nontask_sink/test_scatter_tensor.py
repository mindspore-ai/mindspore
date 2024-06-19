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

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspore.communication.comm_func import scatter_tensor
from mindspore.communication.management import get_rank, create_group


# 'scatter_tensor' function only supports KernelByKernel mode by now. So we set 'jit_level' to 'O0'.
np.random.seed(1)
ms.set_context(jit_level='O0')
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()

class ScatterTensorFuncNet(nn.Cell):
    def construct(self, tensor, src):
        out = scatter_tensor(tensor, src)
        return out

def test_hccl_scatter_tensor_func_in_cell_2p():
    """
    Feature: test 'scatter_tensor' communication function in cell.
    Description: test 'scatter_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(64).reshape([8, 8]).astype(np.float32))
    net = ScatterTensorFuncNet()
    out = net(data, 0)
    if rank == 0:
        gt_rank0 = np.arange(0, 32).reshape([4, 8]).astype(np.float32)
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(32, 64).reshape([4, 8]).astype(np.float32)
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst

def test_hccl_scatter_tensor_func_2p():
    """
    Feature: test 'scatter_tensor' communication function.
    Description: test 'scatter_tensor' communication function.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(64).reshape([8, 8]).astype(np.float32))
    out = scatter_tensor(data, 0)
    if rank == 0:
        gt_rank0 = np.arange(0, 32).reshape([4, 8]).astype(np.float32)
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(32, 64).reshape([4, 8]).astype(np.float32)
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst

def test_scatter_tensor_two_groups():
    """
    Feature: test 'scatter_tensor' communication function in two groups.
    Description: test 'scatter_tensor' communication function in two groups.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.full((64), rank).reshape([4, 16]).astype(np.float32))
    if rank in [0, 2]:
        create_group("group1", [0, 2])
        out = scatter_tensor(data, 0, group="group1")
        exp = np.full((16), 0).reshape([1, 16]).astype(np.float32)
        assert np.allclose(exp, out.asnumpy())
    else:
        create_group("group2", [1, 3])
        out = scatter_tensor(data, 1, group="group2")
        exp = np.full((16), 1).reshape([1, 16]).astype(np.float32)
        assert np.allclose(exp, out.asnumpy())
