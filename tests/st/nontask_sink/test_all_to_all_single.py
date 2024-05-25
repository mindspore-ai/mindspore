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

import os
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspore.communication.comm_func import all_to_all_single
from mindspore.communication.management import get_rank

# 'all_to_all_single' function only supports KernelByKernel mode by now. So we set 'GRAPH_OP_RUN' to 1.
np.random.seed(1)
os.environ['GRAPH_OP_RUN'] = str(1)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()

class AllToAllSingleNet(nn.Cell):
    def construct(self, tensor):
        out = all_to_all_single(tensor)
        return out

def test_hccl_all_to_all_single_func_in_cell_2p():
    """
    Feature: test 'all_to_all_single' communication function in cell.
    Description: test 'all_to_all_single' communication function in cell.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    net = AllToAllSingleNet()
    out = net(data)

    if rank == 0:
        gt_rank0 = np.arange(0, 4).reshape([1, 4]).astype(np.float32)
        gt_rank0 = np.vstack([gt_rank0, gt_rank0])
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(4, 8).reshape([1, 4]).astype(np.float32)
        gt_rank1 = np.vstack([gt_rank1, gt_rank1])
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst

def test_hccl_all_to_all_single_func_2p():
    """
    Feature: test 'all_to_all_single' communication function.
    Description: test 'all_to_all_single' communication function.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    out = all_to_all_single(data)

    if rank == 0:
        gt_rank0 = np.arange(0, 4).reshape([1, 4]).astype(np.float32)
        gt_rank0 = np.vstack([gt_rank0, gt_rank0])
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(4, 8).reshape([1, 4]).astype(np.float32)
        gt_rank1 = np.vstack([gt_rank1, gt_rank1])
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst


test_hccl_all_to_all_single_func_2p()
