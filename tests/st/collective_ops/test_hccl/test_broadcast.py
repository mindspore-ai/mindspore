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
from mindspore.communication.comm_func import broadcast
from mindspore.communication.management import get_rank

# 'broadcast' function only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()


class BoardcastNet(nn.Cell):
    def construct(self, tensor, dst):
        out = broadcast(tensor, dst)
        return out


def test_hccl_broadcast_func_in_cell_2p():
    """
    Feature: test 'broadcast' communication function in cell.
    Description: test 'broadcast' communication function in cell.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    net = BoardcastNet()
    out = net(data, 0)
    print(out)

    if rank == 0:
        gt_rank0 = np.arange(8).reshape([2, 4]).astype(np.float32)
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(8).reshape([2, 4]).astype(np.float32)
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst


def test_hccl_broadcast_func_2p():
    """
    Feature: test 'broadcast' communication function.
    Description: test 'broadcast' communication function.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    out = broadcast(data, 0)
    print(out)

    if rank == 0:
        gt_rank0 = np.arange(8).reshape([2, 4]).astype(np.float32)
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.arange(8).reshape([2, 4]).astype(np.float32)
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst


def test_hccl_broadcast_func_2p_bool():
    """
    Feature: test 'broadcast' communication function.
    Description: test 'broadcast' communication function.
    Expectation: expect correct result.
    """
    rank = get_rank()

    if rank == 0:
        data = ms.Tensor([1, 1, 1], dtype=ms.bool_)
    else:
        data = ms.Tensor([0, 0, 0], dtype=ms.bool_)

    out = broadcast(data, src=0)
    expect_output = ms.Tensor([1, 1, 1], dtype=ms.bool_)
    assert np.allclose(expect_output.asnumpy(), out.asnumpy())


test_hccl_broadcast_func_2p()
