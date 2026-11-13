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

"""Test point-to-point communication and asynchronous completion."""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops import communication as p2p
from mindspore.communication import init
from mindspore.ops.communication import isend, irecv
from mindspore.communication.management import get_rank, get_group_size

# 'isend' and 'irecv' function only supports KernelByKernel mode by now.
np.random.seed(1)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()
rank = get_rank()
size = get_group_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")
x = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)
x2 = np.ones([3, 3, 3, 3]).astype(np.float32)

class SendNet(nn.Cell):
    def construct(self, tensor):
        handle = isend(tensor, rank + size // 2)
        handle.wait()
        return tensor

class RecvNet(nn.Cell):
    def construct(self, tensor):
        handle = irecv(tensor, rank - size // 2)
        handle.wait()
        return tensor

def test_hccl_send_recv_2p():
    """
    Feature: test 'isend' and 'irecv' communication function in cell.
    Description: test 'isend' and 'irecv' communication function in cell.
    Expectation: expect correct result.
    """
    if rank < size / 2:
        _x = ms.Tensor(x)
        send_net = SendNet()
        output = send_net(_x)
    else:
        expect_output = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * (rank-size//2 + 1)
        _x2 = ms.Tensor(x2)
        recv_net = RecvNet()
        output = recv_net(_x2)

        diff = abs(output.asnumpy() - expect_output)
        error = np.ones(shape=output.shape) * 1.0e-5
        assert np.all(diff < error)
        assert expect_output.shape == output.shape

test_hccl_send_recv_2p()


def test_sync_send_payload():
    """
    Feature: Synchronous point-to-point send migration.
    Description: Send a transposed payload through the retained API and receive it on the peer.
    Expectation: The peer receives the original logical layout and values.
    """
    peer = rank + size // 2 if rank < size // 2 else rank - size // 2
    expected = np.arange(6, dtype=np.float32).reshape(2, 3).T
    if rank < size // 2:
        view = ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3)).transpose(1, 0)
        p2p.send(view.contiguous(), dst=peer)
    else:
        buf = ms.Tensor(np.zeros_like(expected))
        result = p2p.recv(buf, src=peer)
        actual = result if isinstance(result, ms.Tensor) else buf
        np.testing.assert_array_equal(actual.asnumpy(), expected)
