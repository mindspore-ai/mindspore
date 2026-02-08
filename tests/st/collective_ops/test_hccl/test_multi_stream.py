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
"""test multi stream"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspore.communication.comm_func import isend, irecv, all_reduce
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.ops import ReduceOp

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()
rank = get_rank()
size = get_group_size()
if rank in [4, 5, 6, 7]:
    create_group("group1", [4, 5, 6, 7])


class SendNet(nn.Cell):
    def construct(self, tensor):
        out = isend(tensor, rank + size // 2)
        return out


class RecvNet(nn.Cell):
    def construct(self, tensor):
        out, handle = irecv(tensor, rank - size // 2)
        handle.wait()
        out, _ = all_reduce(out, ReduceOp.SUM, "group1")
        return out


def test_hccl_multi_stream():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """

    x = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * rank
    x2 = np.ones([3, 3, 3, 3]).astype(np.float32)
    if rank < size / 2:
        _x = ms.Tensor(x)
        send_net = SendNet()
        output = send_net(_x)
    else:
        expect = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * 6
        _x2 = ms.Tensor(x2)
        recv_net = RecvNet()
        output = recv_net(_x2)
        assert np.allclose(output.asnumpy(), expect)


def test_comm_multi_stream():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
    s1 = ms.runtime.Stream()
    with ms.runtime.StreamCtx(s1):
        output, handle = all_reduce(x, op=ReduceOp.SUM, async_op=True)
        # wait allreduce at s1 not the default stream
        handle.wait()
    output = output + 1
    print(output)

def test_comm_multi_stream_raise_error():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """
    with pytest.raises(RuntimeError):
        x = ms.Tensor(np.ones([3, 3, 3, 3]).astype(np.float32))
        s1 = ms.runtime.Stream()
        with ms.runtime.StreamCtx(s1):
            output, handle = all_reduce(x, op=ReduceOp.SUM, async_op=True)
            # don't call handle.wait()
            del handle
        ms.runtime.synchronize()
        output = output + 1
        print(output)
