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
"""testcase for Gather communication in symmetric memory scenario"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import jit
from mindspore.communication import init, get_rank, get_group_size
from mindspore.ops.auto_generate import SignalWaitUntil, PutMemSignal, CreateSymmetricMemory

np.random.seed(0)
ms.set_device("Ascend")
ms.context.set_context(op_timeout=30)
init()
rank = get_rank()
size = get_group_size()

class GatherNet(nn.Cell):
    @jit(backend="ms_backend", jit_level="O0")
    def construct(self, tensor, shape, dtype):
        share= CreateSymmetricMemory()(shape, dtype, group='world_size')
        signal= CreateSymmetricMemory()((1,), ms.int32, group='world_size')
        out, signal = PutMemSignal()(share, ms.Tensor([size * rank]),
                                     tensor,ms.Tensor([0]), ms.Tensor([size]),
                                     signal,ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32), 'add', 0, True)
        if rank == 0:
            out=SignalWaitUntil()(out, signal, ms.Tensor([0]), ms.Tensor([size], dtype=ms.int32), 'eq')
        return out, signal

def test_oneside_gather():
    """
    Description: Verify Gather communication in symmetric memory scenario.
    Expectation:
        1. The output tensor in rank-0 is the same as the expected tensor.
    """
    shape= (size,size)
    dtype= ms.float32
    data= np.ones([size]).astype(np.float32) * rank
    tensor= ms.Tensor(data)
    gather_net= GatherNet()
    output, signal= gather_net(tensor, shape, dtype)
    if rank ==0:
        expected_output= np.array([[i for _ in range(size)]for i in range(size)]).astype(np.float32)
        assert np.allclose(output.asnumpy(), expected_output)
        assert signal.asnumpy() == size
