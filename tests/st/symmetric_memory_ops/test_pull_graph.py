# Copyright 2021 Huawei Technologies Co., Ltd
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
"""testcase for Pull mode in symmetric memory scenario"""

import numpy as np
import mindspore as ms
from mindspore.graph.api import jit
from mindspore import nn, ops
from mindspore.communication import init, get_rank, get_group_size
from mindspore.ops.auto_generate import SignalWaitUntil, PutMem, CreateSymmetricMemory, GetMem, SignalOp

ms.set_device("Ascend")
ms.context.set_context(op_timeout=30)
init()
rank = get_rank()
size = get_group_size()
if size != 2:
    raise RuntimeError("Symmetric memory pull test only support 2 cards.")

class PullNet(nn.Cell):
    @jit(backend="ms_backend", jit_level="O0")
    def construct(self, tensor):
        share = CreateSymmetricMemory()((3, 3, 3), ms.float32, group='world_size')
        signal = CreateSymmetricMemory()((1,), ms.int32, group='world_size')
        if rank == 0:
            share = PutMem()(share, ms.Tensor([0]), tensor, ms.Tensor([0]), ms.Tensor([27]), 0, True)
            signal = ops.Depend()(signal, share)
            signal = SignalOp()(signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32), 'set', 1)
        else:
            share = SignalWaitUntil()(share, signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32), 'eq')
            share = GetMem()(share, ms.Tensor([0]), share, ms.Tensor([0]), ms.Tensor([27]), 0, True)
        return share, signal

def test_pull():
    """
    Description: Verify the pull mode of symmetric memory.
    Expectation:
        1. The output tensor is the same as the expected tensor.
    """
    expected_output = np.ones((3, 3, 3)).astype(np.float32) * 0.01
    data = np.ones((3, 3, 3)).astype(np.float32) * 0.01
    tensor = ms.Tensor(data)
    pull_net = PullNet()
    output, _ = pull_net(tensor)
    diff= abs(output.asnumpy() - expected_output)
    assert np.all(diff < 1e-5)
    assert output.dtype == ms.float32
