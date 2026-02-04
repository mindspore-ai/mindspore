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
"""testcase for Push mode in symmetric memory scenario"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import jit
from mindspore.communication import init, get_rank, get_group_size
from mindspore.ops.auto_generate import SignalWaitUntil, PutMemSignal, CreateSymmetricMemory

ms.set_device("Ascend")
ms.context.set_context(op_timeout=30)
init()
rank = get_rank()
size = get_group_size()
if size != 2:
    raise RuntimeError("Symmetric memory push test only support 2 cards.")

class PushNet(nn.Cell):
    @jit(backend="ms_backend", jit_level="O0")
    def construct(self, tensor):
        share = CreateSymmetricMemory()((3,3,3), ms.float32, group='world_size')
        signal = CreateSymmetricMemory()((1,), ms.int32, group='world_size')
        if rank == 0:
            share, signal = PutMemSignal()(share, ms.Tensor([0]),
                                           tensor, ms.Tensor([0]), ms.Tensor([27]),
                                           signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32),
                                           'set', 1, True)
        else:
            share = SignalWaitUntil()(share, signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32),'eq')
        return share, signal


def test_oneside_push_2p():
    """
    Description: Verify Push communication in symmetric memory scenario.
    Expectation:
        1. The output tensor in rank-1 is the same as the expected tensor.
    """
    data = np.ones((3, 3, 3)).astype(np.float32) * 0.01
    tensor = ms.Tensor(data)
    push_net = PushNet()
    output, _ = push_net(tensor)
    if rank == 1:
        expected_output = np.ones((3,3,3)).astype(np.float32)*0.01
        diff = abs(output.asnumpy() - expected_output)
        error = np.ones([3, 3, 3]) * 1e-5
        assert np.all(diff < error)
