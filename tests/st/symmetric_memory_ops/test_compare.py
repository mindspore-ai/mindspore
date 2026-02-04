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
"""testcase for different compare operations in symmetric memory scenario"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import jit
from mindspore.communication import init, get_rank, get_group_size
from mindspore.ops.auto_generate import SignalWaitUntil, SignalOp, CreateSymmetricMemory

ms.set_device("Ascend")
ms.context.set_context(op_timeout=30)
init()
rank = get_rank()
size = get_group_size()
if size != 2:
    raise RuntimeError("Symmetric memory compare test only support 2 cards.")

class CompareNet(nn.Cell):
    @jit(backend="ms_backend", jit_level="O0")
    def construct(self):
        signal = CreateSymmetricMemory()((6,), ms.int32, group='world_size')
        if rank == 0:
            signal=SignalOp()(signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32), 'set', 1)
            signal=SignalOp()(signal, ms.Tensor([1]), ms.Tensor([1], dtype=ms.int32), 'add', 1)
            signal=SignalOp()(signal, ms.Tensor([2]), ms.Tensor([-1], dtype=ms.int32), 'set', 1)
            signal=SignalOp()(signal, ms.Tensor([3]), ms.Tensor([-1], dtype=ms.int32), 'add', 1)
            signal=SignalOp()(signal, ms.Tensor([4]), ms.Tensor([2], dtype=ms.int32), 'set', 1)
            signal=SignalOp()(signal, ms.Tensor([5]), ms.Tensor([2], dtype=ms.int32), 'add', 1)
        else:
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([0]), ms.Tensor([1], dtype=ms.int32), 'eq')
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([1]), ms.Tensor([1], dtype=ms.int32), 'eq')
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([2]), ms.Tensor([0], dtype=ms.int32), 'lt')
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([3]), ms.Tensor([0], dtype=ms.int32), 'lt')
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([4]), ms.Tensor([1], dtype=ms.int32), 'gt')
            signal=SignalWaitUntil()(signal, signal, ms.Tensor([5]), ms.Tensor([1], dtype=ms.int32), 'gt')
        return signal

def test_compare():
    """
    Description: Verify the set/add op of SignalOp and comparison operations (eq/lt/gt) of SignalWaitUntil.
    Expectation:
        1. SignalWaitUntil can wait for the signal to meet the expectation.
        2. The result signal meets the expectation result after set/add SignalOp.
    """
    compare_net = CompareNet()
    signal = compare_net()
    if rank == 1:
        expected_signal = np.array([1, 1, -1, -1, 2, 2]).astype(np.int32)
        diff= abs(signal.asnumpy() - expected_signal)
        assert np.all(diff==0)
    return signal
