# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.common import mutable
from mindspore.ops.operations._sequence_ops import SequenceStack
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class NetSequenceStack(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.op = SequenceStack(axis=axis)

    def construct(self, seq):
        return self.op(seq)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_stack0():
    """
    Feature: test sequence stack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = np.float32
    data_np = np.array([0] * 16).astype(dtype)
    data_np = np.reshape(data_np, (2, 2, 2, 2))
    x1 = Tensor(data_np)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(dtype))
    x = mutable((x1, x2), True)
    y = ops.stack((x1, x2), 0)
    net = NetSequenceStack(axis=0)
    res = net(x)
    assert np.all(res.asnumpy() == y.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_stack1():
    """
    Feature: test sequence stack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = np.float32
    data_np = np.array([0] * 16).astype(dtype)
    data_np = np.reshape(data_np, (2, 2, 2, 2))
    x1 = Tensor(data_np)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(dtype))
    x = mutable((x1, x2), True)
    y = ops.stack((x1, x2), 1)
    net = NetSequenceStack(axis=1)
    res = net(x)
    assert np.all(res.asnumpy() == y.asnumpy())
