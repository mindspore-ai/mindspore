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
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import mutable
from mindspore.ops.operations._sequence_ops import SequenceConcat
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class NetSequenceConcat(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.op = SequenceConcat(axis=axis)

    def construct(self, seq):
        return self.op(seq)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_tensor_concat0():
    """
    Feature: test sequence concat op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((Tensor([[1, 2], [2, 3]]), Tensor([[2, 3], [3, 4]]), Tensor([[3, 4], [4, 5]])), True)
    expect = Tensor([[1, 2], [2, 3], [2, 3], [3, 4], [3, 4], [4, 5]])
    net = NetSequenceConcat()
    res = net(seq)
    assert np.all(res.asnumpy() == expect.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_tensor_concat1():
    """
    Feature: test sequence concat op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((Tensor([[1, 2], [2, 3]]), Tensor([[2, 3], [3, 4]]), Tensor([[3, 4], [4, 5]])), True)
    expect = Tensor([[1, 2, 2, 3, 3, 4], [2, 3, 3, 4, 4, 5]])
    net = NetSequenceConcat(axis=1)
    res = net(seq)
    assert np.all(res.asnumpy() == expect.asnumpy())
