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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops.operations import _sequence_ops as S
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_slice():
    """
    Feature: test sequence_slice op
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.seq_slice = S.SequenceSlice()

        def construct(self, seq, start, stop, step):
            return self.seq_slice(seq, start, stop, step)

    def func(seq, start, stop, step):
        return seq[start:stop:step]

    seq = (1, 2, 3, 4, 5, 6)
    start = 1
    stop = 3
    step = 1
    net_ms = Net()
    fact = TupleFactory(net_ms, func, (seq, start, stop, step))
    fact.forward_cmp()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_slice_grad():
    """
    Feature: test sequence_slice grad
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.seq_slice = S.SequenceSlice()

        def construct(self, seq, start, stop, step):
            return self.seq_slice(seq, start, stop, step)

    seq = mutable((1, 2, 3, 4, 5, 6), True)
    start = 1
    stop = 3
    step = 1
    dout = mutable((1, 1), True)
    net_ms = Net()
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, start, stop, step, dout))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_slice_mutable():
    """
    Feature: test sequence_slice mutable
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def construct(self, x, a, b):
            out = x[a:b]
            return out

    x = mutable((0, mutable(2), mutable(-3), mutable(-1),
                 mutable(4), -1, -9, mutable(-5), -10))
    a, b = mutable(-9), mutable(-7)
    net = Net()
    out = net(x, a, b)
    ex = x[a:b]
    assert np.allclose(out, ex)
