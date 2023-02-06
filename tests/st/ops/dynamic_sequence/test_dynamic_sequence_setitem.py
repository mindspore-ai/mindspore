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

import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore import context
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE)


class NetSetItem(nn.Cell):
    def construct(self, seq, idx, value):
        return F.tuple_setitem(seq, idx, value)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_setitem():
    """
    Feature: test sequence_setitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((1, 2, 3, 4, 5, 6), True)
    value = 9
    idx = 3
    expect = (1, 2, 3, 9, 5, 6)
    net = NetSetItem()
    res = net(seq, idx, value)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_setitem_grad_0():
    """
    Feature: test sequence setitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetSetItem()
    input_x = mutable((1, 2, 3), True)
    idx = mutable(1)
    value = mutable(8)
    dout = mutable((1, 1, 1), True)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out0 = ", grad_func(input_x, idx, value, dout))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_seq_setitem_grad_1():
    """
    Feature: test sequence setitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetSetItem()
    input_x = mutable((1, 2, 3), True)
    idx = 1
    value = 8
    dout = mutable((1, 1, 1), True)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(input_x, idx, value, dout))
