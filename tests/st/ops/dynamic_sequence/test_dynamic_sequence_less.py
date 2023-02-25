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
from mindspore import context
from mindspore.ops.operations import _sequence_ops as _seq
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class NetTupleLt(nn.Cell):
    def __init__(self):
        super().__init__()
        self.seq_lt = _seq.tuple_lt()

    def construct(self, x, y):
        return self.seq_lt(x, y)


class NetTupleLe(nn.Cell):
    def __init__(self):
        super().__init__()
        self.seq_le = _seq.tuple_le()

    def construct(self, x, y):
        return self.seq_le(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_dyn_le():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    x = mutable((1, 2, 3, 4, 5, 6), True)
    y = mutable((1, 2, 3, 2, 6), True)
    expect = False
    net = NetTupleLe()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_dyn_lt():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    x = mutable((1, 2, 3, 4, 5, 6), True)
    y = (1, 2, 3, 4, 5, 6)
    expect = False
    net = NetTupleLt()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_le():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    x = (1, 2, 3, 4, 5)
    y = (True, 2, 3, 4, 5)
    expect = True
    net = NetTupleLe()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_lt():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    x = (1, 2, 3, 4, 5, 6)
    y = (True, 2, 3, 4, 5)
    expect = False
    net = NetTupleLt()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_getitem_grad():
    """
    Feature: test sequence getitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetTupleLe()
    x = mutable((2, 3, 4, 5, 6), True)
    y = mutable((1, 2, 3, 4, 5, 6), True)
    dout = True
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(x, y, dout))
