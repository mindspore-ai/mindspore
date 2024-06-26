# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
import mindspore
from mindspore import nn
import mindspore.ops as ops
import mindspore.context as context
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    """Net used to test nn.Dropout1d"""
    def __init__(self, p):
        super(Net, self).__init__()
        self.dropout1d = nn.Dropout1d(p)

    def construct(self, x):
        return self.dropout1d(x)


class FNet(nn.Cell):
    """Net used to test ops.dropout1d"""
    def __init__(self, p):
        super(FNet, self).__init__()
        self.p = p

    def construct(self, x):
        out = ops.dropout1d(x, self.p)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dropout1d(mode):
    """
    Feature: dropout1d
    Description: Verify the result of Dropout1d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = np.random.randn(4, 3)
    dropout = Net(p=1.0)
    x = Tensor(x, mindspore.float32)
    dropout.set_train()
    output = dropout(x)
    expect = np.zeros((4, 3))
    np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_f_dropout1d(mode):
    """
    Feature: function api dropout1d
    Description: Verify the result of dropout1d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = np.random.randn(4, 3)
    x = Tensor(x, mindspore.float32)
    net = FNet(p=1.0)
    output = net(x)
    expect = np.zeros((4, 3))
    np.allclose(output.asnumpy(), expect)
