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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self, eps=-1.0):
        super(Net, self).__init__()
        self.grad = G.LogitGrad(eps)

    def construct(self, dy, x):
        return self.grad(dy, x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_logit_grad_graph_float32():
    """
     Feature: LogitGrad gpu TEST.
     Description: 1d test case for LogitGrad with GRAPH_MODE
     Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
    dy = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    expect = np.array([11.11111164, 6.24999952, 4.76190472]).astype(np.float32)
    net = Net()
    output = net(dy, x)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * 1e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_logit_grad_pynative_float32():
    """
     Feature: LogitGrad gpu TEST.
     Description: 1d test case for LogitGrad with PYNATIVE_MODE
     Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
    dy = Tensor(np.array([1.0, 1.0, 1.0]).astype(np.float32))
    expect = np.array([11.11111164, 6.24999952, 4.76190472]).astype(np.float32)
    logitgrad = G.LogitGrad()
    output = logitgrad(dy, x)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * 1e-4
    assert np.all(diff < error)
