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
import mindspore.ops.operations._grad_ops as P
from mindspore import Tensor
from mindspore.common.api import jit


class TraceGradNet(nn.Cell):
    def __init__(self):
        super(TraceGradNet, self).__init__()
        self.trace = P.TraceGrad()

    @jit
    def construct(self, y_grad, x_shape):
        return self.trace(y_grad, x_shape)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_trace_grad_2d_int32():
    """
    Feature: Returns the result of trace grad of the int32 x_shape
    Description: int32 x_shape
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        y_grad = Tensor(25).astype("int32")
        x_shape = Tensor(np.array([3, 3]).astype(np.int32))
        net = TraceGradNet()
        x_grad = net(y_grad, x_shape)
        grad_expect = np.array([[25, 0, 0], [0, 25, 0], [0, 0, 25]]).astype(np.int32)
        assert (x_grad.asnumpy() == grad_expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_trace_grad_2d_double():
    """
    Feature: Returns the result of trace grad of the double y_grad, int32 x_shape
    Description: double y_grad, int32 x_shape
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        y_grad = Tensor(4).astype("int64")
        x_shape = Tensor(np.array([4, 3]).astype(np.int64))
        net = TraceGradNet()
        x_grad = net(y_grad, x_shape)
        grad_expect = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4], [0, 0, 0]]).astype(np.int64)
        assert (x_grad.asnumpy() == grad_expect).all()
