# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops.composite import GradOperation


class NetSigmoidGrad(nn.Cell):
    def __init__(self):
        super(NetSigmoidGrad, self).__init__()
        self.sigmoid_grad = G.SigmoidGrad()

    def construct(self, y, dy):
        return self.sigmoid_grad(y, dy)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, y, y_grad, dout):
        return self.grad(self.network)(y, y_grad, dout)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sigmoid_grad_grad():
    y = Tensor(np.array([[[[-1, 1, 2],
                           [1, -1, 1],
                           [2, 1, -1]]]]).astype(np.float32))
    y_grad = Tensor(np.array([[[[-11, 2, 4],
                                [-1, 1, -1],
                                [-4, 4, -4]]]]).astype(np.float32))
    dout = Tensor(np.array([[[[-11, 2, 4],
                              [-1, 1, -1],
                              [-4, 4, -4]]]]).astype(np.float32))

    expect_ddy = np.array([[[[363., -4., -48.],
                             [-1., 3., -1.],
                             [-48., -16., 48.]]]]).astype(np.float32)

    expect_d2x = np.array([[[[22., 0., -8.],
                             [-0., -2., -0.],
                             [8., 0., 8.]]]]).astype(np.float32)

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    sigmoid_grad_grad = Grad(NetSigmoidGrad())
    ddy, d2x = sigmoid_grad_grad(y, y_grad, dout)
    diff0 = ddy.asnumpy() - expect_ddy
    diff1 = d2x.asnumpy() - expect_d2x
    assert np.all(abs(diff0) < error)
    assert np.all(abs(diff1) < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    sigmoid_grad_grad = Grad(NetSigmoidGrad())
    ddy, d2x = sigmoid_grad_grad(y, y_grad, dout)
    diff0 = ddy.asnumpy() - expect_ddy
    diff1 = d2x.asnumpy() - expect_d2x
    assert np.all(abs(diff0) < error)
    assert np.all(abs(diff1) < error)
