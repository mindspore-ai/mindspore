# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class NetReciprocalGrad(nn.Cell):
    def __init__(self):
        super(NetReciprocalGrad, self).__init__()
        self.grad = G.ReciprocalGrad()

    def construct(self, y, dy):
        return self.grad(y, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reciprocal_grad_float32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    y = Tensor(np.array([[[[-1, 1, 12],
                           [5, 34, 6],
                           [10, 2, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[29, 1, 55],
                            [2.2, 63, 2],
                            [3, 3, 12]]]]).astype(np.float32))
    expect = np.array([[[[-29, -1, -7920],
                         [-55, -72828, -72],
                         [-300, -12, -12]]]]).astype(np.float32)
    net = NetReciprocalGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = Tensor(np.array([[[[-1, 1, 12],
                           [5, 34, 6],
                           [10, 2, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[29, 1, 55],
                            [2.2, 63, 2],
                            [3, 3, 12]]]]).astype(np.float32))
    expect = np.array([[[[-29, -1, -7920],
                         [-55, -72828, -72],
                         [-300, -12, -12]]]]).astype(np.float32)
    net = NetReciprocalGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reciprocal_grad_float16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    y = Tensor(np.array([[0.01, 0.2, 0.22],
                         [10.002, 2, -1]]).astype(np.float16))
    dy = Tensor(np.array([[34, 1, 55],
                          [3, 3, 63]]).astype(np.float16))
    expect = np.array([[-0.0034, -0.03998, -2.662],
                       [-300, -12, -63]]).astype(np.float16)
    net = NetReciprocalGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = Tensor(np.array([[0.01, 0.2, 0.22],
                         [10.002, 2, -1]]).astype(np.float16))
    dy = Tensor(np.array([[34, 1, 55],
                          [3, 3, 63]]).astype(np.float16))
    expect = np.array([[-0.0034, -0.03998, -2.662],
                       [-300, -12, -63]]).astype(np.float16)
    net = NetReciprocalGrad()
    output = net(y, dy)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)
