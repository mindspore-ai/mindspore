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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G


class Conv2dBpropInputInplace(nn.Cell):
    def __init__(self, w1, w2):
        super(Conv2dBpropInputInplace, self).__init__()
        self.conv2d_1 = P.Conv2DBackpropInput(out_channel=256, kernel_size=1)
        self.w1 = Parameter(initializer(w1, w1.shape), name='w1')
        self.conv2d_2 = P.Conv2DBackpropInput(out_channel=256, kernel_size=1)
        self.w2 = Parameter(initializer(w2, w2.shape), name='w2')
        self.add = P.Add()
        self.maxpool = P.MaxPool(kernel_size=3, strides=2, pad_mode='SAME')
        self.maxpool_grad = G.MaxPoolGrad(kernel_size=3, strides=2, pad_mode='SAME')
        self.shape = (32, 64, 56, 56)

    def construct(self, x1, x2, x3):
        dx1 = self.conv2d_1(x1, self.w1, self.shape)
        dx2 = self.conv2d_2(x2, self.w2, self.shape)

        dx = self.add(dx1, dx2)
        y = self.maxpool(x3)
        y = self.maxpool_grad(x3, y, dx)
        return y


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inplace_fusion1():

    np.random.seed(42)
    w1_np = np.random.randn(64, 64, 1, 1)
    w2_np = np.random.randn(256, 64, 1, 1)
    x1_np = np.random.randn(32, 64, 56, 56)
    x2_np = np.random.randn(32, 256, 56, 56)
    x3_np = np.random.randn(32, 64, 112, 112)

    w1 = Tensor(w1_np.astype(np.float32))
    w2 = Tensor(w2_np.astype(np.float32))
    x1 = Tensor(x1_np.astype(np.float32))
    x2 = Tensor(x2_np.astype(np.float32))
    x3 = Tensor(x3_np.astype(np.float32))

    net = Conv2dBpropInputInplace(w1, w2)
    context.set_context(device_target='GPU', mode=context.GRAPH_MODE)
    fusion_output = net(x1, x2, x3)

    context.set_context(device_target='GPU', mode=context.PYNATIVE_MODE)
    no_fusion_output = net(x1, x2, x3)

    assert np.allclose(fusion_output.asnumpy(), no_fusion_output.asnumpy(), atol=2e-5)
