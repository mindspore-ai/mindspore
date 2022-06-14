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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.composite import GradOperation
from mindspore.ops import operations as P
from mindspore.ops.operations import nn_ops as NN


class Net(nn.Cell):
    def __init__(self, out_channel, kernel_size, pad, stride, dilation):
        super(Net, self).__init__()
        self.net = NN.DeformableOffsets(ksize=(kernel_size, kernel_size),
                                        pads=(pad, pad, pad, pad),
                                        strides=(stride, stride, stride, stride),
                                        dilations=(dilation, dilation, dilation, dilation),
                                        deformable_groups=1,
                                        modulated=True,
                                        data_format="NCHW")
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="pad",
                             pad=pad,
                             stride=kernel_size,
                             dilation=1,
                             group=1,
                             data_format="NCHW")

    def construct(self, x, w, offset):
        x = self.net(x, offset)
        return self.conv(x, w)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, w, offset, output_grad):
        return self.grad(self.network)(x, w, offset, output_grad)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_deformable_conv2d_grad():
    """"
    Feature: deformable_conv2d_grad function
    Description: Test case for simplest deformable_conv2d_grad
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
    kernel_size = 2
    stride = 1
    pad = 0
    dilation = 1
    # x shape [1, 64, 2, 2]
    x = Tensor(np.ones([1, 64, 2, 2]).astype(np.float32) * 0.1)
    # weight shape [1, 64, 2, 2]
    weight = Tensor(np.ones([1, 64, 2, 2]).astype(np.float32) * 0.1)
    # offsets shape [1, 12, 1, 1]
    offsets = Tensor(np.ones([1, 12, 1, 1]).astype(np.float32) * 0.1)
    # out_channel, kernel_size, pad, stride, dilation
    dfm_conv2d_net = Net(1, kernel_size, pad, stride, dilation)
    out = dfm_conv2d_net(x, weight, offsets)
    grad_net = Grad(dfm_conv2d_net)
    grad_output = grad_net(x, weight, offsets, out)
    expect_out = np.array([[[[0.2310471]]]]).astype(np.float32)
    expect_grad_x = np.array([[[[0.00187125, 0.00207916], [0.00207916, 0.00231018]]] * 64]).astype(np.float32)
    expect_grad_weight = np.array([[[[0.00231128, 0.00208033], [0.00208033, 0.0018723]]] * 64]).astype((np.float32))
    expect_grad_offset = np.array([[[0]], [[-0.01478]], [[0]], [[-0.01331]],
                                   [[0]], [[0]], [[-0.01478]], [[-0.01331]],
                                   [[0.14785]], [[0.13307]], [[0.13307]], [[0.11976]]]).astype((np.float32))
    assert np.allclose(out.asnumpy(), expect_out, 0.0001, 0.0001)
    assert np.allclose(grad_output[0].asnumpy(), expect_grad_x, 0.0001, 0.0001)
    assert np.allclose(grad_output[1].asnumpy(), expect_grad_weight, 0.0001, 0.0001)
    assert np.allclose(grad_output[2].asnumpy(), expect_grad_offset, 0.0001, 0.0001)
