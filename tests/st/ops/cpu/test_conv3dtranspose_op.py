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
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class NetConv3dTranspose(nn.Cell):
    def __init__(self):
        super(NetConv3dTranspose, self).__init__()
        in_channel = 2
        out_channel = 2
        kernel_size = 2
        self.conv_trans = P.Conv3DTranspose(in_channel, out_channel,
                                            kernel_size,
                                            pad_mode="pad",
                                            pad=1,
                                            stride=1,
                                            dilation=1,
                                            group=1)

    def construct(self, x, w):
        return self.conv_trans(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conv3dtranspose_dshape_1():
    """
    Feature: Test conv3dtranspose dynamic shape.
    Description: Test conv3dtranspose dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetConv3dTranspose()
    input_x_dyn = Tensor(shape=[1, 2, 3, 3, None], dtype=ms.float32)
    input_w_dyn = Tensor(shape=[2, 2, 2, 2, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn, input_w_dyn)
    x = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(2 * 2 * 2 * 2 * 2).reshape(2, 2, 2, 2, 2).astype(np.float32))
    output = net(x, w)
    expect_shape = (1, 2, 2, 2, 2)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conv3dtranspose_dshape_2():
    """
    Feature: Test conv3dtranspose dynamic shape.
    Description: Test conv3dtranspose dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetConv3dTranspose()
    input_x_dyn = Tensor(shape=[None, 2, 3, 3, 3], dtype=ms.float32)
    input_w_dyn = Tensor(shape=[None, 2, 2, 2, 2], dtype=ms.float32)
    net.set_inputs(input_x_dyn, input_w_dyn)
    x = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(2 * 2 * 2 * 2 * 2).reshape(2, 2, 2, 2, 2).astype(np.float32))
    output = net(x, w)
    expect_shape = (1, 2, 2, 2, 2)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conv3d_transpose():
    """
    Feature: Test conv3dtranspose.
    Description: Test conv3dtranspose.
    Expectation: Success.
    """
    x = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    w = Tensor(np.ones((2, 2, 2, 2, 2)).astype(np.float32))
    expect = np.array([[[[[320., 336.],
                          [368., 384.]],
                         [[464., 480.],
                          [512., 528.]]],
                        [[[320., 336.],
                          [368., 384.]],
                         [[464., 480.],
                          [512., 528.]]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    conv3dtranspose = NetConv3dTranspose()
    output = conv3dtranspose(x, w)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conv3d_transpose_vmap():
    """
    Feature: Conv3DTranspose op
    Description: Test vmap rule for Conv3DTranspose op
    Expectation: The dataset is processed as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    conv3d_trans = NetConv3dTranspose()

    batch_dout = Tensor(np.arange(2 * 1 * 2 * 3 * 3 * 3).reshape(2, 1, 2, 3, 3, 3).astype(np.float32))
    weight = Tensor(np.ones([2, 2, 2, 2, 2]).astype(np.float32))
    expected1 = np.array([[[[[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]],
                            [[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]]]],
                          [[[[[1184., 1200.], [1232., 1248.]], [[1328., 1344.], [1376., 1392.]]],
                            [[[1184., 1200.], [1232., 1248.]], [[1328., 1344.], [1376., 1392.]]]]]]).astype(np.float32)
    output1 = vmap(conv3d_trans, (0, None))(batch_dout, weight)
    assert np.allclose(output1.asnumpy(), expected1, 0.0001, 0.0001)

    dout = Tensor(np.arange(1 * 2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).astype(np.float32))
    batch_weight = Tensor(np.ones([2, 2, 2, 2, 2, 2]).astype(np.float32))
    expected2 = np.array([[[[[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]],
                            [[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]]]],
                          [[[[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]],
                            [[[320., 336.], [368., 384.]], [[464., 480.], [512., 528.]]]]]]).astype(np.float32)
    output2 = vmap(conv3d_trans, (None, 0))(dout, batch_weight)
    assert np.allclose(output2.asnumpy(), expected2, 0.0001, 0.0001)
