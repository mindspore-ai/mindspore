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
import mindspore as ms
from mindspore import Tensor
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.functional import vmap


class NetConv3d(nn.Cell):
    def __init__(self):
        super(NetConv3d, self).__init__()
        out_channel = 4
        kernel_size = 2
        self.conv = P.Conv3D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_dshape_1():
    """
    Feature: Test conv3d dynamic shape.
    Description: Test conv3d dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetConv3d()
    input_x_dyn = Tensor(shape=[1, 3, 3, 3, None], dtype=ms.float32)
    input_w_dyn = Tensor(shape=[4, 3, 2, 2, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn, input_w_dyn)
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    output = net(x, w)
    expect_shape = (1, 4, 2, 2, 2)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_dshape_2():
    """
    Feature: Test conv3d dynamic shape.
    Description: Test conv3d dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetConv3d()
    input_x_dyn = Tensor(shape=[None, 3, 3, 3, 3], dtype=ms.float32)
    input_w_dyn = Tensor(shape=[None, 3, 2, 2, 2], dtype=ms.float32)
    net.set_inputs(input_x_dyn, input_w_dyn)
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    output = net(x, w)
    expect_shape = (1, 4, 2, 2, 2)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d():
    x = Tensor(np.arange(1 * 3 * 3 * 3 * 3).reshape(1, 3, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(4 * 3 * 2 * 2 * 2).reshape(4, 3, 2, 2, 2).astype(np.float32))
    expect = np.array([[[[[12960., 13236.],
                          [13788., 14064.]],
                         [[15444., 15720.],
                          [16272., 16548.]]],
                        [[[32256., 33108.],
                          [34812., 35664.]],
                         [[39924., 40776.],
                          [42480., 43332.]]],
                        [[[51552., 52980.],
                          [55836., 57264.]],
                         [[64404., 65832.],
                          [68688., 70116.]]],
                        [[[70848., 72852.],
                          [76860., 78864.]],
                         [[88884., 90888.],
                          [94896., 96900.]]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetConv3d()
    output = net(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetConv3d()
    output = net(x, w)
    assert (output.asnumpy() == expect).all()


class MSConv3dNet(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode='pad', padding=0, stride=1, dilation=1,
                 has_bias=False, weight_init='normal'):
        super(MSConv3dNet, self).__init__()
        self.cv1 = nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             pad_mode=pad_mode,
                             padding=padding,
                             stride=stride,
                             dilation=dilation,
                             group=1,
                             has_bias=has_bias,
                             weight_init=weight_init,
                             data_format='NCDHW')

    def construct(self, x):
        x = self.cv1(x)
        return x


class MSGradNet(nn.Cell):
    def __init__(self, network):
        super(MSGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, x, dy):
        grad_op = self.grad(self.network, self.params)
        output = grad_op(x, dy)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    out_c = 2
    kernel_size = (2, 2, 2)
    x = Tensor(np.array([[[[[1.6924546, 0.05080776, -0.6369957],
                            [0.19091548, 2.1002553, 0.12015896],
                            [0.6172031, 0.30017033, -0.35224986]],
                           [[-1.1425182, -0.34934273, -0.20889424],
                            [0.5866232, 0.8389834, 0.9311021],
                            [0.2855873, 0.8851412, -0.7543979]],
                           [[1.2528682, 0.5129298, -0.29809284],
                            [0.48851815, -0.07557172, 1.1316293],
                            [1.5198169, 2.1855755, -1.3964963]]]]]).astype(dtype))
    dy = Tensor(np.array([[[[[-1.4441139, -0.5044659],
                             [0.16003707, 0.8761689]],
                            [[0.31563494, -2.0222013],
                             [-0.30620402, 0.8279746]]],
                           [[[0.23009473, 0.7620112],
                             [-0.22232814, -0.20075807]],
                            [[0.18656139, 0.41005164],
                             [0.19829972, 0.11900865]]]]]).astype(dtype))
    w = Tensor(np.array([[[[[-0.9358, -0.2679],
                            [0.5304, -0.6917]],
                           [[-0.3968, -0.6872],
                            [-0.8452, -0.6712]]]],
                         [[[[-0.0127, -1.1173],
                            [0.2344, 1.6598]],
                           [[0.7420, -0.1918],
                            [-0.8876, -0.7472]]]]]).astype(dtype))
    w_exp = np.array([[[[[-0.9384, -0.2830],
                         [0.5487, -0.6330]],
                        [[-0.4148, -0.7200],
                         [-0.8572, -0.6079]]]],
                      [[[[-0.0109, -1.1089],
                         [0.2138, 1.6478]],
                        [[0.7450, -0.1866],
                         [-0.8992, -0.7629]]]]]).astype(dtype)
    net = MSConv3dNet(x.shape[1], out_c, kernel_size, weight_init=w)
    grad_net = MSGradNet(net)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    grad_net.set_train(True)
    output = grad_net(x, dy)
    optimizer(output[1])
    assert np.allclose(net.cv1.weight.asnumpy(), w_exp, atol=1.0e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv3d_vmap():
    """
    Feature: Conv3D op
    Description: Test vmap rule for Conv3D op
    Expectation: The dataset is processed as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv3d = NetConv3d()

    batch_x = Tensor(np.arange(2 * 1 * 1 * 3 * 3 * 3).reshape(2, 1, 1, 3, 3, 3).astype(np.float32))
    w = Tensor(np.ones([4, 1, 2, 2, 2]).astype(np.float32))
    expected1 = np.array([[[[[[52., 60.], [76., 84.]], [[124., 132.], [148., 156.]]],
                            [[[52., 60.], [76., 84.]], [[124., 132.], [148., 156.]]],
                            [[[52., 60.], [76., 84.]], [[124., 132.], [148., 156.]]],
                            [[[52., 60.], [76., 84.]], [[124., 132.], [148., 156.]]]]],
                          [[[[[268., 276.], [292., 300.]], [[340., 348.], [364., 372.]]],
                            [[[268., 276.], [292., 300.]], [[340., 348.], [364., 372.]]],
                            [[[268., 276.], [292., 300.]], [[340., 348.], [364., 372.]]],
                            [[[268., 276.], [292., 300.]], [[340., 348.], [364., 372.]]]]]]).astype(np.float32)
    output1 = vmap(conv3d, (0, None))(batch_x, w)
    assert np.allclose(output1.asnumpy(), expected1, 0.0001, 0.0001)

    x = Tensor(np.arange(1 * 1 * 3 * 3 * 3).reshape(1, 1, 3, 3, 3).astype(np.float32))
    batch_w = Tensor(np.arange(2 * 4 * 1 * 2 * 2 * 2).reshape(2, 4, 1, 2, 2, 2).astype(np.float32))
    expected2 = np.array([[[[[[268., 296.], [352., 380.]], [[520., 548.], [604., 632.]]],
                            [[[684., 776.], [960., 1052.]], [[1512., 1604.], [1788., 1880.]]],
                            [[[1100., 1256.], [1568., 1724.]], [[2504., 2660.], [2972., 3128.]]],
                            [[[1516., 1736.], [2176., 2396.]], [[3496., 3716.], [4156., 4376.]]]]],
                          [[[[[1932., 2216.], [2784., 3068.]], [[4488., 4772.], [5340., 5624.]]],
                            [[[2348., 2696.], [3392., 3740.]], [[5480., 5828.], [6524., 6872.]]],
                            [[[2764., 3176.], [4000., 4412.]], [[6472., 6884.], [7708., 8120.]]],
                            [[[3180., 3656.], [4608., 5084.]], [[7464., 7940.], [8892., 9368.]]]]]]
                         ).astype(np.float32)
    output2 = vmap(conv3d, (None, 0))(x, batch_w)
    assert np.allclose(output2.asnumpy(), expected2, 0.0001, 0.0001)
