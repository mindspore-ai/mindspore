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
import pytest
import numpy as np

from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype
from mindspore.ops.operations import nn_ops
from mindspore.ops import functional as F

grad_all = C.GradOperation(get_all=True)


class TestNetwork(nn.Cell):
    def __init__(self):
        super(TestNetwork, self).__init__()
        stride = (1, 1, 1, 1)
        pad = (0, 0, 0, 0)
        ksize = (2, 2)
        self.deformable_offsets_grad_op = G.DeformableOffsetsGrad(stride, pad, ksize)

    def construct(self, dout, x, offsets):
        output = self.deformable_offsets_grad_op(dout, x, offsets)
        return output


def test_grad_infer():
    """
    Feature: CPU operation.
    Description: Test of CPU operation: DeformableOffsetsGrad
    Expectation: No exception raised.
    """
    dout = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    x = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    offsets = Tensor(np.array([0.1] * 12).astype(np.float32).reshape([1, 12, 1, 1]))

    net = TestNetwork()
    grad = net(dout, x, offsets)
    print("grad_x:", grad[0])
    print("grad_offset:", grad[1])
    return grad


class ForwardNet(nn.Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        stride = (1, 1, 1, 1)
        pad = (0, 0, 0, 0)
        ksize = (2, 2)
        self.deformable_offsets_grad_op = nn_ops.DeformableOffsets(stride, pad, ksize)

    def construct(self, x, offsets):
        output = self.deformable_offsets_grad_op(x, offsets)
        return output


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__()
        self.net = net

    def construct(self, *inputs):
        out = self.net(*inputs)
        return out, grad_all(self.net)(*inputs)


def test_auto_diff():
    """
    Feature: CPU operation.
    Description: Test of CPU operation: DeformableOffsetsGrad by auto diff.
    Expectation: No exception raised.
    """
    x = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    offsets = Tensor(np.array([0.1] * 12).astype(np.float32).reshape([1, 12, 1, 1]))
    forward_net = ForwardNet()
    net = BackwardNet(forward_net)
    grad = net(x, offsets)
    print("grad_x:", grad[0])
    print("grad_offset:", grad[1])
    return grad


class NetDeformableOffsetsGrad(nn.Cell):
    def __init__(self, data_format):
        super(NetDeformableOffsetsGrad, self).__init__()
        strides = (1, 1, 1, 1)
        pads = (0, 0, 0, 0)
        ksize = (3, 3)
        self.grad_op = G.DeformableOffsetsGrad(strides, pads, ksize, data_format=data_format)

    def construct(self, grad, input_x, offsets):
        return self.grad_op(grad, input_x, offsets)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_deformable_offsets_grad_nchw(data_type):
    """
    Feature: DeformableOffsetsGrad cpu kernel
    Description: test the rightness of DeformableOffsetsGrad gpu kernel
    Expectation: the output is same as expected result
    """
    net = NetDeformableOffsetsGrad(data_format="NCHW")
    dout = Tensor(np.ones([1, 2, 3, 3]).astype(data_type))
    x = Tensor(np.ones([1, 2, 4, 4]).astype(data_type))
    offsets = Tensor(np.ones([1, 27, 1, 1]).astype(data_type) * 0.1)
    output = net(dout, x, offsets)

    expect_grad_x = np.array([[[0.081, 0.09, 0.09, 0.009],
                               [0.09, 0.1, 0.1, 0.01],
                               [0.09, 0.1, 0.1, 0.01],
                               [0.009, 0.01, 0.01, 0.001]],
                              [[0.081, 0.09, 0.09, 0.009],
                               [0.09, 0.1, 0.1, 0.01],
                               [0.09, 0.1, 0.1, 0.01],
                               [0.009, 0.01, 0.01, 0.001]]]
                             ).astype(data_type)
    expect_grad_offset = np.array([0] * 18 + [2.0] * 9).astype(data_type).reshape([1, 27, 1, 1])
    rtol = 1e-5
    if data_type == np.float16:
        rtol = 1e-3
    assert np.allclose(output[0].asnumpy(), expect_grad_x, rtol)
    assert np.allclose(output[1].asnumpy(), expect_grad_offset, rtol)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('data_type', [np.float16, np.float32])
def test_deformable_offsets_grad_nhwc(data_type):
    """
    Feature: DeformableOffsetsGrad cpu kernel
    Description: test the rightness of DeformableOffsetsGrad gpu kernel
    Expectation: the output is same as expected result
    """
    net = NetDeformableOffsetsGrad(data_format="NHWC")
    dout = Tensor(np.ones([1, 3, 3, 2]).astype(data_type))
    x = Tensor(np.ones([1, 4, 4, 2]).astype(data_type))
    offsets = Tensor(np.ones([1, 1, 1, 27]).astype(data_type) * 0.1)
    output = net(dout, x, offsets)
    expect_grad_x = np.array([[[0.081, 0.081],
                               [0.09, 0.09],
                               [0.09, 0.09],
                               [0.009, 0.009]],
                              [[0.09, 0.09],
                               [0.1, 0.1],
                               [0.1, 0.1],
                               [0.01, 0.01]],
                              [[0.09, 0.09],
                               [0.1, 0.1],
                               [0.1, 0.1],
                               [0.01, 0.01]],
                              [[0.009, 0.009],
                               [0.01, 0.01],
                               [0.01, 0.01],
                               [0.001, 0.001]]
                              ]
                             ).astype(data_type)
    expect_grad_offset = np.array([0] * 18 + [2.0] * 9).astype(data_type).reshape([1, 1, 1, 27])
    rtol = 1e-5
    if data_type == np.float16:
        rtol = 1e-3
    assert np.allclose(output[0].asnumpy(), expect_grad_x, rtol)
    assert np.allclose(output[1].asnumpy(), expect_grad_offset, rtol)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap():
    """"
    Feature: Feature: DeformableOffsetsGrad cpu kernel
    Description: Test case with vmap.
    Expectation: The results are as expected.
    """

    def cal_deformable_offsets_grad(dout, x, offsets):
        net = NetDeformableOffsetsGrad(data_format="NCHW")
        return net(dout, x, offsets)

    dout = Tensor(np.arange(2 * 1 * 2 * 3 * 3).reshape(2, 1, 2, 3, 3), dtype.float32)
    x = Tensor(np.arange(2 * 1 * 2 * 4 * 4).reshape(2, 1, 2, 4, 4), dtype.float32)
    offsets = Tensor(np.arange(2 * 1 * 27 * 1 * 1).reshape(2, 1, 27, 1, 1) * 0.1, dtype.float32)

    vmap_deformable_offset_grad = F.vmap(cal_deformable_offsets_grad, in_axes=(0, 0, 0), out_axes=0)
    out1 = vmap_deformable_offset_grad(dout, x, offsets)

    def manually_batched(dout, x, offsets):
        output_dx = []
        output_d_offsets = []
        for i in range(x.shape[0]):
            dx, d_offsets = cal_deformable_offsets_grad(dout[i], x[i], offsets[i])
            output_dx.append(dx)
            output_d_offsets.append(d_offsets)

        return F.stack(output_dx), F.stack(output_d_offsets)

    out2 = manually_batched(dout, x, offsets)
    assert np.allclose(out1[0].asnumpy(), out2[0].asnumpy())
    assert np.allclose(out1[1].asnumpy(), out2[1].asnumpy())
