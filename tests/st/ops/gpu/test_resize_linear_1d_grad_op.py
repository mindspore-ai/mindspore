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
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops.operations import ResizeLinear1D

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class ResizeLinear1DNet(nn.Cell):
    """ResizeLinear1DNet."""

    def __init__(self, coordinate_transformation_mode="align_corners"):
        """Init."""
        super(ResizeLinear1DNet, self).__init__()
        self.resize = ResizeLinear1D(coordinate_transformation_mode)

    def construct(self, x, size):
        """Construct."""
        return self.resize(x, size)


class ResizeLinear1DGradNet(nn.Cell):
    """ResizeLinear1DGradNet."""

    def __init__(self, forward_cpu_net):
        """Init."""
        super(ResizeLinear1DGradNet, self).__init__()
        self.resize_grad = C.GradOperation(get_all=True, sens_param=True)
        self.forward_cpu_net = forward_cpu_net

    def construct(self, grad_output, input_x, size):
        """Construct."""
        gout = self.resize_grad(self.forward_cpu_net)(
            input_x, size, grad_output)
        return gout


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_resize_linear_1d_grad_align_corners(dtype):
    """
    Feature: ResizeLinear1DGrad cpu kernel align_corners mode
    Description: test the rightness of ResizeLinear1DGrad cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([6], dtype=np.int64))
    grad_output = Tensor(np.array([[[1., 2., 3., 4., 5., 6.],
                                    [7., 8., 9., 10., 11., 12.]]], dtype=dtype))
    net_cpu = ResizeLinear1DNet()
    grad = ResizeLinear1DGradNet(net_cpu)
    output = grad(grad_output, x, size)
    expect = np.array([[[2.8, 8.4, 9.8],
                        [13.6, 22.8, 20.6]]]).astype(dtype)
    print("ms grad input: ", output[0].asnumpy())
    assert np.allclose(output[0].asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_resize_linear_1d_grad_half_pixel(dtype):
    """
    Feature: ResizeLinear1DGrad cpu kernel half_pixel mode
    Description: test the rightness of ResizeLinear1DGrad cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([6], dtype=np.int64))
    grad_output = Tensor(np.array([[[1., 2., 3., 4., 5., 6.],
                                    [7., 8., 9., 10., 11., 12.]]], dtype=dtype))
    net_cpu = ResizeLinear1DNet("half_pixel")
    grad = ResizeLinear1DGradNet(net_cpu)
    output = grad(grad_output, x, size)
    expect = np.array([[[3.25, 7, 10.75],
                        [15.25, 19, 22.75]]]).astype(dtype)
    print("ms grad input: ", output[0].asnumpy())
    assert np.allclose(output[0].asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_resize_linear_1d_grad_same_shape(dtype):
    """
    Feature: ResizeLinear1DGrad cpu kernel same shape
    Description: test the rightness of ResizeLinear1DGrad cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([3], dtype=np.int64))
    grad_output = Tensor(np.array([[[1., 2., 3.],
                                    [7., 8., 9.]]], dtype=dtype))
    net_cpu = ResizeLinear1DNet()
    grad = ResizeLinear1DGradNet(net_cpu)
    output = grad(grad_output, x, size)
    expect = np.array([[[1., 2., 3.],
                        [7., 8., 9.]]]).astype(dtype)
    print("ms grad input: ", output[0].asnumpy())
    assert np.allclose(output[0].asnumpy(), expect)
