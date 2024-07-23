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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetGridSampler2DGrad(nn.Cell):
    def __init__(self):
        super(NetGridSampler2DGrad, self).__init__()
        self.grid_sampler_2d_grad = G.GridSampler2DGrad(interpolation_mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)

    def construct(self, grad, x, grid):
        return self.grid_sampler_2d_grad(grad, x, grid)


class NetGridSampler3DGrad(nn.Cell):
    def __init__(self):
        super(NetGridSampler3DGrad, self).__init__()
        self.grid_sampler_3d_grad = G.GridSampler3DGrad(interpolation_mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)

    def construct(self, grad, x, grid):
        return self.grid_sampler_3d_grad(grad, x, grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_2d_grad_float16():
    """
    Feature: GridSampler2DGrad op.
    Description: test data type is float16 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[1.62434536], [-0.61175641]],
                             [[-0.52817175], [-1.07296862]]],
                            [[[0.86540763], [-2.3015387]],
                             [[1.74481176], [-0.7612069]]]]), ms.float16)
    x = Tensor(np.arange(16).reshape((2, 2, 2, 2)), ms.float16)
    grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)), ms.float16)

    expect_x = np.array([[[[1.8152663e-01, 2.3405522e-01], [2.8468457e-01, 3.1232265e-01]],
                          [[-1.5441670e-01, -2.9868558e-01], [-3.7874258e-01, -7.6929545e-01]]],
                         [[[1.4454526e-02, 2.7964264e-04], [-7.1526930e-02, -1.3793383e+00]],
                          [[4.8538305e-02, 1.7512307e-01], [2.2430333e-01, 5.3564018e-01]]]], np.float16)
    expect_grid = np.array([[[[0.5480869, 1.0961736]], [[-0.8423623, -1.684725]]],
                            [[[1.3051109, 2.6102192]], [[-1.5313722, -3.0627444]]]], np.float16)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-3
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-3

    grid_sampler_2d_grad = NetGridSampler2DGrad()
    output = grid_sampler_2d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_2d_grad_float32():
    """
    Feature: GridSampler2DGrad op.
    Description: test data type is float32 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[1.62434536], [-0.61175641]],
                             [[-0.52817175], [-1.07296862]]],
                            [[[0.86540763], [-2.3015387]],
                             [[1.74481176], [-0.7612069]]]]), ms.float32)
    x = Tensor(np.arange(16).reshape((2, 2, 2, 2)), ms.float32)
    grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)), ms.float32)

    expect_x = np.array([[[[1.8152663e-01, 2.3405522e-01], [2.8468457e-01, 3.1232265e-01]],
                          [[-1.5441670e-01, -2.9868558e-01], [-3.7874258e-01, -7.6929545e-01]]],
                         [[[1.4454526e-02, 2.7964264e-04], [-7.1526930e-02, -1.3793383e+00]],
                          [[4.8538305e-02, 1.7512307e-01], [2.2430333e-01, 5.3564018e-01]]]], np.float32)
    expect_grid = np.array([[[[0.5480869, 1.0961736]], [[-0.8423623, -1.684725]]],
                            [[[1.3051109, 2.6102192]], [[-1.5313722, -3.0627444]]]], np.float32)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-5
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-5

    grid_sampler_2d_grad = NetGridSampler2DGrad()
    output = grid_sampler_2d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_2d_grad_float64():
    """
    Feature: GridSampler2DGrad op.
    Description: test data type is float64 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[1.62434536], [-0.61175641]],
                             [[-0.52817175], [-1.07296862]]],
                            [[[0.86540763], [-2.3015387]],
                             [[1.74481176], [-0.7612069]]]]), ms.float64)
    x = Tensor(np.arange(16).reshape((2, 2, 2, 2)), ms.float64)
    grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)), ms.float64)
    expect_x = np.array([[[[1.81526620e-01, 2.34055154e-01], [2.84684601e-01, 3.12322575e-01]],
                          [[-1.54416692e-01, -2.98685577e-01], [-3.78742596e-01, -7.69295510e-01]]],
                         [[[1.44545354e-02, 2.79674159e-04], [-7.15268792e-02, -1.37933840e+00]],
                          [[4.85383184e-02, 1.75123101e-01], [2.24303344e-01, 5.35640099e-01]]]], np.float64)
    expect_grid = np.array([[[[0.54808681, 1.09617361]], [[-0.84236252, -1.68472504]]],
                            [[[1.3051097, 2.61021939]], [[-1.5313728, -3.0627456]]]], np.float64)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-6
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-6

    grid_sampler_2d_grad = NetGridSampler2DGrad()
    output = grid_sampler_2d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_3d_grad_float16():
    """
    Feature: GridSampler3DGrad op.
    Description: test data type is float16 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[[1.62434536]], [[-0.61175641]]],
                             [[[-0.52817175]], [[-1.07296862]]]],
                            [[[[0.86540763]], [[-2.3015387]]],
                             [[[1.74481176]], [[-0.7612069]]]]]), ms.float16)
    x = Tensor(np.arange(32).reshape((2, 2, 2, 2, 2)), ms.float16)
    grid = Tensor(np.arange(-0.2, 1, 0.1).reshape((2, 2, 1, 1, 3)), ms.float16)

    expect_x = np.array([[[[[0.22947635, 0.13157275], [0.16147566, 0.0755332]],
                           [[0.1964415, 0.09119685], [0.11192339, 0.01496933]]],
                          [[[-0.15474537, -0.14071748], [-0.17269874, -0.17146334]],
                           [[-0.21268564, -0.2115334], [-0.2596092, -0.27768722]]]],
                         [[[[0.01125496, 0.02050772], [0.02340796, 0.00283393]],
                           [[0.01912753, -0.06469222], [-0.13939889, -1.3091719]]],
                          [[[0.02560127, 0.05783327], [0.07337838, 0.15408905]],
                           [[0.09384151, 0.18280618], [0.21644136, 0.17961383]]]]], np.float16)
    expect_grid = np.array([[[[[0.5480868, 1.0961738, 2.192347]]],
                             [[[-0.8423625, -1.6847249, -3.3710938]]]],
                            [[[[1.3051103, 2.610217, 5.2226562]]],
                             [[[-1.531373, -3.062745, -6.1254916]]]]], np.float16)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-3
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-3

    grid_sampler_3d_grad = NetGridSampler3DGrad()
    output = grid_sampler_3d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_3d_grad_float32():
    """
    Feature: GridSampler3DGrad op.
    Description: test data type is float32 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[[1.62434536]], [[-0.61175641]]],
                             [[[-0.52817175]], [[-1.07296862]]]],
                            [[[[0.86540763]], [[-2.3015387]]],
                             [[[1.74481176]], [[-0.7612069]]]]]), ms.float32)
    x = Tensor(np.arange(32).reshape((2, 2, 2, 2, 2)), ms.float32)
    grid = Tensor(np.arange(-0.2, 1, 0.1).reshape((2, 2, 1, 1, 3)), ms.float32)

    expect_x = np.array([[[[[0.22947635, 0.13157275], [0.16147566, 0.0755332]],
                           [[0.1964415, 0.09119685], [0.11192339, 0.01496933]]],
                          [[[-0.15474537, -0.14071748], [-0.17269874, -0.17146334]],
                           [[-0.21268564, -0.2115334], [-0.2596092, -0.27768722]]]],
                         [[[[0.01125496, 0.02050772], [0.02340796, 0.00283393]],
                           [[0.01912753, -0.06469222], [-0.13939889, -1.3091719]]],
                          [[[0.02560127, 0.05783327], [0.07337838, 0.15408905]],
                           [[0.09384151, 0.18280618], [0.21644136, 0.17961383]]]]], np.float32)
    expect_grid = np.array([[[[[0.5480868, 1.0961738, 2.192347]]],
                             [[[-0.8423625, -1.6847249, -3.3694496]]]],
                            [[[[1.3051103, 2.610217, 5.220438]]],
                             [[[-1.531373, -3.062745, -6.1254916]]]]], np.float32)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-6
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-6

    grid_sampler_3d_grad = NetGridSampler3DGrad()
    output = grid_sampler_3d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grid_sampler_3d_grad_float64():
    """
    Feature: GridSampler3DGrad op.
    Description: test data type is float64 in GPU.
    Expectation: success.
    """
    grad = Tensor(np.array([[[[[1.62434536]], [[-0.61175641]]],
                             [[[-0.52817175]], [[-1.07296862]]]],
                            [[[[0.86540763]], [[-2.3015387]]],
                             [[[1.74481176]], [[-0.7612069]]]]]), ms.float64)
    x = Tensor(np.arange(32).reshape((2, 2, 2, 2, 2)), ms.float64)
    grid = Tensor(np.arange(-0.2, 1, 0.1).reshape((2, 2, 1, 1, 3)), ms.float64)

    expect_x = np.array([[[[[0.22947633, 0.13157275], [0.16147564, 0.07553322]],
                           [[0.19644148, 0.09119682], [0.11192337, 0.01496933]]],
                          [[[-0.15474536, -0.14071748], [-0.17269872, -0.17146333]],
                           [[-0.21268567, -0.21153341], [-0.25960918, -0.27768723]]]],
                         [[[[0.01125496, 0.02050773], [0.02340796, 0.00283395]],
                           [[0.01912753, -0.06469218], [-0.13939896, -1.30917204]]],
                          [[[0.02560127, 0.05783328], [0.07337838, 0.15408907]],
                           [[0.09384151, 0.18280619], [0.21644133, 0.17961383]]]]]).astype(np.float64)
    expect_grid = np.array([[[[[0.54808681, 1.09617361, 2.19234722]]],
                             [[[-0.84236252, -1.68472504, -3.36945007]]]],
                            [[[[1.3051097, 2.61021939, 5.22043879]]],
                             [[[-1.5313728, -3.0627456, -6.1254912]]]]]).astype(np.float64)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-6
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-6

    grid_sampler_3d_grad = NetGridSampler3DGrad()
    output = grid_sampler_3d_grad(grad, x, grid)
    diff_x = output[0].asnumpy() - expect_x
    diff_grid = output[1].asnumpy() - expect_grid
    assert np.all(np.abs(diff_x) < error_x)
    assert np.all(np.abs(diff_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_grid_sampler_2d_grad():
    """
    Feature: GridSampler2DGrad GPU op vmap feature.
    Description: test the vmap feature of GridSampler2DGrad.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 2 batches
    grad = Tensor(np.arange(0, 8, 0.5).reshape(2, 2, 2, 2, 1).astype(np.float32))
    input_x = Tensor(np.arange(0, 16, 0.5).reshape(2, 2, 2, 2, 2).astype(np.float32))
    grid = Tensor(np.arange(0, 1.6, 0.1).reshape(2, 2, 2, 1, 2)).astype(np.float32)
    net = NetGridSampler2DGrad()
    expect_x = np.array([[[[[0.07, 0.10500001], [0.12999998, 0.19500001]],
                           [[0.435, 0.54], [0.66499996, 0.86]]],
                          [[[0.225, 0.65], [0.875, 2.75]],
                           [[0.32999998, 0.94499993], [1.27, 3.955]]]],
                         [[[[0.02000001, 0.18000004], [0.38000008, 7.6949997]],
                           [[0.02500001, 0.22500005], [0.4750001, 9.5]]],
                          [[[0, 0], [0, 8.49]],
                           [[0., 0.], [0., 9.855]]]]]).astype(np.float32)
    expect_grid = np.array([[[[[0.25, 0.5]], [[0.49999994, 1.0000002]]],
                             [[[1.25, 2.5000005]], [[1.5, 2.999999]]]],
                            [[[[2.2499986, 4.4999986]], [[-50.350002, -53.]]],
                             [[[-80.537506, -85.274994]], [[-76.5, -81.6]]]]]).astype(np.float32)
    [x_vmap, grid_vmap] = F.vmap(net, in_axes=(0, 0, 0))(grad, input_x, grid)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-6
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-6
    assert np.all(abs(x_vmap.asnumpy() - expect_x) < error_x)
    assert np.all(abs(grid_vmap.asnumpy() - expect_grid) < error_grid)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_vmap_grid_sampler_3d_grad():
    """
    Feature: GridSampler3DGrad GPU op vmap feature.
    Description: test the vmap feature of GridSampler3DGrad.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 3 batches
    grad = Tensor(np.arange(0, 2.4, 0.1).reshape(3, 2, 2, 2, 1, 1).astype(np.float32))
    input_x = Tensor(np.arange(0, 4.8, 0.05).reshape(3, 2, 2, 2, 2, 2).astype(np.float32))
    grid = Tensor(np.arange(-0.6, 3, 0.1).reshape(3, 2, 2, 1, 1, 3)).astype(np.float32)
    net = NetGridSampler3DGrad()
    expect_x = np.array([[[[[[0.02145, 0.01155], [0.0143, 0.0077]],
                            [[0.01755, 0.00945], [0.0117, 0.0063]]],
                           [[[0.14835002, 0.05565], [0.0709, 0.0301]],
                            [[0.08865, 0.03735], [0.0471, 0.0219]]]],
                          [[[[0.049125, 0.060375], [0.074625, 0.100875]],
                            [[0.093375, 0.12712501], [0.15787502, 0.23662502]]],
                           [[[0.072375, 0.08812499], [0.10887501, 0.145625]],
                            [[0.13612501, 0.18337502], [0.22762501, 0.337875]]]]],
                         [[[[[0.0024, 0.0096], [0.0136, 0.05440002]],
                            [[0.02159999, 0.08639999], [0.16515002, 1.3018501]]],
                           [[[0.003, 0.012], [0.017, 0.06800002]],
                            [[0.02699999, 0.10799998], [0.20525001, 1.60475]]]],
                          [[[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 1.1780249]]],
                           [[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 1.368675]]]]],
                         [[[[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.3711]]],
                           [[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.4167]]]],
                          [[[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.031575]]],
                           [[[0., 0.], [0., 0.]],
                            [[0., 0.], [0., 0.034725]]]]]]).astype(np.float32)
    expect_grid = np.array([[[[[[0.005, 0.01, 0.02]]],
                              [[[0.01, 0.02, 0.04]]]],
                             [[[[0.025, 0.04999997, 0.10000003]]],
                              [[[0.02999998, 0.05999997, 0.11999999]]]]],
                            [[[[[0.04499996, 0.08999997, 0.18000007]]],
                              [[[0.04749997, -2.059125, -2.1675]]]],
                             [[[[-2.6214, -2.7756, -2.949075]]],
                              [[[-1.88825, -2.023125, -2.1787503]]]]],
                            [[[[[-1.7586248, -1.9185001, -2.11035]]],
                              [[[-0.94780004, -1.0662752, -1.2186]]]],
                             [[[[-0.4787501, -0.5745, -0.71812487]]],
                              [[[-0.05014996, -0.07522491, -0.15044999]]]]]]).astype(np.float32)
    [x_vmap, grid_vmap] = F.vmap(net, in_axes=(0, 0, 0))(grad, input_x, grid)
    error_x = np.ones(shape=expect_x.shape) * 1.0e-6
    error_grid = np.ones(shape=expect_grid.shape) * 1.0e-6
    assert np.all(abs(x_vmap.asnumpy() - expect_x) < error_x)
    assert np.all(abs(grid_vmap.asnumpy() - expect_grid) < error_grid)
