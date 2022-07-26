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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
