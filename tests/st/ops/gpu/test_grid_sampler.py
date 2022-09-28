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
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore.ops.operations.nn_ops import GridSampler2D, GridSampler3D


class Net2D(Cell):
    def __init__(self, mode, padding_mode, align_corners):
        super(Net2D, self).__init__()
        self.grid_sampler_2d = GridSampler2D(mode, padding_mode, align_corners)

    def construct(self, x0, x1):
        return self.grid_sampler_2d(x0, x1)


class Net3D(Cell):
    def __init__(self, mode, padding_mode, align_corners):
        super(Net3D, self).__init__()
        self.grid_sampler_3d = GridSampler3D(mode, padding_mode, align_corners)

    def construct(self, x0, x1):
        return self.grid_sampler_3d(x0, x1)


def run_net2d(dtype):
    input_np = np.arange(16).reshape((2, 2, 2, 2))
    grid_np = np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2))

    if dtype == np.float16:
        in_tensor = Tensor(input_np, ms.float16)
        grid = Tensor(grid_np, ms.float16)
        expect_out = np.array([[[[1.9], [2.2]], [[5.9], [6.2]]],
                               [[[10.5], [10.8]], [[14.5], [14.8]]]], np.float16)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-3
    elif dtype == np.float32:
        in_tensor = Tensor(input_np, ms.float32)
        grid = Tensor(grid_np, ms.float32)
        expect_out = np.array([[[[1.9], [2.1999998]], [[5.9], [6.2]]],
                               [[[10.5], [10.8]], [[14.5], [14.8]]]], np.float32)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-6
    elif dtype == np.float64:
        in_tensor = Tensor(input_np, ms.float64)
        grid = Tensor(grid_np, ms.float64)
        expect_out = np.array([[[[1.9], [2.2]], [[5.9], [6.2]]],
                               [[[10.5], [10.8]], [[14.5], [14.8]]]], np.float64)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-6

    net = Net2D('bilinear', 'zeros', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


def run_net3d(dtype):
    input_np = np.arange(32).reshape((2, 2, 2, 2, 2))
    grid_np = np.arange(-0.2, 1, 0.1).reshape((2, 2, 1, 1, 3))

    if dtype == np.float16:
        in_tensor = Tensor(input_np, ms.float16)
        grid = Tensor(grid_np, ms.float16)
        expect_out = np.array([[[[[3.3]], [[4.35]]], [[[11.3]], [[12.35]]]],
                               [[[[21.4]], [[22.45]]], [[[29.4]], [[30.45]]]]],
                              np.float16)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-3
    elif dtype == np.float32:
        in_tensor = Tensor(input_np, ms.float32)
        grid = Tensor(grid_np, ms.float32)
        expect_out = np.array([[[[[3.3]], [[4.35]]], [[[11.300001]], [[12.349999]]]],
                               [[[[21.4]], [[22.449999]]], [[[29.4]], [[30.449999]]]]],
                              np.float32)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-6
    elif dtype == np.float64:
        in_tensor = Tensor(input_np, ms.float64)
        grid = Tensor(grid_np, ms.float64)
        expect_out = np.array([[[[[3.3]], [[4.35]]], [[[11.3]], [[12.35]]]],
                               [[[[21.4]], [[22.45]]], [[[29.4]], [[30.45]]]]],
                              np.float64)
        error_out = np.ones(shape=expect_out.shape) * 1.0e-6
    net = Net3D('bilinear', 'zeros', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler2d():
    """
    Feature: GridSampler2D op.
    Description: test data type is float16, float32 and float64 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_net2d(np.float16)
    run_net2d(np.float32)
    run_net2d(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler3d():
    """
    Feature: GridSampler3D op.
    Description: test data type is float16, float32 and float64 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_net3d(np.float16)
    run_net3d(np.float32)
    run_net3d(np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler2d_neg_input():
    """
    Feature: GridSampler2D op.
    Description: test data type is float32 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_np = np.array([[[[1.3315865, 0.715279], [-1.5454003, -0.00838385]],
                          [[0.621336, -0.72008556], [0.26551157, 0.10854852]]],
                         [[[0.00429143, -0.17460021], [0.4330262, 1.2030374]],
                          [[-0.96506566, 1.028274], [0.22863013, 0.44513762]]]])
    grid_np = np.array([[[[-1.1366022, 0.13513687]], [[1.484537, -1.0798049]]],
                        [[[-1.9777282, -1.7433723]], [[0.26607016, 2.3849673]]]])
    in_tensor = Tensor(input_np, ms.float32)
    grid = Tensor(grid_np, ms.float32)
    expect_out = np.array([[[[-0.2807212], [0.5203627]],
                            [[0.3907371], [-0.52385944]]],
                           [[[0.00137821], [0.28305963]],
                            [[-0.30993438], [0.11245471]]]], np.float32)
    error_out = np.ones(shape=expect_out.shape) * 1.0e-6

    net = Net2D('bilinear', 'zeros', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler3d_neg_input():
    """
    Feature: GridSampler3D op.
    Description: test data type is float32 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_np = np.array([[[[[1.7640524, 0.4001572], [0.978738, 2.2408931]],
                           [[1.867558, -0.9772779], [0.95008844, -0.1513572]]],
                          [[[-0.10321885, 0.41059852], [0.14404356, 1.4542735]],
                           [[0.7610377, 0.12167501], [0.44386324, 0.33367434]]]],
                         [[[[1.4940791, -0.20515826], [0.3130677, -0.85409576]],
                           [[-2.5529897, 0.6536186], [0.8644362, -0.742165]]],
                          [[[2.2697546, -1.4543657], [0.04575852, -0.18718386]],
                           [[1.5327792, 1.4693588], [0.15494743, 0.37816253]]]]])
    grid_np = np.array([[[[[-0.88778573, -1.9807965, -0.34791216]]],
                         [[[0.15634897, 1.2302907, 1.2023798]]]],
                        [[[[-0.3873268, -0.30230275, -1.048553]]],
                         [[[-1.420018, -1.7062702, 1.9507754]]]]])
    in_tensor = Tensor(input_np, ms.float32)
    grid = Tensor(grid_np, ms.float32)
    expect_out = np.array([[[[[0.8633592]], [[0.24914137]]], [[[0.0949388]], [[0.30234334]]]],
                           [[[[0.6033937]], [[-0.68442094]]], [[[0.70853865]], [[0.41091672]]]]], np.float32)
    error_out = np.ones(shape=expect_out.shape) * 1.0e-6

    net = Net3D('bilinear', 'zeros', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_grid_sampler_2d():
    """
    Feature: GridSampler2D GPU op vmap feature.
    Description: test the vmap feature of GridSampler2D.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 2 batches
    input_x_np = np.array([[[[[0.5215106, 0.5384331], [0.7693346, 0.4739236]],
                             [[0.5391795, 0.11520209], [0.9797081, 0.24777432]]],
                            [[[0.45743152, 0.5464221], [0.9907652, 0.30161732]],
                             [[0.5704465, 0.90900564], [0.50228757, 0.9520584]]]],
                           [[[[0.60979515, 0.7289113], [0.42028105, 0.5241496]],
                             [[0.42288917, 0.8583189], [0.98225504, 0.7050427]]],
                            [[[0.40967953, 0.74207], [0.37422016, 0.56561804]],
                             [[0.18698247, 0.4010499], [0.33587858, 0.46725345]]]]])
    grid_np = np.array([[[[[0.4249997, 0.4681669], [0.88895506, 0.2020226], [0.6742403, 0.6730695]],
                          [[0.02522387, 0.8492109], [0.39516452, 0.7882344], [0.93104833, 0.98147726]],
                          [[0.14475864, 0.3889179], [0.18994962, 0.9868513], [0.9282293, 0.8602506]]],
                         [[[0.02953037, 0.2053605], [0.80233335, 0.97457963], [0.10763288, 0.6334362]],
                          [[0.21988729, 0.2194931], [0.06795477, 0.65087706], [0.92213994, 0.34084418]],
                          [[0.3268782, 0.5166993], [0.56500953, 0.9950389], [0.8994754, 0.5184141]]]],
                        [[[[0.1166239, 0.9787158], [0.15663764, 0.13361669], [0.05010754, 0.73372984]],
                          [[0.08599498, 0.57037], [0.28230652, 0.9017407], [0.7048581, 0.6080519]],
                          [[0.557095, 0.97886], [0.06154401, 0.8355557], [0.11878787, 0.7070594]]],
                         [[[0.3572435, 0.49456784], [0.19790882, 0.15691541], [0.19227573, 0.05957435]],
                          [[0.6333749, 0.2439932], [0.31420627, 0.2035074], [0.04072937, 0.09606935]],
                          [[0.09063079, 0.6687324], [0.8992046, 0.7573358], [0.85066617, 0.70971775]]]]])
    input_x = Tensor(input_x_np.astype(np.float32))
    grid = Tensor(grid_np.astype(np.float32))
    net = Net2D('bilinear', 'zeros', True)
    expect = np.array([[[[[0.5521302, 0.509145, 0.5242692],
                          [0.6112899, 0.56009036, 0.48460585],
                          [0.5791496, 0.5931649, 0.4882489]],
                         [[0.39940864, 0.22869614, 0.33712122],
                          [0.5831986, 0.44522595, 0.2716822],
                          [0.48002183, 0.5425343, 0.26400435]]],
                        [[[0.58326167, 0.3718621, 0.59033793],
                          [0.54751086, 0.6022081, 0.39914426],
                          [0.52942866, 0.45169112, 0.38578513]],
                         [[0.7381491, 0.9071985, 0.75258183],
                          [0.7767497, 0.74398667, 0.9217864],
                          [0.7993242, 0.85418856, 0.9204311]]]],
                       [[[[0.48037952, 0.566266, 0.5011144],
                          [0.5191704, 0.49666777, 0.54850864],
                          [0.5032763, 0.49165925, 0.507392]],
                         [[0.8257655, 0.75815845, 0.8120483],
                          [0.79469466, 0.799486, 0.7553797],
                          [0.76638407, 0.8202268, 0.8036437]]],
                        [[[0.53724813, 0.53940487, 0.544515],
                          [0.5874621, 0.5510061, 0.52300286],
                          [0.4972005, 0.57651913, 0.5754094]],
                         [[0.40158567, 0.37267873, 0.36736295],
                          [0.41241565, 0.38454783, 0.35639375],
                          [0.3903261, 0.4520942, 0.44693908]]]]])
    out_vmap = F.vmap(net, in_axes=(0, 0))(input_x, grid)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_grid_sampler_3d():
    """
    Feature: GridSampler3D GPU op vmap feature.
    Description: test the vmap feature of GridSampler3D.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # 3 batches
    input_x = Tensor(np.arange(24).reshape((3, 2, 1, 1, 2, 2)).astype(np.float32))
    grid = Tensor(np.arange(-1, 0.8, 0.1).reshape((3, 2, 1, 1, 1, 3)).astype(np.float32))
    net = Net3D('bilinear', 'zeros', True)
    expect = np.array([[[[[[0.10000002]]]], [[[[4.55]]]]],
                       [[[[[9]]]], [[[[13.45]]]]],
                       [[[[[17.9]]]], [[[[22.35]]]]]])
    out_vmap = F.vmap(net, in_axes=(0, 0))(input_x, grid)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
