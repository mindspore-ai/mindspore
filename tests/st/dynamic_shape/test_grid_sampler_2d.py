# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def grid_sampler_2d_forward_func(input_x, grid):
    grid_sampler_2d = ops.GridSampler2D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)(
        input_x, grid)
    return grid_sampler_2d


@test_utils.run_with_cell
def grid_sampler_2d_backward_func(input_x, grid):
    return ops.grad(grid_sampler_2d_forward_func, (0, 1))(input_x, grid)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_2d_forward(mode):
    """
    Feature: Ops.
    Description: test op grid_sampler_2d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[[[0, 1],
                                    [2, 3]],
                                   [[4, 5],
                                    [6, 7]]],
                                  [[[8, 9],
                                    [10, 11]],
                                   [[12, 13],
                                    [14, 15]]]]).astype(np.float32))

    grid = ms.Tensor(np.array([[[[-9.00000000e+00, -8.50000000e+00],
                                 [-8.00000000e+00, -7.50000000e+00],
                                 [-7.00000000e+00, -6.50000000e+00]],
                                [[-6.00000000e+00, -5.50000000e+00],
                                 [-5.00000000e+00, -4.50000000e+00],
                                 [-4.00000000e+00, -3.50000000e+00]],
                                [[-3.00000000e+00, -2.50000000e+00],
                                 [-2.00000000e+00, -1.50000000e+00],
                                 [-1.00000000e+00, -5.00000000e-01]]],
                               [[[0.00000000e+00, 5.00000000e-01],
                                 [1.00000000e+00, 1.50000000e+00],
                                 [2.00000000e+00, 2.50000000e+00]],
                                [[3.00000000e+00, 3.50000000e+00],
                                 [4.00000000e+00, 4.50000000e+00],
                                 [5.00000000e+00, 5.50000000e+00]],
                                [[6.00000000e+00, 6.50000000e+00],
                                 [7.00000000e+00, 7.50000000e+00],
                                 [8.00000000e+00, 8.50000000e+00]]]]).astype(np.float32))
    expect_out = np.array([[[[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0.5]],
                            [[0, 0, 0],
                             [0, 0, 0],
                             [0, 1.5, 4.5]]],
                           [[[10, 8.25, 1.375],
                             [0, 0, 0],
                             [0, 0, 0]],
                            [[14, 11.25, 1.875],
                             [0, 0, 0],
                             [0, 0, 0]]]])

    out = grid_sampler_2d_forward_func(input_x, grid)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_2d_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op grid_sampler_2d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[[[0, 1],
                                    [2, 3]],
                                   [[4, 5],
                                    [6, 7]]],
                                  [[[8, 9],
                                    [10, 11]],
                                   [[12, 13],
                                    [14, 15]]]]).astype(np.float32))

    grid = ms.Tensor(np.array([[[[-9.00000000e+00, -8.50000000e+00],
                                 [-8.00000000e+00, -7.50000000e+00],
                                 [-7.00000000e+00, -6.50000000e+00]],
                                [[-6.00000000e+00, -5.50000000e+00],
                                 [-5.00000000e+00, -4.50000000e+00],
                                 [-4.00000000e+00, -3.50000000e+00]],
                                [[-3.00000000e+00, -2.50000000e+00],
                                 [-2.00000000e+00, -1.50000000e+00],
                                 [-1.00000000e+00, -5.00000000e-01]]],
                               [[[0.00000000e+00, 5.00000000e-01],
                                 [1.00000000e+00, 1.50000000e+00],
                                 [2.00000000e+00, 2.50000000e+00]],
                                [[3.00000000e+00, 3.50000000e+00],
                                 [4.00000000e+00, 4.50000000e+00],
                                 [5.00000000e+00, 5.50000000e+00]],
                                [[6.00000000e+00, 6.50000000e+00],
                                 [7.00000000e+00, 7.50000000e+00],
                                 [8.00000000e+00, 8.50000000e+00]]]]).astype(np.float32))
    expect_out1 = np.array([[[[1.12500000e+00, 0.00000000e+00],
                              [2.50000000e-01, 0.00000000e+00]],
                             [[1.12500000e+00, 0.00000000e+00],
                              [2.50000000e-01, 0.00000000e+00]]],
                            [[[1.25000000e-01, 1.25000000e-01],
                              [3.75000000e-01, 1.25000000e+00]],
                             [[1.25000000e-01, 1.25000000e-01],
                              [3.75000000e-01, 1.25000000e+00]]]])
    expect_out2 = np.array([[[[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[5.00000000e-01, 0.00000000e+00],
                              [1.50000000e+00, 1.00000000e+00],
                              [1.00000000e+00, 2.00000000e+00]]],
                            [[[1.00000000e+00, 2.00000000e+00],
                              [-9.75000000e+00, -1.30000000e+01],
                              [-3.25000000e+00, -6.50000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]]]])
    grads = grid_sampler_2d_backward_func(input_x, grid)
    assert np.allclose(grads[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(grads[1].asnumpy(), expect_out2, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_2d_vmap(mode):
    """
    Feature: test vmap function.
    Description: test grid_sampler_2d op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[[[[[0.00000000e+00, 1.00000000e+00],
                                      [2.00000000e+00, 3.00000000e+00]],
                                     [[4.00000000e+00, 5.00000000e+00],
                                      [6.00000000e+00, 7.00000000e+00]]],
                                    [[[8.00000000e+00, 9.00000000e+00],
                                      [1.00000000e+01, 1.10000000e+01]],
                                     [[1.20000000e+01, 1.30000000e+01],
                                      [1.40000000e+01, 1.50000000e+01]]]]]]).astype(np.float32))
    grid = ms.Tensor(np.array([[[[[[-9.00000000e+00, -8.50000000e+00],
                                   [-8.00000000e+00, -7.50000000e+00],
                                   [-7.00000000e+00, -6.50000000e+00]],
                                  [[-6.00000000e+00, -5.50000000e+00],
                                   [-5.00000000e+00, -4.50000000e+00],
                                   [-4.00000000e+00, -3.50000000e+00]],
                                  [[-3.00000000e+00, -2.50000000e+00],
                                   [-2.00000000e+00, -1.50000000e+00],
                                   [-1.00000000e+00, -5.00000000e-01]]],
                                 [[[0.00000000e+00, 5.00000000e-01],
                                   [1.00000000e+00, 1.50000000e+00],
                                   [2.00000000e+00, 2.50000000e+00]],
                                  [[3.00000000e+00, 3.50000000e+00],
                                   [4.00000000e+00, 4.50000000e+00],
                                   [5.00000000e+00, 5.50000000e+00]],
                                  [[6.00000000e+00, 6.50000000e+00],
                                   [7.00000000e+00, 7.50000000e+00],
                                   [8.00000000e+00, 8.50000000e+00]]]]]]).astype(np.float32))

    expect_out = np.array([[[[[[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0.5]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 1.5, 4.5]]],
                             [[[10, 8.25, 1.375],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[14, 11.25, 1.875],
                               [0, 0, 0],
                               [0, 0, 0]]]]]])

    in_axes = (0, 0)
    grid_sampler_2d_forward_vmap = ops.vmap(grid_sampler_2d_forward_func,
                                            in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sampler_2d_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(input_x, grid)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)
