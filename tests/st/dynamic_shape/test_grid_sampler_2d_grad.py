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
def grid_sampler_2d_grad_forward_func(grad, input_x, grid):
    grid_sampler_2d_grad = ops.auto_generate.GridSampler2DGrad(
        interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)(grad, input_x, grid)
    return grid_sampler_2d_grad


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_2d_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op grid_sampler_2d_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    grad = ms.Tensor(np.array([[[[0, 0, 0],
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
                                 [0, 0, 0]]]]).astype(np.float32))
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
    expect_out1 = np.array([[[[3.75000000e-01, 0.00000000e+00],
                              [1.25000000e-01, 0.00000000e+00]],
                             [[3.93750000e+00, 0.00000000e+00],
                              [1.12500000e+00, 0.00000000e+00]]],
                            [[[1.25000000e+00, 1.25000000e+00],
                              [3.75000000e+00, 1.01093750e+01]],
                             [[1.75000000e+00, 1.75000000e+00],
                              [5.25000000e+00, 1.39218750e+01]]]])
    expect_out2 = np.array([[[[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [2.25000000e+00, 1.50000000e+00],
                              [2.50000000e+00, 5.00000000e+00]]],
                            [[[1.20000000e+01, 2.40000000e+01],
                              [-9.73125000e+01, -1.29750000e+02],
                              [-5.40625000e+00, -1.08125000e+01]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]],
                             [[0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00]]]])
    out = grid_sampler_2d_grad_forward_func(grad, input_x, grid)
    assert (expect_out1 == out[0].asnumpy()).all()
    assert (expect_out2 == out[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_2d_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test grid_sampler_2d op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0, 0)
    grad = ms.Tensor(np.array([[[[[[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0.5]],
                                  [[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 1.5, 4.5]]],
                                 [[[10, 8.25, 1.375],
                                   [0, 0, 0.],
                                   [0, 0, 0.]],
                                  [[14, 11.25, 1.875],
                                   [0., 0, 0.],
                                   [0, 0, 0.]]]]]]).astype(np.float32))
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
    expect_out1 = np.array([[[[[[3.75000000e-01, 0.00000000e+00],
                                [1.25000000e-01, 0.00000000e+00]],
                               [[3.93750000e+00, 0.00000000e+00],
                                [1.12500000e+00, 0.00000000e+00]]],
                              [[[1.25000000e+00, 1.25000000e+00],
                                [3.75000000e+00, 1.01093750e+01]],
                               [[1.75000000e+00, 1.75000000e+00],
                                [5.25000000e+00, 1.39218750e+01]]]]]])
    expect_out2 = np.array([[[[[[0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00]],
                               [[0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00]],
                               [[0.00000000e+00, 0.00000000e+00],
                                [2.25000000e+00, 1.50000000e+00],
                                [2.50000000e+00, 5.00000000e+00]]],
                              [[[1.20000000e+01, 2.40000000e+01],
                                [-9.73125000e+01, -1.29750000e+02],
                                [-5.40625000e+00, -1.08125000e+01]],
                               [[0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00]],
                               [[0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 0.00000000e+00]]]]]])
    grid_sampler_2d_forward_vmap = ops.vmap(grid_sampler_2d_grad_forward_func,
                                            in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sampler_2d_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(grad, input_x, grid)
    assert (expect_out1 == out[0].asnumpy()).all()
    assert (expect_out2 == out[1].asnumpy()).all()
