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
from mindspore import Tensor
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def grid_sampler_3d_forward_func(input_x, grid):
    grid_sampler_3d = ops.GridSampler3D(
        interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)(input_x, grid)
    return grid_sampler_3d


@test_utils.run_with_cell
def grid_sampler_3d_backward_func(input_x, grid):
    return ops.grad(grid_sampler_3d_forward_func, (0, 1))(input_x, grid)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_3d_forward(mode):
    """
    Feature: Ops.
    Description: test op grid_sampler_3d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([[[[[0, 1],
                                  [2, 3]],
                                 [[4, 5],
                                  [6, 7]]],
                                [[[8, 9],
                                  [10, 11]],
                                 [[12, 13],
                                  [14, 15]]]],
                               [[[[16, 17],
                                  [18, 19]],
                                 [[20, 21],
                                  [22, 23]]],
                                [[[24, 25],
                                  [26, 27]],
                                 [[28, 29],
                                  [30, 31]]]]]).astype(np.float32))
    grid = Tensor(
        np.array([[[[[-2.00000000e-01, -1.00000000e-01, 0.00000000e+00]]],
                   [[[1.00000000e-01, 2.00000000e-01, 3.00000000e-01]]]],
                  [[[[4.00000000e-01, 5.00000000e-01, 6.00000000e-01]]],
                   [[[7.00000000e-01, 8.00000000e-01, 9.00000000e-01]]]]]).astype(np.float32))
    except_out = np.array([[[[[3.3000002]],
                             [[4.35]]],
                            [[[11.300001]],
                             [[12.349999]]]],
                           [[[[21.4]],
                             [[22.449997]]],
                            [[[29.399998]],
                             [[30.449999]]]]])

    out = grid_sampler_3d_forward_func(input_x, grid)
    assert np.allclose(out.asnumpy(), except_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_3d_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op grid_sampler_3d.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([[[[[0, 1],
                                  [2, 3]],
                                 [[4, 5],
                                  [6, 7]]],
                                [[[8, 9],
                                  [10, 11]],
                                 [[12, 13],
                                  [14, 15]]]],
                               [[[[16, 17],
                                  [18, 19]],
                                 [[20, 21],
                                  [22, 23]]],
                                [[[24, 25],
                                  [26, 27]],
                                 [[28, 29],
                                  [30, 31]]]]]).astype(np.float32))
    grid = Tensor(
        np.array([[[[[-2.00000000e-01, -1.00000000e-01, 0.00000000e+00]]],
                   [[[1.00000000e-01, 2.00000000e-01, 3.00000000e-01]]]],
                  [[[[4.00000000e-01, 5.00000000e-01, 6.00000000e-01]]],
                   [[[7.00000000e-01, 8.00000000e-01, 9.00000000e-01]]]]]).astype(np.float32))
    expect_out1 = np.array([[[[[2.28000000e-01, 1.87000006e-01],
                               [2.29500026e-01, 2.05500007e-01]],
                              [[2.82000005e-01, 2.52999991e-01],
                               [3.10500026e-01, 3.04499984e-01]]],
                             [[[2.28000000e-01, 1.87000006e-01],
                               [2.29500026e-01, 2.05500007e-01]],
                              [[2.82000005e-01, 2.52999991e-01],
                               [3.10500026e-01, 3.04499984e-01]]]],
                            [[[[1.57500003e-02, 3.92499976e-02],
                               [5.17499968e-02, 1.43249989e-01]],
                              [[7.42500052e-02, 2.20750019e-01],
                               [3.08249980e-01, 1.14674997e+00]]],
                             [[[1.57500003e-02, 3.92499976e-02],
                               [5.17499968e-02, 1.43249989e-01]],
                              [[7.42500052e-02, 2.20750019e-01],
                               [3.08249980e-01, 1.14674997e+00]]]]])
    expect_out2 = np.array([[[[[1.00000000e+00, 1.99999988e+00, 4.00000000e+00]]],
                             [[[1.00000000e+00, 2.00000000e+00, 4.00000000e+00]]]],
                            [[[[1.00000095e+00, 2.00000000e+00, 3.99999952e+00]]],
                             [[[1.00000000e+00, 1.99999905e+00, 4.00000095e+00]]]]])
    grads = grid_sampler_3d_backward_func(input_x, grid)
    assert np.allclose(grads[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(grads[1].asnumpy(), expect_out2, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_3d_vmap(mode):
    """
    Feature: test vmap function.
    Description: test grid_sampler_3d op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0)
    input_x = Tensor(np.array([[[[[[[0.00000000e+00, 1.00000000e+00],
                                    [2.00000000e+00, 3.00000000e+00]],
                                   [[4.00000000e+00, 5.00000000e+00],
                                    [6.00000000e+00, 7.00000000e+00]]],
                                  [[[8.00000000e+00, 9.00000000e+00],
                                    [1.00000000e+01, 1.10000000e+01]],
                                   [[1.20000000e+01, 1.30000000e+01],
                                    [1.40000000e+01, 1.50000000e+01]]]],
                                 [[[[1.60000000e+01, 1.70000000e+01],
                                    [1.80000000e+01, 1.90000000e+01]],
                                   [[2.00000000e+01, 2.10000000e+01],
                                    [2.20000000e+01, 2.30000000e+01]]],
                                  [[[2.40000000e+01, 2.50000000e+01],
                                    [2.60000000e+01, 2.70000000e+01]],
                                   [[2.80000000e+01, 2.90000000e+01],
                                    [3.00000000e+01, 3.10000000e+01]]]]]]]).astype(np.float32))
    grid = Tensor(
        np.array([[[[[[[-2.00000003e-01, -1.00000001e-01, 0.00000000e+00]]],
                     [[[1.00000001e-01, 2.00000003e-01, 3.00000012e-01]]]],
                    [[[[4.00000006e-01, 5.00000000e-01, 6.00000024e-01]]],
                     [[[6.99999988e-01, 8.00000012e-01, 8.99999976e-01]]]]]]]).astype(np.float32))
    except_out = np.array([[[[[[[3.3000002]],
                               [[4.35]]],
                              [[[11.300001]],
                               [[12.349999]]]],
                             [[[[21.4]],
                               [[22.449997]]],
                              [[[29.399998]],
                               [[30.449999]]]]]]])
    grid_sampler_3d_forward_vmap = ops.vmap(grid_sampler_3d_forward_func,
                                            in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sampler_3d_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(input_x, grid)
    assert np.allclose(out.asnumpy(), except_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_grid_sampler_3d_diferent_input_types(mode):
    """
    Feature: Ops.
    Description: test op grid_sampler_3d.
    Expectation: catch the error.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.ones((2, 3, 3, 3, 3)).astype(np.float32))
    grid = Tensor(np.ones((2, 3, 3, 3, 3)).astype(np.int32))
    with pytest.raises(TypeError) as info:
        _ = grid_sampler_3d_forward_func(input_x, grid)
    assert "input type must be same" in str(info.value)
