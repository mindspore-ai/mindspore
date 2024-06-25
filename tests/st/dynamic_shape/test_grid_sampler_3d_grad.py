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
def grid_sampler_3d_grad_forward_func(grad, input_x, grid):
    grid_sampler_3d_grad = ops.auto_generate.GridSampler3DGrad(
        interpolation_mode='bilinear', padding_mode='zeros', align_corners=True)(grad, input_x, grid)
    return grid_sampler_3d_grad


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_3d_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op grid_sampler_3d_grad.
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
    grad = Tensor(np.array([[[[[3.3000002]],
                              [[4.35]]],
                             [[[11.300001]],
                              [[12.349999]]]],
                            [[[[21.4]],
                              [[22.449997]]],
                             [[[29.399998]],
                              [[30.449999]]]]]).astype(np.float32))
    expect_out1 = np.array([[[[[8.18549991e-01, 6.97950065e-01],
                               [8.56575131e-01, 7.99425006e-01]],
                              [[1.05344999e+00, 9.85049963e-01],
                               [1.20892501e+00, 1.23007488e+00]]],
                             [[[2.64255023e+00, 2.19395018e+00],
                               [2.69257545e+00, 2.44342518e+00]],
                              [[3.30945015e+00, 3.00904989e+00],
                               [3.69292498e+00, 3.66607475e+00]]]],
                            [[[[3.37837487e-01, 8.44412446e-01],
                               [1.11453748e+00, 3.10571241e+00]],
                              [[1.60391247e+00, 4.80883789e+00],
                               [6.73121166e+00, 2.53035355e+01]]],
                             [[[4.63837475e-01, 1.15841234e+00],
                               [1.52853727e+00, 4.25171185e+00]],
                              [[2.19791245e+00, 6.57483768e+00],
                               [9.19721127e+00, 3.44775352e+01]]]]])
    expect_out2 = np.array([[[[[7.30000210e+00, 1.45999985e+01, 2.92000084e+01]]],
                             [[[8.34999847e+00, 1.66999989e+01, 3.33999939e+01]]]],
                            [[[[2.54000244e+01, 5.07999573e+01, 1.01599991e+02]]],
                             [[[2.64499817e+01, 5.28999634e+01, 1.05799973e+02]]]]])
    out = grid_sampler_3d_grad_forward_func(grad, input_x, grid)
    assert np.allclose(out[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(out[1].asnumpy(), expect_out2, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_3d_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test grid_sampler_3d_grad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0, 0)
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
    grad = Tensor(np.array(
        [[[[[[[0.00000000e+00]],
             [[1.00000000e+00]]],
            [[[2.00000000e+00]],
             [[3.00000000e+00]]]],
           [[[[4.00000000e+00]],
             [[5.00000000e+00]]],
            [[[6.00000000e+00]],
             [[7.00000000e+00]]]]]]]).astype(np.float32))
    expect_out1 = np.array([[[[[[[6.29999936e-02, 7.70000070e-02],
                                 [9.45000127e-02, 1.15500011e-01]],
                                [[1.16999984e-01, 1.42999992e-01],
                                 [1.75500005e-01, 2.14499995e-01]]],
                               [[[5.18999994e-01, 4.51000035e-01],
                                 [5.53500056e-01, 5.26500046e-01]],
                                [[6.80999994e-01, 6.48999989e-01],
                                 [7.96499968e-01, 8.23499978e-01]]]],
                              [[[[6.37499988e-02, 1.61249995e-01],
                                 [2.13749990e-01, 6.11249983e-01]],
                                [[3.11250031e-01, 9.63750124e-01],
                                 [1.36124992e+00, 5.31374979e+00]]],
                               [[[9.52499956e-02, 2.39749998e-01],
                                 [3.17249984e-01, 8.97750020e-01]],
                                [[4.59750026e-01, 1.40525019e+00],
                                 [1.97774982e+00, 7.60724974e+00]]]]]]])
    expect_out2 = np.array([[[[[[[1.00000000e+00, 1.99999952e+00, 4.00000000e+00]]],
                               [[[1.99999952e+00, 4.00000095e+00, 7.99999809e+00]]]],
                              [[[[5.00000763e+00, 9.99999237e+00, 2.00000000e+01]]],
                               [[[5.99999237e+00, 1.19999847e+01, 2.40000000e+01]]]]]]])
    grid_sampler_3d_grad_forward_vmap = ops.vmap(grid_sampler_3d_grad_forward_func,
                                                 in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sampler_3d_grad_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(grad, input_x, grid)
    assert np.allclose(out[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(out[1].asnumpy(), expect_out2, 1e-04, 1e-04)
