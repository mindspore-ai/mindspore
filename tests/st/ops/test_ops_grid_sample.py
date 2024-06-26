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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import ops
import mindspore as ms


@test_utils.run_with_cell
def grid_sample_forward_func(input_x, grid, mode="bilinear", padding_mode="zeros", align_corners=True):
    return ops.grid_sample(input_x, grid, mode, padding_mode, align_corners)


@test_utils.run_with_cell
def grid_sample_backward_func(input_x, grid, mode="bilinear", padding_mode="zeros", align_corners=True):
    return ops.grad(grid_sample_forward_func, (0, 1))(input_x, grid, mode, padding_mode, align_corners)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
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

    out = grid_sample_forward_func(input_x, grid)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_backward(mode):
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
    grads = grid_sample_backward_func(input_x, grid)
    assert np.allclose(grads[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(grads[1].asnumpy(), expect_out2, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_grid_sampler_vmap(mode):
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
    grid_sample_forward_vmap = ops.vmap(grid_sample_forward_func,
                                        in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sample_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(input_x, grid)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_5d_forward(mode):
    """
    Feature: Ops.
    Description: test ops grid_sample with 5d input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[[[[0, 1],
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
    grid = ms.Tensor(
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

    out = grid_sample_forward_func(input_x, grid)
    assert np.allclose(out.asnumpy(), except_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_5d_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of ops grid_sample with 5d input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[[[[0, 1],
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
    grid = ms.Tensor(
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
    grads = grid_sample_backward_func(input_x, grid)
    assert np.allclose(grads[0].asnumpy(), expect_out1, 1e-04, 1e-04)
    assert np.allclose(grads[1].asnumpy(), expect_out2, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_grid_sampler_5d_vmap(mode):
    """
    Feature: test vmap function.
    Description: test grid_sample ops with 5d input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0)
    input_x = ms.Tensor(np.array([[[[[[[0.00000000e+00, 1.00000000e+00],
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
    grid = ms.Tensor(
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
    grid_sample_forward_vmap = ops.vmap(grid_sample_forward_func,
                                        in_axes=in_axes, out_axes=0)
    nest_vmap = ops.vmap(grid_sample_forward_vmap,
                         in_axes=in_axes, out_axes=0)
    out = nest_vmap(input_x, grid)
    assert np.allclose(out.asnumpy(), except_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_gridsample_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test ops grid_sample dynamic feature with 5d input.
    Expectation: expect correct result.
    """
    ms_data1 = [ms.Tensor(np.random.randn(6, 4, 5, 6).astype(np.float32)),
                ms.Tensor(np.random.randn(6, 6, 6, 2).astype(np.float32))]
    ms_data2 = [ms.Tensor(np.random.randn(4, 2, 2, 3, 4).astype(np.float32)),
                ms.Tensor(np.random.randn(4, 4, 4, 4, 3).astype(np.float32))]
    TEST_OP(grid_sample_forward_func,
            [ms_data1, ms_data2], '', disable_yaml_check=True)
