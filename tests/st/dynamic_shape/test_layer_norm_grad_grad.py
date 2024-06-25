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
def layer_norm_grad_grad_forward_func(x, dy, variance, mean, gamma, d_dx, d_dg, d_db):
    return ops.auto_generate.LayerNormGradGrad(begin_norm_axis=1,
                                               begin_params_axis=1)(x,
                                                                    dy,
                                                                    variance,
                                                                    mean,
                                                                    gamma,
                                                                    d_dx,
                                                                    d_dg,
                                                                    d_db)


def layer_norm_grad_grad_dyn_shape_func(x, dy, variance, mean, gamma, d_dx, d_dg, d_db):
    return ops.auto_generate.LayerNormGradGrad(begin_norm_axis=1,
                                               begin_params_axis=1)(x,
                                                                    dy,
                                                                    variance,
                                                                    mean,
                                                                    gamma,
                                                                    d_dx,
                                                                    d_dg,
                                                                    d_db
                                                                    )


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_grad_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op layer norm grad grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    dy = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    variance = ms.Tensor(np.array([[0.6666667], [0.6666667]]), ms.float32)
    mean = ms.Tensor(np.array([[2.], [2.]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    d_dx = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    d_dg = ms.Tensor(np.array([1., 1., 1.]), ms.float32)
    d_db = ms.Tensor(np.array([1., 1., 1.]), ms.float32)

    sopd_x, sopd_dy, sopd_gamma = layer_norm_grad_grad_forward_func(
        x, dy, variance, mean, gamma, d_dx, d_dg, d_db)
    expect_sopd_x = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)
    expect_sopd_dy = np.array([[-2.24744797e-01, 1.00000000e+00, 2.22474480e+00],
                               [-2.24744797e-01, 1.00000000e+00, 2.22474480e+00]]).astype(np.float32)
    expect_sopd_gamma = np.array(
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]).astype(np.float32)
    assert np.allclose(sopd_x.asnumpy(), expect_sopd_x, rtol=1e-4, atol=1e-4)
    assert np.allclose(sopd_dy.asnumpy(),
                       expect_sopd_dy,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(sopd_gamma.asnumpy(),
                       expect_sopd_gamma,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_grad_grad_dynamic_shape(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of layer norm grad grad.
    Description: test dynamic tensor and dynamic scalar of layer norm grad grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    variance_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    mean_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    gamma_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    d_dx_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    d_dg_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    d_db_dyn = ms.Tensor(shape=[None], dtype=ms.float32)

    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    dy = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    variance = ms.Tensor(np.array([[0.6666667], [0.6666667]]), ms.float32)
    mean = ms.Tensor(np.array([[2.], [2.]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    d_dx = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    d_dg = ms.Tensor(np.array([1., 1., 1.]), ms.float32)
    d_db = ms.Tensor(np.array([1., 1., 1.]), ms.float32)

    test_cell = test_utils.to_cell_obj(layer_norm_grad_grad_dyn_shape_func)
    test_cell.set_inputs(x_dyn, dy_dyn, variance_dyn,
                         mean_dyn, gamma_dyn, d_dx_dyn, d_dg_dyn, d_db_dyn)
    sopd_x, sopd_dy, sopd_gamma = test_cell(
        x, dy, variance, mean, gamma, d_dx, d_dg, d_db)
    expect_sopd_x = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]).astype(np.float32)
    expect_sopd_dy = np.array([[-2.24744797e-01, 1.00000000e+00, 2.22474480e+00],
                               [-2.24744797e-01, 1.00000000e+00, 2.22474480e+00]]).astype(np.float32)
    expect_sopd_gamma = np.array(
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]).astype(np.float32)
    assert np.allclose(sopd_x.asnumpy(), expect_sopd_x, rtol=1e-4, atol=1e-4)
    assert np.allclose(sopd_dy.asnumpy(),
                       expect_sopd_dy,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(sopd_gamma.asnumpy(),
                       expect_sopd_gamma,
                       rtol=1e-4,
                       atol=1e-4)
