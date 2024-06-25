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
def layer_norm_grad_forward_func(x, dy, variance, mean, gamma):
    return ops.auto_generate.LayerNormGrad(begin_norm_axis=1,
                                           begin_params_axis=1)(x,
                                                                dy,
                                                                variance,
                                                                mean,
                                                                gamma)


def layer_norm_grad_dyn_shape_func(x, dy, variance, mean, gamma):
    return ops.auto_generate.LayerNormGrad(begin_norm_axis=1,
                                           begin_params_axis=1)(x,
                                                                dy,
                                                                variance,
                                                                mean,
                                                                gamma
                                                                )


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op layer norm grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    dy = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    variance = ms.Tensor(np.array([[0.6666667], [0.6666667]]), ms.float32)
    mean = ms.Tensor(np.array([[2.], [2.]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    grad_x, grad_gamma, grad_beta = layer_norm_grad_forward_func(
        x, dy, variance, mean, gamma)
    expect_grad_x = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    expect_grad_gamma = np.array([-2.4495, 0.0000, 2.4495]).astype(np.float32)
    expect_grad_beta = np.array([2., 2., 2.]).astype(np.float32)
    assert np.allclose(grad_x.asnumpy(), expect_grad_x, rtol=1e-4, atol=1e-4)
    assert np.allclose(grad_gamma.asnumpy(),
                       expect_grad_gamma,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_beta.asnumpy(),
                       expect_grad_beta,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test layer norm grad op vmap.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    dy = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    variance = ms.Tensor(np.array([[0.6666667], [0.6666667]]), ms.float32)
    mean = ms.Tensor(np.array([[2.], [2.]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)

    batch_x = ops.stack((x, x))
    batch_dy = ops.stack((dy, dy))
    batch_variance = ops.stack((variance, variance))
    batch_mean = ops.stack((mean, mean))
    batch_gamma = ops.stack((gamma, gamma))

    layer_norm_grad_vmap = ops.vmap(layer_norm_grad_forward_func)
    batch_grad_x, batch_grad_gamma, batch_grad_beta = layer_norm_grad_vmap(
        batch_x, batch_dy, batch_variance, batch_mean, batch_gamma)
    expect_batch_grad_x = np.array([[[0., 0., 0.], [0., 0., 0.]],
                                    [[0., 0., 0.], [0., 0.,
                                                    0.]]]).astype(np.float32)
    expect_batch_grad_gamma = np.array([[-2.4494896, 0., 2.4494896],
                                        [-2.4494896, 0.,
                                         2.4494896]]).astype(np.float32)
    expect_batch_grad_beta = np.array([[2., 2., 2.], [2., 2.,
                                                      2.]]).astype(np.float32)
    assert np.allclose(batch_grad_x.asnumpy(),
                       expect_batch_grad_x,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(batch_grad_gamma.asnumpy(),
                       expect_batch_grad_gamma,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(batch_grad_beta.asnumpy(),
                       expect_batch_grad_beta,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_grad_dynamic_shape(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of layer norm grad.
    Description: test dynamic tensor and dynamic scalar of layer norm grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    variance_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    mean_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    gamma_dyn = ms.Tensor(shape=[None], dtype=ms.float32)

    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    dy = ms.Tensor(np.array([[1., 1., 1.], [1., 1., 1.]]), ms.float32)
    variance = ms.Tensor(np.array([[0.6666667], [0.6666667]]), ms.float32)
    mean = ms.Tensor(np.array([[2.], [2.]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)

    test_cell = test_utils.to_cell_obj(layer_norm_grad_dyn_shape_func)
    test_cell.set_inputs(x_dyn, dy_dyn, variance_dyn, mean_dyn, gamma_dyn)
    grad_x, grad_gamma, grad_beta = test_cell(x, dy, variance, mean, gamma)
    expect_grad_x = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    expect_grad_gamma = np.array([-2.4495, 0.0000, 2.4495]).astype(np.float32)
    expect_grad_beta = np.array([2., 2., 2.]).astype(np.float32)
    assert np.allclose(grad_x.asnumpy(), expect_grad_x, rtol=1e-4, atol=1e-4)
    assert np.allclose(grad_gamma.asnumpy(),
                       expect_grad_gamma,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_beta.asnumpy(),
                       expect_grad_beta,
                       rtol=1e-4,
                       atol=1e-4)

    x_2 = ms.Tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), ms.float32)
    dy_2 = ms.Tensor(np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]]),
                     ms.float32)
    variance_2 = ms.Tensor(np.array([[1.25], [1.25]]), ms.float32)
    mean_2 = ms.Tensor(np.array([[2.5], [2.5]]), ms.float32)
    gamma_2 = ms.Tensor(np.ones([4]), ms.float32)
    grad_x_2, grad_gamma_2, grad_beta_2 = test_cell(x_2, dy_2, variance_2,
                                                    mean_2, gamma_2)
    expect_grad_x_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float32)
    expect_grad_gamma_2 = np.array(
        [-2.6832814, -0.8944271, 0.8944271, 2.6832814]).astype(np.float32)
    expect_grad_beta_2 = np.array([2., 2., 2., 2.]).astype(np.float32)
    assert np.allclose(grad_x_2.asnumpy(),
                       expect_grad_x_2,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_gamma_2.asnumpy(),
                       expect_grad_gamma_2,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_beta_2.asnumpy(),
                       expect_grad_beta_2,
                       rtol=1e-4,
                       atol=1e-4)
