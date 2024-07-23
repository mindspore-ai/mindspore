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
def layer_norm_forward_func(input_x, gamma, beta):
    return ops.LayerNorm(begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7)(input_x, gamma, beta)


@test_utils.run_with_cell
def layer_norm_backward_func(input_x, gamma, beta):
    return ops.grad(layer_norm_forward_func, (0, 1, 2))(input_x, gamma, beta)


def layer_norm_dyn_shape_func(input_x, gamma, beta):
    return ops.LayerNorm(begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7)(input_x, gamma, beta)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_forward(mode):
    """
    Feature: Ops.
    Description: test op layer norm.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    beta = ms.Tensor(np.zeros([3]), ms.float32)
    output, mean, variance = layer_norm_forward_func(input_x, gamma, beta)
    expect_output = np.array([[-1.2247, 0.0000, 1.2247],
                              [-1.2247, 0.0000, 1.2247]]).astype(np.float32)
    expect_mean = np.array([[2.], [2.]]).astype(np.float32)
    expect_variance = np.array([[0.6666667], [0.6666667]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    assert np.allclose(mean.asnumpy(), expect_mean, rtol=1e-4, atol=1e-4)
    assert np.allclose(variance.asnumpy(),
                       expect_variance,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op layer norm.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    beta = ms.Tensor(np.zeros([3]), ms.float32)
    grad_input_x, grad_gamma, grad_beta = layer_norm_backward_func(
        input_x, gamma, beta)
    expect_grad_input_x = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    expect_grad_gamma = np.array([-2.4495, 0.0000, 2.4495]).astype(np.float32)
    expect_grad_beta = np.array([2., 2., 2.]).astype(np.float32)
    assert np.allclose(grad_input_x.asnumpy(),
                       expect_grad_input_x,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_gamma.asnumpy(),
                       expect_grad_gamma,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(grad_beta.asnumpy(),
                       expect_grad_beta,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_vmap(mode):
    """
    Feature: test vmap function.
    Description: test layer norm op vmap.
    Expectation: expect correct result.
    """
    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    beta = ms.Tensor(np.zeros([3]), ms.float32)

    batch_input_x = ops.stack((input_x, input_x))
    batch_gamma = ops.stack((gamma, gamma))
    batch_beta = ops.stack((beta, beta))
    layer_norm_vmap = ops.vmap(layer_norm_forward_func)
    outputs, means, variances = layer_norm_vmap(batch_input_x, batch_gamma,
                                                batch_beta)
    expect_outputs = np.array([[[-1.2247, 0.0000, 1.2247],
                                [-1.2247, 0.0000, 1.2247]],
                               [[-1.2247, 0.0000, 1.2247],
                                [-1.2247, 0.0000, 1.2247]]]).astype(np.float32)
    expect_means = np.array([[[2.], [2.]], [[2.], [2.]]]).astype(np.float32)
    expect_variances = np.array([[[0.6666667], [0.6666667]],
                                 [[0.6666667],
                                  [0.6666667]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect_outputs, rtol=1e-4, atol=1e-4)
    assert np.allclose(means.asnumpy(), expect_means, rtol=1e-4, atol=1e-4)
    assert np.allclose(variances.asnumpy(),
                       expect_variances,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_dynamic_shape(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of layer norm.
    Description: test dynamic tensor and dynamic scalar of layer norm.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    gamma_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    beta_dyn = ms.Tensor(shape=[None], dtype=ms.float32)

    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    beta = ms.Tensor(np.zeros([3]), ms.float32)

    test_cell = test_utils.to_cell_obj(layer_norm_dyn_shape_func)
    test_cell.set_inputs(input_x_dyn, gamma_dyn, beta_dyn)
    output, mean, variance = test_cell(input_x, gamma, beta)

    expect_output = np.array([[-1.2247, 0.0000, 1.2247],
                              [-1.2247, 0.0000, 1.2247]]).astype(np.float32)
    expect_mean = np.array([[2.], [2.]]).astype(np.float32)
    expect_variance = np.array([[0.6666667], [0.6666667]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    assert np.allclose(mean.asnumpy(), expect_mean, rtol=1e-4, atol=1e-4)
    assert np.allclose(variance.asnumpy(),
                       expect_variance,
                       rtol=1e-4,
                       atol=1e-4)

    input_x_2 = ms.Tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), ms.float32)
    gamma_2 = ms.Tensor(np.ones([4]), ms.float32)
    beta_2 = ms.Tensor(np.zeros([4]), ms.float32)
    output_2, mean_2, variance_2 = test_cell(input_x_2, gamma_2, beta_2)

    expect_output_2 = np.array([[-1.3416407, -0.4472136, 0.4472136, 1.3416407],
                                [-1.3416407, -0.4472136, 0.4472136,
                                 1.3416407]]).astype(np.float32)
    expect_mean_2 = np.array([[2.5], [2.5]]).astype(np.float32)
    expect_variance_2 = np.array([[1.25], [1.25]]).astype(np.float32)
    assert np.allclose(output_2.asnumpy(),
                       expect_output_2,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(mean_2.asnumpy(), expect_mean_2, rtol=1e-4, atol=1e-4)
    assert np.allclose(variance_2.asnumpy(),
                       expect_variance_2,
                       rtol=1e-4,
                       atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_dynamic_rank(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of layer norm.
    Description: test dynamic tensor and dynamic scalar of layer norm.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    gamma_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    beta_dyn = ms.Tensor(shape=None, dtype=ms.float32)

    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    gamma = ms.Tensor(np.ones([3]), ms.float32)
    beta = ms.Tensor(np.zeros([3]), ms.float32)

    test_cell = test_utils.to_cell_obj(layer_norm_dyn_shape_func)
    test_cell.set_inputs(input_x_dyn, gamma_dyn, beta_dyn)
    output, mean, variance = test_cell(input_x, gamma, beta)

    expect_output = np.array([[-1.2247, 0.0000, 1.2247],
                              [-1.2247, 0.0000, 1.2247]]).astype(np.float32)
    expect_mean = np.array([[2.], [2.]]).astype(np.float32)
    expect_variance = np.array([[0.6666667], [0.6666667]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    assert np.allclose(mean.asnumpy(), expect_mean, rtol=1e-4, atol=1e-4)
    assert np.allclose(variance.asnumpy(),
                       expect_variance,
                       rtol=1e-4,
                       atol=1e-4)

    input_x_2 = ms.Tensor(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), ms.float32)
    gamma_2 = ms.Tensor(np.ones([4]), ms.float32)
    beta_2 = ms.Tensor(np.zeros([4]), ms.float32)
    output_2, mean_2, variance_2 = test_cell(input_x_2, gamma_2, beta_2)

    expect_output_2 = np.array([[-1.3416407, -0.4472136, 0.4472136, 1.3416407],
                                [-1.3416407, -0.4472136, 0.4472136,
                                 1.3416407]]).astype(np.float32)
    expect_mean_2 = np.array([[2.5], [2.5]]).astype(np.float32)
    expect_variance_2 = np.array([[1.25], [1.25]]).astype(np.float32)
    assert np.allclose(output_2.asnumpy(),
                       expect_output_2,
                       rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(mean_2.asnumpy(), expect_mean_2, rtol=1e-4, atol=1e-4)
    assert np.allclose(variance_2.asnumpy(),
                       expect_variance_2,
                       rtol=1e-4,
                       atol=1e-4)
