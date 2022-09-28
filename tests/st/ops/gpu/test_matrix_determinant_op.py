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
import scipy as osp
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.common import dtype as ms_type
from mindspore.ops.functional import vmap


class MatrixDeterminantNet(nn.Cell):
    def __init__(self):
        super(MatrixDeterminantNet, self).__init__()
        self.matrix_determinant = P.MatrixDeterminant()

    def construct(self, input_x):
        output = self.matrix_determinant(input_x)
        return output


def matrix_determinant_scipy_benchmark(input_x):
    """
    Feature: generate a matrix determinant numpy benchmark.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to scipy mindspore MatrixDeterminant.
    """
    result = osp.linalg.det(input_x)
    return result


class LogMatrixDeterminantNet(nn.Cell):
    def __init__(self):
        super(LogMatrixDeterminantNet, self).__init__()
        self.log_matrix_determinant = P.LogMatrixDeterminant()

    def construct(self, input_x):
        output = self.log_matrix_determinant(input_x)
        return output


def log_matrix_determinant_np_benchmark(input_x):
    """
    Feature: generate a log matrix determinant numpy benchmark.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to np mindspore LogMatrixDeterminant.
    """
    result = np.linalg.slogdet(input_x)
    return result


class DeterminantVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(DeterminantVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_shape", [(4, 4), (5, 5)])
@pytest.mark.parametrize("data_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_matrix_determinant(data_shape, data_type):
    """
    Feature: Test MatrixDeterminant.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to scipy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = np.random.random(data_shape).astype(data_type)
    error = 1e-6
    if data_type in (np.float32, np.complex64):
        error = 1e-3
    benchmark_output = matrix_determinant_scipy_benchmark(input_x)
    matrix_determinant = MatrixDeterminantNet()
    output = matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_shape", [(4, 4), (5, 5)])
@pytest.mark.parametrize("data_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_log_matrix_determinant(data_shape, data_type):
    """
    Feature: Test LogMatrixDeterminant.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = np.random.random(data_shape).astype(data_type)
    error = 1e-6
    if data_type in (np.float32, np.complex64):
        error = 1e-3
    benchmark_output = log_matrix_determinant_np_benchmark(input_x)
    log_matrix_determinant = LogMatrixDeterminantNet()
    output = log_matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output[0].asnumpy(), benchmark_output[0], rtol=error, atol=error)
    np.testing.assert_allclose(output[1].asnumpy(), benchmark_output[1], rtol=error, atol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = log_matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output[0].asnumpy(), benchmark_output[0], rtol=error, atol=error)
    np.testing.assert_allclose(output[1].asnumpy(), benchmark_output[1], rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_determinant_dy_shape():
    """
    Feature: Test MatrixDeterMinant DynamicShape.
    Description: The input data type only check float32 is ok.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    matrix_determinant_net = MatrixDeterminantNet()
    loss = 1e-6
    input_shape = (4, 4)
    data_type = np.float32
    ms_data_type = ms_type.float32
    if data_type in (np.float32, np.complex64):
        loss = 1e-3
    input_x_np = np.random.random(input_shape).astype(data_type)
    benchmark_output = matrix_determinant_scipy_benchmark(input_x_np)

    input_dyn = Tensor(shape=[4, None], dtype=ms_data_type)
    matrix_determinant_net.set_inputs(input_dyn)
    ms_result = matrix_determinant_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = matrix_determinant_net(Tensor(input_x_np))
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_log_matrix_determinant_dy_shape():
    """
    Feature: Test LogMatrixDeterMinant DynamicShape.
    Description: The input data type only check float32 is ok.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(mode=context.GRAPH_MODE)
    log_matrix_determinant_net = LogMatrixDeterminantNet()
    loss = 1e-6
    input_shape = (4, 4)
    data_type = np.float32
    ms_data_type = ms_type.float32
    if data_type in (np.float32, np.complex64):
        loss = 1e-3
    input_x_np = np.random.random(input_shape).astype(data_type)
    benchmark_output = log_matrix_determinant_np_benchmark(input_x_np)

    input_dyn = Tensor(shape=[4, None], dtype=ms_data_type)
    log_matrix_determinant_net.set_inputs(input_dyn)
    ms_result = log_matrix_determinant_net(Tensor(input_x_np))
    np.testing.assert_allclose(ms_result[0].asnumpy(), benchmark_output[0], rtol=loss, atol=loss)
    np.testing.assert_allclose(ms_result[1].asnumpy(), benchmark_output[1], rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = log_matrix_determinant_net(Tensor(input_x_np))
    np.testing.assert_allclose(ms_result[0].asnumpy(), benchmark_output[0], rtol=loss, atol=loss)
    np.testing.assert_allclose(ms_result[1].asnumpy(), benchmark_output[1], rtol=loss, atol=loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_matrix_determinant_vmap():
    """
    Feature: test MatrixDeterMinant vmap on GPU.
    Description: inputs(input_x) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss = 1e-6
    data_type = np.float32
    if data_type in (np.float32, np.complex64):
        loss = 1e-3
    # Case : in_axes input_x batch remains 0
    input_x = Tensor(np.array([[[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]],
                               [[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]]).astype(data_type))

    in_axes = 0
    out_axes = 0
    benchmark_output = np.array([[-16.5, 21.], [-16.5, 21.]]).astype(data_type)
    matrix_determinant = MatrixDeterminantNet()
    ms_result = DeterminantVMapNet(matrix_determinant, in_axes, out_axes)(Tensor(input_x))
    assert np.allclose(ms_result.asnumpy(), benchmark_output, rtol=loss, atol=loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_log_matrix_determinant_vmap():
    """
    Feature: test logMatrixDeterMinant vmap on GPU.
    Description: inputs(input_x) with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss = 1e-6
    data_type = np.float32
    if data_type in (np.float32, np.complex64):
        loss = 1e-3
    # Case : in_axes input_x batch remains 0
    input_x = Tensor(np.array([[[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]],
                               [[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]]).astype(data_type))
    in_axes = 0
    out_axes = 0
    benchmark_output_sign = np.array([[-1., 1.], [-1., 1.]]).astype(data_type)
    benchmark_output_determinant = np.array([[2.80336046, 3.04452229], [2.80336046, 3.04452229]]).astype(data_type)
    matrix_determinant = LogMatrixDeterminantNet()
    ms_result = DeterminantVMapNet(matrix_determinant, in_axes, out_axes)(Tensor(input_x))
    assert np.allclose(ms_result[0].asnumpy(), benchmark_output_sign, rtol=loss, atol=loss)
    assert np.allclose(ms_result[1].asnumpy(), benchmark_output_determinant, rtol=loss, atol=loss)
