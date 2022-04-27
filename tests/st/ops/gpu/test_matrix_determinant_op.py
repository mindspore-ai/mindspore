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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_shape", [(4, 4), (5, 5)])
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_matrix_determinant(data_shape, data_type):
    """
    Feature: Test MatrixDeterminant.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to scipy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = np.random.random(data_shape).astype(data_type)
    error = 1e-6
    if data_type == np.float32:
        error = 1e-4
    benchmark_output = matrix_determinant_scipy_benchmark(input_x)
    matrix_determinant = MatrixDeterminantNet()
    output = matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, rtol=error, atol=error)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_shape", [(4, 4), (5, 5)])
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_log_matrix_determinant(data_shape, data_type):
    """
    Feature: Test LogMatrixDeterminant.
    Description: The input shape [..., M, M] need to match output [...].
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = np.random.random(data_shape).astype(data_type)
    error = 1e-6
    if data_type == np.float32:
        error = 1e-4
    benchmark_output = log_matrix_determinant_np_benchmark(input_x)
    log_matrix_determinant = LogMatrixDeterminantNet()
    output = log_matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output[0].asnumpy(), benchmark_output[0], rtol=error, atol=error)
    np.testing.assert_allclose(output[1].asnumpy(), benchmark_output[1], rtol=error, atol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = log_matrix_determinant(Tensor(input_x))
    np.testing.assert_allclose(output[0].asnumpy(), benchmark_output[0], rtol=error, atol=error)
    np.testing.assert_allclose(output[1].asnumpy(), benchmark_output[1], rtol=error, atol=error)
