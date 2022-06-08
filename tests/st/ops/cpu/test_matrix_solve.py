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


import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations.math_ops import MatrixSolve
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('adjoint', [True, False])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('k', [5, 8])
@pytest.mark.parametrize('dtype, error', [(np.float16, 1e-3), (np.float32, 1e-5), (np.float64, 1e-12)])
def test_matrix_solve(adjoint, m, k, dtype, error):
    """
    Feature: ALL To ALL
    Description: test cases for MatrixSolve
    Expectation: the result match to scipy
    """
    np.random.seed(0)
    context.set_context(device_target="CPU")

    matrix = np.random.normal(-10, 10, m * m).reshape((m, m)).astype(dtype)
    rhs = np.random.normal(-10, 10, m * k).reshape((m, k)).astype(dtype)
    matrix_np = np.transpose(matrix) if adjoint else matrix

    result = MatrixSolve(adjoint)(Tensor(matrix), Tensor(rhs)).asnumpy()
    if dtype == np.float16:
        expected = np.linalg.solve(matrix_np.astype(np.float32), rhs.astype(np.float32))
    else:
        expected = np.linalg.solve(matrix_np, rhs)

    assert np.allclose(result, expected, atol=error, rtol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('adjoint', [True, False])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('k', [5, 8])
@pytest.mark.parametrize('dtype, error', [(np.complex64, 1e-5), (np.complex128, 1e-12)])
def test_matrix_solve_complex(adjoint, m, k, dtype, error):
    """
    Feature: ALL To ALL
    Description: test cases for MatrixSolve
    Expectation: the result match to scipy
    """
    np.random.seed(0)
    context.set_context(device_target="CPU")

    matrix = np.random.normal(-10, 10, m * m).reshape((m, m)).astype(dtype)
    matrix.imag = np.random.normal(-10, 10, m * m).reshape((m, m)).astype(dtype)

    rhs = np.random.normal(-10, 10, m * k).reshape((m, k)).astype(dtype)
    rhs.imag = np.random.normal(-10, 10, m * k).reshape((m, k)).astype(dtype)

    matrix_np = np.conj(np.transpose(matrix)) if adjoint else matrix

    result = MatrixSolve(adjoint)(Tensor(matrix), Tensor(rhs)).asnumpy()
    expected = np.linalg.solve(matrix_np, rhs)

    assert np.allclose(result, expected, atol=error, rtol=error)
