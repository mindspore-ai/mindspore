# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test for SolveTriangular"""


import pytest
import numpy as np
from scipy.linalg import solve_triangular
import mindspore.context as context
from mindspore import Tensor
from mindspore.scipy.linalg import solve_triangular as mind_solve

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
np.random.seed(0)


def match(a, b, lower, unit_diagonal, trans):
    sci_x = solve_triangular(
        a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)
    mind_x = mind_solve(Tensor(a), Tensor(
        b), lower=lower, unit_diagonal=unit_diagonal, trans=trans).asnumpy()

    print(sci_x.flatten())
    print(mind_x.flatten())
    print(f'lower: {lower}, unit_diagonal: {unit_diagonal}, trans: {trans}')
    np.testing.assert_almost_equal(sci_x, mind_x, decimal=5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('trans', ["N", "T"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False])
def test_2d(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    # add Identity matrix to make matrix A non-singular
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    b = np.random.random((n, 1)).astype(dtype)
    match(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('trans', ["N", "T"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_1d(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description: test cases for [N x N] X [N]
    Expectation: the result match scipy
    """
    # add Identity matrix to make matrix A non-singular
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    b = np.random.random(n).astype(dtype)
    match(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(4, 5), (10, 20)])
@pytest.mark.parametrize('trans', ["N", "T"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_matrix(shape: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description: test cases for [N x N] X [N]
    Expectation: the result match scipy
    """
    if trans == 'T':
        n, m = shape
    else:
        m, n = shape
    # add Identity matrix to make matrix A non-singular
    a = (np.random.random((m, m)) + np.eye(m)).astype(dtype)
    b = np.random.random((m, n)).astype(dtype)
    match(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)
