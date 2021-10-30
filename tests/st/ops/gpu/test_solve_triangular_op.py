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
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
np.random.seed(0)


class SolveTriangular(PrimitiveWithInfer):
    """
        SolveTriangular op frontend implementation
    """

    @prim_attr_register
    def __init__(self, lower: bool, unit_diagonal: bool, trans: str):
        """Initialize SolveTriangular"""
        self.lower = validator.check_value_type(
            "lower", lower, [bool], self.name)
        self.unit_diagonal = validator.check_value_type(
            "unit_diagonal", unit_diagonal, [bool], self.name)
        self.trans = validator.check_value_type(
            "trans", trans, [str], self.name)

        self.init_prim_io_names(inputs=['A', 'b'], outputs=['output'])

    def __infer__(self, A, b):
        out_shapes = b['shape']
        return {
            'shape': tuple(out_shapes),
            'dtype': A['dtype'],
            'value': None
        }

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid(x_dtype, [mstype.float32, mstype.float64],
                                           self.name, True)
        return x_dtype


def mind_solve(a, b, trans="N", lower=False, unit_diagonal=False,
               overwrite_b=False, debug=None, check_finite=True):
    solve = SolveTriangular(
        lower=lower, unit_diagonal=unit_diagonal, trans=trans)
    return solve(a, b)


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
def test_2D(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
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
def test_1D(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
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
