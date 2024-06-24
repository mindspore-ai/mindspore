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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore.ops.operations.math_ops import MatrixTriangularSolve


class MatrixTriangularSolveTEST(nn.Cell):
    def __init__(self, lower=True, adjoint=False):
        super(MatrixTriangularSolveTEST, self).__init__()
        self.matrix_triangular_solve = MatrixTriangularSolve(lower=lower, adjoint=adjoint)

    def construct(self, matrix, rhs):
        return self.matrix_triangular_solve(matrix, rhs)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matrix_triangular_solve_op_case1():
    """
    Feature: MatrixTriangularSolve GPU operator.
    Description: Test the correctness of result.
    Expectation: Match to tensorflow.
    """
    matrix_triangular_solve_test = MatrixTriangularSolveTEST(lower=True, adjoint=False)
    args_type_list = [mstype.float32, mstype.float64]
    for dtype in args_type_list:
        matrix = Tensor(np.array([[3, 0, 0, 0],
                                  [2, 1, 0, 0],
                                  [1, 0, 1, 0],
                                  [1, 1, 1, 1]]), dtype)
        rhs = Tensor(np.array([[4],
                               [2],
                               [4],
                               [2]]), dtype)

        expect_result = Tensor(np.array([[1.3333334],
                                         [-0.66666675],
                                         [2.6666665],
                                         [-1.3333331]]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matrix_triangular_solve_op_case2():
    """
    Feature: MatrixTriangularSolve GPU operator.
    Description: Test the correctness of result.
    Expectation: Match to tensorflow.
    """
    matrix_triangular_solve_test = MatrixTriangularSolveTEST(lower=True, adjoint=True)
    args_type_list = [mstype.float32, mstype.float64]
    for dtype in args_type_list:
        matrix = Tensor(np.array([[3, 0, 0, 0],
                                  [2, 1, 0, 0],
                                  [1, 0, 1, 0],
                                  [1, 1, 1, 1]]), dtype)
        rhs = Tensor(np.array([[4],
                               [2],
                               [4],
                               [2]]), dtype)

        expect_result = Tensor(np.array([[0],
                                         [0],
                                         [2],
                                         [2]]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matrix_triangular_solve_op_case3():
    """
    Feature: MatrixTriangularSolve GPU operator.
    Description: Test the correctness of result.
    Expectation: Match to tensorflow.
    """
    matrix_triangular_solve_test = MatrixTriangularSolveTEST(lower=False, adjoint=False)
    args_type_list = [mstype.complex64, mstype.complex128]
    for dtype in args_type_list:
        matrix = Tensor(np.array([[3 + 0j, 2j, 3j, 3j],
                                  [0j, 1 + 0j, 2j, 3j],
                                  [0j, 0j, 1 + 0j, 5j],
                                  [0j, 0j, 0j, 1 + 0j]]), dtype)
        rhs = Tensor(np.array([[4j],
                               [2 + 0j],
                               [4 + 0j],
                               [2 + 0j]]), dtype)

        expect_result = Tensor(np.array([[-19.33333333 + 7.33333333j],
                                         [-18. - 14.j],
                                         [4. - 10.j],
                                         [2. + 0.j]]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matrix_triangular_solve_op_case4():
    """
    Feature: MatrixTriangularSolve GPU operator.
    Description: Test the correctness of result.
    Expectation: Match to tensorflow.
    """
    matrix_triangular_solve_test = MatrixTriangularSolveTEST(lower=False, adjoint=True)
    args_type_list = [mstype.complex64, mstype.complex128]
    for dtype in args_type_list:
        matrix = Tensor(np.array([[3 + 0j, 2j, 3j, 3j],
                                  [0j, 1 + 0j, 2j, 3j],
                                  [0j, 0j, 1 + 0j, 5j],
                                  [0j, 0j, 0j, 1 + 0j]]), dtype)
        rhs = Tensor(np.array([[4j],
                               [2 + 0j],
                               [4 + 0j],
                               [2 + 0j]]), dtype)

        expect_result = Tensor(np.array([[0.0000000e+00 + 1.3333334j],
                                         [-6.6666675e-01 + 0.j],
                                         [-1.1920929e-07 - 1.3333335j],
                                         [4.6666675e+00 - 2.000001j]]), dtype)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = matrix_triangular_solve_test(matrix, rhs)
        assert np.allclose(output.asnumpy(), expect_result.asnumpy())
