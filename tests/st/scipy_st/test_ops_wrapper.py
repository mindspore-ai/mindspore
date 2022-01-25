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
"""st for scipy.ops_wrapper."""
import pytest
import numpy as onp
import mindspore.scipy.ops_wrapper as ops_wrapper
from mindspore import context, Tensor
from mindspore.scipy.ops import MatrixBandPartNet
from tests.st.scipy_st.utils import match_matrix

DEFAULT_ALIGNMENT = "LEFT_LEFT"
ALIGNMENT_LIST = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]


def repack_diagonals(packed_diagonals,
                     diag_index,
                     num_rows,
                     num_cols,
                     align=None):
    if align == DEFAULT_ALIGNMENT or align is None:
        return packed_diagonals
    align = align.split("_")
    d_lower, d_upper = diag_index
    batch_dims = packed_diagonals.ndim - (2 if d_lower < d_upper else 1)
    max_diag_len = packed_diagonals.shape[-1]
    index = (slice(None),) * batch_dims
    repacked_diagonals = onp.zeros_like(packed_diagonals)
    for d_index in range(d_lower, d_upper + 1):
        diag_len = min(num_rows + min(0, d_index), num_cols - max(0, d_index))
        row_index = d_upper - d_index
        padding_len = max_diag_len - diag_len
        left_align = (d_index >= 0 and
                      align[0] == "LEFT") or (d_index <= 0 and
                                              align[1] == "LEFT")
        extra_dim = tuple() if d_lower == d_upper else (row_index,)
        packed_last_dim = (slice(None),) if left_align else (slice(0, diag_len, 1),)
        repacked_last_dim = (slice(None),) if left_align else (slice(
            padding_len, max_diag_len, 1),)
        packed_index = index + extra_dim + packed_last_dim
        repacked_index = index + extra_dim + repacked_last_dim

        repacked_diagonals[repacked_index] = packed_diagonals[packed_index]
    return repacked_diagonals


def repack_diagonals_in_tests(tests, num_rows, num_cols, align=None):
    # The original test cases are LEFT_LEFT aligned.
    if align == DEFAULT_ALIGNMENT or align is None:
        return tests
    new_tests = dict()
    # Loops through each case.
    for diag_index, (packed_diagonals, padded_diagonals) in tests.items():
        repacked_diagonals = repack_diagonals(
            packed_diagonals, diag_index, num_rows, num_cols, align=align)
        new_tests[diag_index] = (repacked_diagonals, padded_diagonals)

    return new_tests


def square_cases(align=None, dtype=None):
    mat = onp.array([[[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 1],
                      [3, 4, 5, 6, 7],
                      [8, 9, 1, 2, 3],
                      [4, 5, 6, 7, 8]],
                     [[9, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9],
                      [1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 1],
                      [2, 3, 4, 5, 6]]], dtype=dtype)
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    # tests[d_lower, d_upper] = packed_diagonals
    tests[-1, -1] = (onp.array([[6, 4, 1, 7],
                                [5, 2, 8, 5]], dtype=dtype),
                     onp.array([[[0, 0, 0, 0, 0],
                                 [6, 0, 0, 0, 0],
                                 [0, 4, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 7, 0]],
                                [[0, 0, 0, 0, 0],
                                 [5, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0],
                                 [0, 0, 8, 0, 0],
                                 [0, 0, 0, 5, 0]]], dtype=dtype))
    tests[-4, -3] = (onp.array([[[8, 5],
                                 [4, 0]],
                                [[6, 3],
                                 [2, 0]]], dtype=dtype),
                     onp.array([[[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [8, 0, 0, 0, 0],
                                 [4, 5, 0, 0, 0]],
                                [[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [6, 0, 0, 0, 0],
                                 [2, 3, 0, 0, 0]]], dtype=dtype))
    tests[-2, 1] = (onp.array([[[2, 8, 6, 3, 0],
                                [1, 7, 5, 2, 8],
                                [6, 4, 1, 7, 0],
                                [3, 9, 6, 0, 0]],
                               [[1, 7, 4, 1, 0],
                                [9, 6, 3, 9, 6],
                                [5, 2, 8, 5, 0],
                                [1, 7, 4, 0, 0]]], dtype=dtype),
                    onp.array([[[1, 2, 0, 0, 0],
                                [6, 7, 8, 0, 0],
                                [3, 4, 5, 6, 0],
                                [0, 9, 1, 2, 3],
                                [0, 0, 6, 7, 8]],
                               [[9, 1, 0, 0, 0],
                                [5, 6, 7, 0, 0],
                                [1, 2, 3, 4, 0],
                                [0, 7, 8, 9, 1],
                                [0, 0, 4, 5, 6]]], dtype=dtype))
    tests[2, 4] = (onp.array([[[5, 0, 0],
                               [4, 1, 0],
                               [3, 9, 7]],
                              [[4, 0, 0],
                               [3, 9, 0],
                               [2, 8, 5]]], dtype=dtype),
                   onp.array([[[0, 0, 3, 4, 5],
                               [0, 0, 0, 9, 1],
                               [0, 0, 0, 0, 7],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 2, 3, 4],
                               [0, 0, 0, 8, 9],
                               [0, 0, 0, 0, 5],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]], dtype=dtype))

    return mat, repack_diagonals_in_tests(tests, num_rows, num_cols, align)


def tall_cases(align=None):
    mat = onp.array([[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [9, 8, 7],
                      [6, 5, 4]],
                     [[3, 2, 1],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [9, 8, 7]]])
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    tests[0, 0] = (onp.array([[1, 5, 9],
                              [3, 2, 6]]),
                   onp.array([[[1, 0, 0],
                               [0, 5, 0],
                               [0, 0, 9],
                               [0, 0, 0]],
                              [[3, 0, 0],
                               [0, 2, 0],
                               [0, 0, 6],
                               [0, 0, 0]]]))
    tests[-4, -3] = (onp.array([[[9, 5],
                                 [6, 0]],
                                [[7, 8],
                                 [9, 0]]]),
                     onp.array([[[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [9, 0, 0],
                                 [6, 5, 0]],
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [7, 0, 0],
                                 [9, 8, 0]]]))
    tests[-2, -1] = (onp.array([[[4, 8, 7],
                                 [7, 8, 4]],
                                [[1, 5, 9],
                                 [4, 8, 7]]]),
                     onp.array([[[0, 0, 0],
                                 [4, 0, 0],
                                 [7, 8, 0],
                                 [0, 8, 7],
                                 [0, 0, 4]],
                                [[0, 0, 0],
                                 [1, 0, 0],
                                 [4, 5, 0],
                                 [0, 8, 9],
                                 [0, 0, 7]]]))
    tests[-2, 1] = (onp.array([[[2, 6, 0],
                                [1, 5, 9],
                                [4, 8, 7],
                                [7, 8, 4]],
                               [[2, 3, 0],
                                [3, 2, 6],
                                [1, 5, 9],
                                [4, 8, 7]]]),
                    onp.array([[[1, 2, 0],
                                [4, 5, 6],
                                [7, 8, 9],
                                [0, 8, 7],
                                [0, 0, 4]],
                               [[3, 2, 0],
                                [1, 2, 3],
                                [4, 5, 6],
                                [0, 8, 9],
                                [0, 0, 7]]]))
    tests[1, 2] = (onp.array([[[3, 0],
                               [2, 6]],
                              [[1, 0],
                               [2, 3]]]),
                   onp.array([[[0, 2, 3],
                               [0, 0, 6],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 2, 1],
                               [0, 0, 3],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]]))

    return mat, repack_diagonals_in_tests(tests, num_rows, num_cols, align)


def fat_cases(align=None):
    mat = onp.array([[[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 1, 2, 3]],
                     [[4, 5, 6, 7],
                      [8, 9, 1, 2],
                      [3, 4, 5, 6]]])
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    tests[2, 2] = (onp.array([[3, 8],
                              [6, 2]]),
                   onp.array([[[0, 0, 3, 0],
                               [0, 0, 0, 8],
                               [0, 0, 0, 0]],
                              [[0, 0, 6, 0],
                               [0, 0, 0, 2],
                               [0, 0, 0, 0]]]))
    tests[-2, 0] = (onp.array([[[1, 6, 2],
                                [5, 1, 0],
                                [9, 0, 0]],
                               [[4, 9, 5],
                                [8, 4, 0],
                                [3, 0, 0]]]),
                    onp.array([[[1, 0, 0, 0],
                                [5, 6, 0, 0],
                                [9, 1, 2, 0]],
                               [[4, 0, 0, 0],
                                [8, 9, 0, 0],
                                [3, 4, 5, 0]]]))
    tests[-1, 1] = (onp.array([[[2, 7, 3],
                                [1, 6, 2],
                                [5, 1, 0]],
                               [[5, 1, 6],
                                [4, 9, 5],
                                [8, 4, 0]]]),
                    onp.array([[[1, 2, 0, 0],
                                [5, 6, 7, 0],
                                [0, 1, 2, 3]],
                               [[4, 5, 0, 0],
                                [8, 9, 1, 0],
                                [0, 4, 5, 6]]]))
    tests[0, 3] = (onp.array([[[4, 0, 0],
                               [3, 8, 0],
                               [2, 7, 3],
                               [1, 6, 2]],
                              [[7, 0, 0],
                               [6, 2, 0],
                               [5, 1, 6],
                               [4, 9, 5]]]),
                   onp.array([[[1, 2, 3, 4],
                               [0, 6, 7, 8],
                               [0, 0, 2, 3]],
                              [[4, 5, 6, 7],
                               [0, 9, 1, 2],
                               [0, 0, 5, 6]]]))
    return mat, repack_diagonals_in_tests(tests, num_rows, num_cols, align)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [onp.int])
def test_matrix_set_diag(data_type):
    """
    Feature: ALL TO ALL
    Description: test geneal matrix cases for matrix_set_diag in pynative or graph mode
    Expectation: the result match expected_diag_matrix.
    """
    onp.random.seed(0)
    context.set_context(mode=context.PYNATIVE_MODE)
    for align in ALIGNMENT_LIST:
        for _, tests in [square_cases(align, data_type), tall_cases(align), fat_cases(align)]:
            for k_vec, (diagonal, banded_mat) in tests.items():
                mask = banded_mat[0] == 0
                input_mat = onp.random.randint(10, size=mask.shape)
                expected_diag_matrix = input_mat * mask + banded_mat[0]
                output = ops_wrapper.matrix_set_diag(
                    Tensor(input_mat), Tensor(diagonal[0]), k=k_vec, alignment=align)
                match_matrix(output.astype(onp.float64), Tensor(expected_diag_matrix))

    context.set_context(mode=context.GRAPH_MODE)
    for align in ALIGNMENT_LIST:
        for _, tests in [square_cases(align, data_type), tall_cases(align), fat_cases(align)]:
            for k_vec, (diagonal, banded_mat) in tests.items():
                mask = banded_mat[0] == 0
                input_mat = onp.random.randint(10, size=mask.shape)
                expected_diag_matrix = input_mat * mask + banded_mat[0]
                output = ops_wrapper.matrix_set_diag(
                    Tensor(input_mat), Tensor(diagonal[0]), k=k_vec, alignment=align)
                match_matrix(output.astype(onp.float64), Tensor(expected_diag_matrix))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('band_inputs',
                         [([], 1, 1), ([], 1, 2), ([], 1, 7), ([], 2, 1), ([], 2, 2), ([], 2, 7), ([], 7, 1),
                          ([], 7, 2), ([], 7, 7), ([2], 1, 1), ([2], 1, 2), ([2], 1, 7), ([2], 2, 1), ([2], 2, 2),
                          ([2], 2, 7), ([2], 7, 1), ([2], 7, 2), ([2], 7, 7), ([1, 3, 2], 1, 1), ([1, 3, 2], 1, 2),
                          ([1, 3, 2], 1, 7), ([1, 3, 2], 2, 1), ([1, 3, 2], 2, 2), ([1, 3, 2], 2, 7), ([1, 3, 2], 7, 1),
                          ([1, 3, 2], 7, 2), ([1, 3, 2], 7, 7)])
def test_matrix_band_part_net_cpu(band_inputs):
    """
    Feature: ALL TO ALL
    Description: test general matrix cases for matrix_band_diag in graph mode
    Expectation: the result match expected_diag_band_matrix.
    """
    msp_matrixbandpart = MatrixBandPartNet()
    batch_shape, rows, cols = band_inputs
    for dtype in [onp.int32, onp.float64]:
        mat = onp.ones(batch_shape + [rows, cols]).astype(dtype)
        for lower in -1, 0, 1, rows - 1:
            for upper in -1, 0, 1, cols - 1:
                band_np = mat
                if lower >= 0:
                    band_np = onp.triu(band_np, -lower)
                if upper >= 0:
                    band_np = onp.tril(band_np, upper)
                if batch_shape:
                    band_np = onp.tile(band_np, batch_shape + [1, 1])
                band = msp_matrixbandpart(Tensor(band_np), lower, upper)
                match_matrix(band, Tensor(band_np))
