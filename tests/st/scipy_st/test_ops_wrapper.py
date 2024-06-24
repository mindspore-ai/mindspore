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
import numpy as onp
import pytest
import mindspore.scipy.ops_wrapper as ops_wrapper
from mindspore import context, Tensor
from tests.mark_utils import arg_mark
from tests.st.scipy_st.utils import match_array

DEFAULT_ALIGNMENT = "LEFT_LEFT"
ALIGNMENT_LIST = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]


def pack_diagonals_in_matrix(matrix, num_rows, num_cols, alignment=None):
    if alignment == DEFAULT_ALIGNMENT or alignment is None:
        return matrix
    packed_matrix = dict()
    for diag_index, (diagonals, padded_diagonals) in matrix.items():
        align = alignment.split("_")
        d_lower, d_upper = diag_index
        batch_dims = diagonals.ndim - (2 if d_lower < d_upper else 1)
        max_diag_len = diagonals.shape[-1]
        index = (slice(None),) * batch_dims
        packed_diagonals = onp.zeros_like(diagonals)
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
            packed_diagonals[repacked_index] = diagonals[packed_index]
        packed_matrix[diag_index] = (packed_diagonals, padded_diagonals)
    return packed_matrix


def square_matrix(alignment=None, data_type=None):
    mat = onp.array([[[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 1],
                      [3, 4, 5, 6, 7],
                      [8, 9, 1, 2, 3],
                      [4, 5, 6, 7, 8]],
                     [[9, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9],
                      [1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 1],
                      [2, 3, 4, 5, 6]]], dtype=data_type)
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    # tests[d_lower, d_upper] = packed_diagonals
    tests[-1, -1] = (onp.array([[6, 4, 1, 7],
                                [5, 2, 8, 5]], dtype=data_type),
                     onp.array([[[0, 0, 0, 0, 0],
                                 [6, 0, 0, 0, 0],
                                 [0, 4, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 7, 0]],
                                [[0, 0, 0, 0, 0],
                                 [5, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0],
                                 [0, 0, 8, 0, 0],
                                 [0, 0, 0, 5, 0]]], dtype=data_type))
    tests[-4, -3] = (onp.array([[[8, 5],
                                 [4, 0]],
                                [[6, 3],
                                 [2, 0]]], dtype=data_type),
                     onp.array([[[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [8, 0, 0, 0, 0],
                                 [4, 5, 0, 0, 0]],
                                [[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [6, 0, 0, 0, 0],
                                 [2, 3, 0, 0, 0]]], dtype=data_type))
    tests[-2, 1] = (onp.array([[[2, 8, 6, 3, 0],
                                [1, 7, 5, 2, 8],
                                [6, 4, 1, 7, 0],
                                [3, 9, 6, 0, 0]],
                               [[1, 7, 4, 1, 0],
                                [9, 6, 3, 9, 6],
                                [5, 2, 8, 5, 0],
                                [1, 7, 4, 0, 0]]], dtype=data_type),
                    onp.array([[[1, 2, 0, 0, 0],
                                [6, 7, 8, 0, 0],
                                [3, 4, 5, 6, 0],
                                [0, 9, 1, 2, 3],
                                [0, 0, 6, 7, 8]],
                               [[9, 1, 0, 0, 0],
                                [5, 6, 7, 0, 0],
                                [1, 2, 3, 4, 0],
                                [0, 7, 8, 9, 1],
                                [0, 0, 4, 5, 6]]], dtype=data_type))
    tests[2, 4] = (onp.array([[[5, 0, 0],
                               [4, 1, 0],
                               [3, 9, 7]],
                              [[4, 0, 0],
                               [3, 9, 0],
                               [2, 8, 5]]], dtype=data_type),
                   onp.array([[[0, 0, 3, 4, 5],
                               [0, 0, 0, 9, 1],
                               [0, 0, 0, 0, 7],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]],
                              [[0, 0, 2, 3, 4],
                               [0, 0, 0, 8, 9],
                               [0, 0, 0, 0, 5],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]], dtype=data_type))

    return mat, pack_diagonals_in_matrix(tests, num_rows, num_cols, alignment)


def tall_matrix(alignment=None, data_type=None):
    mat = onp.array([[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [9, 8, 7],
                      [6, 5, 4]],
                     [[3, 2, 1],
                      [1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [9, 8, 7]]], dtype=data_type)
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    tests[0, 0] = (onp.array([[1, 5, 9],
                              [3, 2, 6]], dtype=data_type),
                   onp.array([[[1, 0, 0],
                               [0, 5, 0],
                               [0, 0, 9],
                               [0, 0, 0]],
                              [[3, 0, 0],
                               [0, 2, 0],
                               [0, 0, 6],
                               [0, 0, 0]]], dtype=data_type))
    tests[-4, -3] = (onp.array([[[9, 5],
                                 [6, 0]],
                                [[7, 8],
                                 [9, 0]]], dtype=data_type),
                     onp.array([[[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [9, 0, 0],
                                 [6, 5, 0]],
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [7, 0, 0],
                                 [9, 8, 0]]], dtype=data_type))
    tests[-2, -1] = (onp.array([[[4, 8, 7],
                                 [7, 8, 4]],
                                [[1, 5, 9],
                                 [4, 8, 7]]], dtype=data_type),
                     onp.array([[[0, 0, 0],
                                 [4, 0, 0],
                                 [7, 8, 0],
                                 [0, 8, 7],
                                 [0, 0, 4]],
                                [[0, 0, 0],
                                 [1, 0, 0],
                                 [4, 5, 0],
                                 [0, 8, 9],
                                 [0, 0, 7]]], dtype=data_type))
    tests[-2, 1] = (onp.array([[[2, 6, 0],
                                [1, 5, 9],
                                [4, 8, 7],
                                [7, 8, 4]],
                               [[2, 3, 0],
                                [3, 2, 6],
                                [1, 5, 9],
                                [4, 8, 7]]], dtype=data_type),
                    onp.array([[[1, 2, 0],
                                [4, 5, 6],
                                [7, 8, 9],
                                [0, 8, 7],
                                [0, 0, 4]],
                               [[3, 2, 0],
                                [1, 2, 3],
                                [4, 5, 6],
                                [0, 8, 9],
                                [0, 0, 7]]], dtype=data_type))
    tests[1, 2] = (onp.array([[[3, 0],
                               [2, 6]],
                              [[1, 0],
                               [2, 3]]], dtype=data_type),
                   onp.array([[[0, 2, 3],
                               [0, 0, 6],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 2, 1],
                               [0, 0, 3],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]], dtype=data_type))

    return mat, pack_diagonals_in_matrix(tests, num_rows, num_cols, alignment)


def fat_matrix(alignment=None, data_type=None):
    mat = onp.array([[[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 1, 2, 3]],
                     [[4, 5, 6, 7],
                      [8, 9, 1, 2],
                      [3, 4, 5, 6]]], dtype=data_type)
    num_rows, num_cols = mat.shape[-2:]
    tests = dict()
    tests[2, 2] = (onp.array([[3, 8],
                              [6, 2]], dtype=data_type),
                   onp.array([[[0, 0, 3, 0],
                               [0, 0, 0, 8],
                               [0, 0, 0, 0]],
                              [[0, 0, 6, 0],
                               [0, 0, 0, 2],
                               [0, 0, 0, 0]]], dtype=data_type))
    tests[-2, 0] = (onp.array([[[1, 6, 2],
                                [5, 1, 0],
                                [9, 0, 0]],
                               [[4, 9, 5],
                                [8, 4, 0],
                                [3, 0, 0]]], dtype=data_type),
                    onp.array([[[1, 0, 0, 0],
                                [5, 6, 0, 0],
                                [9, 1, 2, 0]],
                               [[4, 0, 0, 0],
                                [8, 9, 0, 0],
                                [3, 4, 5, 0]]], dtype=data_type))
    tests[-1, 1] = (onp.array([[[2, 7, 3],
                                [1, 6, 2],
                                [5, 1, 0]],
                               [[5, 1, 6],
                                [4, 9, 5],
                                [8, 4, 0]]], dtype=data_type),
                    onp.array([[[1, 2, 0, 0],
                                [5, 6, 7, 0],
                                [0, 1, 2, 3]],
                               [[4, 5, 0, 0],
                                [8, 9, 1, 0],
                                [0, 4, 5, 6]]], dtype=data_type))
    tests[0, 3] = (onp.array([[[4, 0, 0],
                               [3, 8, 0],
                               [2, 7, 3],
                               [1, 6, 2]],
                              [[7, 0, 0],
                               [6, 2, 0],
                               [5, 1, 6],
                               [4, 9, 5]]], dtype=data_type),
                   onp.array([[[1, 2, 3, 4],
                               [0, 6, 7, 8],
                               [0, 0, 2, 3]],
                              [[4, 5, 6, 7],
                               [0, 9, 1, 2],
                               [0, 0, 5, 6]]], dtype=data_type))
    return mat, pack_diagonals_in_matrix(tests, num_rows, num_cols, alignment)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('data_type', [onp.int32, onp.int64, onp.float32, onp.float64])
def test_matrix_set_diag(data_type):
    """
    Feature: ALL TO ALL
    Description: test geneal matrix cases for matrix_set_diag in pynative or graph mode
    Expectation: the result match expected_diag_matrix.
    """
    onp.random.seed(0)
    context.set_context(mode=context.PYNATIVE_MODE)
    for align in ALIGNMENT_LIST:
        for _, tests in [square_matrix(align, data_type), tall_matrix(align, data_type), fat_matrix(align, data_type)]:
            for k_vec, (diagonal, banded_mat) in tests.items():
                mask = banded_mat[0] == 0
                input_mat = onp.random.randint(10, size=mask.shape).astype(dtype=data_type)
                expected_diag_matrix = input_mat * mask + banded_mat[0]
                output = ops_wrapper.matrix_set_diag(
                    Tensor(input_mat), Tensor(diagonal[0]), k=k_vec, alignment=align)
                match_array(output.asnumpy(), expected_diag_matrix)

    context.set_context(mode=context.GRAPH_MODE)
    for align in ALIGNMENT_LIST:
        for _, tests in [square_matrix(align, data_type), tall_matrix(align, data_type), fat_matrix(align, data_type)]:
            for k_vec, (diagonal, banded_mat) in tests.items():
                mask = banded_mat[0] == 0
                input_mat = onp.random.randint(10, size=mask.shape).astype(dtype=data_type)
                expected_diag_matrix = input_mat * mask + banded_mat[0]
                output = ops_wrapper.matrix_set_diag(
                    Tensor(input_mat), Tensor(diagonal[0]), k=k_vec, alignment=align)
                match_array(output.asnumpy(), expected_diag_matrix)
