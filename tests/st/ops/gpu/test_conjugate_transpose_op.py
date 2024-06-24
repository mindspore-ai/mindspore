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
from mindspore import Tensor, complex64
from mindspore.ops.operations import array_ops as P
import mindspore.common.dtype as ms


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_bool_3x3():
    """
    Feature:  ConjugateTranspose 2 input and 1 output.
    Description: Compatible with Tensorflow's ConjugateTranspose.
    Expectation: The result matches numpy implementation.
    """
    input_c = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    input_c = Tensor(input_c, ms.bool_)
    perm = (1, 0)
    expected_out = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.bool)
    conjugate_transpose_net = P.ConjugateTranspose()
    conjugate_transpose_ms_out = conjugate_transpose_net(input_c, perm)

    np.testing.assert_almost_equal(conjugate_transpose_ms_out, expected_out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_float64_3x3():
    """
    Feature:  ConjugateTranspose 2 input and 1 output.
    Description: Compatible with Tensorflow's ConjugateTranspose.
    Expectation: The result matches numpy implementation.
    """
    input_c = np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7]])
    input_c = Tensor(input_c, ms.float64)
    perm = (1, 0)
    expected_out = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], np.float64)
    conjugate_transpose_net = P.ConjugateTranspose()
    conjugate_transpose_ms_out = conjugate_transpose_net(input_c, perm)

    np.testing.assert_almost_equal(conjugate_transpose_ms_out, expected_out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_float32_4x4():
    """
    Feature:  ConjugateTranspose 2 input and 1 output.
    Description: Compatible with Tensorflow's ConjugateTranspose.
    Expectation: The result matches numpy implementation.
    """
    m = 16
    input_c = np.arange(m).reshape(4, 4)
    input_c = Tensor(input_c, ms.float32)
    perm = (1, 0)
    expected_out = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]], np.float32)
    conjugate_transpose_net = P.ConjugateTranspose()
    conjugate_transpose_ms_out = conjugate_transpose_net(input_c, perm)

    np.testing.assert_almost_equal(conjugate_transpose_ms_out, expected_out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_int32_2x2x2():
    """
    Feature:  ConjugateTranspose 2 input and 1 output.
    Description: Compatible with Tensorflow's ConjugateTranspose.
    Expectation: The result matches numpy implementation.
    """
    m = 8
    input_c = np.arange(m).reshape(2, 2, 2)
    input_c = Tensor(input_c, ms.int32)
    perm = (1, 2, 0)
    expected_out = np.array([[[0, 4], [1, 5]], [[2, 6], [3, 7]]], np.int32)
    conjugate_transpose_net = P.ConjugateTranspose()
    conjugate_transpose_ms_out = conjugate_transpose_net(input_c, perm)

    np.testing.assert_almost_equal(conjugate_transpose_ms_out, expected_out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_zero_rank():
    """
    Feature:  ConjugateTranspose input with zero rank.
    Description: Compatible with Tensorflow's ConjugateTranspose.
    Expectation: no core dump.
    """
    perm = ()
    input_c = Tensor(np.random.uniform(-10, 10, size=())).astype(complex64)
    conjugate_transpose_net = P.ConjugateTranspose()
    _ = conjugate_transpose_net(input_c, perm)
