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
import pytest
from mindspore import Tensor
from mindspore.ops.operations import math_ops as P
import mindspore.common.dtype as ms


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_complex_abs_complex64_3x3():
    """
    Feature:  ComplexAbs 1 input and 1 output.
    Description: Compatible with Tensorflow's ComplexAbs.
    Expectation: The result matches numpy implementation.
    """
    m = 3
    input_c = np.arange(m)[:, None] + np.complex('j') * np.arange(m)
    input_c = Tensor(input_c, ms.complex64)
    expected_out = np.array([[0, 1, 2], [1, 2. ** 0.5, 5. ** 0.5], [2, 5. ** 0.5, 8. ** 0.5]], np.float32)
    complex_abs_net = P.ComplexAbs()
    complex_abs_ms_out = complex_abs_net(input_c)

    np.testing.assert_almost_equal(complex_abs_ms_out, expected_out)


def test_complex_abs_complex128_3x3():
    """
    Feature:  ComplexAbs 1 input and 1 output.
    Description: Compatible with Tensorflow's ComplexAbs.
    Expectation: The result matches numpy implementation.
    """
    m = 3
    input_c = np.arange(m)[:, None] + np.complex('j') * np.arange(m)
    input_c = Tensor(input_c, ms.complex128)
    expected_out = np.array([[0, 1, 2], [1, 2. ** 0.5, 5. ** 0.5], [2, 5. ** 0.5, 8. ** 0.5]], np.float64)
    complex_abs_net = P.ComplexAbs()
    complex_abs_ms_out = complex_abs_net(input_c)

    np.testing.assert_almost_equal(complex_abs_ms_out, expected_out)


def test_complex_abs_complex128_1x1():
    """
    Feature:  ComplexAbs 1 input and 1 output.
    Description: Compatible with Tensorflow's ComplexAbs.
    Expectation: The result matches numpy implementation.
    """
    input_c = np.array([3]) + np.complex('j') * np.array([4])
    input_c = Tensor(input_c, ms.complex64)
    expected_out = np.array([5], np.float32)
    complex_abs_net = P.ComplexAbs()
    complex_abs_ms_out = complex_abs_net(input_c)

    np.testing.assert_almost_equal(complex_abs_ms_out, expected_out)
