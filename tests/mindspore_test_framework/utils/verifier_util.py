# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Utils for verifier."""

import numpy as np


def count_unequal_element(expect, result, rtol, atol):
    """
    Count unequal element.

    Args:
        expect (numpy ndarray): Expected result.
        result (numpy ndarray): Actual result.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:
    """
    if expect.shape != result.shape:
        raise ValueError(f'expect.shape {expect.shape}, result.shape {result.shape}')
    total_count = len(expect.flatten())
    error = np.abs(expect - result)
    count = np.count_nonzero(np.less_equal(error, atol + np.abs(result) * rtol))
    if ((total_count - count) / total_count) >= rtol:
        raise ValueError(f'expect {expect}, but got {result}, '
                         f'{total_count - count} / {total_count} elements out of tolerance, '
                         f'absolute_tolerance {atol}, relative_tolerance {rtol}')
    print(f'expect {expect}, got {result}, '
          f'{total_count - count} / {total_count} elements out of tolerance, '
          f'absolute_tolerance {atol}, relative_tolerance {rtol}')


def tolerance_assert(expect, result, rtol, atol):
    """
    Verify if results are in expected tolerance.

    Args:
        expect (numpy ndarray): Expected result.
        result (numpy ndarray): Actual result.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:
    """
    if not np.allclose(expect, result, rtol, atol):
        count_unequal_element(expect, result, rtol, atol)
