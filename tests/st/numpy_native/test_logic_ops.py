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
"""unit tests for numpy logical operations"""

import pytest
import numpy as onp

import mindspore.numpy as mnp
from mindspore import context

from .utils import rand_int, rand_bool, run_binop_test, run_logical_test, match_res, \
    match_all_arrays, to_tensor

context.set_context(mode=context.PYNATIVE_MODE)


class Cases():
    def __init__(self):
        self.arrs = [
            rand_int(2),
            rand_int(2, 3),
            rand_int(2, 3, 4),
            rand_int(2, 3, 4, 5),
        ]

        # scalars expanded across the 0th dimension
        self.scalars = [
            rand_int(),
            rand_int(1),
            rand_int(1, 1),
            rand_int(1, 1, 1, 1),
        ]

        # arrays of the same size expanded across the 0th dimension
        self.expanded_arrs = [
            rand_int(2, 3),
            rand_int(1, 2, 3),
            rand_int(1, 1, 2, 3),
            rand_int(1, 1, 1, 2, 3),
        ]

        # arrays which can be broadcast
        self.broadcastables = [
            rand_int(5),
            rand_int(6, 1),
            rand_int(7, 1, 5),
            rand_int(8, 1, 6, 1)
        ]

        # Boolean arrays
        self.boolean_arrs = [
            rand_bool(),
            rand_bool(5),
            rand_bool(6, 1),
            rand_bool(7, 1, 5),
            rand_bool(8, 1, 6, 1)
        ]

        # array which contains infs and nans
        self.infs = onp.array([[1.0, onp.nan], [onp.inf, onp.NINF], [2.3, -4.5], [onp.nan, 0.0]])


test_case = Cases()


def mnp_not_equal(a, b):
    return mnp.not_equal(a, b)


def onp_not_equal(a, b):
    return onp.not_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_not_equal():
    run_binop_test(mnp_not_equal, onp_not_equal, test_case)


def mnp_less_equal(a, b):
    return mnp.less_equal(a, b)


def onp_less_equal(a, b):
    return onp.less_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less_equal():
    run_binop_test(mnp_less_equal, onp_less_equal, test_case)


def mnp_less(a, b):
    return mnp.less(a, b)


def onp_less(a, b):
    return onp.less(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less():
    run_binop_test(mnp_less, onp_less, test_case)


def mnp_greater_equal(a, b):
    return mnp.greater_equal(a, b)


def onp_greater_equal(a, b):
    return onp.greater_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_greater_equal():
    run_binop_test(mnp_greater_equal, onp_greater_equal, test_case)


def mnp_greater(a, b):
    return mnp.greater(a, b)


def onp_greater(a, b):
    return onp.greater(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_greater():
    run_binop_test(mnp_greater, onp_greater, test_case)


def mnp_equal(a, b):
    return mnp.equal(a, b)


def onp_equal(a, b):
    return onp.equal(a, b)


def test_equal():
    run_binop_test(mnp_equal, onp_equal, test_case)


def mnp_isfinite(x):
    return mnp.isfinite(x)


def onp_isfinite(x):
    return onp.isfinite(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isfinite():
    match_res(mnp_isfinite, onp_isfinite, test_case.infs)


def mnp_isnan(x):
    return mnp.isnan(x)


def onp_isnan(x):
    return onp.isnan(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isnan():
    match_res(mnp_isnan, onp_isnan, test_case.infs)


def mnp_isinf(x):
    return mnp.isinf(x)


def onp_isinf(x):
    return onp.isinf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isinf():
    match_res(mnp_isinf, onp_isinf, test_case.infs)


def mnp_isposinf(x):
    return mnp.isposinf(x)


def onp_isposinf(x):
    return onp.isposinf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isposinf():
    match_res(mnp_isposinf, onp_isposinf, test_case.infs)


def mnp_isneginf(x):
    return mnp.isneginf(x)


def onp_isneginf(x):
    return onp.isneginf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isneginf():
    match_res(mnp_isneginf, onp_isneginf, test_case.infs)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isscalar():
    assert mnp.isscalar(1) == onp.isscalar(1)
    assert mnp.isscalar(2.3) == onp.isscalar(2.3)
    assert mnp.isscalar([4.5]) == onp.isscalar([4.5])
    assert mnp.isscalar(False) == onp.isscalar(False)
    assert mnp.isscalar(to_tensor(True)) == onp.isscalar(onp.array(True))
    assert mnp.isscalar('numpy') == onp.isscalar('numpy')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isclose():
    a = [0, 1, 2, float('inf'), float('inf'), float('nan')]
    b = [0, 1, -2, float('-inf'), float('inf'), float('nan')]
    match_all_arrays(mnp.isclose(a, b), onp.isclose(a, b))
    match_all_arrays(mnp.isclose(a, b, equal_nan=True), onp.isclose(a, b, equal_nan=True))

    a = rand_int(2, 3, 4, 5)
    diff = (onp.random.random((2, 3, 4, 5)).astype("float32") - 0.5) / 1000
    b = a + diff
    match_all_arrays(mnp.isclose(to_tensor(a), to_tensor(b), atol=1e-3), onp.isclose(a, b, atol=1e-3))
    match_all_arrays(mnp.isclose(to_tensor(a), to_tensor(b), atol=1e-3, rtol=1e-4),
                     onp.isclose(a, b, atol=1e-3, rtol=1e-4))
    match_all_arrays(mnp.isclose(to_tensor(a), to_tensor(b), atol=1e-2, rtol=1e-6),
                     onp.isclose(a, b, atol=1e-2, rtol=1e-6))

    a = rand_int(2, 3, 4, 5)
    b = rand_int(4, 5)
    match_all_arrays(mnp.isclose(to_tensor(a), to_tensor(b)), onp.isclose(a, b))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_in1d():
    xi = [rand_int(), rand_int(1), rand_int(10)]
    yi = [rand_int(), rand_int(1), rand_int(10)]
    for x in xi:
        for y in yi:
            match_res(mnp.in1d, onp.in1d, x, y)
            match_res(mnp.in1d, onp.in1d, x, y, invert=True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isin():
    xi = [rand_int(), rand_int(1), rand_int(10), rand_int(2, 3)]
    yi = [rand_int(), rand_int(1), rand_int(10), rand_int(2, 3)]
    for x in xi:
        for y in yi:
            match_res(mnp.in1d, onp.in1d, x, y)
            match_res(mnp.in1d, onp.in1d, x, y, invert=True)


def mnp_logical_or(x1, x2):
    return mnp.logical_or(x1, x2)


def onp_logical_or(x1, x2):
    return onp.logical_or(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logical_or():
    run_logical_test(mnp_logical_or, onp_logical_or, test_case)


def mnp_logical_xor(x1, x2):
    return mnp.logical_xor(x1, x2)


def onp_logical_xor(x1, x2):
    return onp.logical_xor(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logical_xor():
    run_logical_test(mnp_logical_xor, onp_logical_xor, test_case)


def mnp_logical_and(x1, x2):
    return mnp.logical_and(x1, x2)


def onp_logical_and(x1, x2):
    return onp.logical_and(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logical_and():
    run_logical_test(mnp_logical_and, onp_logical_and, test_case)


def mnp_logical_not(x):
    return mnp.logical_not(x)


def onp_logical_not(x):
    return onp.logical_not(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logical_not():
    for arr in test_case.boolean_arrs:
        expected = onp_logical_not(arr)
        actual = mnp_logical_not(to_tensor(arr))
        onp.testing.assert_equal(actual.asnumpy().tolist(), expected.tolist())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_array_equal():
    a = [0, 1, 2, float('inf'), float('nan')]
    b = [0, 1, 2, float('inf'), float('nan')]
    match_all_arrays(mnp.array_equal(a, b), onp.array_equal(a, b))
    a = [0, 1, 2]
    b = [[0, 1, 2], [0, 1, 2]]
    assert mnp.array_equal(a, b) == onp.array_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_array_equiv():
    a = [0, 1, 2, float('inf'), float('nan')]
    b = [0, 1, 2, float('inf'), float('nan')]
    match_all_arrays(mnp.array_equal(a, b), onp.array_equal(a, b))
    a = [0, 1, 2]
    b = [[0, 1, 2], [0, 1, 2]]
    assert mnp.array_equal(a, b) == onp.array_equal(a, b)


def mnp_signbit(*arrs):
    arr1 = arrs[0]
    arr2 = arrs[1]
    a = mnp.signbit(arr1)
    b = mnp.signbit(arr2, dtype=mnp.bool_)
    return a, b


def onp_signbit(*arrs):
    arr1 = arrs[0]
    arr2 = arrs[1]
    a = onp.signbit(arr1)
    b = onp.signbit(arr2, dtype='bool')
    return a, b


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_signbit():
    onp_arrs = [onp.arange(-10, 10).astype('float32'), onp.arange(-10, 10).astype('int32')]
    mnp_arrs = [mnp.arange(-10, 10).astype('float32'), mnp.arange(-10, 10).astype('int32')]
    for actual, expected in zip(mnp_signbit(*mnp_arrs), onp_signbit(*onp_arrs)):
        onp.testing.assert_equal(actual.asnumpy().tolist(), expected.tolist())


def mnp_sometrue(x):
    a = mnp.sometrue(x)
    b = mnp.sometrue(x, axis=0)
    c = mnp.sometrue(x, axis=(0, -1))
    d = mnp.sometrue(x, axis=(0, 1), keepdims=True)
    e = mnp.sometrue(x, axis=(0, 1), keepdims=-1)
    f = mnp.sometrue(x, axis=(0, 1), keepdims=0)
    return a, b, c, d, e, f


def onp_sometrue(x):
    a = onp.sometrue(x)
    b = onp.sometrue(x, axis=0)
    c = onp.sometrue(x, axis=(0, -1))
    d = onp.sometrue(x, axis=(0, 1), keepdims=True)
    e = onp.sometrue(x, axis=(0, 1), keepdims=-1)
    f = onp.sometrue(x, axis=(0, 1), keepdims=0)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sometrue():
    onp_arr = onp.full((3, 2), [True, False])
    mnp_arr = to_tensor(onp_arr)
    for actual, expected in zip(mnp_sometrue(mnp_arr), onp_sometrue(onp_arr)):
        onp.testing.assert_equal(actual.asnumpy().tolist(), expected.tolist())
