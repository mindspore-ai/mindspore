# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""unit tests for numpy math operations"""

import pytest
import numpy as onp

import mindspore.numpy as mnp

from .utils import rand_int, rand_bool, run_binop_test, run_unary_test, run_multi_test, \
    run_single_test, match_res, match_array, match_meta, match_all_arrays, to_tensor

class Cases():
    def __init__(self):
        self.arrs = [
            rand_int(2),
            rand_int(2, 3),
            rand_int(2, 3, 4),
        ]

        # scalars expanded across the 0th dimension
        self.scalars = [
            rand_int(),
            rand_int(1),
            rand_int(1, 1),
        ]

        # arrays of the same size expanded across the 0th dimension
        self.expanded_arrs = [
            rand_int(2, 3),
            rand_int(1, 2, 3),
            rand_int(1, 1, 2, 3),
        ]

        # arrays with last dimension aligned
        self.aligned_arrs = [
            rand_int(2, 3),
            rand_int(1, 4, 3),
            rand_int(5, 1, 2, 3),
            rand_int(4, 2, 1, 1, 3),
        ]

        # arrays which can be broadcast
        self.broadcastables = [
            rand_int(5),
            rand_int(6, 1),
            rand_int(7, 1, 5),
        ]

        # boolean arrays which can be broadcast
        self.bool_broadcastables = [
            rand_bool(),
            rand_bool(1),
            rand_bool(5),
            rand_bool(6, 1),
            rand_bool(7, 1, 5),
            rand_bool(8, 1, 6, 1),
        ]

        # core dimension 0 is matched for each
        # pair of array[i] and array[i + 1]
        self.core_broadcastables = [
            rand_int(3),
            rand_int(3),
            rand_int(6),
            rand_int(6, 4),
            rand_int(5, 2),
            rand_int(2),
            rand_int(2, 9),
            rand_int(9, 8),
            rand_int(6),
            rand_int(2, 6, 5),
            rand_int(9, 2, 7),
            rand_int(7),
            rand_int(5, 2, 4),
            rand_int(6, 1, 4, 9),
            rand_int(7, 1, 5, 3, 2),
            rand_int(8, 1, 6, 1, 2, 9),
        ]

        # arrays with dimensions of size 1
        self.nested_arrs = [
            rand_int(1),
            rand_int(1, 2),
            rand_int(3, 1, 8),
            rand_int(1, 3, 9, 1),
        ]


test_case = Cases()


def mnp_add(x1, x2):
    return mnp.add(x1, x2)


def onp_add(x1, x2):
    return onp.add(x1, x2)


def mnp_subtract(x1, x2):
    return mnp.subtract(x1, x2)


def onp_subtract(x1, x2):
    return onp.subtract(x1, x2)


def mnp_mutiply(x1, x2):
    return mnp.multiply(x1, x2)


def onp_multiply(x1, x2):
    return onp.multiply(x1, x2)


def mnp_divide(x1, x2):
    return mnp.divide(x1, x2)


def onp_divide(x1, x2):
    return onp.divide(x1, x2)


def mnp_true_divide(x1, x2):
    return mnp.true_divide(x1, x2)


def onp_true_divide(x1, x2):
    return onp.true_divide(x1, x2)


def mnp_power(x1, x2):
    return mnp.power(x1, x2)


def onp_power(x1, x2):
    return onp.power(x1, x2)


def mnp_float_power(x1, x2):
    return mnp.float_power(x1, x2)


def onp_float_power(x1, x2):
    return onp.float_power(x1, x2)


def mnp_minimum(a, b):
    return mnp.minimum(a, b)


def onp_minimum(a, b):
    return onp.minimum(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add():
    run_binop_test(mnp_add, onp_add, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_subtract():
    run_binop_test(mnp_subtract, onp_subtract, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multiply():
    run_binop_test(mnp_mutiply, onp_multiply, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_divide():
    run_binop_test(mnp_divide, onp_divide, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_true_divide():
    run_binop_test(mnp_true_divide, onp_true_divide, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_power():
    run_binop_test(mnp_power, onp_power, test_case, error=1e-5)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_float_power():
    run_binop_test(mnp_float_power, onp_float_power, test_case, error=1e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_minimum():
    run_binop_test(mnp_minimum, onp_minimum, test_case)
    x = onp.random.randint(-10, 10, 20).astype(onp.float32)
    y = onp.random.randint(-10, 10, 20).astype(onp.float32)
    x[onp.random.randint(0, 10, 3)] = onp.nan
    y[onp.random.randint(0, 10, 3)] = onp.nan
    x[onp.random.randint(0, 10, 3)] = onp.NINF
    y[onp.random.randint(0, 10, 3)] = onp.NINF
    x[onp.random.randint(0, 10, 3)] = onp.PINF
    y[onp.random.randint(0, 10, 3)] = onp.PINF
    match_res(mnp_minimum, onp_minimum, x, y)
    match_res(mnp_minimum, onp_minimum, y, x)


def mnp_tensordot(x, y):
    a = mnp.tensordot(x, y)
    b = mnp.tensordot(x, y, axes=0)
    c = mnp.tensordot(x, y, axes=1)
    d = mnp.tensordot(x, y, axes=2)
    e = mnp.tensordot(x, y, axes=(3, 0))
    f = mnp.tensordot(x, y, axes=[2, 1])
    g = mnp.tensordot(x, y, axes=((2, 3), (0, 1)))
    h = mnp.tensordot(x, y, axes=[[3, 2], [1, 0]])
    return a, b, c, d, e, f, g, h


def onp_tensordot(x, y):
    a = onp.tensordot(x, y)
    b = onp.tensordot(x, y, axes=0)
    c = onp.tensordot(x, y, axes=1)
    d = onp.tensordot(x, y, axes=2)
    e = onp.tensordot(x, y, axes=(3, 0))
    f = onp.tensordot(x, y, axes=[2, 1])
    g = onp.tensordot(x, y, axes=((2, 3), (0, 1)))
    h = onp.tensordot(x, y, axes=[[3, 2], [1, 0]])
    return a, b, c, d, e, f, g, h


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensordot():
    x = rand_int(4, 2, 7, 7)
    y = rand_int(7, 7, 6)
    run_multi_test(mnp_tensordot, onp_tensordot, (x, y))


def mnp_std(x):
    a = mnp.std(x)
    b = mnp.std(x, axis=None)
    c = mnp.std(x, axis=0)
    d = mnp.std(x, axis=1)
    e = mnp.std(x, axis=(-1, 1))
    f = mnp.std(x, axis=(0, 1, 2))
    g = mnp.std(x, axis=None, ddof=1, keepdims=True)
    h = mnp.std(x, axis=0, ddof=1, keepdims=True)
    i = mnp.std(x, axis=(2), ddof=1, keepdims=True)
    return a, b, c, d, e, f, g, h, i


def onp_std(x):
    a = onp.std(x)
    b = onp.std(x, axis=None)
    c = onp.std(x, axis=0)
    d = onp.std(x, axis=1)
    e = onp.std(x, axis=(-1, 1))
    f = onp.std(x, axis=(0, 1, 2))
    g = onp.std(x, axis=None, ddof=1, keepdims=True)
    h = onp.std(x, axis=0, ddof=1, keepdims=True)
    i = onp.std(x, axis=(2), ddof=1, keepdims=True)
    return a, b, c, d, e, f, g, h, i


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_std():
    arr1 = rand_int(2, 3, 4, 5)
    arr2 = rand_int(4, 5, 4, 3, 3)
    run_single_test(mnp_std, onp_std, arr1, error=1e-5)
    run_single_test(mnp_std, onp_std, arr2, error=1e-5)


def mnp_nanstd(x):
    a = mnp.nanstd(x)
    b = mnp.nanstd(x, axis=None)
    c = mnp.nanstd(x, axis=0)
    d = mnp.nanstd(x, axis=1)
    e = mnp.nanstd(x, axis=(-1, 1))
    f = mnp.nanstd(x, axis=(0, 1, 2))
    g = mnp.nanstd(x, axis=None, ddof=1, keepdims=True)
    h = mnp.nanstd(x, axis=0, ddof=1, keepdims=True)
    i = mnp.nanstd(x, axis=(2), ddof=1, keepdims=True)
    return a, b, c, d, e, f, g, h, i


def onp_nanstd(x):
    a = onp.nanstd(x)
    b = onp.nanstd(x, axis=None)
    c = onp.nanstd(x, axis=0)
    d = onp.nanstd(x, axis=1)
    e = onp.nanstd(x, axis=(-1, 1))
    f = onp.nanstd(x, axis=(0, 1, 2))
    g = onp.nanstd(x, axis=None, ddof=1, keepdims=True)
    h = onp.nanstd(x, axis=0, ddof=1, keepdims=True)
    i = onp.nanstd(x, axis=(2), ddof=1, keepdims=True)
    return a, b, c, d, e, f, g, h, i


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nanstd():
    arr1 = rand_int(2, 3, 4, 5)
    arr1[0][2][1][3] = onp.nan
    arr1[1][0][2][4] = onp.nan
    arr1[1][1][1][1] = onp.nan
    arr2 = rand_int(4, 5, 4, 3, 3)
    arr2[3][1][2][1][0] = onp.nan
    arr2[1][1][1][1][1] = onp.nan
    arr2[0][4][3][0][2] = onp.nan
    run_single_test(mnp_nanstd, onp_nanstd, arr1, error=1e-5)
    run_single_test(mnp_nanstd, onp_nanstd, arr2, error=1e-5)


def mnp_var(x):
    a = mnp.var(x)
    b = mnp.var(x, axis=0)
    c = mnp.var(x, axis=(0))
    d = mnp.var(x, axis=(0, 1, 2))
    e = mnp.var(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
    return a, b, c, d, e


def onp_var(x):
    a = onp.var(x)
    b = onp.var(x, axis=0)
    c = onp.var(x, axis=(0))
    d = onp.var(x, axis=(0, 1, 2))
    e = onp.var(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
    return a, b, c, d, e


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_var():
    arr1 = rand_int(2, 3, 4, 5)
    arr2 = rand_int(4, 5, 4, 3, 3)
    run_single_test(mnp_var, onp_var, arr1, error=1e-5)
    run_single_test(mnp_var, onp_var, arr2, error=1e-5)


def mnp_nanvar(x):
    a = mnp.var(x)
    b = mnp.var(x, axis=0)
    c = mnp.var(x, axis=(0))
    d = mnp.var(x, axis=(0, 1, 2))
    e = mnp.var(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
    return a, b, c, d, e


def onp_nanvar(x):
    a = onp.var(x)
    b = onp.var(x, axis=0)
    c = onp.var(x, axis=(0))
    d = onp.var(x, axis=(0, 1, 2))
    e = onp.var(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
    return a, b, c, d, e


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nanvar():
    arr1 = rand_int(2, 3, 4, 5)
    arr1[0][2][1][3] = onp.nan
    arr1[1][0][2][4] = onp.nan
    arr1[1][1][1][1] = onp.nan
    arr2 = rand_int(4, 5, 4, 3, 3)
    arr2[3][1][2][1][0] = onp.nan
    arr2[1][1][1][1][1] = onp.nan
    arr2[0][4][3][0][2] = onp.nan
    run_single_test(mnp_nanvar, onp_nanvar, arr1, error=1e-5)
    run_single_test(mnp_nanvar, onp_nanvar, arr2, error=1e-5)


def mnp_average(x):
    a = mnp.average(x)
    b = mnp.average(x, axis=None)
    c = mnp.average(x, axis=0)
    d = mnp.average(x, axis=1)
    e = mnp.average(x, axis=(-2, 1))
    f = mnp.average(x, axis=(0, 1, 2, 3))
    g = mnp.average(x, axis=None, weights=x)
    h = mnp.average(x, axis=0, weights=x)
    i = mnp.average(x, axis=(1, 2, 3), weights=x)
    return a, b, c, d, e, f, g, h, i


def onp_average(x):
    a = onp.average(x)
    b = onp.average(x, axis=None)
    c = onp.average(x, axis=0)
    d = onp.average(x, axis=1)
    e = onp.average(x, axis=(-2, 1))
    f = onp.average(x, axis=(0, 1, 2, 3))
    g = onp.average(x, axis=None, weights=x)
    h = onp.average(x, axis=0, weights=x)
    i = onp.average(x, axis=(1, 2, 3), weights=x)
    return a, b, c, d, e, f, g, h, i


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_average():
    arr1 = rand_int(2, 3, 4, 5)
    arr2 = rand_int(4, 5, 1, 3, 1)
    run_single_test(mnp_average, onp_average, arr1, error=1e-5)
    run_single_test(mnp_average, onp_average, arr2, error=1e-5)


def mnp_count_nonzero(x):
    a = mnp.count_nonzero(x)
    b = mnp.count_nonzero(x, axis=None)
    c = mnp.count_nonzero(x, axis=0)
    d = mnp.count_nonzero(x, axis=1)
    e = mnp.count_nonzero(x, axis=(-2, 1))
    f = mnp.count_nonzero(x, axis=(0, 1, 2, 3))
    return a, b, c, d, e, f


def onp_count_nonzero(x):
    a = onp.count_nonzero(x)
    b = onp.count_nonzero(x, axis=None)
    c = onp.count_nonzero(x, axis=0)
    d = onp.count_nonzero(x, axis=1)
    e = onp.count_nonzero(x, axis=(-2, 1))
    f = onp.count_nonzero(x, axis=(0, 1, 2, 3))
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_count_nonzero():
    # minus 5 to make some values below zero
    arr1 = rand_int(2, 3, 4, 5) - 5
    arr2 = rand_int(4, 5, 4, 3, 3) - 5
    run_single_test(mnp_count_nonzero, onp_count_nonzero, arr1)
    run_single_test(mnp_count_nonzero, onp_count_nonzero, arr2)


def mnp_inner(a, b):
    return mnp.inner(a, b)


def onp_inner(a, b):
    return onp.inner(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_inner():
    for arr1 in test_case.aligned_arrs:
        for arr2 in test_case.aligned_arrs:
            match_res(mnp_inner, onp_inner, arr1, arr2)

    for scalar1 in test_case.scalars:
        for scalar2 in test_case.scalars:
            match_res(mnp_inner, onp_inner,
                      scalar1, scalar2)


def mnp_dot(a, b):
    return mnp.dot(a, b)


def onp_dot(a, b):
    return onp.dot(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dot():
    # test case (1D, 1D)
    match_res(mnp_dot, onp_dot, rand_int(3), rand_int(3))

    # test case (2D, 2D)
    match_res(mnp_dot, onp_dot, rand_int(4, 7), rand_int(7, 2))

    # test case (0D, _) (_, 0D)
    match_res(mnp_dot, onp_dot, rand_int(), rand_int(1, 9, 3))
    match_res(mnp_dot, onp_dot, rand_int(8, 5, 6, 3), rand_int())

    # test case (ND, 1D)
    match_res(mnp_dot, onp_dot, rand_int(2, 4, 5), rand_int(5))

    # test case (ND, MD)
    match_res(mnp_dot, onp_dot, rand_int(5, 4, 1, 8), rand_int(8, 3))

    for i in range(8):
        match_res(mnp_dot, onp_dot,
                  test_case.core_broadcastables[2*i], test_case.core_broadcastables[2*i + 1])


def mnp_outer(a, b):
    return mnp.outer(a, b)


def onp_outer(a, b):
    return onp.outer(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_outer():
    run_binop_test(mnp_outer, onp_outer, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_type_promotion():
    arr = rand_int(2, 3)
    onp_sum = onp_add(arr, arr)

    a = to_tensor(arr, dtype=mnp.float16)
    b = to_tensor(arr, dtype=mnp.float32)
    c = to_tensor(arr, dtype=mnp.int32)

    match_array(mnp_add(a, b).asnumpy(), onp_sum)
    match_array(mnp_add(b, c).asnumpy(), onp_sum)


def mnp_absolute(x):
    return mnp.absolute(x)


def onp_absolute(x):
    return onp.absolute(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_absolute():
    arr = rand_int(2, 3)

    a = to_tensor(arr, dtype=mnp.float16)
    b = to_tensor(arr, dtype=mnp.float32)
    c = to_tensor(arr, dtype=mnp.uint8)
    d = to_tensor(arr, dtype=mnp.bool_)

    match_array(mnp_absolute(a).asnumpy(), onp_absolute(a.asnumpy()))
    match_array(mnp_absolute(b).asnumpy(), onp_absolute(b.asnumpy()))
    match_array(mnp_absolute(c).asnumpy(), onp_absolute(c.asnumpy()))
    match_array(mnp_absolute(d).asnumpy(), onp_absolute(d.asnumpy()))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_deg2rad_rad2deg():
    arrs = [rand_int(2, 3), rand_int(1, 2, 4), rand_int(2, 4)]
    for arr in arrs:
        match_res(mnp.deg2rad, onp.deg2rad, arr)
        match_res(mnp.rad2deg, onp.rad2deg, arr)


def mnp_ptp(x):
    a = mnp.ptp(x)
    b = mnp.ptp(x, keepdims=True)
    c = mnp.ptp(x, axis=(0, 1))
    d = mnp.ptp(x, axis=-1)
    return a, b, c, d


def onp_ptp(x):
    a = onp.ptp(x)
    b = onp.ptp(x, keepdims=True)
    c = onp.ptp(x, axis=(0, 1))
    d = onp.ptp(x, axis=-1)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptp():
    arrs = [rand_int(2, 3), rand_int(1, 2, 4), rand_int(2, 4)]
    for arr in arrs:
        match_res(mnp_ptp, onp_ptp, arr)


def mnp_add_dtype(x1, x2):
    return mnp.add(x1, x2, dtype=mnp.float32)


def onp_add_dtype(x1, x2):
    return onp.add(x1, x2, dtype=onp.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add_dtype():
    x1 = rand_int(2, 3).astype('int32')
    x2 = rand_int(2, 3).astype('int32')
    arrs = (x1, x2)
    mnp_arrs = map(to_tensor, arrs)
    mnp_res = mnp_add_dtype(*mnp_arrs)
    onp_res = onp_add_dtype(*arrs)
    for actual, expected in zip(mnp_res, onp_res):
        assert actual.asnumpy().dtype == expected.dtype


def mnp_matmul(x1, x2):
    return mnp.matmul(x1, x2)


def onp_matmul(x1, x2):
    return onp.matmul(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul():
    for scalar1 in test_case.scalars[1:]:
        for scalar2 in test_case.scalars[1:]:
            match_res(mnp_matmul, onp_matmul,
                      scalar1, scalar2)
    for i in range(8):
        match_res(mnp_matmul, onp_matmul,
                  test_case.core_broadcastables[2*i],
                  test_case.core_broadcastables[2*i + 1])


def mnp_square(x):
    return mnp.square(x)


def onp_square(x):
    return onp.square(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_square():
    run_unary_test(mnp_square, onp_square, test_case)


def mnp_sqrt(x):
    return mnp.sqrt(x)


def onp_sqrt(x):
    return onp.sqrt(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sqrt():
    run_unary_test(mnp_sqrt, onp_sqrt, test_case)


def mnp_reciprocal(x):
    return mnp.reciprocal(x)


def onp_reciprocal(x):
    return onp.reciprocal(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reciprocal():
    run_unary_test(mnp_reciprocal, onp_reciprocal, test_case)


def mnp_log(x):
    return mnp.log(x)


def onp_log(x):
    return onp.log(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_log():
    run_unary_test(mnp.log, onp.log, test_case, error=1e-5)


def mnp_log1p(x):
    return mnp.log1p(x)


def onp_log1p(x):
    return onp.log1p(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_log1p():
    run_unary_test(mnp_log1p, onp_log1p, test_case, error=1e-5)


def mnp_logaddexp(x1, x2):
    return mnp.logaddexp(x1, x2)


def onp_logaddexp(x1, x2):
    return onp.logaddexp(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logaddexp():
    test_cases = [
        onp.random.randint(1, 5, (2)).astype('float16'),
        onp.random.randint(1, 5, (3, 2)).astype('float16'),
        onp.random.randint(1, 5, (1, 3, 2)).astype('float16'),
        onp.random.randint(1, 5, (5, 6, 3, 2)).astype('float16')]
    for _, x1 in enumerate(test_cases):
        for _, x2 in enumerate(test_cases):
            expected = onp_logaddexp(x1, x2)
            actual = mnp_logaddexp(to_tensor(x1), to_tensor(x2))
            onp.testing.assert_almost_equal(actual.asnumpy().tolist(), expected.tolist(),
                                            decimal=2)


def mnp_log2(x):
    return mnp.log2(x)


def onp_log2(x):
    return onp.log2(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_log2():
    run_unary_test(mnp_log2, onp_log2, test_case, error=1e-5)


def mnp_logaddexp2(x1, x2):
    return mnp.logaddexp2(x1, x2)


def onp_logaddexp2(x1, x2):
    return onp.logaddexp2(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logaddexp2():
    test_cases = [
        onp.random.randint(1, 5, (2)).astype('float16'),
        onp.random.randint(1, 5, (3, 2)).astype('float16'),
        onp.random.randint(1, 5, (1, 3, 2)).astype('float16'),
        onp.random.randint(1, 5, (5, 6, 3, 2)).astype('float16')]
    for _, x1 in enumerate(test_cases):
        for _, x2 in enumerate(test_cases):
            expected = onp_logaddexp2(x1, x2)
            actual = mnp_logaddexp2(to_tensor(x1), to_tensor(x2))
            onp.testing.assert_almost_equal(actual.asnumpy().tolist(), expected.tolist(),
                                            decimal=2)


def mnp_log10(x):
    return mnp.log10(x)


def onp_log10(x):
    return onp.log10(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_log10():
    run_unary_test(mnp_log10, onp_log10, test_case, error=1e-5)


def mnp_maximum(x1, x2):
    return mnp.maximum(x1, x2)


def onp_maximum(x1, x2):
    return onp.maximum(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum():
    run_binop_test(mnp_maximum, onp_maximum, test_case)
    x = onp.random.randint(-10, 10, 20).astype(onp.float32)
    y = onp.random.randint(-10, 10, 20).astype(onp.float32)
    x[onp.random.randint(0, 10, 3)] = onp.nan
    y[onp.random.randint(0, 10, 3)] = onp.nan
    x[onp.random.randint(0, 10, 3)] = onp.NINF
    y[onp.random.randint(0, 10, 3)] = onp.NINF
    x[onp.random.randint(0, 10, 3)] = onp.PINF
    y[onp.random.randint(0, 10, 3)] = onp.PINF
    match_res(mnp_maximum, onp_maximum, x, y)
    match_res(mnp_maximum, onp_maximum, y, x)


def mnp_clip(x):
    a = mnp.clip(x, to_tensor(10.0), to_tensor([2,]))
    b = mnp.clip(x, 0, 1)
    c = mnp.clip(x, to_tensor(0), to_tensor(10), dtype=mnp.float32)
    return a, b, c


def onp_clip(x):
    a = onp.clip(x, onp.asarray(10.0), onp.asarray([2,]))
    b = onp.clip(x, 0, 1)
    c = onp.clip(x, onp.asarray(0), onp.asarray(10), dtype=onp.float32)
    return a, b, c


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_clip():
    run_unary_test(mnp_clip, onp_clip, test_case)


def mnp_amax(x, mask):
    a = mnp.amax(x)
    b = mnp.amax(x, axis=-3)
    c = mnp.amax(x, keepdims=True)
    d = mnp.amax(x, initial=3)
    e = mnp.amax(x, axis=(0, 1), keepdims=True)
    f = mnp.amax(x, initial=4, where=mask)
    g = mnp.amax(x, initial=5, where=mask, keepdims=True)
    h = mnp.amax(x, axis=(1, 2, 3), initial=6, where=mask)
    return a, b, c, d, e, f, g, h


def onp_amax(x, mask):
    a = onp.amax(x)
    b = onp.amax(x, axis=-3)
    c = onp.amax(x, keepdims=True)
    d = onp.amax(x, initial=3)
    e = onp.amax(x, axis=(0, 1), keepdims=True)
    f = onp.amax(x, initial=4, where=mask)
    g = onp.amax(x, initial=5, where=mask, keepdims=True)
    h = onp.amax(x, axis=(1, 2, 3), initial=6, where=mask)
    return a, b, c, d, e, f, g, h


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_amax():
    a = rand_int(2, 3, 4, 5).astype('float32')
    mask = rand_bool(2, 3, 4, 5)
    run_multi_test(mnp_amax, onp_amax, (a, mask))


def mnp_amin(x, mask):
    a = mnp.amin(x)
    b = mnp.amin(x, axis=-3)
    c = mnp.amin(x, keepdims=True)
    d = mnp.amin(x, initial=-1)
    e = mnp.amin(x, axis=(0, 1), keepdims=True)
    f = mnp.amin(x, initial=-2)
    g = mnp.amin(x, initial=-3, keepdims=True)
    h = mnp.amin(x, axis=(1, 2, 3), initial=-4, where=mask)
    return a, b, c, d, e, f, g, h


def onp_amin(x, mask):
    a = onp.amin(x)
    b = onp.amin(x, axis=-3)
    c = onp.amin(x, keepdims=True)
    d = onp.amin(x, initial=-1)
    e = onp.amin(x, axis=(0, 1), keepdims=True)
    f = onp.amin(x, initial=-2)
    g = onp.amin(x, initial=-3, keepdims=True)
    h = onp.amin(x, axis=(1, 2, 3), initial=-4, where=mask)
    return a, b, c, d, e, f, g, h


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_amin():
    a = rand_int(2, 3, 4, 5).astype('float32')
    mask = rand_bool(2, 3, 4, 5)
    run_multi_test(mnp_amin, onp_amin, (a, mask))


def mnp_hypot(x1, x2):
    return mnp.hypot(x1, x2)


def onp_hypot(x1, x2):
    return onp.hypot(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hypot():
    run_binop_test(mnp_hypot, onp_hypot, test_case)


def mnp_heaviside(x1, x2):
    return mnp.heaviside(x1, x2)


def onp_heaviside(x1, x2):
    return onp.heaviside(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_heaviside():
    broadcastables = test_case.broadcastables
    for b1 in broadcastables:
        for b2 in broadcastables:
            b = onp.subtract(b1, b2)
            match_res(mnp_heaviside, onp_heaviside, b, b1)
            match_res(mnp_heaviside, onp_heaviside, b, b2)


def mnp_floor(x):
    return mnp.floor(x)


def onp_floor(x):
    return onp.floor(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor():
    run_unary_test(mnp_floor, onp_floor, test_case)
    x = rand_int(2, 3) * onp.random.rand(2, 3)
    match_res(mnp_floor, onp_floor, x)
    match_res(mnp_floor, onp_floor, -x)


def mnp_floor_divide(x, y):
    return mnp.floor_divide(x, y)


def onp_floor_divde(x, y):
    return onp.floor_divide(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor_divide():
    run_binop_test(mnp_floor_divide, onp_floor_divde, test_case)


def mnp_remainder(x, y):
    return mnp.remainder(x, y)


def onp_remainder(x, y):
    return onp.remainder(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_remainder():
    x = rand_int(2, 3)
    y = rand_int(2, 3)
    match_res(mnp_remainder, onp_remainder, x, y)


def mnp_mod(x, y):
    return mnp.mod(x, y)


def onp_mod(x, y):
    return onp.mod(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mod():
    x = rand_int(2, 3)
    y = rand_int(2, 3)
    match_res(mnp_mod, onp_mod, x, y)


def mnp_fmod(x, y):
    return mnp.fmod(x, y)


def onp_fmod(x, y):
    return onp.fmod(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fmod():
    x = rand_int(2, 3)
    y = rand_int(2, 3)
    match_res(mnp_fmod, onp_fmod, x, y)


def mnp_fix(x):
    return mnp.fix(x)


def onp_fix(x):
    return onp.fix(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fix():
    x = rand_int(2, 3)
    y = rand_int(2, 3)
    floats = onp.divide(onp.subtract(x, y), y)
    match_res(mnp_fix, onp_fix, floats, error=1e-5)


def mnp_trunc(x):
    return mnp.trunc(x)


def onp_trunc(x):
    return onp.trunc(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trunc():
    x = rand_int(2, 3)
    y = rand_int(2, 3)
    floats = onp.divide(onp.subtract(x, y), y)
    match_res(mnp_trunc, onp_trunc, floats, error=1e-5)


def mnp_exp(x):
    return mnp.exp(x)


def onp_exp(x):
    return onp.exp(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exp():
    run_unary_test(mnp_exp, onp_exp, test_case, error=5)


def mnp_expm1(x):
    return mnp.expm1(x)


def onp_expm1(x):
    return onp.expm1(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expm1():
    run_unary_test(mnp_expm1, onp_expm1, test_case, error=5)


def mnp_exp2(x):
    return mnp.exp2(x)


def onp_exp2(x):
    return onp.exp2(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exp2():
    run_unary_test(mnp_exp2, onp_exp2, test_case, error=5)


def mnp_kron(x, y):
    return mnp.kron(x, y)


def onp_kron(x, y):
    return onp.kron(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_kron():
    run_binop_test(mnp_kron, onp_kron, test_case)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cross():
    x = onp.arange(8).reshape(2, 2, 1, 2)
    y = onp.arange(4).reshape(1, 2, 2)
    match_res(mnp.cross, onp.cross, x, y)
    match_res(mnp.cross, onp.cross, x, y, axisa=-3, axisb=1, axisc=2)
    match_res(mnp.cross, onp.cross, x, y, axisa=-3, axisb=1, axisc=2, axis=1)
    x = onp.arange(18).reshape(2, 3, 1, 3)
    y = onp.arange(9).reshape(1, 3, 3)
    match_res(mnp.cross, onp.cross, x, y)
    match_res(mnp.cross, onp.cross, x, y, axisa=-3, axisb=1, axisc=2)
    match_res(mnp.cross, onp.cross, x, y, axisa=-3, axisb=1, axisc=2, axis=1)


def mnp_ceil(x):
    return mnp.ceil(x)


def onp_ceil(x):
    return onp.ceil(x)


@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ceil():
    run_unary_test(mnp_ceil, onp_ceil, test_case)


def mnp_positive(x):
    return mnp.positive(x)


def onp_positive(x):
    return onp.positive(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_positive():
    arr = onp.arange(-6, 6).reshape((2, 2, 3)).astype('float32')
    onp_pos = onp_positive(arr)
    mnp_pos = mnp_positive(to_tensor(arr))
    match_array(mnp_pos.asnumpy(), onp_pos)


def mnp_negative(x):
    return mnp.negative(x)


def onp_negative(x):
    return onp.negative(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_negative():
    arr = onp.arange(-6, 6).reshape((2, 2, 3)).astype('float32')
    onp_neg = onp_negative(arr)
    mnp_neg = mnp_negative(to_tensor(arr))
    match_array(mnp_neg.asnumpy(), onp_neg, 1e-5)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cumsum():
    x = mnp.ones((16, 16), dtype="bool")
    match_array(mnp.cumsum(x).asnumpy(), onp.cumsum(x.asnumpy()))
    match_array(mnp.cumsum(x, axis=0).asnumpy(),
                onp.cumsum(x.asnumpy(), axis=0))
    match_meta(mnp.cumsum(x).asnumpy(), onp.cumsum(x.asnumpy()))

    x = rand_int(3, 4, 5)
    match_array(mnp.cumsum(to_tensor(x), dtype="bool").asnumpy(),
                onp.cumsum(x, dtype="bool"))
    match_array(mnp.cumsum(to_tensor(x), axis=-1).asnumpy(),
                onp.cumsum(x, axis=-1))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_promote_types():
    assert mnp.promote_types(mnp.int32, mnp.bool_) == mnp.int32
    assert mnp.promote_types(int, mnp.bool_) == mnp.int32
    assert mnp.promote_types("float32", mnp.int64) == mnp.float32
    assert mnp.promote_types(mnp.int64, mnp.float16) == mnp.float16
    assert mnp.promote_types(int, float) == mnp.float32


def mnp_diff(input_tensor):
    a = mnp.diff(input_tensor, 2, append=3.0)
    b = mnp.diff(input_tensor, 4, prepend=6, axis=-2)
    c = mnp.diff(input_tensor, 0, append=3.0, axis=-1)
    d = mnp.diff(input_tensor, 1, prepend=input_tensor)
    e = mnp.ediff1d(input_tensor, to_end=input_tensor)
    f = mnp.ediff1d(input_tensor)
    g = mnp.ediff1d(input_tensor, to_begin=3)
    return a, b, c, d, e, f, g


def onp_diff(input_array):
    a = onp.diff(input_array, 2, append=3.0)
    b = onp.diff(input_array, 4, prepend=6, axis=-2)
    c = onp.diff(input_array, 0, append=3.0, axis=-1)
    d = onp.diff(input_array, 1, prepend=input_array)
    e = onp.ediff1d(input_array, to_end=input_array)
    f = onp.ediff1d(input_array)
    g = onp.ediff1d(input_array, to_begin=3)
    return a, b, c, d, e, f, g


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_diff():
    arr = rand_int(3, 4, 5)
    match_res(mnp_diff, onp_diff, arr)
    arr = rand_int(1, 4, 6, 3)
    match_res(mnp_diff, onp_diff, arr)


def mnp_sin(x):
    return mnp.sin(x)


def onp_sin(x):
    return onp.sin(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sin():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_sin(arr)
    actual = mnp_sin(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_cos(x):
    return mnp.cos(x)


def onp_cos(x):
    return onp.cos(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cos():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_cos(arr)
    actual = mnp_cos(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_tan(x):
    return mnp.tan(x)


def onp_tan(x):
    return onp.tan(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tan():
    arr = onp.array([-0.75, -0.5, 0, 0.5, 0.75]).astype('float32')
    expect = onp_tan(arr)
    actual = mnp_tan(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_arcsin(x):
    return mnp.arcsin(x)


def onp_arcsin(x):
    return onp.arcsin(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arcsin():
    arr = onp.random.uniform(-1, 1, 12).astype('float32')
    onp_asin = onp_arcsin(arr)
    mnp_asin = mnp_arcsin(to_tensor(arr))
    match_array(mnp_asin.asnumpy(), onp_asin, error=3)


def mnp_arccos(x):
    return mnp.arccos(x)


def onp_arccos(x):
    return onp.arccos(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arccos():
    arr = onp.random.uniform(-1, 1, 12).astype('float32')
    onp_acos = onp_arccos(arr)
    mnp_acos = mnp_arccos(to_tensor(arr))
    match_array(mnp_acos.asnumpy(), onp_acos, error=2)


def mnp_arctan(x):
    return mnp.arctan(x)


def onp_arctan(x):
    return onp.arctan(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arctan():
    arr = onp.random.uniform(-1, 1, 12).astype('float32')
    onp_atan = onp_arctan(arr)
    mnp_atan = mnp_arctan(to_tensor(arr))
    match_array(mnp_atan.asnumpy(), onp_atan, error=5)


def mnp_sinh(x):
    return mnp.sinh(x)


def onp_sinh(x):
    return onp.sinh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sinh():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_sinh(arr)
    actual = mnp_sinh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_cosh(x):
    return mnp.cosh(x)


def onp_cosh(x):
    return onp.cosh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cosh():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_cosh(arr)
    actual = mnp_cosh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_tanh(x):
    return mnp.tanh(x)


def onp_tanh(x):
    return onp.tanh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tanh():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_tanh(arr)
    actual = mnp_tanh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_arcsinh(x):
    return mnp.arcsinh(x)


def onp_arcsinh(x):
    return onp.arcsinh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arcsinh():
    arr = onp.random.rand(2, 3, 4).astype('float32')
    expect = onp_arcsinh(arr)
    actual = mnp_arcsinh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_arccosh(x):
    return mnp.arccosh(x)


def onp_arccosh(x):
    return onp.arccosh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arccosh():
    arr = onp.random.randint(1, 100, size=(2, 3)).astype('float32')
    expect = onp_arccosh(arr)
    actual = mnp_arccosh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_arctanh(x):
    return mnp.arctanh(x)


def onp_arctanh(x):
    return onp.arctanh(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arctanh():
    arr = onp.random.uniform(-0.9, 1, 10).astype('float32')
    expect = onp_arctanh(arr)
    actual = mnp_arctanh(to_tensor(arr))
    match_array(actual.asnumpy(), expect, error=5)


def mnp_arctan2(x, y):
    return mnp.arctan2(x, y)


def onp_arctan2(x, y):
    return onp.arctan2(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arctan2():
    run_binop_test(mnp_arctan2, onp_arctan2, test_case, error=5)


def mnp_convolve(mode):
    a = mnp.convolve([1, 2, 3, 4, 5], 2, mode=mode)
    b = mnp.convolve([1, 2, 3, 4, 5], [2, 3], mode=mode)
    c = mnp.convolve([1, 2], [2, 5, 10], mode=mode)
    d = mnp.convolve(mnp.array([1, 2, 3, 4, 5]), mnp.array([1, 2, 3, 4, 5]), mode=mode)
    e = mnp.convolve([1, 2, 3, 4, 5], 2, mode=mode)
    return a, b, c, d, e


def onp_convolve(mode):
    a = onp.convolve([1, 2, 3, 4, 5], 2, mode=mode)
    b = onp.convolve([1, 2, 3, 4, 5], [2, 3], mode=mode)
    c = onp.convolve([1, 2], [2, 5, 10], mode=mode)
    d = onp.convolve(onp.array([1, 2, 3, 4, 5]), onp.array([1, 2, 3, 4, 5]), mode=mode)
    e = onp.convolve([1, 2, 3, 4, 5], 2, mode=mode)
    return a, b, c, d, e


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_convolve():
    for mode in ['full', 'same', 'valid']:
        mnp_res = mnp_convolve(mode)
        onp_res = onp_convolve(mode)
        match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cov():
    x = onp.random.random((3, 4)).tolist()
    mnp_res = mnp.cov(x)
    onp_res = onp.cov(x)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x[0])
    onp_res = onp.cov(x[0])
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    w1 = [0, 1, 2, 3]
    w2 = [4, 5, 6, 7]
    mnp_res = mnp.cov(x, fweights=w1)
    onp_res = onp.cov(x, fweights=w1)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x, aweights=w2)
    onp_res = onp.cov(x, aweights=w2)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x, fweights=w1, aweights=w2)
    onp_res = onp.cov(x, fweights=w1, aweights=w2)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x, fweights=w1, aweights=w2, ddof=3)
    onp_res = onp.cov(x, fweights=w1, aweights=w2, ddof=3)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x, fweights=w1, aweights=w2, bias=True)
    onp_res = onp.cov(x, fweights=w1, aweights=w2, bias=True)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.cov(x, fweights=w1[0:3], aweights=w2[0:3], rowvar=False, bias=True)
    onp_res = onp.cov(x, fweights=w1[0:3], aweights=w2[0:3], rowvar=False, bias=True)
    match_all_arrays(mnp_res, onp_res, error=1e-5)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trapz():
    y = rand_int(2, 3, 4, 5)
    match_res(mnp.trapz, onp.trapz, y)
    match_res(mnp.trapz, onp.trapz, y, x=[-5, -3, 0, 7, 10])
    match_res(mnp.trapz, onp.trapz, y, dx=2, axis=3)
    match_res(mnp.trapz, onp.trapz, y, x=[1, 5, 6, 9], dx=3, axis=-2)


def mnp_gcd(x, y):
    return mnp.gcd(x, y)


def onp_gcd(x, y):
    return onp.gcd(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gcd():
    x = onp.arange(-12, 12).reshape(2, 3, 4)
    y = onp.arange(24).reshape(2, 3, 4)
    match_res(mnp_gcd, onp_gcd, x, y)


def mnp_lcm(x, y):
    return mnp.lcm(x, y)


def onp_lcm(x, y):
    return onp.lcm(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lcm():
    x = onp.arange(-12, 12).reshape(2, 3, 4)
    y = onp.arange(24).reshape(2, 3, 4)
    match_res(mnp_lcm, onp_lcm, x, y)


def mnp_nansum(x):
    a = mnp.nansum(x)
    b = mnp.nansum(x, keepdims=True)
    c = mnp.nansum(x, axis=-2)
    d = mnp.nansum(x, axis=0, keepdims=True)
    e = mnp.nansum(x, axis=(-2, 3))
    f = mnp.nansum(x, axis=(-3, -1), keepdims=True)
    return a, b, c, d, e, f


def onp_nansum(x):
    a = onp.nansum(x)
    b = onp.nansum(x, keepdims=True)
    c = onp.nansum(x, axis=-2)
    d = onp.nansum(x, axis=0, keepdims=True)
    e = onp.nansum(x, axis=(-2, 3))
    f = onp.nansum(x, axis=(-3, -1), keepdims=True)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nansum():
    x = rand_int(2, 3, 4, 5)
    x[0][2][1][3] = onp.nan
    x[1][0][2][4] = onp.nan
    x[1][1][1][1] = onp.nan
    run_multi_test(mnp_nansum, onp_nansum, (x,))


def mnp_nanmean(x):
    a = mnp.nanmean(x)
    b = mnp.nanmean(x, keepdims=True)
    c = mnp.nanmean(x, axis=-2)
    d = mnp.nanmean(x, axis=0, keepdims=True)
    e = mnp.nanmean(x, axis=(-2, 3))
    f = mnp.nanmean(x, axis=(-3, -1), keepdims=True)
    return a, b, c, d, e, f


def onp_nanmean(x):
    a = onp.nanmean(x)
    b = onp.nanmean(x, keepdims=True)
    c = onp.nanmean(x, axis=-2)
    d = onp.nanmean(x, axis=0, keepdims=True)
    e = onp.nanmean(x, axis=(-2, 3))
    f = onp.nanmean(x, axis=(-3, -1), keepdims=True)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nanmean():
    x = rand_int(2, 3, 4, 5)
    x[0][2][1][3] = onp.nan
    x[1][0][2][4] = onp.nan
    x[1][1][1][1] = onp.nan
    run_multi_test(mnp_nanmean, onp_nanmean, (x,))


def mnp_mean(*arrs):
    arr1 = arrs[0]
    arr2 = arrs[1]
    arr3 = arrs[2]
    a = mnp.mean(arr1)
    b = mnp.mean(arr2, keepdims=True)
    c = mnp.mean(arr3, keepdims=False)
    d = mnp.mean(arr2, axis=0, keepdims=True)
    e = mnp.mean(arr3, axis=(0, -1))
    f = mnp.mean(arr3, axis=-1, keepdims=True)
    return a, b, c, d, e, f


def onp_mean(*arrs):
    arr1 = arrs[0]
    arr2 = arrs[1]
    arr3 = arrs[2]
    a = onp.mean(arr1)
    b = onp.mean(arr2, keepdims=True)
    c = onp.mean(arr3, keepdims=False)
    d = onp.mean(arr2, axis=0, keepdims=True)
    e = onp.mean(arr3, axis=(0, -1))
    f = onp.mean(arr3, axis=-1, keepdims=True)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mean():
    run_multi_test(mnp_mean, onp_mean, test_case.arrs, error=3)
    run_multi_test(mnp_mean, onp_mean, test_case.expanded_arrs, error=3)
    run_multi_test(mnp_mean, onp_mean, test_case.scalars, error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_innner():
    with pytest.raises(ValueError):
        mnp.inner(to_tensor(test_case.arrs[0]),
                  to_tensor(test_case.arrs[1]))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_add():
    with pytest.raises(ValueError):
        mnp.add(to_tensor(test_case.arrs[1]), to_tensor(test_case.arrs[2]))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_mean():
    with pytest.raises(ValueError):
        mnp.mean(to_tensor(test_case.arrs[0]), (-1, 0))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_amax():
    with pytest.raises(TypeError):
        mnp.amax(mnp.array([[1, 2], [3, 4]]).astype(mnp.float32), initial=[1.0, 2.0])
