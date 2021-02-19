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
    run_single_test, match_res, match_array

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

        # empty arrays
        self.empty_arrs = [
            rand_int(0),
            rand_int(4, 0),
            rand_int(2, 0, 2),
            rand_int(5, 0, 7, 0),
        ]

        # arrays of the same size expanded across the 0th dimension
        self.expanded_arrs = [
            rand_int(2, 3),
            rand_int(1, 2, 3),
            rand_int(1, 1, 2, 3),
            rand_int(1, 1, 1, 2, 3),
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
            rand_int(8, 1, 6, 1)
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
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_minimum():
    run_binop_test(mnp_minimum, onp_minimum, test_case)


def mnp_add_kwargs(x, y, where=None, out=None):
    return mnp.add(x, y, where=where, out=out)


def onp_add_kwargs(x, y, where=None, out=None):
    return onp.add(x, y, where=where, out=out)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add_kwargs():
    for where in test_case.bool_broadcastables[:2]:
        for x in test_case.broadcastables[:2]:
            for y in test_case.broadcastables[:2]:
                shape_out = onp.broadcast(where, x, y).shape
                out = rand_int(*shape_out)
                match_res(mnp_add_kwargs, onp_add_kwargs, x, y, where, out)


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


def mnp_var(x):
    a = mnp.std(x)
    b = mnp.std(x, axis=0)
    c = mnp.std(x, axis=(0))
    d = mnp.std(x, axis=(0, 1, 2))
    e = mnp.std(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
    return a, b, c, d, e


def onp_var(x):
    a = onp.std(x)
    b = onp.std(x, axis=0)
    c = onp.std(x, axis=(0))
    d = onp.std(x, axis=(0, 1, 2))
    e = onp.std(x, axis=(-1, 1, 2), ddof=1, keepdims=True)
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

    a = mnp.asarray(arr, dtype='float16')
    b = mnp.asarray(arr, dtype='float32')
    c = mnp.asarray(arr, dtype='int32')

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

    a = mnp.asarray(arr, dtype='float16')
    b = mnp.asarray(arr, dtype='float32')
    c = mnp.asarray(arr, dtype='uint8')
    d = mnp.asarray(arr, dtype='bool')

    match_array(mnp_absolute(a).asnumpy(), onp_absolute(a.asnumpy()))
    match_array(mnp_absolute(b).asnumpy(), onp_absolute(b.asnumpy()))
    match_array(mnp_absolute(c).asnumpy(), onp_absolute(c.asnumpy()))
    match_array(mnp_absolute(d).asnumpy(), onp_absolute(d.asnumpy()))

    where = rand_int(2, 3).astype('bool')
    out = rand_int(2, 3)
    match_array(mnp.absolute(a, out=mnp.asarray(out), where=mnp.asarray(where)).asnumpy(),
                onp.absolute(a.asnumpy(), out=out, where=where))


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


def mnp_add_dtype(x1, x2, out, where):
    a = mnp.add(x1, x2, dtype=mnp.float16)
    b = mnp.add(x1, x2, out=out, dtype=mnp.float16)
    c = mnp.add(x1, x2, where=where, dtype=mnp.float16)
    d = mnp.add(x1, x2, out=out, where=where, dtype=mnp.float16)
    return a, b, c, d


def onp_add_dtype(x1, x2, out, where):
    a = onp.add(x1, x2, dtype=onp.float16)
    b = onp.add(x1, x2, out=out, dtype=onp.float16)
    c = onp.add(x1, x2, where=where, dtype=onp.float16)
    d = onp.add(x1, x2, out=out, where=where, dtype=onp.float16)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add_dtype():
    x1 = rand_int(2, 3).astype('int32')
    x2 = rand_int(2, 3).astype('int32')
    out = rand_int(2, 3).astype('float32')
    where = rand_bool(2, 3)
    arrs = (x1, x2, out, where)
    mnp_arrs = map(mnp.array, arrs)
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


def mnp_maximum(x1, x2):
    return mnp.maximum(x1, x2)


def onp_maximum(x1, x2):
    return onp.maximum(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maximum():
    run_binop_test(mnp_maximum, onp_maximum, test_case)


def mnp_clip(x):
    a = mnp.clip(x, mnp.asarray(10.0), mnp.asarray([2,]))
    b = mnp.clip(x, 0, 1)
    c = mnp.clip(x, mnp.asarray(0), mnp.asarray(10), dtype=mnp.float32)
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
    f = mnp.amin(x, initial=-2, where=mask)
    g = mnp.amin(x, initial=-3, where=mask, keepdims=True)
    h = mnp.amin(x, axis=(1, 2, 3), initial=-4, where=mask)
    return a, b, c, d, e, f, g, h


def onp_amin(x, mask):
    a = onp.amin(x)
    b = onp.amin(x, axis=-3)
    c = onp.amin(x, keepdims=True)
    d = onp.amin(x, initial=-1)
    e = onp.amin(x, axis=(0, 1), keepdims=True)
    f = onp.amin(x, initial=-2, where=mask)
    g = onp.amin(x, initial=-3, where=mask, keepdims=True)
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
    run_binop_test(mnp_remainder, onp_remainder, test_case)


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
    run_binop_test(mnp_mod, onp_mod, test_case)


def mnp_fmod(x, y):
    return mnp.fmod(x, y)


def onp_fmod(x, y):
    return onp.fmod(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fmod():
    run_binop_test(mnp_fmod, onp_fmod, test_case)


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
    run_binop_test(mnp_fmod, onp_fmod, test_case, error=1e-5)


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


def mnp_positive(x, out, where):
    return mnp.positive(x, out=out, where=where)


def onp_positive(x, out, where):
    return onp.positive(x, out=out, where=where)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_positive():
    arr = onp.arange(-6, 6).reshape((2, 2, 3)).astype('float32')
    out_lst = [onp.ones((2, 2, 3)).astype('float32'), onp.ones((5, 2, 2, 3)).astype('float32')]
    where_lst = [onp.full((2, 2, 3), [True, False, True]), onp.full((2, 3), False)]
    for out in out_lst:
        for where in where_lst:
            onp_pos = onp_positive(arr, out=out, where=where)
            mnp_pos = mnp_positive(mnp.asarray(arr), mnp.asarray(out), mnp.asarray(where))
            match_array(mnp_pos.asnumpy(), onp_pos)


def mnp_negative(x, out, where):
    return mnp.negative(x, out=out, where=where)


def onp_negative(x, out, where):
    return onp.negative(x, out=out, where=where)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_negative():
    arr = onp.arange(-6, 6).reshape((2, 2, 3)).astype('float32')
    out_lst = [onp.ones((2, 2, 3)).astype('float32'), onp.ones((5, 2, 2, 3)).astype('float32')]
    where_lst = [onp.full((2, 2, 3), [True, False, True]), onp.full((2, 3), False)]
    for out in out_lst:
        for where in where_lst:
            onp_neg = onp_negative(arr, out=out, where=where)
            mnp_neg = mnp_negative(mnp.asarray(arr), mnp.asarray(out), mnp.asarray(where))
            match_array(mnp_neg.asnumpy(), onp_neg, 1e-5)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_innner():
    with pytest.raises(ValueError):
        mnp.inner(mnp.asarray(test_case.arrs[0]),
                  mnp.asarray(test_case.arrs[1]))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_add():
    with pytest.raises(ValueError):
        mnp.add(mnp.asarray(test_case.arrs[1]), mnp.asarray(test_case.arrs[2]))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_exception_mean():
    with pytest.raises(ValueError):
        mnp.mean(mnp.asarray(test_case.arrs[0]), (-1, 0))
