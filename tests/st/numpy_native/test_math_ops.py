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
from functools import partial

import pytest
import numpy as onp

import mindspore.numpy as mnp
from mindspore import context


def rand_int(*shape):
    """return an random integer array with parameter shape"""
    res = onp.random.randint(low=1, high=5, size=shape)
    if isinstance(res, onp.ndarray):
        return res.astype(onp.float32)
    return float(res)


# return an random boolean array
def rand_bool(*shape):
    return onp.random.rand(*shape) > 0.5


class Cases():
    def __init__(self):
        self.device_cpu = context.get_context('device_target')

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
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


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


def mnp_power(x1, x2):
    return mnp.power(x1, x2)


def onp_power(x1, x2):
    return onp.power(x1, x2)


def mnp_inner(a, b):
    return mnp.inner(a, b)


def onp_inner(a, b):
    return onp.inner(a, b)


def mnp_dot(a, b):
    return mnp.dot(a, b)


def onp_dot(a, b):
    return onp.dot(a, b)


def mnp_outer(a, b):
    return mnp.outer(a, b)


def onp_outer(a, b):
    return onp.outer(a, b)


def mnp_add_kwargs(x, y, where=None, out=None):
    return mnp.add(x, y, where=where, out=out)


def onp_add_kwargs(x, y, where=None, out=None):
    return onp.add(x, y, where=where, out=out)


def mnp_subtract_kwargs(x, y, where=None, out=None):
    return mnp.subtract(x, y, where=where, out=out)


def onp_subtract_kwargs(x, y, where=None, out=None):
    return onp.subtract(x, y, where=where, out=out)


def mnp_multiply_kwargs(x, y, where=None, out=None):
    return mnp.multiply(x, y, where=where, out=out)


def onp_multiply_kwargs(x, y, where=None, out=None):
    return onp.multiply(x, y, where=where, out=out)


def mnp_divide_kwargs(x, y, where=None, out=None):
    return mnp.divide(x, y, where=where, out=out)


def onp_divide_kwargs(x, y, where=None, out=None):
    return onp.divide(x, y, where=where, out=out)


def mnp_power_kwargs(x, y, where=None, out=None):
    return mnp.power(x, y, where=where, out=out)


def onp_power_kwargs(x, y, where=None, out=None):
    return onp.power(x, y, where=where, out=out)


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


def run_binop_test(mnp_fn, onp_fn):
    for arr in test_case.arrs:
        match_res(mnp_fn, onp_fn, arr, arr)

        for scalar in test_case.scalars:
            match_res(mnp_fn, onp_fn, arr, scalar)
            match_res(mnp_fn, onp_fn, scalar, arr)

    for scalar1 in test_case.scalars:
        for scalar2 in test_case.scalars:
            match_res(mnp_fn, onp_fn, scalar1, scalar2)

    for expanded_arr1 in test_case.expanded_arrs:
        for expanded_arr2 in test_case.expanded_arrs:
            match_res(mnp_fn, onp_fn, expanded_arr1, expanded_arr2)

    for broadcastable1 in test_case.broadcastables:
        for broadcastable2 in test_case.broadcastables:
            match_res(mnp_fn, onp_fn, broadcastable1, broadcastable2)


def run_multi_test(mnp_fn, onp_fn, arrs):
    mnp_arrs = map(mnp.asarray, arrs)
    for actual, expected in zip(mnp_fn(*mnp_arrs), onp_fn(*arrs)):
        match_array(actual.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_add():
    run_binop_test(mnp_add, onp_add)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_subtract():
    run_binop_test(mnp_subtract, onp_subtract)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multiply():
    run_binop_test(mnp_mutiply, onp_multiply)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_divide():
    run_binop_test(mnp_divide, onp_divide)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_power():
    run_binop_test(mnp_power, onp_power)


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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_outer():
    run_binop_test(mnp_outer, onp_outer)


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


# check if the output from mnp function and onp function applied on the arrays are matched


def match_res(mnp_fn, onp_fn, *arrs):
    mnp_arrs = map(partial(mnp.asarray, dtype='float32'), arrs)
    mnp_res = mnp_fn(*mnp_arrs)
    onp_res = onp_fn(*arrs)
    if isinstance(mnp_res, (tuple, list)):
        for actual, expected in zip(mnp_res, onp_res):
            match_array(actual.asnumpy(), expected)
    else:
        match_array(mnp_res.asnumpy(), onp_res)


def match_array(actual, expected, error=5):
    if error > 0:
        onp.testing.assert_almost_equal(actual.tolist(), expected.tolist(),
                                        decimal=error)
    else:
        onp.testing.assert_equal(actual.tolist(), expected.tolist())


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
