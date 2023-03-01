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
"""unit tests for numpy array operations"""

import pytest
import numpy as onp
import mindspore.numpy as mnp
import mindspore.ops.functional as F
from mindspore import context
from mindspore import set_seed
from mindspore.common import dtype as mstype
from mindspore.common.api import _pynative_executor

from .utils import rand_int, rand_bool, match_array, match_res, match_meta, \
    match_all_arrays, run_multi_test, to_tensor

context.set_context(mode=context.PYNATIVE_MODE)

class Cases():
    def __init__(self):
        self.all_shapes = [
            1, 2, (1,), (2,), (1, 2, 3), [1], [2], [1, 2, 3]
        ]
        self.onp_dtypes = [onp.int32, 'int32', int,
                           onp.float32, 'float32', float,
                           onp.uint32, 'uint32',
                           onp.bool_, 'bool', bool]

        self.mnp_dtypes = [mnp.int32, 'int32', int,
                           mnp.float32, 'float32', float,
                           mnp.uint32, 'uint32',
                           mnp.bool_, 'bool', bool]

        self.empty_support_type = [mnp.int32, mnp.float32, mnp.uint32, mnp.bool_]

        self.array_sets = [1, 1.1, True, [1, 0, True], [1, 1.0, 2], (1,),
                           [(1, 2, 3), (4, 5, 6)], onp.random.random(  # pylint: disable=no-member
                               (100, 100)).astype(onp.float32).tolist(),
                           onp.random.random((100, 100)).astype(onp.bool).tolist()]

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
            rand_int(1, 1, 1),
        ]

        # arrays of the same size expanded across the 0th dimension
        self.expanded_arrs = [
            rand_int(2, 3),
            rand_int(1, 2, 3),
            rand_int(1, 1, 2, 3),
            rand_int(1, 1, 1, 2, 3),
        ]

        # arrays with dimensions of size 1
        self.nested_arrs = [
            rand_int(1),
            rand_int(1, 2),
            rand_int(3, 1, 8),
            rand_int(1, 3, 9, 1),
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

        self.mnp_prototypes = [
            mnp.ones((2, 3, 4)),
            mnp.ones((1, 3, 1, 2, 5)),
            mnp.ones((2, 7, 1)),
            [mnp.ones(3), (1, 2, 3), mnp.ones(3), [4, 5, 6]],
            ([(1, 2), mnp.ones(2)], (mnp.ones(2), [3, 4])),
        ]

        self.onp_prototypes = [
            onp.ones((2, 3, 4)),
            onp.ones((1, 3, 1, 2, 5)),
            onp.ones((2, 7, 1)),
            [onp.ones(3), (1, 2, 3), onp.ones(3), [4, 5, 6]],
            ([(1, 2), onp.ones(2)], (onp.ones(2), [3, 4])),
        ]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_asarray():
    test_case = Cases()
    for array in test_case.array_sets:
        # Check for dtype matching
        actual = onp.asarray(array)
        expected = mnp.asarray(array).asnumpy()
        # Since we set float32/int32 as the default dtype in mindspore, we need
        # to make a conversion between numpy.asarray and mindspore.numpy.asarray
        if actual.dtype is onp.dtype('float64'):
            assert expected.dtype == onp.dtype('float32')
        elif actual.dtype is onp.dtype('int64'):
            assert expected.dtype == onp.dtype('int32')
        else:
            assert actual.dtype == expected.dtype
        match_array(actual, expected, error=7)

        for i in range(len(test_case.onp_dtypes)):
            actual = onp.asarray(array, test_case.onp_dtypes[i])
            expected = mnp.asarray(array, test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected, error=7)

    # Additional tests for nested tensor mixture
    mnp_input = [(mnp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.asarray(onp_input)
    expected = mnp.asarray(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_array():
    # array's function is very similar to asarray, so we mainly test the
    # `copy` argument.
    test_case = Cases()
    for array in test_case.array_sets:
        arr1 = mnp.asarray(array)
        arr2 = mnp.array(arr1, copy=False)
        arr3 = mnp.array(arr1)
        arr4 = mnp.asarray(array, dtype='int32')
        arr5 = mnp.asarray(arr4, dtype=mnp.int32)
        assert arr1 is arr2
        assert arr1 is not arr3
        assert arr4 is arr5

    # Additional tests for nested tensor/numpy_array mixture
    mnp_input = [(mnp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.array(onp_input)
    expected = mnp.array(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_asfarray():
    test_case = Cases()
    for array in test_case.array_sets:
        # Check for dtype matching
        actual = onp.asfarray(array)
        expected = mnp.asfarray(array).asnumpy()
        # Since we set float32/int32 as the default dtype in mindspore, we need
        # to make a conversion between numpy.asarray and mindspore.numpy.asarray
        if actual.dtype is onp.dtype('float64'):
            assert expected.dtype == onp.dtype('float32')
        else:
            assert actual.dtype == expected.dtype
        match_array(actual, expected, error=7)

        for i in range(len(test_case.onp_dtypes)):
            actual = onp.asfarray(array, test_case.onp_dtypes[i])
            expected = mnp.asfarray(array, test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected, error=7)

    # Additional tests for nested tensor/numpy_array mixture
    mnp_input = [(mnp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.asfarray(onp_input)
    expected = mnp.asfarray(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_zeros():
    test_case = Cases()
    for shape in test_case.all_shapes:
        for i in range(len(test_case.onp_dtypes)):
            actual = onp.zeros(shape, test_case.onp_dtypes[i])
            expected = mnp.zeros(shape, test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected)
        actual = onp.zeros(shape)
        expected = mnp.zeros(shape).asnumpy()
        match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ones():
    test_case = Cases()
    for shape in test_case.all_shapes:
        for i in range(len(test_case.onp_dtypes)):
            actual = onp.ones(shape, test_case.onp_dtypes[i])
            expected = mnp.ones(shape, test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected)
        actual = onp.ones(shape)
        expected = mnp.ones(shape).asnumpy()
        match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_full():
    actual = onp.full((2, 2), [1, 2])
    expected = mnp.full((2, 2), [1, 2]).asnumpy()
    match_array(actual, expected)

    actual = onp.full((2, 3), True)
    expected = mnp.full((2, 3), True).asnumpy()
    match_array(actual, expected)

    actual = onp.full((3, 4, 5), 7.5)
    expected = mnp.full((3, 4, 5), 7.5).asnumpy()
    match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_eye():
    test_case = Cases()
    for i in range(len(test_case.onp_dtypes)):
        for m in range(1, 5):
            actual = onp.eye(m, dtype=test_case.onp_dtypes[i])
            expected = mnp.eye(m, dtype=test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected)
            for n in range(1, 5):
                for k in range(0, 5):
                    actual = onp.eye(m, n, k, dtype=test_case.onp_dtypes[i])
                    expected = mnp.eye(
                        m, n, k, dtype=test_case.mnp_dtypes[i]).asnumpy()
                    match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_identity():
    test_case = Cases()
    for i in range(len(test_case.onp_dtypes)):
        for m in range(1, 5):
            actual = onp.identity(m, dtype=test_case.onp_dtypes[i])
            expected = mnp.identity(m, dtype=test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arange():
    actual = onp.arange(10)
    expected = mnp.arange(10).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(0, 10)
    expected = mnp.arange(0, 10).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(10, step=0.1)
    expected = mnp.arange(10, step=0.1).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.arange(0.1, 9.9)
    expected = mnp.arange(0.1, 9.9).asnumpy()
    match_array(actual, expected, error=6)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linspace():
    actual = onp.linspace(2.0, 3.0, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.linspace(2.0, 3.0, num=5, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.linspace(
        2.0, 3.0, num=5, endpoint=False, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5, endpoint=False).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.linspace(2.0, 3.0, num=5, retstep=True, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5, retstep=True)
    match_array(actual[0], expected[0].asnumpy())
    assert actual[1] == expected[1].asnumpy()

    actual = onp.linspace(2.0, [3, 4, 5], num=5,
                          endpoint=False, dtype=onp.float32)
    expected = mnp.linspace(
        2.0, [3, 4, 5], num=5, endpoint=False).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.linspace(2.0, [[3, 4, 5]], num=5, endpoint=False, axis=2)
    expected = mnp.linspace(2.0, [[3, 4, 5]], num=5, endpoint=False, axis=2).asnumpy()
    match_array(actual, expected, error=6)

    start = onp.random.random([2, 1, 4]).astype("float32")
    stop = onp.random.random([1, 5, 1]).astype("float32")
    actual = onp.linspace(start, stop, num=20, retstep=True,
                          endpoint=False, dtype=onp.float32)
    expected = mnp.linspace(to_tensor(start), to_tensor(stop), num=20,
                            retstep=True, endpoint=False)
    match_array(actual[0], expected[0].asnumpy(), error=6)
    match_array(actual[1], expected[1].asnumpy(), error=6)

    actual = onp.linspace(start, stop, num=20, retstep=True,
                          endpoint=False, dtype=onp.int16)
    expected = mnp.linspace(to_tensor(start), to_tensor(stop), num=20,
                            retstep=True, endpoint=False, dtype=mnp.int16)
    match_array(actual[0], expected[0].asnumpy(), error=6)
    match_array(actual[1], expected[1].asnumpy(), error=6)

    for axis in range(2):
        actual = onp.linspace(start, stop, num=20, retstep=False,
                              endpoint=False, dtype=onp.float32, axis=axis)
        expected = mnp.linspace(to_tensor(start), to_tensor(stop), num=20,
                                retstep=False, endpoint=False, dtype=mnp.float32, axis=axis)
        match_array(actual, expected.asnumpy(), error=6)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logspace():
    actual = onp.logspace(2.0, 3.0, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0).asnumpy()
    match_array(actual, expected, error=3)

    actual = onp.logspace(2.0, 3.0, num=5, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0, num=5).asnumpy()
    match_array(actual, expected, error=3)

    actual = onp.logspace(
        2.0, 3.0, num=5, endpoint=False, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0, num=5, endpoint=False).asnumpy()
    match_array(actual, expected, error=3)

    actual = onp.logspace(2.0, [3, 4, 5], num=5, base=2,
                          endpoint=False, dtype=onp.float32)
    expected = mnp.logspace(
        2.0, [3, 4, 5], num=5, base=2, endpoint=False).asnumpy()
    match_array(actual, expected, error=3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty():
    test_case = Cases()
    for shape in test_case.all_shapes:
        for mnp_dtype, onp_dtype in zip(test_case.mnp_dtypes,
                                        test_case.onp_dtypes):
            if mnp_dtype not in test_case.empty_support_type:
                continue
            actual = mnp.empty(shape, mnp_dtype).asnumpy()
            expected = onp.empty(shape, onp_dtype)
            match_meta(actual, expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_like():
    test_case = Cases()
    for mnp_proto, onp_proto in zip(test_case.mnp_prototypes, test_case.onp_prototypes):
        actual = mnp.empty_like(mnp_proto).asnumpy()
        expected = onp.empty_like(onp_proto)
        assert actual.shape == expected.shape

        for mnp_dtype, onp_dtype in zip(test_case.mnp_dtypes,
                                        test_case.onp_dtypes):
            if mnp_dtype not in test_case.empty_support_type:
                continue
            actual = mnp.empty_like(mnp_proto, dtype=mnp_dtype).asnumpy()
            expected = onp.empty_like(onp_proto, dtype=onp_dtype)
            match_meta(actual, expected)


def run_x_like(mnp_fn, onp_fn):
    test_case = Cases()
    for mnp_proto, onp_proto in zip(test_case.mnp_prototypes, test_case.onp_prototypes):
        actual = mnp_fn(mnp_proto).asnumpy()
        expected = onp_fn(onp_proto)
        match_array(actual, expected)

        for shape in test_case.all_shapes:
            actual = mnp_fn(mnp_proto, shape=shape).asnumpy()
            expected = onp_fn(onp_proto, shape=shape)
            match_array(actual, expected)
            for mnp_dtype, onp_dtype in zip(test_case.mnp_dtypes,
                                            test_case.onp_dtypes):
                actual = mnp_fn(mnp_proto, dtype=mnp_dtype).asnumpy()
                expected = onp_fn(onp_proto, dtype=onp_dtype)
                match_array(actual, expected)

                actual = mnp_fn(mnp_proto, dtype=mnp_dtype,
                                shape=shape).asnumpy()
                expected = onp_fn(onp_proto, dtype=onp_dtype, shape=shape)
                match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ones_like():
    run_x_like(mnp.ones_like, onp.ones_like)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_zeros_like():
    run_x_like(mnp.zeros_like, onp.zeros_like)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_full_like():
    test_case = Cases()
    for mnp_proto, onp_proto in zip(test_case.mnp_prototypes, test_case.onp_prototypes):
        shape = onp.zeros_like(onp_proto).shape
        fill_value = rand_int()
        actual = mnp.full_like(mnp_proto, to_tensor(fill_value)).asnumpy()
        expected = onp.full_like(onp_proto, fill_value)
        match_array(actual, expected)

        for i in range(len(shape) - 1, 0, -1):
            fill_value = rand_int(*shape[i:])
            actual = mnp.full_like(mnp_proto, to_tensor(fill_value)).asnumpy()
            expected = onp.full_like(onp_proto, fill_value)
            match_array(actual, expected)

            fill_value = rand_int(1, *shape[i + 1:])
            actual = mnp.full_like(mnp_proto, to_tensor(fill_value)).asnumpy()
            expected = onp.full_like(onp_proto, fill_value)
            match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tri_triu_tril():
    x = mnp.ones((16, 32), dtype="bool")
    match_array(mnp.tril(x).asnumpy(), onp.tril(x.asnumpy()))
    match_array(mnp.tril(x, -1).asnumpy(), onp.tril(x.asnumpy(), -1))
    match_array(mnp.triu(x).asnumpy(), onp.triu(x.asnumpy()))
    match_array(mnp.triu(x, -1).asnumpy(), onp.triu(x.asnumpy(), -1))

    x = mnp.ones((64, 64), dtype="uint8")
    match_array(mnp.tril(x).asnumpy(), onp.tril(x.asnumpy()))
    match_array(mnp.tril(x, 25).asnumpy(), onp.tril(x.asnumpy(), 25))
    match_array(mnp.triu(x).asnumpy(), onp.triu(x.asnumpy()))
    match_array(mnp.triu(x, 25).asnumpy(), onp.triu(x.asnumpy(), 25))

    match_array(mnp.tri(64, 64).asnumpy(), onp.tri(64, 64))
    match_array(mnp.tri(64, 64, -10).asnumpy(), onp.tri(64, 64, -10))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nancumsum():
    x = rand_int(2, 3, 4, 5)
    x[0][2][1][3] = onp.nan
    x[1][0][2][4] = onp.nan
    x[1][1][1][1] = onp.nan
    match_res(mnp.nancumsum, onp.nancumsum, x)
    match_res(mnp.nancumsum, onp.nancumsum, x, axis=-2)
    match_res(mnp.nancumsum, onp.nancumsum, x, axis=0)
    match_res(mnp.nancumsum, onp.nancumsum, x, axis=3)


def mnp_diagonal(arr):
    return mnp.diagonal(arr, offset=2, axis1=-1, axis2=0)


def onp_diagonal(arr):
    return onp.diagonal(arr, offset=2, axis1=-1, axis2=0)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_diagonal():

    arr = rand_int(3, 5)
    for i in [-1, 0, 2]:
        match_res(mnp.diagonal, onp.diagonal, arr, offset=i, axis1=0, axis2=1)
        match_res(mnp.diagonal, onp.diagonal, arr, offset=i, axis1=1, axis2=0)

    arr = rand_int(7, 4, 9)
    for i in [-1, 0, 2]:
        match_res(mnp.diagonal, onp.diagonal, arr, offset=i, axis1=0, axis2=-1)
        match_res(mnp.diagonal, onp.diagonal, arr, offset=i, axis1=-2, axis2=2)
        match_res(mnp.diagonal, onp.diagonal, arr,
                  offset=i, axis1=-1, axis2=-2)


def mnp_trace(arr):
    return mnp.trace(arr, offset=4, axis1=1, axis2=2)


def onp_trace(arr):
    return onp.trace(arr, offset=4, axis1=1, axis2=2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_trace():

    arr = rand_int(3, 5)
    match_res(mnp.trace, onp.trace, arr, offset=-1, axis1=0, axis2=1)

    arr = rand_int(7, 4, 9)
    match_res(mnp.trace, onp.trace, arr, offset=0, axis1=-2, axis2=2)


def mnp_meshgrid(*xi):
    a = mnp.meshgrid(*xi)
    b = mnp.meshgrid(*xi, sparse=True)
    c = mnp.meshgrid(*xi, indexing='ij')
    d = mnp.meshgrid(*xi, sparse=False, indexing='ij')
    return a, b, c, d


def onp_meshgrid(*xi):
    a = onp.meshgrid(*xi)
    b = onp.meshgrid(*xi, sparse=True)
    c = onp.meshgrid(*xi, indexing='ij')
    d = onp.meshgrid(*xi, sparse=False, indexing='ij')
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_meshgrid():
    xi = (onp.full(3, 2), onp.full(1, 5), onp.full(
        (2, 3), 9), onp.full((4, 5, 6), 7))
    for i in range(len(xi)):
        arrs = xi[i:]
        mnp_arrs = map(to_tensor, arrs)
        for mnp_res, onp_res in zip(mnp_meshgrid(*mnp_arrs), onp_meshgrid(*arrs)):
            match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_diagflat():
    arrs = [rand_int(2, 3)]
    for arr in arrs:
        for i in [-2, 0, 7]:
            match_res(mnp.diagflat, onp.diagflat, arr, k=i)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_diag():
    arrs = [rand_int(7), rand_int(5, 5), rand_int(3, 8), rand_int(9, 6)]
    for arr in arrs:
        for i in [-10, -5, -1, 0, 2, 5, 6, 10]:
            match_res(mnp.diag, onp.diag, arr, k=i)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_diag_indices():
    mnp_res = mnp.diag_indices(5, 7)
    onp_res = onp.diag_indices(5, 7)
    match_all_arrays(mnp_res, onp_res)


def mnp_ix_(*args):
    return mnp.ix_(*args)


def onp_ix_(*args):
    return onp.ix_(*args)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ix_():
    arrs = [rand_int(i + 1) for i in range(10)]
    for i in range(10):
        test_arrs = arrs[:i + 1]
        match_res(mnp_ix_, onp_ix_, *test_arrs)


def mnp_indices():
    a = mnp.indices((2, 3))
    b = mnp.indices((2, 3, 4), sparse=True)
    return a, b


def onp_indices():
    a = onp.indices((2, 3))
    b = onp.indices((2, 3, 4), sparse=True)
    return a, b


def test_indices():
    run_multi_test(mnp_indices, onp_indices, ())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_geomspace():
    start = onp.arange(1, 7).reshape(2, 3)
    end = [1000, 2000, 3000]
    match_array(mnp.geomspace(1, 256, num=9).asnumpy(),
                onp.geomspace(1, 256, num=9), error=1)
    match_array(mnp.geomspace(1, 256, num=8, endpoint=False).asnumpy(),
                onp.geomspace(1, 256, num=8, endpoint=False), error=1)
    match_array(mnp.geomspace(to_tensor(start), end, num=4).asnumpy(),
                onp.geomspace(start, end, num=4), error=1)
    match_array(mnp.geomspace(to_tensor(start), end, num=4, endpoint=False).asnumpy(),
                onp.geomspace(start, end, num=4, endpoint=False), error=1)
    match_array(mnp.geomspace(to_tensor(start), end, num=4, axis=-1).asnumpy(),
                onp.geomspace(start, end, num=4, axis=-1), error=1)
    match_array(mnp.geomspace(to_tensor(start), end, num=4, endpoint=False, axis=-1).asnumpy(),
                onp.geomspace(start, end, num=4, endpoint=False, axis=-1), error=1)

    start = onp.arange(1, 1 + 2*3*4*5).reshape(2, 3, 4, 5)
    end = [1000, 2000, 3000, 4000, 5000]
    for i in range(-5, 5):
        match_array(mnp.geomspace(to_tensor(start), end, num=4, axis=i).asnumpy(),
                    onp.geomspace(start, end, num=4, axis=i), error=1)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vander():
    arrs = [rand_int(i + 3) for i in range(3)]
    for i in range(3):
        mnp_vander = mnp.vander(to_tensor(arrs[i]))
        onp_vander = onp.vander(arrs[i])
        match_all_arrays(mnp_vander, onp_vander, error=1e-4)
        mnp_vander = mnp.vander(to_tensor(arrs[i]), N=2, increasing=True)
        onp_vander = onp.vander(arrs[i], N=2, increasing=True)
        match_all_arrays(mnp_vander, onp_vander, error=1e-4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bartlett():
    for i in [-3, -1, 0, 1, 5, 6, 10, 15]:
        match_all_arrays(mnp.bartlett(i), onp.bartlett(i), error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_blackman():
    for i in [-3, -1, 0, 1, 5, 6, 10, 15]:
        match_all_arrays(mnp.blackman(i), onp.blackman(i), error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hamming():
    for i in [-3, -1, 0, 1, 5, 6, 10, 15]:
        match_all_arrays(mnp.hamming(i), onp.hamming(i), error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hanning():
    for i in [-3, -1, 0, 1, 5, 6, 10, 15]:
        match_all_arrays(mnp.hanning(i), onp.hanning(i), error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_triu_indices():
    m = rand_int().tolist()
    n = rand_int().tolist()
    k = rand_int().tolist()
    mnp_res = mnp.triu_indices(n, k, m)
    onp_res = onp.triu_indices(n, k, m)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tril_indices():
    m = rand_int().tolist()
    n = rand_int().tolist()
    k = rand_int().tolist()
    mnp_res = mnp.tril_indices(n, k, m)
    onp_res = onp.tril_indices(n, k, m)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_triu_indices_from():
    m = int(rand_int().tolist())
    n = int(rand_int().tolist())
    t = mnp.asarray(rand_int(m, n).tolist())
    k = rand_int().tolist()
    mnp_res = mnp.triu_indices_from(t, k)
    onp_res = onp.triu_indices_from(t.asnumpy(), k)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tril_indices_from():
    m = int(rand_int().tolist())
    n = int(rand_int().tolist())
    t = mnp.asarray(rand_int(m, n).tolist())
    k = rand_int().tolist()
    mnp_res = mnp.tril_indices_from(t, k)
    onp_res = onp.tril_indices_from(t.asnumpy(), k)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_histogram_bin_edges():
    x = onp.random.randint(-10, 10, 10)
    match_res(mnp.histogram_bin_edges, onp.histogram_bin_edges, x, onp.arange(5))
    match_res(mnp.histogram_bin_edges, onp.histogram_bin_edges, x, bins=(1, 2, 3), range=None, error=3)
    match_res(mnp.histogram_bin_edges, onp.histogram_bin_edges, x, bins=10, range=(2, 20), error=3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_randn():
    """
    Feature: Numpy method randn.
    Description: Test numpy method randn.
    Expectation: No exception.
    """
    set_seed(1)
    t1 = mnp.randn(1, 2, 3)
    t2 = mnp.randn(1, 2, 3)
    assert onp.array_equal(t1.asnumpy(), t2.asnumpy()) is False

    set_seed(1)
    t3 = mnp.randn(1, 2, 3)
    assert (t1.asnumpy() == t3.asnumpy()).all()

    with pytest.raises(ValueError):
        mnp.randn(dtype="int32")
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.randn(dtype=mstype.int32)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        mnp.randn({1})
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        mnp.randn(1, 1.2, 2)
        _pynative_executor.sync()



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rand():
    """
    Feature: Numpy method rand.
    Description: Test numpy method rand.
    Expectation: No exception.
    """
    set_seed(1)
    t1 = mnp.rand(1, 2, 3)
    t2 = mnp.rand(1, 2, 3)
    assert onp.array_equal(t1.asnumpy(), t2.asnumpy()) is False

    set_seed(1)
    t3 = mnp.rand(1, 2, 3)
    assert (t1.asnumpy() == t3.asnumpy()).all()

    with pytest.raises(ValueError):
        mnp.rand(dtype="int32")
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.rand(dtype=mstype.int32)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        mnp.rand({1})
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        mnp.rand(1, 1.2, 2)
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_randint():
    """
    Feature: Numpy method randint.
    Description: Test numpy method randint.
    Expectation: No exception.
    """
    set_seed(1)
    t1 = mnp.randint(1, 5, 3)
    t2 = mnp.randint(1, 5, 3)
    assert onp.array_equal(t1.asnumpy(), t2.asnumpy()) is False

    set_seed(1)
    t3 = mnp.randint(1, 5, 3)
    assert (t1.asnumpy() == t3.asnumpy()).all()

    with pytest.raises(TypeError):
        mnp.randint(1.2)
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.randint(0)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        mnp.randint(1, 1.2)
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.randint(2, 1)
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.randint(1, dtype="float")
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        mnp.randint(1, dtype=mstype.float32)
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ops_arange():
    """
    Feature: Ops function arange.
    Description: Test ops function arange.
    Expectation: No exception.
    """
    actual = onp.arange(5)
    expected = F.arange(5).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(0, 5)
    expected = F.arange(0, 5).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(5, step=0.2)
    expected = F.arange(5, step=0.2).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(0.1, 0.9)
    expected = F.arange(0.1, 0.9).asnumpy()
    match_array(actual, expected)

    with pytest.raises(TypeError):
        F.arange([1])
        _pynative_executor.sync()
    with pytest.raises(ValueError):
        F.arange(10, 1)
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_asarray_exception():
    with pytest.raises(TypeError):
        mnp.asarray({1, 2, 3})
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_linspace_exception():
    with pytest.raises(TypeError):
        mnp.linspace(0, 1, num=2.5)
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_empty_like_exception():
    with pytest.raises(ValueError):
        mnp.empty_like([[1, 2, 3], [4, 5]])
        _pynative_executor.sync()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pad():
    x_np = onp.random.random([2, 3, 4]).astype("float32")
    x_ms = mnp.asarray(x_np.tolist())

    # pad constant
    mnp_res = mnp.pad(x_ms, ((1, 1), (2, 2), (3, 4)))
    onp_res = onp.pad(x_np, ((1, 1), (2, 2), (3, 4)))
    match_all_arrays(mnp_res, onp_res, error=1e-5)
    mnp_res = mnp.pad(x_ms, ((1, 1), (2, 3), (4, 5)), constant_values=((3, 4), (5, 6), (7, 8)))
    onp_res = onp.pad(x_np, ((1, 1), (2, 3), (4, 5)), constant_values=((3, 4), (5, 6), (7, 8)))
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad statistic
    mnp_res = mnp.pad(x_ms, ((1, 1), (2, 2), (3, 4)), mode="mean", stat_length=((1, 2), (2, 10), (3, 4)))
    onp_res = onp.pad(x_np, ((1, 1), (2, 2), (3, 4)), mode="mean", stat_length=((1, 2), (2, 10), (3, 4)))
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad edge
    mnp_res = mnp.pad(x_ms, ((1, 1), (2, 2), (3, 4)), mode="edge")
    onp_res = onp.pad(x_np, ((1, 1), (2, 2), (3, 4)), mode="edge")
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad wrap
    mnp_res = mnp.pad(x_ms, ((1, 1), (2, 2), (3, 4)), mode="wrap")
    onp_res = onp.pad(x_np, ((1, 1), (2, 2), (3, 4)), mode="wrap")
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad linear_ramp
    mnp_res = mnp.pad(x_ms, ((1, 3), (5, 2), (3, 0)), mode="linear_ramp", end_values=((0, 10), (9, 1), (-10, 99)))
    onp_res = onp.pad(x_np, ((1, 3), (5, 2), (3, 0)), mode="linear_ramp", end_values=((0, 10), (9, 1), (-10, 99)))
    match_all_arrays(mnp_res, onp_res, error=1e-5)


def pad_with_msfunc(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def pad_with_npfunc(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_gpu():
    x_np = onp.random.random([2, 1, 4, 3]).astype("float32")
    x_ms = mnp.asarray(x_np.tolist())

    # pad symmetric odd
    mnp_res = mnp.pad(x_ms, ((10, 3), (5, 2), (3, 0), (2, 6)), mode='symmetric', reflect_type='odd')
    onp_res = onp.pad(x_np, ((10, 3), (5, 2), (3, 0), (2, 6)), mode='symmetric', reflect_type='odd')
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad symmetric even
    mnp_res = mnp.pad(x_ms, ((10, 13), (5, 12), (3, 0), (2, 6)), mode='symmetric', reflect_type='even')
    onp_res = onp.pad(x_np, ((10, 13), (5, 12), (3, 0), (2, 6)), mode='symmetric', reflect_type='even')
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad reflect odd
    mnp_res = mnp.pad(x_ms, ((10, 3), (5, 2), (3, 0), (2, 6)), mode='reflect', reflect_type='odd')
    onp_res = onp.pad(x_np, ((10, 3), (5, 2), (3, 0), (2, 6)), mode='reflect', reflect_type='odd')
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad reflect even
    mnp_res = mnp.pad(x_ms, ((10, 13)), mode='reflect', reflect_type='even')
    onp_res = onp.pad(x_np, ((10, 13)), mode='reflect', reflect_type='even')
    match_all_arrays(mnp_res, onp_res, error=1e-5)

    # pad func
    x_np = onp.random.random([2, 4]).astype("float32")
    x_ms = mnp.asarray(x_np.tolist())
    mnp_res = mnp.pad(x_ms, ((5, 5)), mode=pad_with_msfunc, padder=99)
    onp_res = onp.pad(x_np, ((5, 5)), mode=pad_with_npfunc, padder=99)
    match_all_arrays(mnp_res, onp_res, error=1e-5)
