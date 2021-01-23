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

import functools

import pytest
import numpy as onp

import mindspore.numpy as mnp
from mindspore.nn import Cell


class Cases():
    def __init__(self):
        self.all_shapes = [
            0, 1, 2, (), (1,), (2,), (1, 2, 3), [], [1], [2], [1, 2, 3]
        ]
        self.onp_dtypes = [onp.int32, 'int32', int,
                           onp.float32, 'float32', float,
                           onp.uint32, 'uint32',
                           onp.bool_, 'bool', bool]

        self.mnp_dtypes = [mnp.int32, 'int32', int,
                           mnp.float32, 'float32', float,
                           mnp.uint32, 'uint32',
                           mnp.bool_, 'bool', bool]

        self.array_sets = [1, 1.1, True, [1, 0, True], [1, 1.0, 2], (1,),
                           [(1, 2, 3), (4, 5, 6)], onp.random.random(  # pylint: disable=no-member
                               (100, 100)).astype(onp.float32),
                           onp.random.random((100, 100)).astype(onp.bool)]

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
            mnp.ones((0, 3, 0, 2, 5)),
            onp.ones((2, 7, 0)),
            onp.ones(()),
            [mnp.ones(3), (1, 2, 3), onp.ones(3), [4, 5, 6]],
            ([(1, 2), mnp.ones(2)], (onp.ones(2), [3, 4])),
        ]

        self.onp_prototypes = [
            onp.ones((2, 3, 4)),
            onp.ones((0, 3, 0, 2, 5)),
            onp.ones((2, 7, 0)),
            onp.ones(()),
            [onp.ones(3), (1, 2, 3), onp.ones(3), [4, 5, 6]],
            ([(1, 2), onp.ones(2)], (onp.ones(2), [3, 4])),
        ]


def match_array(actual, expected, error=0):
    if error > 0:
        onp.testing.assert_almost_equal(actual.tolist(), expected.tolist(),
                                        decimal=error)
    else:
        onp.testing.assert_equal(actual.tolist(), expected.tolist())


def check_all_results(onp_results, mnp_results, error=0):
    """Check all results from numpy and mindspore.numpy"""
    for i, _ in enumerate(onp_results):
        match_array(onp_results[i], mnp_results[i].asnumpy())


def run_non_kw_test(mnp_fn, onp_fn):
    """Run tests on functions with non keyword arguments"""
    test_case = Cases()
    for i in range(len(test_case.arrs)):
        arrs = test_case.arrs[:i]
        match_res(mnp_fn, onp_fn, *arrs)

    for i in range(len(test_case.scalars)):
        arrs = test_case.scalars[:i]
        match_res(mnp_fn, onp_fn, *arrs)

    for i in range(len(test_case.expanded_arrs)):
        arrs = test_case.expanded_arrs[:i]
        match_res(mnp_fn, onp_fn, *arrs)

    for i in range(len(test_case.nested_arrs)):
        arrs = test_case.nested_arrs[:i]
        match_res(mnp_fn, onp_fn, *arrs)


def rand_int(*shape):
    """return an random integer array with parameter shape"""
    res = onp.random.randint(low=1, high=5, size=shape)
    if isinstance(res, onp.ndarray):
        return res.astype(onp.float32)
    return float(res)


# return an random boolean array
def rand_bool(*shape):
    return onp.random.rand(*shape) > 0.5


def match_res(mnp_fn, onp_fn, *arrs, **kwargs):
    """Checks results from applying mnp_fn and onp_fn on arrs respectively"""
    mnp_arrs = map(functools.partial(mnp.asarray, dtype='float32'), arrs)
    mnp_res = mnp_fn(*mnp_arrs, **kwargs)
    onp_res = onp_fn(*arrs, **kwargs)
    match_all_arrays(mnp_res, onp_res)


def match_all_arrays(mnp_res, onp_res, error=0):
    if isinstance(mnp_res, (tuple, list)):
        assert len(mnp_res) == len(onp_res)
        for actual, expected in zip(mnp_res, onp_res):
            match_array(actual.asnumpy(), expected, error)
    else:
        match_array(mnp_res.asnumpy(), onp_res, error)


def match_meta(actual, expected):
    # float64 and int64 are not supported, and the default type for
    # float and int are float32 and int32, respectively
    if expected.dtype == onp.float64:
        expected = expected.astype(onp.float32)
    elif expected.dtype == onp.int64:
        expected = expected.astype(onp.int32)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


# Test np.transpose and np.ndarray.transpose
def mnp_transpose(input_tensor):
    a = mnp.transpose(input_tensor, (0, 2, 1))
    b = mnp.transpose(input_tensor, [2, 1, 0])
    c = mnp.transpose(input_tensor, (1, 0, 2))
    d = mnp.transpose(input_tensor)
    return a, b, c, d


def onp_transpose(input_array):
    a = onp.transpose(input_array, (0, 2, 1))
    b = onp.transpose(input_array, [2, 1, 0])
    c = onp.transpose(input_array, (1, 0, 2))
    d = onp.transpose(input_array)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transpose():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_transposed = onp_transpose(onp_array)
    m_transposed = mnp_transpose(mnp_array)
    check_all_results(o_transposed, m_transposed)


# Test np.expand_dims
def mnp_expand_dims(input_tensor):
    a = mnp.expand_dims(input_tensor, 0)
    b = mnp.expand_dims(input_tensor, -1)
    c = mnp.expand_dims(input_tensor, axis=2)
    d = mnp.expand_dims(input_tensor, axis=-2)
    return a, b, c, d


def onp_expand_dims(input_array):
    a = onp.expand_dims(input_array, 0)
    b = onp.expand_dims(input_array, -1)
    c = onp.expand_dims(input_array, axis=2)
    d = onp.expand_dims(input_array, axis=-2)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expand_dims():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_expanded = onp_expand_dims(onp_array)
    m_expanded = mnp_expand_dims(mnp_array)
    check_all_results(o_expanded, m_expanded)


# Test np.squeeze
def mnp_squeeze(input_tensor):
    a = mnp.squeeze(input_tensor)
    b = mnp.squeeze(input_tensor, 0)
    c = mnp.squeeze(input_tensor, axis=None)
    d = mnp.squeeze(input_tensor, axis=-3)
    e = mnp.squeeze(input_tensor, (2,))
    f = mnp.squeeze(input_tensor, (0, 2))
    return a, b, c, d, e, f


def onp_squeeze(input_array):
    a = onp.squeeze(input_array)
    b = onp.squeeze(input_array, 0)
    c = onp.squeeze(input_array, axis=None)
    d = onp.squeeze(input_array, axis=-3)
    e = onp.squeeze(input_array, (2,))
    f = onp.squeeze(input_array, (0, 2))
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze():
    onp_array = onp.random.random((1, 3, 1, 4, 2)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_squeezed = onp_squeeze(onp_array)
    m_squeezed = mnp_squeeze(mnp_array)
    check_all_results(o_squeezed, m_squeezed)

    onp_array = onp.random.random((1, 1, 1, 1, 1)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_squeezed = onp_squeeze(onp_array)
    m_squeezed = mnp_squeeze(mnp_array)
    check_all_results(o_squeezed, m_squeezed)


# Test np.rollaxis
def mnp_rollaxis(input_tensor):
    a = mnp.rollaxis(input_tensor, 0, 1)
    b = mnp.rollaxis(input_tensor, 0, 2)
    c = mnp.rollaxis(input_tensor, 2, 1)
    d = mnp.rollaxis(input_tensor, 2, 2)
    e = mnp.rollaxis(input_tensor, 0)
    f = mnp.rollaxis(input_tensor, 1)
    return a, b, c, d, e, f


def onp_rollaxis(input_array):
    a = onp.rollaxis(input_array, 0, 1)
    b = onp.rollaxis(input_array, 0, 2)
    c = onp.rollaxis(input_array, 2, 1)
    d = onp.rollaxis(input_array, 2, 2)
    e = onp.rollaxis(input_array, 0)
    f = onp.rollaxis(input_array, 1)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rollaxis():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_rolled = onp_rollaxis(onp_array)
    m_rolled = mnp_rollaxis(mnp_array)
    check_all_results(o_rolled, m_rolled)


# Test np.swapaxes
def mnp_swapaxes(input_tensor):
    a = mnp.swapaxes(input_tensor, 0, 1)
    b = mnp.swapaxes(input_tensor, 1, 0)
    c = mnp.swapaxes(input_tensor, 1, 1)
    d = mnp.swapaxes(input_tensor, 2, 1)
    e = mnp.swapaxes(input_tensor, 1, 2)
    f = mnp.swapaxes(input_tensor, 2, 2)
    return a, b, c, d, e, f


def onp_swapaxes(input_array):
    a = onp.swapaxes(input_array, 0, 1)
    b = onp.swapaxes(input_array, 1, 0)
    c = onp.swapaxes(input_array, 1, 1)
    d = onp.swapaxes(input_array, 2, 1)
    e = onp.swapaxes(input_array, 1, 2)
    f = onp.swapaxes(input_array, 2, 2)
    return a, b, c, d, e, f


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_swapaxes():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_swaped = onp_swapaxes(onp_array)
    m_swaped = mnp_swapaxes(mnp_array)
    check_all_results(o_swaped, m_swaped)


# Test np.reshape
def mnp_reshape(input_tensor):
    a = mnp.reshape(input_tensor, (3, 8))
    b = mnp.reshape(input_tensor, [3, -1])
    c = mnp.reshape(input_tensor, (-1, 12))
    d = mnp.reshape(input_tensor, (-1,))
    e = mnp.reshape(input_tensor, 24)
    f = mnp.reshape(input_tensor, [2, 4, -1])
    g = input_tensor.reshape(3, 8)
    h = input_tensor.reshape(3, -1)
    i = input_tensor.reshape([-1, 3])
    j = input_tensor.reshape(-1)
    return a, b, c, d, e, f, g, h, i, j


def onp_reshape(input_array):
    a = onp.reshape(input_array, (3, 8))
    b = onp.reshape(input_array, [3, -1])
    c = onp.reshape(input_array, (-1, 12))
    d = onp.reshape(input_array, (-1,))
    e = onp.reshape(input_array, 24)
    f = onp.reshape(input_array, [2, 4, -1])
    g = input_array.reshape(3, 8)
    h = input_array.reshape(3, -1)
    i = input_array.reshape([-1, 3])
    j = input_array.reshape(-1)
    return a, b, c, d, e, f, g, h, i, j


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reshape():
    onp_array = onp.random.random((2, 3, 4)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_reshaped = onp_reshape(onp_array)
    m_reshaped = mnp_reshape(mnp_array)
    check_all_results(o_reshaped, m_reshaped)


# Test np.ravel
def mnp_ravel(input_tensor):
    a = mnp.ravel(input_tensor)
    return a


def onp_ravel(input_array):
    a = onp.ravel(input_array)
    return a


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ravel():
    onp_array = onp.random.random((2, 3, 4)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_ravel = onp_ravel(onp_array)
    m_ravel = mnp_ravel(mnp_array).asnumpy()
    match_array(o_ravel, m_ravel)


# Test np.concatenate
def mnp_concatenate(input_tensor):
    a = mnp.concatenate(input_tensor, None)
    b = mnp.concatenate(input_tensor, 0)
    c = mnp.concatenate(input_tensor, 1)
    d = mnp.concatenate(input_tensor, 2)
    return a, b, c, d


def onp_concatenate(input_array):
    a = onp.concatenate(input_array, None)
    b = onp.concatenate(input_array, 0)
    c = onp.concatenate(input_array, 1)
    d = onp.concatenate(input_array, 2)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concatenate():
    onp_array = onp.random.random((5, 4, 3, 2)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_concatenate = onp_concatenate(onp_array)
    m_concatenate = mnp_concatenate(mnp_array)
    check_all_results(o_concatenate, m_concatenate)


def construct_arrays(n=1, ndim=1, axis=None, low=1, high=5):
    onp_array_lst = []
    mnp_array_lst = []
    shape = onp.random.randint(low=low, high=high, size=ndim)
    new_shape = [sh for sh in shape]
    while n > 0:
        n -= 1
        onp_array1 = onp.random.randint(
            low=low, high=high, size=shape).astype(onp.float32)
        onp_array_lst.append(onp_array1)
        mnp_array_lst.append(mnp.asarray(onp_array1))
        if axis is not None and axis < ndim:
            new_shape[axis] += onp.random.randint(2)
            onp_array2 = onp.random.randint(
                low=low, high=high, size=new_shape).astype(onp.float32)
            onp_array_lst.append(onp_array2)
            mnp_array_lst.append(mnp.asarray(onp_array2))
    return onp_array_lst, mnp_array_lst

# Test np.xstack


def prepare_array_sequences(n_lst, ndim_lst, axis=None, low=1, high=5):
    onp_seq_lst = []
    mnp_seq_lst = []
    for n in n_lst:
        for ndim in ndim_lst:
            onp_array_lst, mnp_array_lst = construct_arrays(
                n=n, ndim=ndim, axis=axis, low=low, high=high)
            onp_seq_lst.append(onp_array_lst)
            mnp_seq_lst.append(mnp_array_lst)
    return onp_seq_lst, mnp_seq_lst


def mnp_column_stack(input_tensor):
    return mnp.column_stack(input_tensor)


def onp_column_stack(input_array):
    return onp.column_stack(input_array)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_column_stack():
    onp_seq_lst, mnp_seq_lst = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[1, 2, 3, 4], axis=1)
    for i, onp_seq in enumerate(onp_seq_lst):
        onp_seq = onp_seq_lst[i]
        mnp_seq = mnp_seq_lst[i]
        o_column_stack = onp_column_stack(onp_seq)
        m_column_stack = mnp_column_stack(mnp_seq)
        check_all_results(o_column_stack, m_column_stack)


def mnp_hstack(input_tensor):
    return mnp.hstack(input_tensor)


def onp_hstack(input_array):
    return onp.hstack(input_array)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hstack():
    onp_seq_lst0, mnp_seq_lst0 = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[2, 3, 4], axis=1)
    onp_seq_lst1, mnp_seq_lst1 = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[1], axis=0)
    onp_seq_lst = onp_seq_lst0 + onp_seq_lst1
    mnp_seq_lst = mnp_seq_lst0 + mnp_seq_lst1
    for i, onp_seq in enumerate(onp_seq_lst):
        mnp_seq = mnp_seq_lst[i]
        o_hstack = onp_hstack(onp_seq)
        m_hstack = mnp_hstack(mnp_seq)
        check_all_results(o_hstack, m_hstack)


def mnp_dstack(input_tensor):
    return mnp.dstack(input_tensor)


def onp_dstack(input_array):
    return onp.dstack(input_array)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dstack():
    onp_seq_lst, mnp_seq_lst = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[1, 2, 3, 4], axis=2)
    for i, onp_seq in enumerate(onp_seq_lst):
        mnp_seq = mnp_seq_lst[i]
        o_dstack = onp_dstack(onp_seq)
        m_dstack = mnp_dstack(mnp_seq)
        check_all_results(o_dstack, m_dstack)


def mnp_vstack(input_tensor):
    return mnp.vstack(input_tensor)


def onp_vstack(input_array):
    return onp.vstack(input_array)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vstack():
    onp_seq_lst0, mnp_seq_lst0 = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[2, 3, 4], axis=0)
    onp_seq_lst1, mnp_seq_lst1 = prepare_array_sequences(
        n_lst=[1, 5], ndim_lst=[1])
    onp_seq_lst = onp_seq_lst0 + onp_seq_lst1
    mnp_seq_lst = mnp_seq_lst0 + mnp_seq_lst1
    for i, onp_seq in enumerate(onp_seq_lst):
        mnp_seq = mnp_seq_lst[i]
        o_vstack = onp_vstack(onp_seq)
        m_vstack = mnp_vstack(mnp_seq)
        check_all_results(o_vstack, m_vstack)
# Test np.atleastxd


def mnp_atleast1d(*arys):
    return mnp.atleast_1d(*arys)


def onp_atleast1d(*arys):
    return onp.atleast_1d(*arys)


def mnp_atleast2d(*arys):
    return mnp.atleast_2d(*arys)


def onp_atleast2d(*arys):
    return onp.atleast_2d(*arys)


def mnp_atleast3d(*arys):
    return mnp.atleast_3d(*arys)


def onp_atleast3d(*arys):
    return onp.atleast_3d(*arys)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atleast1d():
    run_non_kw_test(mnp_atleast1d, onp_atleast1d)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atleast2d():
    run_non_kw_test(mnp_atleast2d, onp_atleast2d)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atleast3d():
    run_non_kw_test(mnp_atleast3d, onp_atleast3d)


# Test np.where
def mnp_where(condition, x, y):
    return mnp.where(condition, x, y)


def onp_where(condition, x, y):
    return onp.where(condition, x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_where():
    test_case = Cases()
    for condition1 in test_case.bool_broadcastables[:2]:
        for x in test_case.broadcastables[:2]:
            for y in test_case.broadcastables[:2]:
                for condition2 in test_case.broadcastables[:2]:
                    match_res(mnp_where, onp_where, condition1, x, y)
                    match_res(mnp_where, onp_where, condition2, x, y)


# Test ndarray.flatten
def mnp_ndarray_flatten(input_tensor):
    a = input_tensor.flatten()
    b = input_tensor.flatten(order='F')
    c = input_tensor.flatten(order='C')
    return a, b, c


def onp_ndarray_flatten(input_array):
    a = input_array.flatten()
    b = input_array.flatten(order='F')
    c = input_array.flatten(order='C')
    return a, b, c


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ndarray_flatten():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_flatten = onp_ndarray_flatten(onp_array)
    m_flatten = mnp_ndarray_flatten(mnp_array)
    check_all_results(o_flatten, m_flatten)


# Test ndarray.transpose
def mnp_ndarray_transpose(input_tensor):
    a = input_tensor.T
    b = input_tensor.transpose()
    c = input_tensor.transpose((0, 2, 1))
    d = input_tensor.transpose([0, 2, 1])
    return a, b, c, d


def onp_ndarray_transpose(input_array):
    a = input_array.T
    b = input_array.transpose()
    c = input_array.transpose((0, 2, 1))
    d = input_array.transpose([0, 2, 1])
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ndarray_transpose():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_transposed = onp_ndarray_transpose(onp_array)
    m_transposed = mnp_ndarray_transpose(mnp_array)
    check_all_results(o_transposed, m_transposed)


# Test ndarray.astype
def mnp_ndarray_astype(input_tensor):
    a = input_tensor.astype("float16")
    b = input_tensor.astype(onp.float64)
    c = input_tensor.astype(mnp.bool_)
    return a, b, c


def onp_ndarray_astype(input_array):
    a = input_array.astype("float16")
    b = input_array.astype(onp.float64)
    c = input_array.astype(onp.bool_)
    return a, b, c


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ndarray_astype():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_astype = onp_ndarray_astype(onp_array)
    m_astype = mnp_ndarray_astype(mnp_array)
    for arr1, arr2 in zip(o_astype, m_astype):
        assert arr1.dtype == arr2.asnumpy().dtype


def onp_concatenate_type_promotion(onp_array1, onp_array2, onp_array3, onp_array4):
    o_concatenate = onp.concatenate((onp_array1,
                                     onp_array2,
                                     onp_array3,
                                     onp_array4), -1)
    return o_concatenate


def mnp_concatenate_type_promotion(mnp_array1, mnp_array2, mnp_array3, mnp_array4):
    m_concatenate = mnp.concatenate([mnp_array1,
                                     mnp_array2,
                                     mnp_array3,
                                     mnp_array4], -1)
    return m_concatenate


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concatenate_type_promotion():
    onp_array = onp.random.random((5, 1)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    onp_array1 = onp_array.astype(onp.float16)
    onp_array2 = onp_array.astype(onp.bool_)
    onp_array3 = onp_array.astype(onp.float32)
    onp_array4 = onp_array.astype(onp.int32)

    mnp_array1 = mnp_array.astype(onp.float16)
    mnp_array2 = mnp_array.astype(onp.bool_)
    mnp_array3 = mnp_array.astype(onp.float32)
    mnp_array4 = mnp_array.astype(onp.int32)
    o_concatenate = onp_concatenate_type_promotion(
        onp_array1, onp_array2, onp_array3, onp_array4).astype('float32')
    m_concatenate = mnp_concatenate_type_promotion(
        mnp_array1, mnp_array2, mnp_array3, mnp_array4)
    check_all_results(o_concatenate, m_concatenate, error=1e-7)


def mnp_stack(*arrs):
    a = mnp.stack(arrs, axis=-4)
    b = mnp.stack(arrs, axis=-3)
    c = mnp.stack(arrs, axis=0)
    d = mnp.stack(arrs, axis=3)
    e = mnp.stack(arrs, axis=2)
    return a, b, c, d, e


def onp_stack(*arrs):
    a = onp.stack(arrs, axis=-4)
    b = onp.stack(arrs, axis=-3)
    c = onp.stack(arrs, axis=0)
    d = onp.stack(arrs, axis=3)
    e = onp.stack(arrs, axis=2)
    return a, b, c, d, e


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_stack():
    arr = rand_int(3, 4, 5, 6)
    match_res(mnp.stack, onp.stack, arr)
    for i in range(-4, 4):
        match_res(mnp.stack, onp.stack, arr, axis=i)

    arr = rand_int(7, 4, 0, 3)
    match_res(mnp.stack, onp.stack, arr)
    for i in range(-4, 4):
        match_res(mnp.stack, onp.stack, arr, axis=i)

    arrs = [rand_int(3, 4, 5) for i in range(10)]
    match_res(mnp.stack, onp.stack, arrs)
    match_res(mnp.stack, onp.stack, tuple(arrs))
    match_res(mnp_stack, onp_stack, *arrs)
    for i in range(-4, 4):
        match_res(mnp.stack, onp.stack, arrs, axis=i)

    arrs = [rand_int(3, 0, 5, 8, 0) for i in range(5)]
    match_res(mnp.stack, onp.stack, arrs)
    match_res(mnp.stack, onp.stack, tuple(arrs))
    match_res(mnp_stack, onp_stack, *arrs)
    for i in range(-6, 6):
        match_res(mnp.stack, onp.stack, arrs, axis=i)


class ReshapeExpandSqueeze(Cell):
    def __init__(self):
        super(ReshapeExpandSqueeze, self).__init__()

    def construct(self, x):
        x = mnp.expand_dims(x, 2)
        x = mnp.reshape(x, (1, 2, 3, 4, 1, 1))
        x = mnp.squeeze(x)
        return x


class TransposeConcatRavel(Cell):
    def __init__(self):
        super(TransposeConcatRavel, self).__init__()

    def construct(self, x1, x2, x3):
        x1 = mnp.transpose(x1, [0, 2, 1])
        x2 = x2.transpose(0, 2, 1)
        x = mnp.concatenate((x1, x2, x3), -1)
        x = mnp.ravel(x)
        return x


class RollSwap(Cell):
    def __init__(self):
        super(RollSwap, self).__init__()

    def construct(self, x):
        x = mnp.rollaxis(x, 2)
        x = mnp.swapaxes(x, 0, 1)
        return x


test_case_array_ops = [
    ('ReshapeExpandSqueeze', {
        'block': ReshapeExpandSqueeze(),
        'desc_inputs': [mnp.ones((2, 3, 4))]}),

    ('TransposeConcatRavel', {
        'block': TransposeConcatRavel(),
        'desc_inputs': [mnp.ones((2, 3, 4)),
                        mnp.ones((2, 3, 4)),
                        mnp.ones((2, 4, 1))]}),

    ('RollSwap', {
        'block': RollSwap(),
        'desc_inputs': [mnp.ones((2, 3, 4))]})
]

test_case_lists = [test_case_array_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_expand_dims_exception():
    with pytest.raises(TypeError):
        mnp.expand_dims(mnp.ones((3, 3)), 1.2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_swapaxes_exception():
    with pytest.raises(ValueError):
        mnp.swapaxes(mnp.ones((3, 3)), 1, 10)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_flatten():
    lst = [[1.0, 2.0], [3.0, 4.0]]
    tensor_list = mnp.asarray(lst)
    assert tensor_list.flatten().asnumpy().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert tensor_list.flatten(order='F').asnumpy().tolist() == [
        1.0, 3.0, 2.0, 4.0]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_reshape():
    lst = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    tensor_list = mnp.asarray(lst)
    with pytest.raises(TypeError):
        tensor_list = tensor_list.reshape({0, 1, 2})
    with pytest.raises(ValueError):
        tensor_list = tensor_list.reshape(1, 2, 3)
    assert tensor_list.reshape([-1, 4]).shape == (2, 4)
    assert tensor_list.reshape(1, -1, 4).shape == (1, 2, 4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_squeeze():
    lst = [[[1.0], [2.0], [3.0]]]
    tensor_list = mnp.asarray(lst)
    with pytest.raises(TypeError):
        tensor_list = tensor_list.squeeze(1.2)
    with pytest.raises(ValueError):
        tensor_list = tensor_list.squeeze(4)
    assert tensor_list.squeeze().shape == (3,)
    assert tensor_list.squeeze(axis=2).shape == (1, 3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_ravel():
    lst = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    tensor_list = mnp.asarray(lst)
    assert tensor_list.ravel().shape == (8,)
    assert tensor_list.ravel().asnumpy().tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_swapaxes():
    lst = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor_list = mnp.asarray(lst)
    with pytest.raises(TypeError):
        tensor_list = tensor_list.swapaxes(0, (1,))
    with pytest.raises(ValueError):
        tensor_list = tensor_list.swapaxes(0, 3)
    assert tensor_list.swapaxes(0, 1).shape == (3, 2)
