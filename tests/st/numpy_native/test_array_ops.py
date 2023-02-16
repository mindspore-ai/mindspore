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
from mindspore import context, Tensor, int32
from mindspore.nn import Cell

from .utils import rand_int, run_non_kw_test, check_all_results, match_array, \
    rand_bool, match_res, run_multi_test, to_tensor, match_all_arrays

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
            [mnp.ones(3), (1, 2, 3), onp.ones(3), [4, 5, 6]],
            ([(1, 2), mnp.ones(2)], (onp.ones(2), [3, 4])),
        ]

        self.onp_prototypes = [
            onp.ones((2, 3, 4)),
            [onp.ones(3), (1, 2, 3), onp.ones(3), [4, 5, 6]],
            ([(1, 2), onp.ones(2)], (onp.ones(2), [3, 4])),
        ]


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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
    o_squeezed = onp_squeeze(onp_array)
    m_squeezed = mnp_squeeze(mnp_array)
    check_all_results(o_squeezed, m_squeezed)

    onp_array = onp.random.random((1, 1, 1, 1, 1)).astype('float32')
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
    o_concatenate = onp_concatenate(onp_array)
    m_concatenate = mnp_concatenate(mnp_array)
    check_all_results(o_concatenate, m_concatenate)


def mnp_append(arr1, arr2):
    a = mnp.append(arr1, arr2)
    b = mnp.append(arr1, arr2, axis=0)
    c = mnp.append(arr1, arr2, axis=-1)
    return a, b, c

def onp_append(arr1, arr2):
    a = onp.append(arr1, arr2)
    b = onp.append(arr1, arr2, axis=0)
    c = onp.append(arr1, arr2, axis=-1)
    return a, b, c

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_append():
    onp_array = onp.random.random((4, 3, 2)).astype('float32')
    onp_value = onp.random.random((4, 3, 2)).astype('float32')
    mnp_array = to_tensor(onp_array)
    mnp_value = to_tensor(onp_value)
    onp_res = onp_append(onp_array, onp_value)
    mnp_res = mnp_append(mnp_array, mnp_value)
    check_all_results(onp_res, mnp_res)


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
        mnp_array_lst.append(to_tensor(onp_array1))
        if axis is not None and axis < ndim:
            new_shape[axis] += onp.random.randint(2)
            onp_array2 = onp.random.randint(
                low=low, high=high, size=new_shape).astype(onp.float32)
            onp_array_lst.append(onp_array2)
            mnp_array_lst.append(to_tensor(onp_array2))
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
    onp_seq_lst, mnp_seq_lst = prepare_array_sequences(
        n_lst=[1], ndim_lst=[2], axis=0)
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
    run_non_kw_test(mnp_atleast1d, onp_atleast1d, Cases())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atleast2d():
    run_non_kw_test(mnp_atleast2d, onp_atleast2d, Cases())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_atleast3d():
    run_non_kw_test(mnp_atleast3d, onp_atleast3d, Cases())


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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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
    mnp_array = to_tensor(onp_array)
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

    arrs = [rand_int(3, 4, 5) for i in range(10)]
    match_res(mnp.stack, onp.stack, arrs)
    match_res(mnp.stack, onp.stack, tuple(arrs))
    match_res(mnp_stack, onp_stack, *arrs)
    for i in range(-4, 4):
        match_res(mnp.stack, onp.stack, arrs, axis=i)


def mnp_roll(input_tensor):
    a = mnp.roll(input_tensor, -3)
    b = mnp.roll(input_tensor, [-2, -3], 1)
    c = mnp.roll(input_tensor, (3, 0, -5), (-1, -2, 0))
    d = mnp.roll(input_tensor, (4,), [0, 0, 1])
    return a, b, c, d


def onp_roll(input_array):
    a = onp.roll(input_array, -3)
    b = onp.roll(input_array, [-2, -3], 1)
    c = onp.roll(input_array, (3, 0, -5), (-1, -2, 0))
    d = onp.roll(input_array, (4,), [0, 0, 1])
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_roll():
    arr = rand_int(3, 4, 5)
    match_res(mnp_roll, onp_roll, arr)
    arr = rand_int(1, 4, 6).astype("int64")
    match_res(mnp_roll, onp_roll, arr)


def mnp_moveaxis(a):
    a = mnp.moveaxis(a, 3, 3)
    b = mnp.moveaxis(a, -1, 4)
    c = mnp.moveaxis(a, (2, 1, 4), (0, 3, 2))
    d = mnp.moveaxis(a, [-2, -5], [2, -4])
    return a, b, c, d


def onp_moveaxis(a):
    a = onp.moveaxis(a, 3, 3)
    b = onp.moveaxis(a, -1, 4)
    c = onp.moveaxis(a, (2, 1, 4), (0, 3, 2))
    d = onp.moveaxis(a, [-2, -5], [2, -4])
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_moveaxis():
    a = rand_int(2, 4, 5, 9, 6)
    match_res(mnp_moveaxis, onp_moveaxis, a)


def mnp_tile(x):
    a = mnp.tile(x, 1)
    b = mnp.tile(x, 3)
    c = mnp.tile(x, [5, 1])
    d = mnp.tile(x, [5, 1, 2, 3, 7])
    return a, b, c, d


def onp_tile(x):
    a = onp.tile(x, 1)
    b = onp.tile(x, 3)
    c = onp.tile(x, [5, 1])
    d = onp.tile(x, [5, 1, 2, 3, 7])
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tile():
    a = rand_int(2, 3, 4)
    match_res(mnp_tile, onp_tile, a)


def mnp_broadcast_to(x):
    a = mnp.broadcast_to(x, (2, 3))
    b = mnp.broadcast_to(x, (8, 1, 3))
    return a, b


def onp_broadcast_to(x):
    a = onp.broadcast_to(x, (2, 3))
    b = onp.broadcast_to(x, (8, 1, 3))
    return a, b


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_broadcast_to():
    x = rand_int()
    match_res(mnp_broadcast_to, onp_broadcast_to, x)
    x = rand_int(3)
    match_res(mnp_broadcast_to, onp_broadcast_to, x)
    x = rand_int(1, 3)
    match_res(mnp_broadcast_to, onp_broadcast_to, x)


def mnp_broadcast_arrays(*args):
    return mnp.broadcast_arrays(*args)


def onp_broadcast_arrays(*args):
    return onp.broadcast_arrays(*args)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_broadcast_arrays():
    test_case = Cases()
    broadcastables = test_case.broadcastables
    for i in range(len(broadcastables)):
        arrs = broadcastables[i:]
        match_res(mnp_broadcast_arrays, onp_broadcast_arrays, *arrs)


def mnp_flip(x):
    a = mnp.flip(x)
    b = mnp.flip(x, 0)
    c = mnp.flip(x, 1)
    d = mnp.flip(x, (-3, -1))
    return a, b, c, d


def onp_flip(x):
    a = onp.flip(x)
    b = onp.flip(x, 0)
    c = onp.flip(x, 1)
    d = onp.flip(x, (-3, -1))
    return a, b, c, d


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_flip():
    x = rand_int(2, 3, 4)
    run_multi_test(mnp_flip, onp_flip, (x,))


def mnp_flipud(x):
    return mnp.flipud(x)


def onp_flipud(x):
    return  onp.flipud(x)


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_flipud():
    x = rand_int(2, 3, 4)
    run_multi_test(mnp_flipud, onp_flipud, (x,))


def mnp_fliplr(x):
    return mnp.fliplr(x)


def onp_fliplr(x):
    return onp.fliplr(x)


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fliplr():
    x = rand_int(2, 3, 4)
    run_multi_test(mnp_fliplr, onp_fliplr, (x,))


def mnp_split(input_tensor):
    a = mnp.split(input_tensor, indices_or_sections=1)
    b = mnp.split(input_tensor, indices_or_sections=3)
    return a, b


def onp_split(input_array):
    a = onp.split(input_array, indices_or_sections=1)
    b = onp.split(input_array, indices_or_sections=3)
    return a, b


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_split():
    onp_arrs = [
        onp.random.randint(1, 5, size=(9, 4, 5)).astype('float32')
    ]
    mnp_arrs = [to_tensor(arr) for arr in onp_arrs]
    for onp_arr, mnp_arr in zip(onp_arrs, mnp_arrs):
        o_split = onp_split(onp_arr)
        m_split = mnp_split(mnp_arr)
        for expect_lst, actual_lst in zip(o_split, m_split):
            for expect, actual in zip(expect_lst, actual_lst):
                match_array(expect, actual.asnumpy())


def mnp_array_split(input_tensor):
    a = mnp.array_split(input_tensor, indices_or_sections=4, axis=2)
    b = mnp.array_split(input_tensor, indices_or_sections=3, axis=1)
    c = mnp.array_split(input_tensor, indices_or_sections=6)
    return a, b, c


def onp_array_split(input_array):
    a = onp.array_split(input_array, indices_or_sections=4, axis=2)
    b = onp.array_split(input_array, indices_or_sections=3, axis=1)
    c = onp.array_split(input_array, indices_or_sections=6)
    return a, b, c


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_array_split():
    onp_arr = onp.random.randint(1, 5, size=(9, 7, 13)).astype('float32')
    mnp_arr = to_tensor(onp_arr)
    o_split = onp_split(onp_arr)
    m_split = mnp_split(mnp_arr)
    for expect_lst, actual_lst in zip(o_split, m_split):
        for expect, actual in zip(expect_lst, actual_lst):
            match_array(expect, actual.asnumpy())


def mnp_vsplit(input_tensor):
    a = mnp.vsplit(input_tensor, indices_or_sections=3)
    return a


def onp_vsplit(input_array):
    a = onp.vsplit(input_array, indices_or_sections=3)
    return a


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vsplit():
    onp_arrs = [
        onp.random.randint(1, 5, size=(9, 4, 5)).astype('float32')
    ]
    mnp_arrs = [to_tensor(arr) for arr in onp_arrs]
    for onp_arr, mnp_arr in zip(onp_arrs, mnp_arrs):
        o_vsplit = onp_vsplit(onp_arr)
        m_vsplit = mnp_vsplit(mnp_arr)
        for expect_lst, actual_lst in zip(o_vsplit, m_vsplit):
            for expect, actual in zip(expect_lst, actual_lst):
                match_array(expect, actual.asnumpy())


def mnp_hsplit(input_tensor):
    a = mnp.hsplit(input_tensor, indices_or_sections=3)
    return a


def onp_hsplit(input_array):
    a = onp.hsplit(input_array, indices_or_sections=3)
    return a


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hsplit():
    onp_arrs = [
        onp.random.randint(1, 5, size=(4, 9, 5)).astype('float32')
    ]
    mnp_arrs = [to_tensor(arr) for arr in onp_arrs]
    for onp_arr, mnp_arr in zip(onp_arrs, mnp_arrs):
        o_hsplit = onp_hsplit(onp_arr)
        m_hsplit = mnp_hsplit(mnp_arr)
        for expect_lst, actual_lst in zip(o_hsplit, m_hsplit):
            for expect, actual in zip(expect_lst, actual_lst):
                match_array(expect, actual.asnumpy())


def mnp_dsplit(input_tensor):
    a = mnp.dsplit(input_tensor, indices_or_sections=3)
    return a

def onp_dsplit(input_array):
    a = onp.dsplit(input_array, indices_or_sections=3)
    return a

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dsplit():
    onp_arrs = [
        onp.random.randint(1, 5, size=(5, 4, 9)).astype('float32')
    ]
    mnp_arrs = [to_tensor(arr) for arr in onp_arrs]
    for onp_arr, mnp_arr in zip(onp_arrs, mnp_arrs):
        o_dsplit = onp_dsplit(onp_arr)
        m_dsplit = mnp_dsplit(mnp_arr)
        for expect_lst, actual_lst in zip(o_dsplit, m_dsplit):
            for expect, actual in zip(expect_lst, actual_lst):
                match_array(expect, actual.asnumpy())


def mnp_take_along_axis(*arrs):
    x = arrs[0]
    a = mnp.take_along_axis(x, arrs[1], axis=None)
    b = mnp.take_along_axis(x, arrs[2], axis=1)
    c = mnp.take_along_axis(x, arrs[3], axis=-1)
    d = mnp.take_along_axis(x, arrs[4], axis=0)
    return a, b, c, d


def onp_take_along_axis(*arrs):
    x = arrs[0]
    a = onp.take_along_axis(x, arrs[1], axis=None)
    b = onp.take_along_axis(x, arrs[2], axis=1)
    c = onp.take_along_axis(x, arrs[3], axis=-1)
    d = onp.take_along_axis(x, arrs[4], axis=0)
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_take_along_axis():
    x = rand_int(6, 7, 8, 9)
    indices1 = rand_int(2).astype(onp.int32)
    indices2 = rand_int(6, 3, 8, 1).astype(onp.int32)
    indices3 = rand_int(6, 1, 8, 5).astype(onp.int32)
    indices4 = rand_int(4, 1, 1, 1).astype(onp.int32)
    run_multi_test(mnp_take_along_axis, onp_take_along_axis,
                   (x, indices1, indices2, indices3, indices4))

    e = onp.random.randint(0, 9, (4, 10, 2))
    neighbours = onp.random.randint(0, 9, (4, 10, 3))
    e = e[:, :, None]
    neighbours = neighbours[:, :, :, None]

    op = onp.take_along_axis(e, neighbours, 1)

    new_neighbours = Tensor(neighbours, int32)
    new_e = Tensor(e, int32)
    output = mnp.take_along_axis(new_e, new_neighbours, 1)
    onp.testing.assert_almost_equal(list(op), list(output),
                                    decimal=0)


def mnp_take(x, indices):
    a = mnp.take(x, indices)
    b = mnp.take(x, indices, axis=-1)
    c = mnp.take(x, indices, axis=0, mode='wrap')
    d = mnp.take(x, indices, axis=1, mode='clip')
    return a, b, c, d


def onp_take(x, indices):
    a = onp.take(x, indices)
    b = onp.take(x, indices, axis=-1)
    c = onp.take(x, indices, axis=0, mode='wrap')
    d = onp.take(x, indices, axis=1, mode='clip')
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_take():
    x = rand_int(2, 3, 4, 5)
    indices = rand_int(2, 3).astype(onp.int32)
    run_multi_test(mnp_take, onp_take, (x, indices))


def mnp_repeat(x):
    a = mnp.repeat(x, 2)
    b = mnp.repeat(x, 3, axis=0)
    c = mnp.repeat(x, (4, 1, 5), axis=1)
    d = mnp.repeat(x, (3, 2, 1, 0, 4), axis=-1)
    e = mnp.repeat(x, 0)
    return a, b, c, d, e


def onp_repeat(x):
    a = onp.repeat(x, 2)
    b = onp.repeat(x, 3, axis=0)
    c = onp.repeat(x, (4, 1, 5), axis=1)
    d = onp.repeat(x, (3, 2, 1, 0, 4), axis=-1)
    e = onp.repeat(x, 0)
    return a, b, c, d, e


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_repeat():
    x = rand_int(2, 3, 4, 5)
    run_multi_test(mnp_repeat, onp_repeat, (x,))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_select():
    choicelist = rand_int(2, 3, 4, 5)
    condlist = choicelist > 2
    match_res(mnp.select, onp.select, condlist, choicelist)
    match_res(mnp.select, onp.select, condlist, choicelist, default=10)

    condlist = rand_bool(5, 4, 1, 3)
    choicelist = rand_int(5, 3)
    match_res(mnp.select, onp.select, condlist, choicelist)
    match_res(mnp.select, onp.select, condlist, choicelist, default=10)

    condlist = rand_bool(3, 1, 7)
    choicelist = rand_int(3, 5, 2, 1)
    match_res(mnp.select, onp.select, condlist, choicelist)
    match_res(mnp.select, onp.select, condlist, choicelist, default=10)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_choose():
    x = rand_int(2, 1, 4).astype(onp.int32)
    y = rand_int(3, 2, 5, 4).astype(onp.int32)
    match_res(mnp.choose, onp.choose, x, y, mode='wrap', dtype=mnp.int32)
    match_res(mnp.choose, onp.choose, x, y, mode='clip', dtype=mnp.int32)

    x = rand_int(5, 3, 1, 7).astype(onp.int32)
    y1 = rand_int(7).astype(onp.int32)
    y2 = rand_int(1, 3, 1).astype(onp.int32)
    y3 = rand_int(5, 1, 1, 7).astype(onp.int32)
    onp_arrays = (x, (y1, y2, y3))
    mnp_arrays = (to_tensor(x), tuple(map(to_tensor, (y1, y2, y3))))
    match_all_arrays(mnp.choose(*mnp_arrays, mode='wrap'), onp.choose(*onp_arrays, mode='wrap'))
    match_all_arrays(mnp.choose(*mnp_arrays, mode='clip'), onp.choose(*onp_arrays, mode='clip'))


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
    tensor_list = to_tensor(lst)
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
    tensor_list = to_tensor(lst)
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
    tensor_list = to_tensor(lst)
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
    tensor_list = to_tensor(lst)
    assert tensor_list.ravel().shape == (8,)
    assert tensor_list.ravel().asnumpy().tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def mnp_rot90(input_tensor):
    a = mnp.rot90(input_tensor)
    b = mnp.rot90(input_tensor, 2)
    c = mnp.rot90(input_tensor, 3)
    d = mnp.rot90(input_tensor, 4)
    e = mnp.rot90(input_tensor, 5, (0, -1))
    f = mnp.rot90(input_tensor, 1, (2, 0))
    g = mnp.rot90(input_tensor, -3, (-1, -2))
    h = mnp.rot90(input_tensor, 3, (2, 1))
    return a, b, c, d, e, f, g, h


def onp_rot90(input_array):
    a = onp.rot90(input_array)
    b = onp.rot90(input_array, 2)
    c = onp.rot90(input_array, 3)
    d = onp.rot90(input_array, 4)
    e = onp.rot90(input_array, 5, (0, -1))
    f = onp.rot90(input_array, 1, (2, 0))
    g = onp.rot90(input_array, -3, (-1, -2))
    h = onp.rot90(input_array, 3, (2, 1))
    return a, b, c, d, e, f, g, h


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rot90():
    onp_array = rand_int(3, 4, 5).astype('float32')
    mnp_array = to_tensor(onp_array)
    o_rot = onp_rot90(onp_array)
    m_rot = mnp_rot90(mnp_array)
    check_all_results(o_rot, m_rot)


def mnp_size(x):
    a = mnp.size(x)
    b = mnp.size(x, axis=0)
    return a, b


def onp_size(x):
    a = onp.size(x)
    b = onp.size(x, axis=0)
    return a, b


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_size():
    onp_arr = onp.random.rand(2, 3, 4).astype('float32')
    mnp_arr = to_tensor(onp_arr)
    for actual, expected in zip(mnp_size(mnp_arr), onp_size(onp_arr)):
        match_array(actual, expected)


def mnp_array_str(x):
    return mnp.array_str(x)


def onp_array_str(x):
    return onp.array_str(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_array_str():
    onp_arr = onp.random.rand(2, 3, 4).astype('float32')
    mnp_arr = to_tensor(onp_arr)
    for actual, expected in zip(mnp_size(mnp_arr), onp_size(onp_arr)):
        match_array(actual, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_along_axis():
    onp_arr = rand_int(5, 3, 7)
    mnp_arr = to_tensor(onp_arr)
    for i in range(-3, 3):
        mnp_res = mnp.apply_along_axis(mnp.diag, i, mnp_arr)
        onp_res = onp.apply_along_axis(onp.diag, i, onp_arr)
        match_all_arrays(mnp_res, onp_res)
    mnp_res = mnp.apply_along_axis(lambda x: x[0], 2, mnp_arr)
    onp_res = onp.apply_along_axis(lambda x: x[0], 2, onp_arr)
    match_all_arrays(mnp_res, onp_res)
    mnp_res = mnp.apply_along_axis(lambda x, y, offset=0: (x[4] - y)*offset, 2, mnp_arr, 1, offset=3)
    onp_res = onp.apply_along_axis(lambda x, y, offset=0: (x[4] - y)*offset, 2, onp_arr, 1, offset=3)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_resize():
    x = rand_int(3, 5)
    mnp_x = to_tensor(x)

    x.resize(2, 4, refcheck=False)
    mnp_x = mnp_x.resize(2, 4)
    match_array(mnp_x.asnumpy(), x)

    x.resize((3, 1), refcheck=False)
    mnp_x = mnp_x.resize((3, 1))
    match_array(mnp_x.asnumpy(), x)

    x.resize(7, 4, refcheck=False)
    mnp_x = mnp_x.resize(7, 4)
    match_array(mnp_x.asnumpy(), x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_piecewise():
    x = rand_int(2, 4)
    mnp_x = to_tensor(x)
    condlist = [x < 2, x == 2, x > 2]
    mnp_condlist = [mnp_x < 2, mnp_x == 2, mnp_x > 2]
    funclist = [lambda x, offset=0: x - offset, lambda x, offset=0: x, lambda x, offset=0: x*offset]
    mnp_res = mnp.piecewise(mnp_x, mnp_condlist, funclist, offset=2)
    onp_res = onp.piecewise(x, condlist, funclist, offset=2)
    match_all_arrays(mnp_res, onp_res)

    funclist = [-1, 0, 1]
    mnp_res = mnp.piecewise(mnp_x, mnp_condlist, funclist)
    onp_res = onp.piecewise(x, condlist, funclist)
    match_all_arrays(mnp_res, onp_res)

    condlist = [x > 10, x < 0]
    mnp_x = to_tensor(x)
    mnp_condlist = [mnp_x > 10, mnp_x < 0]
    funclist = [lambda x: x - 2, lambda x: x - 1, lambda x: x*2]
    mnp_res = mnp.piecewise(mnp_x, mnp_condlist, funclist)
    onp_res = onp.piecewise(x, condlist, funclist)
    match_all_arrays(mnp_res, onp_res)

    x = 2
    condlist = True
    funclist = [lambda x: x - 1]
    mnp_res = mnp.piecewise(x, condlist, funclist)
    onp_res = onp.piecewise(x, condlist, funclist)
    match_all_arrays(mnp_res, onp_res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unravel_index():
    shapes = [(2, 6, 3)]
    dims = [(5, 4, 7), 5*4*7]
    for shape in shapes:
        x = onp.random.randint(0, 5*4*7, shape)
        for dim in dims:
            for order in ('C', 'F'):
                mnp_res = mnp.unravel_index(to_tensor(x), dim, order=order)
                onp_res = onp.unravel_index(x, dim, order=order)
                match_all_arrays(mnp_res, onp_res)


def mnp_apply_over_axes(x):
    a = mnp.apply_over_axes(mnp.sum, x, axes=0)
    b = mnp.apply_over_axes(mnp.sum, x, axes=(0, 1))
    c = mnp.apply_over_axes(mnp.std, x, axes=1)
    d = mnp.apply_over_axes(mnp.mean, x, axes=(-1,))
    return a, b, c, d


def onp_apply_over_axes(x):
    a = onp.apply_over_axes(onp.sum, x, axes=0)
    b = onp.apply_over_axes(onp.sum, x, axes=(0, 1))
    c = onp.apply_over_axes(onp.std, x, axes=1)
    d = onp.apply_over_axes(onp.mean, x, axes=(-1,))
    return a, b, c, d


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_over_axes():
    arrs = [
        onp.random.rand(2, 2).astype('float32'),
        onp.random.rand(3, 2, 2).astype('float32'),
        onp.random.rand(5, 4, 3, 3).astype('float32'),
    ]
    for x in arrs:
        for expected, actual in zip(onp_apply_over_axes(x),
                                    mnp_apply_over_axes(to_tensor(x))):
            match_array(actual.asnumpy(), expected, error=5)


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_choose():
    x = rand_int(2, 1, 4).astype(onp.int32)
    mnp_x = to_tensor(x)
    y = rand_int(3, 2, 5, 4).astype(onp.int32)
    match_res(mnp_x.choose, x.choose, y, mode='wrap')
    match_res(mnp_x.choose, x.choose, y, mode='clip')

    x = rand_int(5, 3, 1, 7).astype(onp.int32)
    mnp_x = to_tensor(x)
    y1 = rand_int(7).astype(onp.int32)
    y2 = rand_int(1, 3, 1).astype(onp.int32)
    y3 = rand_int(5, 1, 1, 7).astype(onp.int32)
    onp_arrays = (y1, y2, y3)
    mnp_arrays = tuple(map(to_tensor, (y1, y2, y3)))
    match_all_arrays(mnp_x.choose(mnp_arrays, mode='wrap'), x.choose(onp_arrays, mode='wrap'))
    match_all_arrays(mnp_x.choose(mnp_arrays, mode='clip'), x.choose(onp_arrays, mode='clip'))
