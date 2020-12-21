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
"""unit tests for numpy array operations"""

import functools

import pytest
import numpy as onp

import mindspore.context as context
import mindspore.numpy as mnp
from mindspore.nn import Cell

from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config


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
                           [(1, 2, 3), (4, 5, 6)], onp.random.random(
                               (100, 100)).astype(onp.float32),
                           onp.random.random((100, 100)).astype(onp.bool)]


def match_array(actual, expected, error=0):
    if error > 0:
        onp.testing.assert_almost_equal(actual.tolist(), expected.tolist(),
                                        decimal=error)
    else:
        onp.testing.assert_equal(actual.tolist(), expected.tolist())


def check_all_results(onp_results, mnp_results):
    """Check all results from numpy and mindspore.numpy"""
    for i, _ in enumerate(onp_results):
        match_array(onp_results[i], mnp_results[i].asnumpy())


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

    # Additional tests for nested tensor/numpy_array mixture
    mnp_input = [(onp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.asarray(onp_input)
    expected = mnp.asarray(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


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
    mnp_input = [(onp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.asarray(onp_input)
    expected = mnp.asarray(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


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
    mnp_input = [(onp.ones(3,), mnp.ones(3)), [[1, 1, 1], (1, 1, 1)]]
    onp_input = [(onp.ones(3,), onp.ones(3)), [[1, 1, 1], (1, 1, 1)]]

    actual = onp.asarray(onp_input)
    expected = mnp.asarray(mnp_input).asnumpy()
    match_array(actual, expected, error=7)


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


def test_full():
    actual = onp.full((2, 2), [1, 2])
    expected = mnp.full((2, 2), [1, 2]).asnumpy()
    match_array(actual, expected)

    actual = onp.full((2, 0), onp.inf)
    expected = mnp.full((2, 0), mnp.inf).asnumpy()
    match_array(actual, expected)

    actual = onp.full((2, 3), True)
    expected = mnp.full((2, 3), True).asnumpy()
    match_array(actual, expected)

    actual = onp.full((3, 4, 5), 7.5)
    expected = mnp.full((3, 4, 5), 7.5).asnumpy()
    match_array(actual, expected)


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


def test_identity():
    test_case = Cases()
    for i in range(len(test_case.onp_dtypes)):
        for m in range(1, 5):
            actual = onp.identity(m, dtype=test_case.onp_dtypes[i])
            expected = mnp.identity(m, dtype=test_case.mnp_dtypes[i]).asnumpy()
            match_array(actual, expected)


def test_arange():
    actual = onp.arange(10)
    expected = mnp.arange(10).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(0, 10)
    expected = mnp.arange(0, 10).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(start=10)
    expected = mnp.arange(start=10).asnumpy()
    match_array(actual, expected)

    actual = onp.arange(start=10, step=0.1)
    expected = mnp.arange(start=10, step=0.1).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.arange(10, step=0.1)
    expected = mnp.arange(10, step=0.1).asnumpy()
    match_array(actual, expected, error=6)

    actual = onp.arange(0.1, 9.9)
    expected = mnp.arange(0.1, 9.9).asnumpy()
    match_array(actual, expected, error=6)


def test_linspace():
    actual = onp.linspace(2.0, 3.0, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0).asnumpy()
    match_array(actual, expected, error=7)

    actual = onp.linspace(2.0, 3.0, num=5, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5).asnumpy()
    match_array(actual, expected, error=7)

    actual = onp.linspace(
        2.0, 3.0, num=5, endpoint=False, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5, endpoint=False).asnumpy()
    match_array(actual, expected, error=7)

    actual = onp.linspace(2.0, 3.0, num=5, retstep=True, dtype=onp.float32)
    expected = mnp.linspace(2.0, 3.0, num=5, retstep=True)
    match_array(actual[0], expected[0].asnumpy())
    assert actual[1] == expected[1]

    actual = onp.linspace(2.0, [3, 4, 5], num=5,
                          endpoint=False, dtype=onp.float32)
    expected = mnp.linspace(
        2.0, [3, 4, 5], num=5, endpoint=False).asnumpy()
    match_array(actual, expected)


def test_logspace():
    actual = onp.logspace(2.0, 3.0, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0).asnumpy()
    match_array(actual, expected)

    actual = onp.logspace(2.0, 3.0, num=5, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0, num=5).asnumpy()
    match_array(actual, expected)

    actual = onp.logspace(
        2.0, 3.0, num=5, endpoint=False, dtype=onp.float32)
    expected = mnp.logspace(2.0, 3.0, num=5, endpoint=False).asnumpy()
    match_array(actual, expected)

    actual = onp.logspace(2.0, [3, 4, 5], num=5,
                          endpoint=False, dtype=onp.float32)
    expected = mnp.logspace(
        2.0, [3, 4, 5], num=5, endpoint=False).asnumpy()
    match_array(actual, expected)


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

# Test np.reshape


def mnp_reshape(input_tensor):
    a = mnp.reshape(input_tensor, (3, 8))
    b = mnp.reshape(input_tensor, [3, -1])
    c = mnp.reshape(input_tensor, (-1, 12))
    d = mnp.reshape(input_tensor, (-1,))
    e = mnp.reshape(input_tensor, 24)
    f = mnp.reshape(input_tensor, [2, 4, -1])
    return a, b, c, d, e, f


def onp_reshape(input_array):
    a = onp.reshape(input_array, (3, 8))
    b = onp.reshape(input_array, [3, -1])
    c = onp.reshape(input_array, (-1, 12))
    d = onp.reshape(input_array, (-1,))
    e = onp.reshape(input_array, 24)
    f = onp.reshape(input_array, [2, 4, -1])
    return a, b, c, d, e, f

# Test np.ravel


def mnp_ravel(input_tensor):
    a = mnp.ravel(input_tensor)
    return a


def onp_ravel(input_array):
    a = onp.ravel(input_array)
    return a

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


def test_transpose():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_transposed = onp_transpose(onp_array)
    m_transposed = mnp_transpose(mnp_array)
    check_all_results(o_transposed, m_transposed)


def test_expand_dims():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_expanded = onp_expand_dims(onp_array)
    m_expanded = mnp_expand_dims(mnp_array)
    check_all_results(o_expanded, m_expanded)


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


def test_rollaxis():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_rolled = onp_rollaxis(onp_array)
    m_rolled = mnp_rollaxis(mnp_array)
    check_all_results(o_rolled, m_rolled)


def test_swapaxes():
    onp_array = onp.random.random((3, 4, 5)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_swaped = onp_swapaxes(onp_array)
    m_swaped = mnp_swapaxes(mnp_array)
    check_all_results(o_swaped, m_swaped)


def test_reshape():
    onp_array = onp.random.random((2, 3, 4)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_reshaped = onp_reshape(onp_array)
    m_reshaped = mnp_reshape(mnp_array)
    check_all_results(o_reshaped, m_reshaped)


def test_ravel():
    onp_array = onp.random.random((2, 3, 4)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_ravel = onp_ravel(onp_array)
    m_ravel = mnp_ravel(mnp_array).asnumpy()
    match_array(o_ravel, m_ravel)


def test_concatenate():
    onp_array = onp.random.random((5, 4, 3, 2)).astype('float32')
    mnp_array = mnp.asarray(onp_array)
    o_concatenate = onp_concatenate(onp_array)
    m_concatenate = mnp_concatenate(mnp_array)
    check_all_results(o_concatenate, m_concatenate)


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


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case


def test_expand_dims_exception():
    with pytest.raises(TypeError):
        mnp.expand_dims(mnp.ones((3, 3)), 1.2)


def test_asarray_exception():
    with pytest.raises(TypeError):
        mnp.asarray({1, 2, 3})


def test_swapaxes_exception():
    with pytest.raises(ValueError):
        mnp.swapaxes(mnp.ones((3, 3)), 1, 10)


def test_linspace_exception():
    with pytest.raises(TypeError):
        mnp.linspace(0, 1, num=2.5)
