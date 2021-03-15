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
"""utility functions for mindspore.numpy st tests"""
import functools
import numpy as onp
from mindspore import Tensor
import mindspore.numpy as mnp


def match_array(actual, expected, error=0):

    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(expected, int):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(actual.tolist(), expected.tolist(),
                                        decimal=error)
    else:
        onp.testing.assert_equal(actual.tolist(), expected.tolist())


def check_all_results(onp_results, mnp_results, error=0):
    """Check all results from numpy and mindspore.numpy"""
    for i, _ in enumerate(onp_results):
        match_array(onp_results[i], mnp_results[i].asnumpy())


def check_all_unique_results(onp_results, mnp_results):
    """
    Check all results from numpy and mindspore.numpy.

    Args:
        onp_results (Union[tuple of numpy.arrays, numpy.array])
        mnp_results (Union[tuple of Tensors, Tensor])
    """
    for i, _ in enumerate(onp_results):
        if isinstance(onp_results[i], tuple):
            for j in range(len(onp_results[i])):
                match_array(onp_results[i][j],
                            mnp_results[i][j].asnumpy(), error=7)
        else:
            match_array(onp_results[i], mnp_results[i].asnumpy(), error=7)


def run_non_kw_test(mnp_fn, onp_fn, test_case):
    """Run tests on functions with non keyword arguments"""
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
    dtype = kwargs.get('dtype', mnp.float32)
    kwargs.pop('dtype', None)
    mnp_arrs = map(functools.partial(Tensor, dtype=dtype), arrs)
    error = kwargs.get('error', 0)
    kwargs.pop('error', None)
    mnp_res = mnp_fn(*mnp_arrs, **kwargs)
    onp_res = onp_fn(*arrs, **kwargs)
    match_all_arrays(mnp_res, onp_res, error=error)


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


def run_binop_test(mnp_fn, onp_fn, test_case, error=0):
    for arr in test_case.arrs:
        match_res(mnp_fn, onp_fn, arr, arr, error=error)

        for scalar in test_case.scalars:
            match_res(mnp_fn, onp_fn, arr, scalar, error=error)
            match_res(mnp_fn, onp_fn, scalar, arr, error=error)

    for scalar1 in test_case.scalars:
        for scalar2 in test_case.scalars:
            match_res(mnp_fn, onp_fn, scalar1, scalar2, error=error)

    for expanded_arr1 in test_case.expanded_arrs:
        for expanded_arr2 in test_case.expanded_arrs:
            match_res(mnp_fn, onp_fn, expanded_arr1, expanded_arr2, error=error)

    for broadcastable1 in test_case.broadcastables:
        for broadcastable2 in test_case.broadcastables:
            match_res(mnp_fn, onp_fn, broadcastable1, broadcastable2, error=error)


def run_unary_test(mnp_fn, onp_fn, test_case, error=0):
    for arr in test_case.arrs:
        match_res(mnp_fn, onp_fn, arr, error=error)

    for arr in test_case.scalars:
        match_res(mnp_fn, onp_fn, arr, error=error)

    for arr in test_case.expanded_arrs:
        match_res(mnp_fn, onp_fn, arr, error=error)


def run_multi_test(mnp_fn, onp_fn, arrs, error=0):
    mnp_arrs = map(Tensor, arrs)
    for actual, expected in zip(mnp_fn(*mnp_arrs), onp_fn(*arrs)):
        match_all_arrays(actual, expected, error)


def run_single_test(mnp_fn, onp_fn, arr, error=0):
    mnp_arr = Tensor(arr)
    for actual, expected in zip(mnp_fn(mnp_arr), onp_fn(arr)):
        if isinstance(expected, tuple):
            for actual_arr, expected_arr in zip(actual, expected):
                match_array(actual_arr.asnumpy(), expected_arr, error)
        match_array(actual.asnumpy(), expected, error)


def run_logical_test(mnp_fn, onp_fn, test_case):
    for x1 in test_case.boolean_arrs:
        for x2 in test_case.boolean_arrs:
            match_res(mnp_fn, onp_fn, x1, x2, dtype=mnp.bool_)

def to_tensor(obj, dtype=None):
    if dtype is None:
        res = Tensor(obj)
        if res.dtype == mnp.float64:
            res = res.astype(mnp.float32)
        if res.dtype == mnp.int64:
            res = res.astype(mnp.int32)
    else:
        res = Tensor(obj, dtype)
    return res
