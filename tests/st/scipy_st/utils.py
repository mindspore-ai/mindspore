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
"""utility functions for mindspore.scipy st tests"""
import numpy as onp
from mindspore import Tensor
import mindspore.numpy as mnp
from mindspore.common import dtype as mstype


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


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(expected, (int, tuple)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def match_matrix(actual, expected, error=0, err_msg=''):
    if actual.shape != expected.shape:
        raise ValueError(
            err_msg.join(f" actual shape {actual.shape} is not equal to expected input shape {expected.shape}"))
    sub_abs = mnp.abs(mnp.subtract(actual, expected))
    no_zero_max = sub_abs.max()
    if no_zero_max > Tensor(error, dtype=mstype.float64):
        raise ValueError(
            err_msg.join(f" actual value: {actual} is not equal to expected input value: {expected}"))


def create_full_rank_matrix(shape, dtype):
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            'Full rank matrix must be a square matrix, but has shape: ', shape)

    invertible = False
    a = None
    while not invertible:
        a = onp.random.random(shape).astype(dtype)
        try:
            onp.linalg.inv(a)
            invertible = True
        except onp.linalg.LinAlgError:
            pass

    return a


def create_random_rank_matrix(shape, dtype):
    if len(shape) < 2:
        raise ValueError(
            'random rank matrix must shape bigger than two dims, but has shape: ', shape)
    return onp.random.random(shape).astype(dtype)


def create_sym_pos_matrix(shape, dtype):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ', shape)

    n = shape[-1]
    a = (onp.random.random(shape) + onp.eye(n)).astype(dtype)
    return onp.dot(a, a.T)
