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


def match_array(actual, expected, error=0):
    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(expected, (int, tuple)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error)
    else:
        onp.testing.assert_equal(actual, expected)


def create_batch_full_rank_matrix(shape, dtype):
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


def create_full_rank_matrix(m, n, dtype):
    a_rank = 0
    a = onp.random.random((m, n)).astype(dtype)
    while a_rank != m:
        a = (a + onp.eye(m, n)).astype(dtype)
        a_rank = onp.linalg.matrix_rank(a)
    return a


def create_sym_pos_matrix(m, n, dtype):
    a = (onp.random.random((m, n)) + onp.eye(m, n)).astype(dtype)
    return onp.dot(a, a.T)
