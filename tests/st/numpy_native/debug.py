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

import numpy as onp
import mindspore.numpy as mnp
from .utils import match_all_arrays, to_tensor


def test_pad_inner():
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


def mnp_logaddexp(x1, x2):
    return mnp.logaddexp(x1, x2)


def onp_logaddexp(x1, x2):
    return onp.logaddexp(x1, x2)

def mnp_logaddexp2(x1, x2):
    return mnp.logaddexp2(x1, x2)


def onp_logaddexp2(x1, x2):
    return onp.logaddexp2(x1, x2)


def test_logaddexp_inner():
    test_cases = [
        onp.random.randint(1, 5, (5, 6, 3, 2)).astype('float16')]
    for _, x1 in enumerate(test_cases):
        for _, x2 in enumerate(test_cases):
            expected = onp_logaddexp(x1, x2)
            actual = mnp_logaddexp(to_tensor(x1), to_tensor(x2))
            onp.testing.assert_almost_equal(actual.asnumpy().tolist(), expected.tolist(),
                                            decimal=2)


def test_logaddexp2_inner():
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
