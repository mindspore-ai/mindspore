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
"""unit tests for numpy logical operations"""

import pytest
import numpy as onp

import mindspore.numpy as mnp

from .utils import rand_int, run_binop_test, match_res


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

        # arrays of the same size expanded across the 0th dimension
        self.expanded_arrs = [
            rand_int(2, 3),
            rand_int(1, 2, 3),
            rand_int(1, 1, 2, 3),
            rand_int(1, 1, 1, 2, 3),
        ]

        # arrays which can be broadcast
        self.broadcastables = [
            rand_int(5),
            rand_int(6, 1),
            rand_int(7, 1, 5),
            rand_int(8, 1, 6, 1)
        ]

        # array which contains infs and nans
        self.infs = onp.array([[1.0, onp.nan], [onp.inf, onp.NINF], [2.3, -4.5], [onp.nan, 0.0]])


test_case = Cases()


def mnp_not_equal(a, b):
    return mnp.not_equal(a, b)


def onp_not_equal(a, b):
    return onp.not_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_not_equal():
    run_binop_test(mnp_not_equal, onp_not_equal, test_case)


def mnp_less_equal(a, b):
    return mnp.less_equal(a, b)


def onp_less_equal(a, b):
    return onp.less_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less_equal():
    run_binop_test(mnp_less_equal, onp_less_equal, test_case)


def mnp_less(a, b):
    return mnp.less(a, b)


def onp_less(a, b):
    return onp.less(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_less():
    run_binop_test(mnp_less, onp_less, test_case)


def mnp_greater_equal(a, b):
    return mnp.greater_equal(a, b)


def onp_greater_equal(a, b):
    return onp.greater_equal(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_greater_equal():
    run_binop_test(mnp_greater_equal, onp_greater_equal, test_case)


def mnp_greater(a, b):
    return mnp.greater(a, b)


def onp_greater(a, b):
    return onp.greater(a, b)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_greater():
    run_binop_test(mnp_greater, onp_greater, test_case)


def mnp_equal(a, b):
    return mnp.equal(a, b)


def onp_equal(a, b):
    return onp.equal(a, b)


def test_equal():
    run_binop_test(mnp_equal, onp_equal, test_case)


def mnp_isfinite(x):
    return mnp.isfinite(x)


def onp_isfinite(x):
    return onp.isfinite(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isfinite():
    match_res(mnp_isfinite, onp_isfinite, test_case.infs)


def mnp_isnan(x):
    return mnp.isnan(x)


def onp_isnan(x):
    return onp.isnan(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isnan():
    match_res(mnp_isnan, onp_isnan, test_case.infs)


def mnp_isinf(x):
    return mnp.isinf(x)


def onp_isinf(x):
    return onp.isinf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isinf():
    match_res(mnp_isinf, onp_isinf, test_case.infs)


def mnp_isposinf(x):
    return mnp.isposinf(x)


def onp_isposinf(x):
    return onp.isposinf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isposinf():
    match_res(mnp_isposinf, onp_isposinf, test_case.infs)


def mnp_isneginf(x):
    return mnp.isneginf(x)


def onp_isneginf(x):
    return onp.isneginf(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_isneginf():
    match_res(mnp_isneginf, onp_isneginf, test_case.infs)


def test_isscalar():
    assert mnp.isscalar(1) == onp.isscalar(1)
    assert mnp.isscalar(2.3) == onp.isscalar(2.3)
    assert mnp.isscalar([4.5]) == onp.isscalar([4.5])
    assert mnp.isscalar(False) == onp.isscalar(False)
    assert mnp.isscalar(mnp.array(True)) == onp.isscalar(onp.array(True))
    assert mnp.isscalar('numpy') == onp.isscalar('numpy')
