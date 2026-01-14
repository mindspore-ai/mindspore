# Copyright 2024 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark


class AddNet(nn.Cell):
    def construct(self, x, other, *, alpha=2):
        out = x.add(other, alpha=alpha)
        return out


class AddNetWoAlpha(nn.Cell):
    def construct(self, x, other):
        out = x.add(other)
        return out


def _assert(res, expect, rtol=1e-4):
    np.testing.assert_allclose(res, expect, rtol)


def check_add_with_alpha(input_x, input_y, input_a, input_b):
    net = AddNet()
    res1 = net(input_x, input_y)
    expect1 = input_a + 2 * input_b
    _assert(res1.asnumpy(), expect1)


def check_add(input_x, input_y, input_a, input_b):
    net = AddNetWoAlpha()
    res = net(input_x, input_y)
    res2 = net(input_x, input_b.tolist())
    res3 = net(input_x, tuple(input_b.tolist()))
    expect = input_a + input_b
    _assert(res.asnumpy(), expect)
    _assert(res2.asnumpy(), expect)
    _assert(res3.asnumpy(), expect)


def tensor_factory(nptype, mstype):
    a = np.array(np.arange(20).reshape((10, 2)), dtype=nptype)
    b = np.array(np.arange(20).reshape((10, 2)), dtype=nptype)
    x = ms.Tensor(a, dtype=mstype)
    y = ms.Tensor(b, dtype=mstype)
    return a, b, x, y


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_add(mode):
    """
    Feature: Tensor.add.
    Description: Verify the result of add.
    Expectation: expect correct result.
    """
    np_dtypes = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8]
    ms_dtypes = [ms.float32, ms.float64, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]
    ms.set_context(mode=mode)
    for np_dtype, ms_dtype in zip(np_dtypes, ms_dtypes):
        a, b, x, y = tensor_factory(np_dtype, ms_dtype)
        check_add(x, y, a, b)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_add_with_alpha(mode):
    """
    Feature: Tensor.add.
    Description: Verify the result of add.
    Expectation: expect correct result.
    """
    np_dtypes = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8]
    ms_dtypes = [ms.float32, ms.float64, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]
    ms.set_context(mode=mode)
    for np_dtype, ms_dtype in zip(np_dtypes, ms_dtypes):
        a, b, x, y = tensor_factory(np_dtype, ms_dtype)
        check_add_with_alpha(x, y, a, b)
