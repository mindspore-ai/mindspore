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


class InplaceAddNet(nn.Cell):
    def construct(self, x, other, *, alpha=2):
        out = x.add_(other, alpha=alpha)
        return out, x


class InplaceAddNetWoAlpha(nn.Cell):
    def construct(self, x, other):
        out = x.add_(other)
        return out, x


class BuildinInplaceAddNet(nn.Cell):
    def construct(self, x, other):
        x += other
        return x


def _assert(res, expect, rtol=1e-4):
    np.testing.assert_allclose(res, expect, rtol)


def check_add(input_x, input_y, input_a, input_b, inplace):
    net = InplaceAddNetWoAlpha()
    expect = input_a + input_b

    res1 = net(input_x.copy(), input_y)
    _assert(res1[0].asnumpy(), expect)
    if inplace:
        _assert(res1[1].asnumpy(), expect)

def check_add_with_alpha(input_x, input_y, input_a, input_b, inplace):
    net = InplaceAddNet()
    res = net(input_x, input_y)
    expect = input_a + 2 * input_b
    _assert(res[0].asnumpy(), expect)
    if inplace:
        _assert(res[1].asnumpy(), expect)


def check_buildin_add(input_x, input_y, input_a, input_b):
    net = BuildinInplaceAddNet()
    res1 = net(input_x.copy(), input_y)
    expect = input_a + input_b
    _assert(res1.asnumpy(), expect)


def tensor_factory(nptype, mstype):
    a = np.array(np.arange(20).reshape((10, 2)), dtype=nptype)
    b = np.array(np.arange(20).reshape((10, 2)), dtype=nptype)
    x = ms.Tensor(a, dtype=mstype)
    y = ms.Tensor(b, dtype=mstype)
    return a, b, x, y


def tensor_scalar_factory(nptype, mstype):
    a = np.array(np.arange(20).reshape((10, 2)), dtype=nptype)
    x = ms.Tensor(a, dtype=mstype)
    s = 10
    return a, x, s


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_add_(mode):
    """
    Feature: Tensor.add_ and Tensor.__iadd__.
    Description: Verify the result of add.
    Expectation: expect correct result.
    """
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    elif mode == 'ge':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    inplace = False
    if ms.context.get_context("device_target") == "Ascend" and mode == 'pynative':
        inplace = True
    np_dtypes = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8]
    ms_dtypes = [ms.float32, ms.float64, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8]
    for np_dtype, ms_dtype in zip(np_dtypes, ms_dtypes):
        # a + b
        a, b, x, y = tensor_factory(np_dtype, ms_dtype)
        check_add(x, y, a, b, inplace)
        a, x, s = tensor_scalar_factory(np_dtype, ms_dtype)
        check_add(x, s, a, s, inplace)
        # a + b * alpha
        a, b, x, y = tensor_factory(np_dtype, ms_dtype)
        check_add_with_alpha(x, y, a, b, inplace)
        a, x, s = tensor_scalar_factory(np_dtype, ms_dtype)
        check_add_with_alpha(x, s, a, s, inplace)
        # a += b
        a, b, x, y = tensor_factory(np_dtype, ms_dtype)
        check_buildin_add(x, y, a, b)
        a, x, s = tensor_scalar_factory(np_dtype, ms_dtype)
        check_buildin_add(x, s, a, s)
