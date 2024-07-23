# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Rand(nn.Cell):
    def construct(self, size, dtype):
        return ops.rand(size, dtype=dtype)


class RandLike(nn.Cell):
    def construct(self, x, dtype):
        return ops.rand_like(x, dtype=dtype)


class Randn(nn.Cell):
    def construct(self, size, dtype):
        return ops.randn(size, dtype=dtype)


class RandnLike(nn.Cell):
    def construct(self, x, dtype):
        return ops.randn_like(x, dtype=dtype)


class RandInt(nn.Cell):
    def construct(self, low, high, size, dtype):
        return ops.randint(low, high, size, dtype=dtype)


class RandIntLike(nn.Cell):
    def construct(self, x, low, high, dtype):
        return ops.randint_like(x, low, high, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, ms.float32])
def test_rand_functions(mode, dtype):
    r"""
    Feature: ops.rand, ops.randn, ops.rand_like, ops.randn_like
    Description: Verify the result of ops.rand, ops.randn, ops.rand_like, ops.randn_like
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), ms.float16)
    size = (2, 3)
    net1 = Rand()
    net2 = Randn()
    net3 = RandLike()
    net4 = RandnLike()
    out1 = net1(size, dtype)
    out2 = net2(size, dtype)
    out3 = net3(x, dtype)
    out4 = net4(x, dtype)
    if dtype is None:
        assert out1.dtype == ms.float32
        assert out2.dtype == ms.float32
        assert out3.dtype == ms.float16
        assert out4.dtype == ms.float32
    else:
        assert out1.dtype == dtype
        assert out2.dtype == dtype
        assert out3.dtype == dtype
        assert out4.dtype == dtype

    assert out1.shape == size
    assert out2.shape == size
    assert out3.shape == x.shape
    assert out4.shape == x.shape


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, ms.int32])
def test_randint_functions(mode, dtype):
    r"""
    Feature: ops.randint, ops.randint_like
    Description: Verify the result of ops.randint, ops.randint_like
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), ms.int32)
    net = RandInt()
    net2 = RandIntLike()
    out = net(0, 10, (2, 3), dtype=dtype)
    out2 = net2(x, low=0, high=15, dtype=dtype)
    if dtype is None:
        assert out.dtype == ms.int64
        assert out2.dtype == ms.int32
    else:
        assert out.dtype == dtype
        assert out2.dtype == dtype
    assert out.shape == (2, 3)
    assert out2.shape == x.shape
    assert out.max() < 10 and out.min() >= 0
    assert out2.max() < 15 and out2.min() >= 0
