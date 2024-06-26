# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@ms.jit
def greater_forward_func(x, y):
    return ops.greater(x, y)


@ms.jit
def greater_backward_func(x, y):
    return ops.grad(greater_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_greater_forward():
    """
    Feature: Ops.
    Description: test op greater.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([False, True, False])
    out = greater_forward_func(x, y)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_greater_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op greater.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    expect_out = np.array([0, 0, 0])
    grads = greater_backward_func(x, y)
    assert np.allclose(grads.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_greater_vmap():
    """
    Feature: test vmap function.
    Description: test greater op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    x = ms.Tensor(np.array([[[1, 2, 3]]]), ms.int32)
    y = ms.Tensor(np.array([[[1, 1, 4]]]), ms.int32)
    expect_out = np.array([[[False]], [[True]], [[False]]])
    nest_vmap = ops.vmap(ops.vmap(
        greater_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x, y)
    assert np.allclose(out.asnumpy(), expect_out)
