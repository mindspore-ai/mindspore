# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import mindspore as ms
from mindspore.common import jit
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


@jit
def mul_scalar_by_one(x, y):
    P.AssignAdd()(x, 1)
    x = x * 1
    P.AssignAdd()(x, y)
    return x, P.Cos()(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_mul_scalar_by_one():
    """
    Feature: Keep multiply scalar by one.
    Description: Keep multiply scalar by one.
    Expectation: Run success.
    """
    x = ms.Tensor(2.0)
    y = ms.Tensor(3.0)
    x, c = mul_scalar_by_one(x, y)
    assert x == 6.0
    assert np.allclose(c.asnumpy(), np.cos(6.0), 0.0001, 0.0001)


@jit
def mul_by_one(x, y):
    P.AssignAdd()(x, 1)
    x = x * ms.Tensor(1.0)
    P.AssignAdd()(x, y)
    return x, P.Cos()(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_mul_by_one():
    """
    Feature: Keep multiply by one.
    Description: Keep multiply by one.
    Expectation: Run success.
    """
    x = ms.Tensor(2.0)
    y = ms.Tensor(3.0)
    x, c = mul_by_one(x, y)
    assert x == 6.0
    assert np.allclose(c.asnumpy(), np.cos(6.0), 0.0001, 0.0001)


@jit
def add_by_zero(x, y):
    P.AssignAdd()(x, 1)
    x = x + 0
    P.AssignAdd()(x, y)
    return x, P.Cos()(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_add_by_zero():
    """
    Feature: Keep add by zero.
    Description: Keep add by zero.
    Expectation: Run success.
    """
    x = ms.Tensor(2.0)
    y = ms.Tensor(3.0)
    x, c = add_by_zero(x, y)
    assert x == 6.0
    assert np.allclose(c.asnumpy(), np.cos(6.0), 0.0001, 0.0001)
