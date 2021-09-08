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

#!/usr/bin/env python3

import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE)
input_x = Tensor(np.ones([1, 1, 120, 640]), dtype=mstype.float32)
input_y = Tensor(np.full((1, 1, 120, 640), 4), dtype=mstype.float32)
ret_output_2 = Tensor(np.full((1, 1, 120, 640), 3.125), dtype=mstype.float32)


@pytest.mark.level1
@pytest.mark.timeout(60)
@pytest.mark.env_Ascend_1p
@pytest.mark.env_Gpu_1p
@pytest.mark.env_CPU
@pytest.mark.Function
def test_ms_function_nested_local():
    @ms_function
    def function1(x, y):
        x = x ** y
        x /= y
        x += y
        x -= 1
        x %= 2
        return x

    @ms_function
    def function11(x, y):
        r = function1(x, y)
        out = r + r
        return out

    @ms_function
    def function2(x, y):
        r1 = function1(x, y)
        r2 = function11(x, y)
        z = r1 * r2
        return z

    with pytest.raises(TypeError) as info:
        output2 = function2(input_x, input_y)
        print(output2)
    assert "Not support nested calling of local ms_function, please delete decorator of 'function11'." in str(
        info.value)


@ms_function
def function1_g(x, y):
    x = x ** y
    x /= y
    x += y
    x -= 1
    x %= 2
    return x

@ms_function
def function11_g(x, y):
    r = function1_g(x, y)
    out = r + r
    return out

@pytest.mark.level1
@pytest.mark.timeout(60)
@pytest.mark.env_Ascend_1p
@pytest.mark.env_Gpu_1p
@pytest.mark.env_CPU
@pytest.mark.Function
def test_ms_function_nested_global():
    @ms_function
    def function2_g(x, y):
        r1 = function1_g(x, y)
        r2 = function11_g(x, y)
        z = r1 * r2
        return z

    output2 = function2_g(input_x, input_y)
    assert np.allclose(output2.asnumpy(), ret_output_2.asnumpy(), 0.0001, 0.0001)
