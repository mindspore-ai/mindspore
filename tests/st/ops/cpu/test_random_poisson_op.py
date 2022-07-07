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

import pytest
import numpy as np
import mindspore as ms
from mindspore.ops import functional as F
from mindspore import Tensor


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [ms.float64, ms.float32, ms.float16, ms.int64, ms.int32])
@pytest.mark.parametrize("shape_dtype", [ms.int64, ms.int32])
@pytest.mark.parametrize("rate_dtype", [ms.float64, ms.float32, ms.float16, ms.int64, ms.int32])
def test_poisson_function_op(dtype, shape_dtype, rate_dtype):
    """
    Feature: Poisson functional interface
    Description: test output shape of the poisson functional interface.
    Expectation: match to tensorflow output.
    """

    shape = Tensor(np.array([3, 5]), shape_dtype)
    mean = Tensor(np.array([0.5]), rate_dtype)
    output = F.poisson(shape, mean, seed=1, dtype=dtype)
    assert output.shape == (3, 5, 1)
    assert output.dtype == dtype

    shape = Tensor(np.array([3, 2]), shape_dtype)
    mean = Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), rate_dtype)
    output = F.poisson(shape, mean, seed=5, dtype=dtype)
    assert output.shape == (3, 2, 2, 2)
    assert output.dtype == dtype
