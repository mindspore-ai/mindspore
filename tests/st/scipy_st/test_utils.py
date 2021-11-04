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
"""st for scipy.utils"""

import pytest
import numpy as onp
from mindspore import context, Tensor
from mindspore.scipy.utils import _safe_normalize


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(10,), (10, 1)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_safe_normalize(mode, shape, dtype):
    """
    Feature: ALL TO ALL
    Description: test cases for _safe_normalize
    Expectation: the result match scipy
    """
    context.set_context(mode=mode)
    x = onp.random.random(shape).astype(dtype)
    normalized_x, x_norm = _safe_normalize(Tensor(x))

    normalized_x = normalized_x.asnumpy()
    x_norm = x_norm.asnumpy()
    assert onp.allclose(onp.sum(normalized_x ** 2), 1)
    assert onp.allclose(x / x_norm, normalized_x)
