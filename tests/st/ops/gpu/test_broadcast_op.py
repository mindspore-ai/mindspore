# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
import mindspore.context as context
import numpy as np

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nobroadcast():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    x1_np = np.random.rand(10, 20).astype(np.float32)
    x2_np = np.random.rand(10, 20).astype(np.float32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    x1_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    x2_np = np.random.rand(1, 4, 1, 6).astype(np.float32)

    output_ms = P.Minimum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.minimum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Maximum()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.maximum(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Greater()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np > x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Less()(Tensor(x1_np), Tensor(x2_np))
    output_np = x1_np < x2_np
    assert np.allclose(output_ms.asnumpy(), output_np)

    output_ms = P.Pow()(Tensor(x1_np), Tensor(x2_np))
    output_np = np.power(x1_np, x2_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
