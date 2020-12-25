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

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_topk():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = P.TopK(False)(Tensor(x_np), k)
    assert np.allclose(ms_output[0].asnumpy(), x_np)

    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    k = 2
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 1024).astype(np.float32)
    k = 512
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    # sorted elements num greater than max thread per block
    x_np = np.random.rand(512, 2048).astype(np.float32)
    k = 1
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 2048).astype(np.float32)
    k = 2048
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    # sorted elements num greater than max share memory per block
    x_np = np.random.rand(512, 40960).astype(np.float32)
    k = 1
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 40960).astype(np.float32)
    k = 40960
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 40960).astype(np.float32)
    k = 40960
    ms_output = P.TopK(False)(Tensor(x_np), k)
    assert np.allclose(ms_output[0].asnumpy(), x_np)
