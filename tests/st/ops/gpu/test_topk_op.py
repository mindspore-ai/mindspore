# Copyright 2020-21 Huawei Technologies Co., Ltd
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
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_small_2d():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = P.TopK(False)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_3d():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.rand(2, 256, 128).astype(np.float32)
    k = 4
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    k = 2
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_big_2d():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.rand(512, 1024).astype(np.float32)
    k = 512
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    # sorted elements num greater than max thread per block
    x_np = np.random.rand(128, 2048).astype(np.float32)
    k = 1
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(32, 2048).astype(np.float32)
    k = 2048
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    # sorted elements num greater than max share memory per block
    x_np = np.random.rand(16, 40960).astype(np.float32)
    k = 1
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_big_k():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.rand(8, 40960).astype(np.float32)
    k = 4096
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_topk_1d():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.rand(12).astype(np.float32)
    k = 4
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np)[::-1][0:k]

    assert np.allclose(ms_output[0].asnumpy(), np_output)
    x_np = np.random.rand(1200).astype(np.float32)
    k = 256
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np)[::-1][0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(250000).astype(np.float32)
    k = 2000
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np)[::-1][0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(10240).astype(np.float32)
    k = 4096
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np)[::-1][0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(720).astype(np.float32)
    k = 720
    ms_output = P.TopK(True)(Tensor(x_np), k)
    np_output = np.sort(x_np)[::-1][0:k]
    assert np.allclose(ms_output[0].asnumpy()[:k], np_output)
