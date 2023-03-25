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
import mindspore as ms
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class TopkNet(nn.Cell):

    def __init__(self, isSorted=True):
        super(TopkNet, self).__init__()
        self.op = P.TopK(sorted=isSorted)

    def construct(self, x, k):
        return self.op(x, k)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_topk_dynamic_shape():
    """
    Feature: test TopK op in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = TopkNet()

    x_dyn = Tensor(shape=[2, None], dtype=ms.float16)
    k = 3
    net.set_inputs(x_dyn, k)

    x = Tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], ms.float16)
    values, indices = net(x, k)
    expect_shape = (2, 3)
    assert values.asnumpy().shape == expect_shape
    assert indices.asnumpy().shape == expect_shape


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.level1
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
