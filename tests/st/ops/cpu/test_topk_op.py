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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_top_k_functional():
    """
    Feature: test_top_k_functional
    Description: test top_k functional API
    Expectation: the output is as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = F.topk(Tensor(x_np), k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = F.topk(Tensor(x_np), k, sorted=False)
    assert np.allclose(ms_output[0].asnumpy(), x_np)

    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    k = 2
    ms_output = F.topk(Tensor(x_np), k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 1024).astype(np.float32)
    k = 512
    ms_output = F.topk(Tensor(x_np), k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_top_k_tensor():
    """
    Feature: test_top_k_tensor
    Description: test top_k tensor API
    Expectation: the output is as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = Tensor(x_np).topk(k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(3, 4).astype(np.float32)
    k = 4
    ms_output = Tensor(x_np).topk(k, sorted=False)
    assert np.allclose(ms_output[0].asnumpy(), x_np)

    x_np = np.random.rand(2, 3, 4).astype(np.float32)
    k = 2
    ms_output = Tensor(x_np).topk(k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)

    x_np = np.random.rand(512, 1024).astype(np.float32)
    k = 512
    ms_output = Tensor(x_np).topk(k, sorted=True)
    np_output = np.sort(x_np, axis=-1)[..., ::-1][..., 0:k]
    assert np.allclose(ms_output[0].asnumpy(), np_output)
