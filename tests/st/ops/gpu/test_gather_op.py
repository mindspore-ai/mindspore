# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class GatherNet(nn.Cell):
    def __init__(self, dim=0):
        super(GatherNet, self).__init__()
        self.gather = P.GatherD()
        self.dim = dim

    def construct(self, x, index):
        return self.gather(x, self.dim, index)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp32_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float32)
    output = P.GatherD()(x, dim, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp32_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int64)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float32)
    output = P.GatherD()(x, dim, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp16_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float16)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float16)
    output = P.GatherD()(x, dim, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_pynative_fp16_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float16)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int64)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float16)
    output = P.GatherD()(x, dim, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp32_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float32)
    gather = GatherNet(dim)
    output = gather(x, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp32_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float32)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int64)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float32)
    gather = GatherNet(dim)
    output = gather(x, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp16_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float16)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int32)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float16)
    gather = GatherNet(dim)
    output = gather(x, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

def test_gather_graph_fp16_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    error = 1e-3
    x = Tensor(np.array([[1.303, 2.333], [3.232, 4.235]]), ms.float16)
    dim = 1
    index = Tensor(np.array([[0, 0], [1, 0]]), ms.int64)
    expect = np.array([[1.303, 1.303], [4.235, 3.232]], np.float16)
    gather = GatherNet(dim)
    output = gather(x, index)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
