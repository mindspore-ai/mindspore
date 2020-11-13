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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class RCnet(nn.Cell):
    def __init__(self, num_sample, seed=0, dtype=ms.int64):
        super(RCnet, self).__init__()
        self.rc = P.RandomCategorical(dtype)
        self.num_sample = num_sample
        self.seed = seed

    def construct(self, logits):
        return self.rc(logits, self.num_sample, self.seed)

TARGET = "GPU"

def test_rc_graph_fp16_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    random_cateogoric = RCnet(num_sample, seed, dtype)
    output = random_cateogoric(x)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_graph_fp32_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float32)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    random_cateogoric = RCnet(num_sample, seed, dtype)
    output = random_cateogoric(x)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_graph_fp64_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float64)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    random_cateogoric = RCnet(num_sample, seed, dtype)
    output = random_cateogoric(x)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_graph_fp16_int16():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int16
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int16)

    random_cateogoric = RCnet(num_sample, seed, dtype)
    output = random_cateogoric(x)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_graph_fp16_int32():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int32
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int32)

    random_cateogoric = RCnet(num_sample, seed, dtype)
    output = random_cateogoric(x)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_pynative_fp16_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    output = P.RandomCategorical(dtype)(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_pynative_fp32_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float32)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    output = P.RandomCategorical(dtype)(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_pynative_fp64_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float64)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    output = P.RandomCategorical(dtype)(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_pynative_fp16_int16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int16
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int16)

    output = P.RandomCategorical(dtype)(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)

def test_rc_pynative_fp16_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int32
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int32)

    output = P.RandomCategorical(dtype)(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)
