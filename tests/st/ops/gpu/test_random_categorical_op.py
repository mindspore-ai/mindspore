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
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class RCnet(nn.Cell):
    def __init__(self, dtype=ms.int64):
        super(RCnet, self).__init__()
        self.rc = P.RandomCategorical(dtype)

    def construct(self, logits, num_sample, seed):
        return self.rc(logits, num_sample, seed)

TARGET = "GPU"

def test_rc_graph_fp16_int64():
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
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

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rc_pynative_fp16_int32_dynamic_shape():
    """
    Feature: random_cateogoric operation dynamic shape test
    Description: test random_cateogoric dynamic shape operation
    Expectation: random_cateogoric output == expect
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int32
    expect = np.array([[4, 3, 2, 4, 4, 4, 3, 4, 1, 3], [4, 3, 2, 4, 4, 4, 3, 4, 1, 3]], dtype=np.int32)
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    random_cateogoric = RCnet(dtype)
    random_cateogoric.set_inputs(x_dyn, num_sample, seed)
    output = random_cateogoric(x, num_sample, seed)
    diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)
