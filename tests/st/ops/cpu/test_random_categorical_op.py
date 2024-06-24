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

import platform
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

TARGET = "CPU"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp16_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect

    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp32_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float32)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp64_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float64)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp16_int16():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int16
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int16)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int16)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int16)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp16_int32():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int32
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int32)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int32)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int32)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp16_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp32_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float32)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp64_int64():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float64)
    num_sample = 10
    seed = 5
    dtype = ms.int64
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int64)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp16_int16():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int16
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int16)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int16)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int16)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp16_int32():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 10
    seed = 5
    dtype = ms.int32
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int32)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int32)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int32)

    random_cateogoric = RCnet(dtype)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_pynative_fp16_int32_result_random():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 2
    seed = 0
    dtype = ms.int32
    diff = 0
    random_cateogoric = RCnet(dtype)
    expect = random_cateogoric(x, num_sample, seed)
    for _ in range(10):
        output = random_cateogoric(x, num_sample, seed)
        diff += abs(output.asnumpy() - expect)
    assert np.any(diff != 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rc_graph_fp16_int32_result_random():
    """
    Feature: RandomCategorical cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=TARGET)

    x = Tensor(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), ms.float16)
    num_sample = 2
    seed = 0
    dtype = ms.int32
    diff = 0
    random_cateogoric = RCnet(dtype)
    expect = random_cateogoric(x, num_sample, seed)
    for _ in range(10):
        output = random_cateogoric(x, num_sample, seed)
        diff += abs(output.asnumpy() - expect)
    assert np.any(diff != 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    expect = np.array([[4, 4, 2, 4, 4, 3, 4, 4, 4, 4], [4, 4, 2, 4, 4, 3, 4, 4, 4, 4]], dtype=np.int32)
    expect_mac = np.array([[4, 4, 1, 4, 4, 0, 4, 3, 4, 3], [4, 4, 1, 4, 4, 0, 4, 3, 4, 3]], dtype=np.int64)
    expect_windows = np.array([[3, 3, 4, 4, 3, 4, 4, 4, 3, 3], [3, 3, 4, 4, 3, 4, 4, 4, 3, 3]], dtype=np.int64)
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    random_cateogoric = RCnet(dtype)
    random_cateogoric.set_inputs(x_dyn, num_sample, seed)
    output = random_cateogoric(x, num_sample, seed)
    if platform.system().lower() == "darwin":
        diff = output.asnumpy() - expect_mac
    elif platform.system().lower() == "windows":
        diff = output.asnumpy() - expect_windows
    else:
        diff = output.asnumpy() - expect
    assert expect.dtype == output.asnumpy().dtype
    assert np.all(diff == 0)
