# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test hyper offload with sliding window"""
import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore.graph._parse import compile_config

ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="0.05GB")

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
    compile_config.ENABLE_HYPER_OFFLOAD_SLIDE="1"
    yield
    os.unsetenv("MS_DEV_HYPER_OFFLOAD")
    compile_config.ENABLE_HYPER_OFFLOAD_SLIDE="0"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=50.00)
def test_memory_with_sliding_window_base():
    """
    Feature: Hyper offload with sliding window method.
    Description: Basic case.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        y = 2 * x
        z = y + y
        t = x + x
        w = x * x
        m = x - 2
        n = y + 5
        q = y - 1
        v = z + w + n + m + q + t
        return v

    x = Tensor(np.ones([200, 300, 50]), ms.float32)
    output = func(x)
    expect_output = 14*x
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=44.0)
def test_memory_with_sliding_window_with_control_flow():
    """
    Feature: Hyper offload with sliding window method.
    Description: Basic control flow case.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, a):
        y = 2 * x
        z = y + y
        t = x + x
        w = x
        if a > 0:
            w = x + x
            b = w + z
            t = x + b
        else:
            w = x + 2
            t = x + z
        m = x - 2
        n = y + 5
        q = y - 1
        v = z + w + n + m + q + t
        return v

    x = Tensor(np.ones([200, 300, 20]), ms.float32)
    output = func(x, Tensor(2))
    expect_output = 20*x
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=44.0)
def test_memory_with_sliding_window_with_control_flow2():
    """
    Feature: Hyper offload with sliding window method.
    Description: Nested control flow case in true branch.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, a, flag):
        y = 2 * x
        z = y + y
        t = x + x
        w = x
        if a > 0:
            w = x * x
            if flag:
                b = w + z
                t = x + b
            else:
                b = w - z
                t = x - b
        else:
            w = x + 2
            t = x + z
        m = x - 2
        n = y + 5
        q = y - 1
        v = z + w + n + m + q + t
        return v

    x = Tensor(np.ones([200, 300, 20]), ms.float32)
    output = func(x, Tensor(2), Tensor(True))
    expect_output = 18*x
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=40.0)
def test_memory_with_sliding_window_with_control_flow3():
    """
    Feature: Hyper offload with sliding window method.
    Description: Nested control flow case in false branch.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, a, flag):
        y = 2 * x
        z = y + y
        t = x + x
        w = x
        if a > 0:
            w = x * x
            t = x + z
        else:
            w = x + 2
            if flag:
                b = w + z
                t = x + b
            else:
                b = w - z
                t = x - b
        m = x - 2
        n = y + 5
        q = y - 1
        v = z + w + n + m + q + t
        return v

    x = Tensor(np.ones([200, 300, 20]), ms.float32)
    output = func(x, Tensor(2), Tensor(True))
    expect_output = 17*x
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.0)
def test_memory_with_sliding_window_with_tuple_output():
    """
    Feature: Hyper offload with sliding window method.
    Description: Tuple output.
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        y = 2 * x
        z = y + y
        t = x + x
        w = x * x
        m = x - 2
        n = y + 5
        q = y - 1
        return (z, t + w + m + n + q)

    x = Tensor(np.ones([200, 300, 20]), ms.float32)
    output = func(x)
    expect_output = 10*x
    assert np.allclose(output[1].asnumpy(), expect_output.asnumpy())
