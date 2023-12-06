# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor, ops, grad, jit
import mindspore as ms
import numpy as np


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_simple_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream api.
    Expectation: hal.Stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    a = Tensor(2.0)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        ops.abs(a)
    s1.synchronize()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_set_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.set_cur_stream api.
    Expectation: hal.set_cur_stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    cur_stream = ms.hal.current_stream()
    assert cur_stream == ms.hal.default_stream()
    s1 = ms.hal.Stream()
    ms.hal.set_cur_stream(s1)
    assert ms.hal.current_stream() == s1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_stream_query():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream.query api.
    Expectation: hal.Stream.query api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    a = Tensor(np.ones([1024, 2048]), ms.float32)
    b = Tensor(np.ones([2048, 4096]), ms.float32)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)
        assert not s1.query()

    s1.synchronize()
    assert s1.query()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_wait_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.Stream.wait_stream api.
    Expectation: hal.Stream.wait_stream api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)

    with ms.hal.StreamCtx(s2):
        s2.wait_stream(s1)
        ops.matmul(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_synchronize():
    """
    Feature: Hal stream api.
    Description: Test hal.synchronize api.
    Expectation: hal.synchronize api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)

    with ms.hal.StreamCtx(s2):
        ops.matmul(a, b)
    ms.hal.synchronize()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_jit_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.StreamCtx api.
    Expectation: hal.StreamCtx api performs as expected in jit.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()

    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    a *= 4
    with ms.hal.StreamCtx(s1):
        @jit
        def jit_func(a):
            return a + 2
        jit_func(a)
        ops.matmul(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_grad_stream():
    """
    Feature: Hal stream api.
    Description: Test hal.StreamCtx api.
    Expectation: hal.StreamCtx api performs as expected in grad.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def f(x):
        return ops.sin(x)
    grad_fn = grad(f)

    a = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), ms.float32)
    s1 = ms.hal.Stream()
    a *= 4
    with ms.hal.StreamCtx(s1):
        grad_fn(a)
    s1.synchronize()
