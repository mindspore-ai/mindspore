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
from mindspore import Tensor, ops
import mindspore as ms
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_event_elapsed_time():
    """
    Feature: Hal event api.
    Description: Test hal.event.elapsed_time.
    Expectation: hal.event.elapsed_time performs as expected.
    """
    start = ms.hal.Event(enable_timing=True)
    end = ms.hal.Event(enable_timing=True)
    start.record()
    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    ops.matmul(a, b)
    end.record()
    start.synchronize()
    end.synchronize()

    elapsed_time = start.elapsed_time(end)
    assert elapsed_time > 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_event_wait():
    """
    Feature: Hal event api.
    Description: Test hal.event.wait.
    Expectation: hal.device.wait as expected.
    """
    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    event = ms.hal.Event()
    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)
        event.record()

    with ms.hal.StreamCtx(s2):
        event.query()
        event.wait()
        ops.matmul(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_event_sync():
    """
    Feature: Hal event sync.
    Description: Test hal.device.sync.
    Expectation: hal.device.sync as expected.
    """
    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    event = ms.hal.Event()
    a = Tensor(np.ones([1, 2]), ms.float32)
    b = Tensor(np.ones([2,]), ms.float32)
    c = Tensor(np.ones([2,]), ms.float32)
    with ms.hal.StreamCtx(s1):
        ops.matmul(a, b)
        event.record()
        c += 2

    with ms.hal.StreamCtx(s2):
        event.synchronize()
        ops.matmul(a, b)
