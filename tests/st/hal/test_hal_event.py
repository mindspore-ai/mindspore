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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_hal_event_args():
    """
    Feature: Hal event api.
    Description: Test hal.event args.
    Expectation: hal.event performs as expected.
    """
    ev1 = ms.hal.Event()
    assert ev1 is not None

    ev2 = ms.hal.Event(enable_timing=True, blocking=True)
    assert ev2 is not None


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
    context.set_context(mode=context.PYNATIVE_MODE)

    start = ms.hal.Event(enable_timing=True)
    end = ms.hal.Event(enable_timing=True)
    start.record()
    a = Tensor(np.ones([50, 50]), ms.float32)
    ops.matmul(a, a)
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
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.hal.Stream()
    s2 = ms.hal.Stream()

    ev1 = ms.hal.Event()
    ev2 = ms.hal.Event()
    a = Tensor(np.random.randn(20, 20), ms.float32)
    with ms.hal.StreamCtx(s1):
        b = ops.matmul(a, a)
        ev1.record()

    with ms.hal.StreamCtx(s2):
        ev1.wait()
        c = ops.matmul(b, b)

    ev2.wait()
    ev2.synchronize()
    assert ev1.query() is True
    assert ev1.query() is True
    assert np.allclose(ops.matmul(a, a).asnumpy(), b.asnumpy())
    assert np.allclose(ops.matmul(b, b).asnumpy(), c.asnumpy())


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
    context.set_context(mode=context.PYNATIVE_MODE)

    stream = ms.hal.Stream()

    ev1 = ms.hal.Event(True, False)
    ev2 = ms.hal.Event(True, False)

    ev1.record(stream)
    a = Tensor(np.random.randn(5000, 5000), ms.float32)
    with ms.hal.StreamCtx(stream):
        ops.matmul(a, a)
        stream.record_event(ev2)
        assert ev2.query() is False
        assert stream.query() is False

    ev1.synchronize()
    ev2.synchronize()
    assert ev2.query() is True
    stream.synchronize()
    assert stream.query() is True
    elapsed_time = ev1.elapsed_time(ev2)
    assert elapsed_time > 0
