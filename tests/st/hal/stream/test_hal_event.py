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
import mindspore.context as context
from mindspore import Tensor, ops
import mindspore as ms
import numpy as np
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.device_utils import set_device


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@test_utils.run_test_with_On
def test_hal_event_args():
    """
    Feature: runtime event api.
    Description: Test runtime.event args.
    Expectation: runtime.event performs as expected.
    """
    set_device()
    ev1 = ms.runtime.Event()
    assert ev1 is not None

    ev2 = ms.runtime.Event(enable_timing=True, blocking=True)
    assert ev2 is not None


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_hal_event_elapsed_time():
    """
    Feature: runtime event api.
    Description: Test runtime.event.elapsed_time.
    Expectation: runtime.event.elapsed_time performs as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    start = ms.runtime.Event(enable_timing=True)
    end = ms.runtime.Event(enable_timing=True)
    start.record()
    a = Tensor(np.ones([50, 50]), ms.float32)
    ops.matmul(a, a)
    end.record()
    start.synchronize()
    end.synchronize()

    elapsed_time = start.elapsed_time(end)
    assert elapsed_time > 0


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=6.00)
def test_hal_event_wait():
    """
    Feature: runtime event api.
    Description: Test runtime.event.wait.
    Expectation: runtime.device.wait as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    s1 = ms.runtime.Stream()
    s2 = ms.runtime.Stream()

    ev1 = ms.runtime.Event()
    ev2 = ms.runtime.Event()
    a = Tensor(np.random.randn(20, 20), ms.float32)
    with ms.runtime.StreamCtx(s1):
        b = ops.matmul(a, a)
        ev1.record()

    with ms.runtime.StreamCtx(s2):
        ev1.wait()
        c = ops.matmul(b, b)
        ev2.record()

    ev2.wait()
    ev2.synchronize()
    assert ev1.query() is True
    assert ev2.query() is True
    assert np.allclose(ops.matmul(a, a).asnumpy(), b.asnumpy())
    assert np.allclose(ops.matmul(b, b).asnumpy(), c.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.0)
def test_hal_event_sync():
    """
    Feature: runtime event sync.
    Description: Test runtime.device.sync.
    Expectation: runtime.device.sync as expected.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    stream = ms.runtime.Stream()

    ev1 = ms.runtime.Event(True, False)
    ev2 = ms.runtime.Event(True, False)

    ev1.record(stream)
    a = Tensor(np.random.randn(5000, 5000), ms.float32)
    with ms.runtime.StreamCtx(stream):
        ops.bmm(a, a)
        stream.record_event(ev2)

    ev1.synchronize()
    ev2.synchronize()
    assert ev2.query() is True
    stream.synchronize()
    assert stream.query() is True
    elapsed_time = ev1.elapsed_time(ev2)
    assert elapsed_time > 0
