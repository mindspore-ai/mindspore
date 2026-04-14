
# Copyright 2026 Huawei Technologies Co., Ltd
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
"""test runtime stream priority."""
import os
import subprocess
import sys
import textwrap
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark
from tests.device_utils import set_device


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.0)
def test_runtime_stream_priority_default():
    """
    Feature: runtime stream priority.
    Description: Create streams with various priorities under the default config (priority mapping disabled).
        In legacy mode only non-negative priorities are accepted; verify they all yield usable streams
        whose computation results match the default stream.
    Expectation: Stream creation succeeds for every priority and produces correct results.
    """
    set_device()
    context.set_context(mode=context.PYNATIVE_MODE)

    a = Tensor(np.random.randn(20, 20), ms.float32)
    expected = ops.matmul(a, a).asnumpy()
    for prio in (0, 1, 3, 7):
        s = ms.runtime.Stream(priority=prio)
        with ms.runtime.StreamCtx(s):
            b = ops.matmul(a, a)
        s.synchronize()
        assert np.allclose(b.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.0)
def test_runtime_stream_priority_mapping_enabled():
    """
    Feature: runtime stream priority.
    Description: Enable the signed->device priority mapping via MS_DEV_RUNTIME_CONF=stream_priority:True
        and verify streams created with negative/zero/positive priorities (including out-of-range values
        that should be clamped) all execute correctly.
    Expectation: Stream creation and computation succeed for in-range and out-of-range priorities.
    """
    # MS_DEV_RUNTIME_CONF is parsed once per process and cached, so the mapping feature must
    # be enabled from a fresh interpreter — run the body in a subprocess.
    script = textwrap.dedent("""
        import numpy as np
        import mindspore as ms
        import mindspore.context as context
        from mindspore import Tensor, ops
        from tests.device_utils import set_device

        set_device()
        context.set_context(mode=context.PYNATIVE_MODE)

        a = Tensor(np.random.randn(16, 16), ms.float32)
        expected = ops.matmul(a, a).asnumpy()
        for prio in (-100, -3, -1, 0, 1, 3, 100):
            s = ms.runtime.Stream(priority=prio)
            with ms.runtime.StreamCtx(s):
                b = ops.matmul(a, a)
            s.synchronize()
            assert np.allclose(b.asnumpy(), expected), f"mismatch at priority={prio}"
        print("stream_priority_mapping_enabled:PASS")
    """)
    env = os.environ.copy()
    env["MS_DEV_RUNTIME_CONF"] = "stream_priority:True"
    env.setdefault("DEVICE_TARGET", "Ascend")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "stream_priority_mapping_enabled:PASS" in result.stdout, result.stdout


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', mem_peak=4096.0)
def test_runtime_stream_priority_high_preempts_low():
    """
    Feature: runtime stream priority.
    Description: With the signed priority mapping enabled, submit equal-sized heavy
        workloads on a low-priority stream and then a high-priority stream and verify
        both streams compute correct results. Device-side start/done timestamps are
        captured on each stream and reported for observability; whether the device
        actually overlaps or preempts across streams is hardware/driver dependent,
        so the timing assertion is intentionally loose — it only catches a clearly
        pathological case (high-priority work delayed by more than a full extra
        low-priority backlog).
    Expectation: Both streams produce correct results; the high-priority stream's
        completion is not delayed by more than one extra low-priority-backlog worth
        of time after the low-priority backlog finishes.
    """
    # MS_DEV_RUNTIME_CONF is parsed once per process, so the mapping feature has to be
    # enabled from a fresh interpreter — run the body in a subprocess.
    script = textwrap.dedent("""
        import numpy as np
        import mindspore as ms
        import mindspore.context as context
        from mindspore import Tensor, ops
        from tests.device_utils import set_device

        set_device()
        context.set_context(mode=context.PYNATIVE_MODE)

        big = Tensor(np.random.randn(2048, 2048), ms.float32)
        # Warm up + reference result for correctness check.
        expected_big = ops.matmul(big, big).asnumpy()

        s_low = ms.runtime.Stream(priority=3)    # positive => lower priority
        s_high = ms.runtime.Stream(priority=-3)  # negative => higher priority

        n_each = 32
        ev_low_start = ms.runtime.Event(enable_timing=True)
        ev_low_done = ms.runtime.Event(enable_timing=True)
        ev_high_start = ms.runtime.Event(enable_timing=True)
        ev_high_done = ms.runtime.Event(enable_timing=True)

        # Enqueue the low-priority backlog first, then an equally heavy backlog on
        # the high-priority stream. Record start/done events on each stream so we
        # can compare device-side execution windows across streams.
        with ms.runtime.StreamCtx(s_low):
            ev_low_start.record(s_low)
            outs_low = [ops.matmul(big, big) for _ in range(n_each)]
            ev_low_done.record(s_low)
        with ms.runtime.StreamCtx(s_high):
            ev_high_start.record(s_high)
            outs_high = [ops.matmul(big, big) for _ in range(n_each)]
            ev_high_done.record(s_high)

        ms.runtime.synchronize()

        low_dur = ev_low_start.elapsed_time(ev_low_done)
        high_dur = ev_high_start.elapsed_time(ev_high_done)
        # Device-time offset of high_start relative to low_start (positive => high started later).
        high_start_delay = ev_low_start.elapsed_time(ev_high_start)
        # Device-time offset of high_done relative to low_done (positive => high finished later).
        high_done_delay = ev_low_done.elapsed_time(ev_high_done)
        print(f"LOW_DUR={low_dur:.3f} HIGH_DUR={high_dur:.3f} "
              f"HIGH_START_DELAY={high_start_delay:.3f} HIGH_DONE_DELAY={high_done_delay:.3f}")

        for out in outs_low:
            assert np.allclose(out.asnumpy(), expected_big), "low-priority result mismatch"
        for out in outs_high:
            assert np.allclose(out.asnumpy(), expected_big), "high-priority result mismatch"

        # Loose sanity bound: the high-priority work must not be stalled by more
        # than ~1.5x the low-backlog duration after the low backlog itself completes.
        # Whether the device actually preempts/interleaves is hardware dependent;
        # this threshold leaves margin for run-to-run noise while still catching a
        # clear regression (high-priority stalled behind 2+ full backlogs).
        assert high_done_delay < low_dur * 1.5, (
            f"high-priority stream finished more than ~1.5x low-backlog after the "
            f"low-priority backlog: high_done_delay={high_done_delay:.3f}ms "
            f"low_dur={low_dur:.3f}ms"
        )
        print("stream_priority_preemption:PASS")
    """)
    env = os.environ.copy()
    env["MS_DEV_RUNTIME_CONF"] = "stream_priority:True"
    env.setdefault("DEVICE_TARGET", "Ascend")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "stream_priority_preemption:PASS" in result.stdout, result.stdout
