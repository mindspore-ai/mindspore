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
"""Hal event class"""
from mindspore._c_expression import Event as Event_
from mindspore._c_expression import current_stream as current_stream_


class Event(Event_):
    r"""
    Wrapper around a device event.

    Device events are synchronization markers that can be used to monitor the device’s progress,
    to accurately measure timing, and to synchronize device streams.

    The underlying device events are lazily initialized when the event is first recorded or exported to another
    process. After creation, only streams on the same device may record the event.
    However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional) – indicates if the event should measure time (default: ``False``)
        blocking (bool, optional): if ``True``, `wait` will be blocking (default: ``False``)

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> start = ms.hal.Event(enable_timing=True)
        >>> end = ms.hal.Event(enable_timing=True)
        >>> s1 = ms.hal.Stream()
        >>> s2 = ms.hal.Stream()
        >>> a = Tensor(np.ones([2, 2]), ms.float32)
        >>> b = Tensor(np.ones([2, 2]), ms.float32)
        >>> c = Tensor(np.ones([2, 2]), ms.float32)
        >>> with ms.hal.StreamCtx(s1):
        >>>     d = ops.matmul(a, b)
        >>>     start.record()
        >>> c += 2
        >>> end.record()
        >>> with ms.hal.StreamCtx(s2):
        >>>     start.wait()
        >>>     end.synchronize()
        >>>     e = c + d
        >>> ms.hal.synchronize()
        >>> print(e)
        [[5. 5.]
        [5. 5.]]
        >>> elapsed_time = start.elapsed_time(end)
        >>> assert elapsed_time > 0
    """
    def __init__(self, enable_timing=False, blocking=False):
        # pylint: disable=useless-super-delegation
        super().__init__(enable_timing, blocking)

    def record(self, stream=None):
        r"""
        Records the event in a given stream.

        Uses ``mindspore.hal.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = current_stream_()
        super().record(stream)

    def wait(self, stream=None):
        r"""
        Makes all future work submitted to the given stream wait for this
        event.

        Use ``mindspore.hal.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = current_stream_()
        super().wait(stream)

    def synchronize(self):
        r"""
        Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        # pylint: disable=useless-super-delegation
        super().synchronize()

    def query(self):
        r"""
        Checks if all work currently captured by event has completed.

        Outputs:
            A boolean indicating if all work currently captured by event has completed.
        """
        # pylint: disable=useless-super-delegation
        return super().query()

    def elapsed_time(self, end_event):
        r"""
        Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.

        Outputs:
        end_event (Event): end event.
        """
        # pylint: disable=useless-super-delegation
        return super().elapsed_time(end_event)
