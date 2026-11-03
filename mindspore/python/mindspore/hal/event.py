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
from mindspore._c_expression import Stream as Stream_
from mindspore._c_expression import current_stream as current_stream_
from mindspore import _checkparam as Validator
from mindspore import log as logger

function_event_status = {'Event': False, 'wait': False}


class Event(Event_):
    r"""
    Wrapper around a device event, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.Event` instead.

    Device events are synchronization markers that can be used to monitor the device’s progress,
    to accurately measure timing, and to synchronize device streams.

    The underlying device events are lazily initialized when the event is first recorded.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time. Default ``False``.
        blocking (bool, optional): if ``True``, `wait` will be blocking. Default ``False``.

    Examples:
        >>> import mindspore
        >>> start = mindspore.hal.Event(enable_timing=True)
        >>> end = mindspore.hal.Event(enable_timing=True)
        >>> s1 = mindspore.hal.Stream()
        >>> s2 = mindspore.hal.Stream()
        >>> a = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
        >>> c = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
        >>> with mindspore.hal.StreamCtx(s1):
        ...     d = mindspore.ops.matmul(a, b)
        ...     start.record()
        >>> c += 2
        >>> end.record()
        >>> with mindspore.hal.StreamCtx(s2):
        ...     start.synchronize()
        ...     end.synchronize()
        ...     e = c + d
        >>> mindspore.hal.synchronize()
        >>> print(e)
        [[5. 5.]
         [5. 5.]]
        >>> elapsed_time = start.elapsed_time(end)
    """

    def __init__(self, enable_timing=False, blocking=False):
        if not function_event_status['Event']:
            function_event_status['Event'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.Event(enable_timing=True) is deprecated."
                " Please use mindspore.runtime.Event(enable_timing=True)")
        # pylint: disable=useless-super-delegation
        Validator.check_bool(enable_timing, "enable_timing", "Event")
        Validator.check_bool(blocking, "blocking", "Event")
        super().__init__(enable_timing, blocking)

    def record(self, stream=None):
        r"""
        Record the event in a given stream.

        Uses :func:`mindspore.hal.current_stream()` if no `stream` is specified. The
        stream's device must match the event's device.

        Args:
            stream (Stream, optional): a stream to record. If this argument is ``None``,
                current stream will be used. Default ``None``.
        """
        if stream is None:
            stream = current_stream_()
        if not isinstance(stream, Stream_):
            raise TypeError(f"For 'record', the argument 'stream' should be Stream,"
                            f" but got {type(stream)}.")
        super().record(stream)

    def wait(self, stream=None):
        r"""
        Make all future work submitted to the given stream wait for this
        event.

        Use :func:`mindspore.hal.current_stream()` if no `stream` is specified.

        Args:
            stream (Stream, optional): a stream to wait. If this argument is ``None``,
                current stream will be used. Default ``None``.

        Examples:
            >>> import mindspore
            >>> event = mindspore.hal.Event()
            >>> s1 = mindspore.hal.Stream()
            >>> s2 = mindspore.hal.Stream()
            >>> a = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = mindspore.ops.matmul(a, b)
            ...     event.record()
            >>> event.wait()
            >>> d = c + 2
            >>> mindspore.hal.synchronize()
            >>> print(d)
            [[4. 4.]
             [4. 4.]]
        """
        if not function_event_status['wait']:
            function_event_status['wait'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.Event() is deprecated."
                " Please use mindspore.runtime.Event()")
        if stream is None:
            stream = current_stream_()
        if not isinstance(stream, Stream_):
            raise TypeError(f"For 'wait', the argument 'stream' should be Stream,"
                            f" but got {type(stream)}.")
        super().wait(stream)

    def synchronize(self):
        r"""
        Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        # pylint: disable=useless-super-delegation
        super().synchronize()

    def query(self):
        r"""
        Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has completed.

        Examples:
            >>> import mindspore
            >>> a = mindspore.tensor(mindspore.ops.ones([1024, 2048]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([2048, 4096]), mindspore.float32)
            >>> s1 = mindspore.hal.Stream()
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = mindspore.ops.matmul(a, b)
            ...     ev = s1.record_event()
            >>> s1.synchronize()
            >>> assert ev.query()
        """
        # pylint: disable=useless-super-delegation
        return super().query()

    def elapsed_time(self, end_event):
        r"""
        Return the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.

        Args:
            end_event (Event): end event.

        Returns:
            float, the time elapsed in milliseconds.
        """
        # pylint: disable=useless-super-delegation
        if not isinstance(end_event, Event):
            raise TypeError(f"For 'elapsed_time', the argument 'end_event' should be Event,"
                            f" but got {type(end_event)}.")
        return super().elapsed_time(end_event)
