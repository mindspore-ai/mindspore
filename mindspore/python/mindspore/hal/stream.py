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
"""Hal stream class"""
from mindspore._c_expression import Stream as Stream_
from mindspore._c_expression import set_cur_stream as set_cur_stream_
from mindspore._c_expression import synchronize as synchronize_
from mindspore._c_expression import current_stream as current_stream_
from mindspore._c_expression import default_stream as default_stream_
from mindspore._c_expression import communication_stream as communication_stream_

from mindspore import _checkparam as Validator
from mindspore import log as logger
from .event import Event

function_stream_status = {'record_event': False, 'wait_event': False, 'wait_stream': False,
                          'query': False, 'synchronize': False, 'set_cur_stream': False,
                          'current_stream': False, 'default_stream': False,
                          'communication_stream': False, 'StreamCtx': False}

class Stream(Stream_):
    r"""
    Wrapper around a device stream, this api will be deprecated and removed in future versions, please use
    the api :class:`mindspore.runtime.Stream` instead.

    A device stream is a linear sequence of execution that belongs to a specific device,
    independent from other streams.

    Args:
        priority (int, optional): priority of the stream, lower numbers represent higher priorities.
            By default, streams have priority ``0``.
        kwargs (dict): keyword arguments.
    """

    def __init__(self, priority=0, **kwargs):
        self.init_finished = False
        if 'stream' in kwargs and kwargs['stream'] is not None:
            super().__init__(kwargs['stream'])

        Validator.check_is_int(priority, 'priority', "Stream")
        if 'stream_id' in kwargs:
            Validator.check_is_int(kwargs['stream_id'], 'stream_id', "Stream")
            super().__init__(priority, kwargs['stream_id'])
        else:
            super().__init__(priority)
        self.init_finished = True

    def record_event(self, event=None):
        r"""
        Record an event.

        Args:
            event (Event, optional): event to record. If this argument is ``None``,
                a new one will be allocated. Default is ``None``.

        Returns:
            Event, recorded event.

        Examples:
            >>> import mindspore
            >>> a = mindspore.tensor(mindspore.ops.ones([3, 3]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([3, 3]), mindspore.float32)
            >>> s1 = mindspore.hal.Stream()
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = a + b
            ...     event = s1.record_event()
            ...     d = a * b
            >>> cur_stream = mindspore.hal.current_stream()
            >>> cur_stream.wait_event(event)
            >>> e = c + 3
            >>> print(e)
            [[5. 5. 5.]
             [5. 5. 5.]
             [5. 5. 5.]]
        """
        if not function_stream_status['record_event']:
            function_stream_status['record_event'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.Stream().record_event() is deprecated."
                " Please use mindspore.runtime.Stream().record_event()"
            )
        if event is None:
            event = Event()
        if not isinstance(event, Event):
            raise TypeError(f"For 'record_event', the argument 'event' should be Event,"
                            f" but got {type(event)}.")
        event.record(self)
        return event

    def wait_event(self, event):
        r"""
        Make all future work submitted to the stream wait for an event.

        Args:
            event (Event): an event to wait for.

        Examples:
            >>> import mindspore
            >>> a = mindspore.tensor(mindspore.ops.ones([3, 3]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([3, 3]), mindspore.float32)
            >>> s1 = mindspore.hal.Stream()
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = a + b
            ...     event = s1.record_event()
            ...     d = a * b
            >>> cur_stream = mindspore.hal.current_stream()
            >>> cur_stream.wait_event(event)
            >>> e = c + 3
            >>> print(e)
            [[5. 5. 5.]
             [5. 5. 5.]
             [5. 5. 5.]]
        """
        if not function_stream_status['wait_event']:
            function_stream_status['wait_event'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.current_stream().wait_event(event) is deprecated."
                " Please use mindspore.runtime.current_stream().wait_event(event)"
            )
        if not isinstance(event, Event):
            raise TypeError(f"For 'wait_event', the argument 'event' should be Event,"
                            f" but got {type(event)}.")
        event.wait(self)

    def wait_stream(self, stream):
        r"""
        Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        Examples:
            >>> import mindspore
            >>> s1 = mindspore.hal.Stream()
            >>> s2 = mindspore.hal.Stream()
            >>> a = mindspore.tensor(mindspore.ops.ones([1, 2]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([2, 2]), mindspore.float32)
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = mindspore.ops.matmul(a, b)
            >>> with mindspore.hal.StreamCtx(s2):
            ...     s2.wait_stream(s1)
            ...     d = mindspore.ops.matmul(c, b)
            >>> mindspore.hal.synchronize()
            >>> print(d)
            [[4. 4.]]
        """
        if not function_stream_status['wait_stream']:
            function_stream_status['wait_stream'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.Stream() is deprecated."
                " Please use mindspore.runtime.Stream()"
            )
        if not isinstance(stream, Stream):
            raise TypeError(f"For 'wait_stream', the argument 'stream' should be Stream,"
                            f" but got {type(stream)}.")
        self.wait_event(stream.record_event())

    def synchronize(self):
        r"""
        Wait for all the kernels in this stream to complete.

        Examples:
            >>> import mindspore
            >>> a = mindspore.tensor(mindspore.ops.ones([1024, 2048]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([2048, 4096]), mindspore.float32)
            >>> s1 = mindspore.hal.Stream()
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = mindspore.ops.matmul(a, b)
            >>> s1.synchronize()
            >>> assert s1.query()
        """
        # pylint: disable=useless-super-delegation
        super().synchronize()

    def query(self):
        r"""
        Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.

        Examples:
            >>> import mindspore
            >>> a = mindspore.tensor(mindspore.ops.ones([1024, 2048]), mindspore.float32)
            >>> b = mindspore.tensor(mindspore.ops.ones([2048, 4096]), mindspore.float32)
            >>> s1 = mindspore.hal.Stream()
            >>> with mindspore.hal.StreamCtx(s1):
            ...     c = mindspore.ops.matmul(a, b)
            >>> s1.synchronize()
            >>> assert s1.query()
        """
        if not function_stream_status['query']:
            function_stream_status['query'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.Stream() is deprecated."
                " Please use mindspore.runtime.Stream()"
            )
        # pylint: disable=useless-super-delegation
        return super().query()

    def __eq__(self, other):
        if not isinstance(other, Stream):
            raise TypeError(f"For '__eq__', the argument 'other' should be Stream,"
                            f" but got {type(other)}.")
        return super().__eq__(other)

    def __hash__(self):
        return hash((self.id, self.device_id))

    def __repr__(self):
        if self.init_finished:
            return super().__repr__()
        return ''


def synchronize():
    r"""
    Synchronize all streams on current device, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.synchronize` instead.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1024, 2048]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([2048, 4096]), mindspore.float32)
        >>> s1 = mindspore.hal.Stream()
        >>> with mindspore.hal.StreamCtx(s1):
        ...     c = mindspore.ops.matmul(a, b)
        >>> mindspore.hal.synchronize()
        >>> assert s1.query()
    """
    if not function_stream_status['synchronize']:
        function_stream_status['synchronize'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.synchronize() is deprecated."
            " Please use mindspore.runtime.synchronize()"
        )
    synchronize_()


def set_cur_stream(stream):
    r"""
    Set the current stream, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.set_cur_stream` instead.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.

    Examples:
        >>> import mindspore
        >>> cur_stream = mindspore.hal.current_stream()
        >>> assert cur_stream == mindspore.hal.default_stream()
        >>> s1 = mindspore.hal.Stream()
        >>> mindspore.hal.set_cur_stream(s1)
        >>> assert mindspore.hal.current_stream() == s1
        >>> mindspore.hal.set_cur_stream(mindspore.hal.default_stream())
    """
    if not function_stream_status['set_cur_stream']:
        function_stream_status['set_cur_stream'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.Stream() is deprecated."
            " Please use mindspore.runtime.Stream()"
        )
    if stream is None:
        return
    if not isinstance(stream, Stream):
        raise TypeError(f"For 'set_cur_stream', the argument 'stream' should be Stream,"
                        f" but got {type(stream)}.")
    set_cur_stream_(stream)


def current_stream():
    r"""
    Return current stream used on this device, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.current_stream` instead.

    Returns:
        stream (Stream), current stream.

    Examples:
        >>> import mindspore
        >>> cur_stream = mindspore.hal.current_stream()
        >>> assert cur_stream == mindspore.hal.default_stream()
    """
    if not function_stream_status['current_stream']:
        function_stream_status['current_stream'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.current_stream() is deprecated."
            " Please use mindspore.runtime.current_stream()"
        )
    return Stream(stream=current_stream_())


def default_stream():
    r"""
    Return default stream on this device, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.default_stream` instead.

    Returns:
        stream (Stream), default stream.

    Examples:
        >>> import mindspore
        >>> cur_stream = mindspore.hal.current_stream()
        >>> assert cur_stream == mindspore.hal.default_stream()
    """
    if not function_stream_status['default_stream']:
        function_stream_status['default_stream'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.default_stream() is deprecated."
            " Please use mindspore.runtime.default_stream()"
        )
    return Stream(stream=default_stream_())


def communication_stream():
    r"""
    Return communication stream on this device, this api will be deprecated and removed in future versions, please use
    the api :func:`mindspore.runtime.communication_stream` instead.

    Returns:
        stream (Stream), communication stream.

    Examples:
        >>> import mindspore
        >>> mindspore.hal.communication_stream()
        Stream(device_name=Ascend, device_id:0, stream id:1)
    """
    if not function_stream_status['communication_stream']:
        function_stream_status['communication_stream'] = True
        logger.warning(
            "WARN_DEPRECATED: The usage of mindspore.hal.communication_stream() is deprecated."
            " Please use mindspore.runtime.communication_stream()"
        )
    return Stream(stream=communication_stream_())


class StreamCtx():
    r"""
    Context-manager that selects a given stream, this api will be deprecated and removed in future versions, please use
    the api :class:`mindspore.runtime.StreamCtx` instead.

    All kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        ctx_stream (Stream): selected stream. This manager is a no-op if it's ``None``.

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor(mindspore.ops.ones([1024, 2048]), mindspore.float32)
        >>> b = mindspore.tensor(mindspore.ops.ones([2048, 4096]), mindspore.float32)
        >>> s1 = mindspore.hal.Stream()
        >>> with mindspore.hal.StreamCtx(s1):
        ...     c = mindspore.ops.matmul(a, b)
        >>> mindspore.hal.synchronize()
        >>> assert s1.query()
    """

    def __init__(self, ctx_stream):
        if not function_stream_status['StreamCtx']:
            function_stream_status['StreamCtx'] = True
            logger.warning(
                "WARN_DEPRECATED: The usage of mindspore.hal.StreamCtx(s1) is deprecated."
                " Please use mindspore.runtime.StreamCtx(s1)"
            )
        if ctx_stream is not None and not isinstance(ctx_stream, Stream):
            raise TypeError(f"For 'StreamCtx', the argument 'ctx_stream' should be Stream,"
                            f" but got {type(ctx_stream)}.")
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        if self.stream is None:
            return
        self.prev_stream = current_stream()
        set_cur_stream(self.stream)
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream is None:
            return
        set_cur_stream(self.prev_stream)
        return
