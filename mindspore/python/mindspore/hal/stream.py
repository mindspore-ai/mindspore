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
from .event import Event


class Stream(Stream_):
    """
    Python stream class wrapping around hardware stream interfaces.
    """
    def __init__(self, priority=0, **kwargs):
        if 'stream_id' in kwargs:
            super().__init__(priority, kwargs['stream_id'])
        else:
            super().__init__(priority)

    def record_event(self, event=None):
        """
        Record event in this stream.
        This event captures tasks on this stream at the time of this call.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def wait_event(self, event):
        """
        Wait on the specified event.
        This stream will wait till the tasks captured by this event are completed.
        """
        event.wait(self)

    def wait_stream(self, stream):
        """
        Synchronize with specified stream: wait for tasks on another stream to complete.
        """
        self.wait_event(stream.record_event())


def synchronize():
    """
    Synchronize all streams on current device.(Each MindSpore process only occupies one device)
    """
    synchronize_()

def set_cur_stream(stream):
    """
    Set current stream to specified stream.
    """
    set_cur_stream_(stream)

def current_stream():
    """
    Return current stream used on this device.
    """
    return current_stream_()

def default_stream():
    """
    Return default stream on this device.
    """
    return default_stream_()

class StreamCtx():
    """
    All MindSpore operators within this context will be executed in the given stream.

    Args:
        ctx_stream (Stream): Stream this context wraps.
    """
    def __init__(self, ctx_stream):
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
