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
class Stream():
    """
    Python stream class wrapping around hardware stream interfaces.
    """
    def __init__(self):
        # Use _c_expression hal class.
        pass

    def synchronize(self):
        """
        Wait for tasks on this stream to complete.
        """
        return

    def record_event(self, event: Event):
        """
        Record event in this stream.
        This event captures tasks on this stream at the time of this call.
        """
        return

    def wait_event(self, event: Event):
        """
        Wait on the specified event.
        This stream will wait till the tasks captured by this event are completed.
        """
        return

    def sync_stream(self, stream: Stream):
        """
        Synchronize with specified stream: wait for tasks on another stream to complete.
        """
        return

    def query(self):
        """
        Query this stream's completion status.
        """
        return

def synchronize(stream: Stream = None):
    """
    Synchronize specified stream. If stream is `None`, synchronize all streams on current device.
    Note:
        Each MindSpore process only occupies one device.
    """
    return

def set_cur_stream(stream: Stream):
    """
    Set current stream to specified stream.
    """
    return

def current_stream():
    """
    Return current stream used on this device.
    """
    return

def default_stream():
    """
    Return default stream on this device.
    """
    return

class StreamCtx():
    """
    All MindSpore operators within this context will be executed in the given stream.

    Args:
        ctx_stream (Stream): Stream this context wraps.
    """
    def __init__(self, ctx_stream: Stream):
        self.stream = ctx_stream

    def __enter__(self):
        return

    def __exit__(self, except_type, except_value, traceback):
        return
