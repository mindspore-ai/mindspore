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
    """
    Python event class wrapping around hardware event interfaces.
    """
    def __init__(self, enable_timing=False, blocking=False):
        # pylint: disable=useless-super-delegation
        super().__init__(enable_timing, blocking)

    def record(self, stream=None):
        """
        Record event in specified stream.
        This event captures tasks on specified stream at the time of this call.
        """
        if stream is None:
            stream = current_stream_()
        super().record(stream)

    def wait(self, stream=None):
        """
        Make the specified stream wait on this event.
        The specified stream will wait till the tasks captured by this event are completed.
        """
        if stream is None:
            stream = current_stream_()
        super().wait(stream)
