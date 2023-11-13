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
class Event():
    """
    Python event class wrapping around hardware event interfaces.
    """
    def __init__(self):
        # Use _c_expression hal class.
        pass

    def synchronize(self):
        """
        Wait for tasks captured by this event to complete.
        """
        return

    def record(self, stream: Stream):
        """
        Record event in specified stream.
        This event captures tasks on specified stream at the time of this call.
        """
        return

    def wait(self, stream: Stream):
        """
        Make the specified stream wait on this event.
        The specified stream will wait till the tasks captured by this event are completed.
        """
        return

    def query(self):
        """
        Query completion status of all tasks captured by this event.
        """
        return


def elapsed_time(start_event: Event, end_event: Event):
    """
    Return the elapsed time between two events.
    """
    return
