# Copyright 2020 Huawei Technologies Co., Ltd
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
"""The container of metadata used in profiler parser."""


class HWTSContainer:
    """
    HWTS output container.

    Args:
        split_list (list): The split list of metadata in HWTS output file.
    """
    def __init__(self, split_list):
        self._op_name = ''
        self._duration = None
        self._status = split_list[0]
        self._task_id = split_list[6]
        self._cycle_counter = float(split_list[7])
        self._stream_id = split_list[8]

    @property
    def status(self):
        """Get the status of the operator, i.e. Start or End."""
        return self._status

    @property
    def task_id(self):
        """Get the task id of the operator."""
        return self._task_id

    @property
    def cycle_counter(self):
        """Get the cycle counter."""
        return self._cycle_counter

    @property
    def stream_id(self):
        """Get the stream id of the operator."""
        return self._stream_id

    @property
    def op_name(self):
        """Get the name of the operator."""
        return self._op_name

    @op_name.setter
    def op_name(self, name):
        """Set the name of the operator."""
        self._op_name = name

    @property
    def duration(self):
        """Get the duration of the operator execution."""
        return self._duration

    @duration.setter
    def duration(self, value):
        """Set the duration of the operator execution."""
        self._duration = value


class TimelineContainer:
    """
    A container of operator computation metadata.

    Args:
        split_list (list): The split list of metadata in op_compute output file.
    """
    def __init__(self, split_list):
        self._op_name = split_list[0]
        self._stream_id = str(split_list[1])
        self._start_time = float(split_list[2])
        self._duration = float(split_list[3])
        self._pid = None
        if len(split_list) == 5:
            self._pid = int(split_list[4])

    @property
    def op_name(self):
        """Get the name of the operator."""
        return self._op_name

    @property
    def stream_id(self):
        """Get the stream id of the operator."""
        return self._stream_id

    @property
    def start_time(self):
        """Get the execution start time of the operator."""
        return self._start_time

    @property
    def duration(self):
        """Get the duration of the operator execution."""
        return self._duration

    @property
    def pid(self):
        """Get the pid of the operator execution."""
        return self._pid
