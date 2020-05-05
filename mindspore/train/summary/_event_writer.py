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
"""Writes events to disk in a logdir."""
import os
import stat
import time

from mindspore import log as logger

from ..._c_expression import EventWriter_
from ._summary_adapter import package_init_event


class _WrapEventWriter(EventWriter_):
    """
    Wrap the c++ EventWriter object.

    Args:
        full_file_name (str): Include directory and file name.
    """

    def __init__(self, full_file_name):
        if full_file_name is not None:
            EventWriter_.__init__(self, full_file_name)


class EventRecord:
    """
    Creates a `EventFileWriter` and write event to file.

    Args:
        full_file_name (str): Summary event file path and file name.
        flush_time (int): The flush seconds to flush the pending events to disk. Default: 120.
    """

    def __init__(self, full_file_name: str, flush_time: int = 120):
        self.full_file_name = full_file_name

        # The first event will be flushed immediately.
        self.flush_time = flush_time
        self.next_flush_time = 0

        # create event write object
        self.event_writer = self._create_event_file()
        self._init_event_file()

        # count the events
        self.event_count = 0

    def _create_event_file(self):
        """Create the event write file."""
        with open(self.full_file_name, 'w'):
            os.chmod(self.full_file_name, stat.S_IWUSR | stat.S_IRUSR)

        # create c++ event write object
        event_writer = _WrapEventWriter(self.full_file_name)
        return event_writer

    def _init_event_file(self):
        """Send the init event to file."""
        self.event_writer.Write((package_init_event()).SerializeToString())
        self.flush()
        return True

    def write_event_to_file(self, event_str):
        """Write the event to file."""
        self.event_writer.Write(event_str)

    def get_data_count(self):
        """Return the event count."""
        return self.event_count

    def flush_cycle(self):
        """Flush file by timer."""
        self.event_count = self.event_count + 1
        # Flush the event writer every so often.
        now = int(time.time())
        if now > self.next_flush_time:
            self.flush()
            # update the flush time
            self.next_flush_time = now + self.flush_time

    def count_event(self):
        """Count event."""
        logger.debug("Write the event count is %r", self.event_count)
        self.event_count = self.event_count + 1
        return self.event_count

    def flush(self):
        """Flush the event file to disk."""
        self.event_writer.Flush()

    def close(self):
        """Flush the event file to disk and close the file."""
        self.flush()
        self.event_writer.Shut()
