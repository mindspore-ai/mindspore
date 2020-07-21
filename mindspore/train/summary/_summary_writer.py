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
from shutil import disk_usage

from ..._c_expression import EventWriter_
from ._summary_adapter import package_init_event


class BaseWriter:
    """BaseWriter to be subclass."""

    def __init__(self, filepath, max_file_size=None) -> None:
        self._filepath, self._max_file_size = filepath, max_file_size
        self._writer: EventWriter_ = None

    def init_writer(self):
        """Write some metadata etc."""

    @property
    def writer(self) -> EventWriter_:
        """Get the writer."""
        if self._writer is not None:
            return self._writer

        with open(self._filepath, 'w'):
            os.chmod(self._filepath, stat.S_IWUSR | stat.S_IRUSR)
        self._writer = EventWriter_(self._filepath)
        self.init_writer()
        return self._writer

    def write(self, plugin, data):
        """Write data to file."""
        if self.writer and disk_usage(self._filepath).free < len(data) * 32:
            raise RuntimeError(f'The disk space may be soon exhausted by the {type(self).__name__}.')
        if self._max_file_size is None:
            self.writer.Write(data)
        elif self._max_file_size > 0:
            self._max_file_size -= len(data)
            self.writer.Write(data)
        else:
            raise RuntimeError(f"The file written by the {type(self).__name__} "
                               f"has exceeded the specified max file size.")

    def flush(self):
        """Flush the writer."""
        if self._writer is not None:
            self._writer.Flush()

    def close(self):
        """Close the writer."""
        if self._writer is not None:
            self._writer.Shut()


class SummaryWriter(BaseWriter):
    """SummaryWriter for write summaries."""

    def init_writer(self):
        """Write some metadata etc."""
        self.writer.Write(package_init_event().SerializeToString())

    def write(self, plugin, data):
        """Write data to file."""
        if plugin in ('summary', 'graph'):
            super().write(plugin, data)


class LineageWriter(BaseWriter):
    """LineageWriter for write lineage."""

    def write(self, plugin, data):
        """Write data to file."""
        if plugin in ('dataset_graph', 'train_lineage', 'eval_lineage', 'custom_lineage_data'):
            super().write(plugin, data)
