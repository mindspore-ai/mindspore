# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

import os
import stat
from urllib.parse import quote
from shutil import disk_usage

import numpy as np

from mindspore.train.summary.enums import PluginEnum, WriterPluginEnum
from mindspore import log as logger
from mindspore.train._utils import _make_directory
from mindspore.train.summary._summary_adapter import package_init_event
from mindspore._c_expression import security
if not security.enable_security():
    from mindspore._c_expression import EventWriter_


FREE_DISK_SPACE_TIMES = 32
FILE_MODE = 0o400


class BaseWriter:
    """BaseWriter to be subclass."""

    def __init__(self, filepath, max_file_size=None) -> None:
        self._filepath, self._max_file_size = filepath, max_file_size
        self._writer: EventWriter_ = None

    def init_writer(self):
        """Write some metadata etc."""

    @property
    def writer(self):
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
        # 8: data length
        # 4: crc32 of data length
        # 4: crc32 of data
        metadata_length = 8 + 4 + 4
        required_length = len(data) + metadata_length
        if self.writer and disk_usage(self._filepath).free < required_length * FREE_DISK_SPACE_TIMES:
            raise RuntimeError(f"The disk space may be soon exhausted by the '{self._filepath}'.")
        if self._max_file_size is None:
            self.writer.Write(data)
        elif self._max_file_size >= required_length:
            self._max_file_size -= required_length
            self.writer.Write(data)
        else:
            raise RuntimeWarning(f"'max_file_size' reached: There are {self._max_file_size} bytes remaining, "
                                 f"but the '{self._filepath}' requires to write {required_length} bytes.")

    def flush(self):
        """Flush the writer."""
        if self._writer is not None:
            self._writer.Flush()

    def close(self):
        """Close the writer."""
        try:
            os.chmod(self._filepath, FILE_MODE)
        except FileNotFoundError:
            logger.debug("The summary file %r has been removed.", self._filepath)
        if self._writer is not None:
            self._writer.Shut()


class SummaryWriter(BaseWriter):
    """SummaryWriter for write summaries."""

    def init_writer(self):
        """Write some metadata etc."""
        self.write(WriterPluginEnum.SUMMARY.value, package_init_event().SerializeToString())

    def write(self, plugin, data):
        """Write data to file."""
        if plugin in (WriterPluginEnum.SUMMARY.value, PluginEnum.GRAPH.value):
            super().write(plugin, data)


class LineageWriter(BaseWriter):
    """LineageWriter for write lineage."""

    def write(self, plugin, data):
        """Write data to file."""
        if plugin in (PluginEnum.DATASET_GRAPH.value, PluginEnum.TRAIN_LINEAGE.value, PluginEnum.EVAL_LINEAGE.value,
                      PluginEnum.CUSTOM_LINEAGE_DATA.value):
            super().write(plugin, data)


class ExportWriter(BaseWriter):
    """ExportWriter for export data."""

    def write(self, plugin, data):
        """Write data to file."""
        if plugin == WriterPluginEnum.EXPORTER.value:
            self.export_data(data, data.get('export_option'))

    def flush(self):
        """Flush the writer."""

    def close(self):
        """Close the writer."""

    def export_data(self, data, export_option):
        """
        export the tensor data.

        Args:
            data (dict): Export data info.
            export_option (Union[None, str]): The export options.
        """
        options = {
            'npy': self._export_npy
        }

        if export_option in options:
            options[export_option](data, self._filepath)

    @staticmethod
    def _export_npy(data, export_dir):
        """
        export the tensor data as npy.

        Args:
            data (dict): Export data info.
            export_dir (str): The path of export dir.
        """
        tag = quote(data.get('tag'), safe="")
        step = int(data.get('step'))
        np_value = data.get('value')
        path = _make_directory(os.path.join(export_dir, 'tensor'))

        #  128 is the typical length of header of npy file
        metadata_length = 128
        required_length = np_value.nbytes + metadata_length
        if disk_usage(path).free < required_length * FREE_DISK_SPACE_TIMES:
            raise RuntimeError(f"The disk space may be soon exhausted by the '{path}'.")

        np_path = "{}/{}_{}.npy".format(path, tag, step)
        np.save(np_path, np_value)
        os.chmod(np_path, FILE_MODE)
