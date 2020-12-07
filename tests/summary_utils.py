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
"""Summary reader."""
import struct

import mindspore.train.summary_pb2 as summary_pb2

_HEADER_SIZE = 8
_HEADER_CRC_SIZE = 4
_DATA_CRC_SIZE = 4


class _EndOfSummaryFileException(Exception):
    """Indicates the summary file is exhausted."""


class SummaryReader:
    """
    Basic summary read function.

    Args:
        canonical_file_path (str): The canonical summary file path.
        ignore_version_event (bool): Whether ignore the version event at the beginning of summary file.
    """

    def __init__(self, canonical_file_path, ignore_version_event=True):
        self._file_path = canonical_file_path
        self._ignore_version_event = ignore_version_event
        self._file_handler = None

    def __enter__(self):
        self._file_handler = open(self._file_path, "rb")
        if self._ignore_version_event:
            self.read_event()
        return self

    def __exit__(self, *unused_args):
        self._file_handler.close()
        return False

    def read_event(self):
        """Read next event."""
        file_handler = self._file_handler
        header = file_handler.read(_HEADER_SIZE)
        if not header:
            return None
        data_len = struct.unpack('Q', header)[0]
        # Ignore crc check.
        file_handler.read(_HEADER_CRC_SIZE)

        event_str = file_handler.read(data_len)
        # Ignore crc check.
        file_handler.read(_DATA_CRC_SIZE)
        summary_event = summary_pb2.Event.FromString(event_str)

        return summary_event
