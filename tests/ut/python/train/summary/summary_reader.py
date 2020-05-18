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
import mindspore.train.summary_pb2 as summary_pb2
import struct

_HEADER_SIZE = 8
_HEADER_CRC_SIZE = 4
_DATA_CRC_SIZE = 4


class SummaryReader:
    """Read events from summary file."""

    def __init__(self, file_name):
        self._file_name = file_name
        self._file_handler = open(self._file_name, "rb")
        # skip version event
        self.read_event()

    def read_event(self):
        """Read next event."""
        file_handler = self._file_handler
        header = file_handler.read(_HEADER_SIZE)
        data_len = struct.unpack('Q', header)[0]
        file_handler.read(_HEADER_CRC_SIZE)
        event_str = file_handler.read(data_len)
        file_handler.read(_DATA_CRC_SIZE)
        summary_event = summary_pb2.Event.FromString(event_str)
        return summary_event
