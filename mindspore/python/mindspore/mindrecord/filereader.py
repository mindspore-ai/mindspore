# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This module is to read data from MindRecord.
"""
import platform

from .shardreader import ShardReader
from .shardheader import ShardHeader
from .shardutils import populate_data
from .shardutils import check_parameter
from .common.exceptions import ParamTypeError

__all__ = ['FileReader']


class FileReader:
    """
    Class to read MindRecord files.

    Note:
        If `file_name` is a file path, it tries to load all MindRecord files generated \
        in a conversion, and throws an exception if a MindRecord file is missing.
        If `file_name` is file path list, only the MindRecord files in the list are loaded.

    Args:
        file_name (str, list[str]): One of MindRecord file path or file path list.
        num_consumer (int, optional): Number of reader workers which load data. Default: 4.
            It should not be smaller than 1 or larger than the number of processor cores.
        columns (list[str], optional): A list of fields where corresponding data would be read. Default: None.
        operator (int, optional): Reserved parameter for operators. Default: None.

    Raises:
        ParamValueError: If `file_name` , `num_consumer` or `columns` is invalid.

    Examples:
        >>> from mindspore.mindrecord import FileReader
        >>>
        >>> mindrecord_file = "/path/to/mindrecord/file"
        >>> reader = FileReader(file_name=mindrecord_file)
        >>>
        >>> # create iterator for mindrecord and get saved data
        >>> for _, item in enumerate(reader.get_next()):
        ...     ori_data = item
        >>> reader.close()
    """

    @check_parameter
    def __init__(self, file_name, num_consumer=4, columns=None, operator=None):
        if columns:
            if isinstance(columns, list):
                self._columns = columns
            else:
                raise ParamTypeError('columns', 'list')
        else:
            self._columns = None

        self._reader = ShardReader()
        self._file_name = ""
        if platform.system().lower() == "windows":
            if isinstance(file_name, list):
                self._file_name = [item.replace("\\", "/") for item in file_name]
            else:
                self._file_name = file_name.replace("\\", "/")
        else:
            self._file_name = file_name

        self._reader.open(self._file_name, num_consumer, columns, operator)
        self._header = ShardHeader(self._reader.get_header())
        self._reader.launch()

    def get_next(self):
        """
        Yield a batch of data according to columns at a time.

        Returns:
            dict, a batch whose keys are the same as columns.

        Raises:
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        iterator = self._reader.get_next()
        while iterator:
            for blob, raw in iterator:
                yield populate_data(raw, blob, self._columns, self._header.blob_fields, self._header.schema)
            iterator = self._reader.get_next()

    def close(self):
        """Stop reader worker and close file."""
        self._reader.close()

    def schema(self):
        """
        Get the schema of the MindRecord.

        Returns:
            dict, the schema info.
        """
        return self._header.schema

    def len(self):
        """
        Get the number of the samples in MindRecord.

        Returns:
            int, the number of the samples in MindRecord.
        """
        return self._reader.len()
