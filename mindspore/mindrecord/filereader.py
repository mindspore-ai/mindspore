# Copyright 2019 Huawei Technologies Co., Ltd
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
This module is to read data from mindrecord.
"""
from .shardreader import ShardReader
from .shardheader import ShardHeader
from .shardutils import populate_data
from .shardutils import MIN_CONSUMER_COUNT, MAX_CONSUMER_COUNT, check_filename
from .common.exceptions import ParamValueError, ParamTypeError

__all__ = ['FileReader']


class FileReader:
    """
    Class to read MindRecord File series.

    Args:
       file_name (str, list[str]): One of MindRecord File or a file list.
       num_consumer(int, optional): Number of consumer threads which load data to memory (default=4).
           It should not be smaller than 1 or larger than the number of CPUs.
       columns (list[str], optional): A list of fields where corresponding data would be read (default=None).
       operator(int, optional): Reserved parameter for operators (default=None).

    Raises:
        ParamValueError: If file_name, num_consumer or columns is invalid.
    """

    def __init__(self, file_name, num_consumer=4, columns=None, operator=None):
        if isinstance(file_name, list):
            for f in file_name:
                check_filename(f)
        else:
            check_filename(file_name)

        if num_consumer is not None:
            if isinstance(num_consumer, int):
                if num_consumer < MIN_CONSUMER_COUNT or num_consumer > MAX_CONSUMER_COUNT():
                    raise ParamValueError("Consumer number should between {} and {}."
                                          .format(MIN_CONSUMER_COUNT, MAX_CONSUMER_COUNT()))
            else:
                raise ParamValueError("Consumer number is illegal.")
        else:
            raise ParamValueError("Consumer number is illegal.")
        if columns:
            if isinstance(columns, list):
                self._columns = columns
            else:
                raise ParamTypeError('columns', 'list')
        else:
            self._columns = None
        self._reader = ShardReader()
        self._reader.open(file_name, num_consumer, columns, operator)
        self._header = ShardHeader(self._reader.get_header())
        self._reader.launch()

    def get_next(self):
        """
        Yield a batch of data according to columns at a time.

        Yields:
            dictionary: keys are the same as columns.

        Raises:
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        iterator = self._reader.get_next()
        while iterator:
            for blob, raw in iterator:
                yield populate_data(raw, blob, self._columns, self._header.blob_fields, self._header.schema)
            iterator = self._reader.get_next()

    def close(self):
        """Stop reader worker and close File."""
        self._reader.close()
