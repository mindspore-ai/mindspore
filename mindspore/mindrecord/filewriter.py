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
This module is to write data into mindrecord.
"""
import os
import re
import stat
import numpy as np
from mindspore import log as logger
from .shardwriter import ShardWriter
from .shardreader import ShardReader
from .shardheader import ShardHeader
from .shardindexgenerator import ShardIndexGenerator
from .shardutils import MIN_SHARD_COUNT, MAX_SHARD_COUNT, VALID_ATTRIBUTES, VALID_ARRAY_ATTRIBUTES, \
    check_filename, VALUE_TYPE_MAP
from .common.exceptions import ParamValueError, ParamTypeError, MRMInvalidSchemaError, MRMDefineIndexError

__all__ = ['FileWriter']


class FileWriter:
    """
    Class to write user defined raw data into MindRecord File series.

    Note:
        The mindrecord file may fail to be read if the file name is modified.

    Args:
        file_name (str): File name of MindRecord File.
        shard_num (int, optional): The Number of MindRecord File (default=1).
            It should be between [1, 1000].

    Raises:
        ParamValueError: If `file_name` or `shard_num` is invalid.
    """

    def __init__(self, file_name, shard_num=1):
        check_filename(file_name)
        self._file_name = file_name

        if shard_num is not None:
            if isinstance(shard_num, int):
                if shard_num < MIN_SHARD_COUNT or shard_num > MAX_SHARD_COUNT:
                    raise ParamValueError("Shard number should between {} and {}."
                                          .format(MIN_SHARD_COUNT, MAX_SHARD_COUNT))
            else:
                raise ParamValueError("Shard num is illegal.")
        else:
            raise ParamValueError("Shard num is illegal.")

        self._shard_num = shard_num
        self._index_generator = True
        suffix_shard_size = len(str(self._shard_num - 1))

        if self._shard_num == 1:
            self._paths = [self._file_name]
        else:
            self._paths = ["{}{}".format(self._file_name,
                                         str(x).rjust(suffix_shard_size, '0'))
                           for x in range(self._shard_num)]

        self._append = False
        self._header = ShardHeader()
        self._writer = ShardWriter()
        self._generator = None

    @classmethod
    def open_for_append(cls, file_name):
        """
        Open MindRecord file and get ready to append data.

        Args:
            file_name (str): String of MindRecord file name.

        Returns:
            FileWriter, file writer for the opened MindRecord file.

        Raises:
            ParamValueError: If file_name is invalid.
            FileNameError: If path contains invalid characters.
            MRMOpenError: If failed to open MindRecord File.
            MRMOpenForAppendError: If failed to open file for appending data.
        """
        check_filename(file_name)
        # construct ShardHeader
        reader = ShardReader()
        reader.open(file_name)
        header = ShardHeader(reader.get_header())
        reader.close()

        instance = cls("append")
        instance.init_append(file_name, header)
        return instance

    def init_append(self, file_name, header):
        self._append = True
        self._file_name = file_name
        self._header = header
        self._writer.open_for_append(file_name)

    def add_schema(self, content, desc=None):
        """
        Return a schema id if schema is added successfully, or raise an exception.

        Args:
            content (dict): Dictionary of user defined schema.
            desc (str, optional): String of schema description (default=None).

        Returns:
            int, schema id.

        Raises:
            MRMInvalidSchemaError: If schema is invalid.
            MRMBuildSchemaError: If failed to build schema.
            MRMAddSchemaError: If failed to add schema.
        """
        ret, error_msg = self._validate_schema(content)
        if ret is False:
            raise MRMInvalidSchemaError(error_msg)
        schema = self._header.build_schema(content, desc)
        return self._header.add_schema(schema)

    def add_index(self, index_fields):
        """
        Select index fields from schema to accelerate reading.

        Args:
            index_fields (list[str]): Fields would be set as index which should be primitive type.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            ParamTypeError: If index field is invalid.
            MRMDefineIndexError: If index field is not primitive type.
            MRMAddIndexError: If failed to add index field.
            MRMGetMetaError: If the schema is not set or failed to get meta.
        """
        if not index_fields or not isinstance(index_fields, list):
            raise ParamTypeError('index_fields', 'list')

        for field in index_fields:
            if field in self._header.blob_fields:
                raise MRMDefineIndexError("Failed to set field {} since it's not primitive type.".format(field))
            if not isinstance(field, str):
                raise ParamTypeError('index field', 'str')
        return self._header.add_index_fields(index_fields)

    def _verify_based_on_schema(self, raw_data):
        """
        Verify data according to schema and remove invalid data if validation failed.

        1) allowed data type contains: "int32", "int64", "float32", "float64", "string", "bytes".

        Args:
           raw_data (list[dict]): List of raw data.
        """
        error_data_dic = {}
        schema_content = self._header.schema
        for field in schema_content:
            for i, v in enumerate(raw_data):
                if i in error_data_dic:
                    continue

                if field not in v:
                    error_data_dic[i] = "for schema, {} th data is wrong, " \
                                        "there is not '{}' object in the raw data.".format(i, field)
                    continue
                field_type = type(v[field]).__name__
                if field_type not in VALUE_TYPE_MAP:
                    error_data_dic[i] = "for schema, {} th data is wrong, " \
                                        "data type for '{}' is not matched.".format(i, field)
                    continue

                if schema_content[field]["type"] not in VALUE_TYPE_MAP[field_type]:
                    error_data_dic[i] = "for schema, {} th data is wrong, " \
                                        "data type for '{}' is not matched.".format(i, field)
                    continue

                if field_type == 'ndarray':
                    if 'shape' not in schema_content[field]:
                        error_data_dic[i] = "for schema, {} th data is wrong, " \
                                            "data type for '{}' is not matched.".format(i, field)
                    else:
                        try:
                            np.reshape(v[field], schema_content[field]['shape'])
                        except ValueError:
                            error_data_dic[i] = "for schema, {} th data is wrong, " \
                                                "data type for '{}' is not matched.".format(i, field)
        error_data_dic = sorted(error_data_dic.items(), reverse=True)
        for i, v in error_data_dic:
            raw_data.pop(i)
            logger.warning(v)

    def open_and_set_header(self):
        """
        Open writer and set header.
        """
        if not self._writer.is_open:
            self._writer.open(self._paths)
        if not self._writer.get_shard_header():
            self._writer.set_shard_header(self._header)

    def write_raw_data(self, raw_data, parallel_writer=False):
        """
        Write raw data and generate sequential pair of MindRecord File and \
        validate data based on predefined schema by default.

        Args:
           raw_data (list[dict]): List of raw data.
           parallel_writer (bool, optional): Load data parallel if it equals to True (default=False).

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            ParamTypeError: If index field is invalid.
            MRMOpenError: If failed to open MindRecord File.
            MRMValidateDataError: If data does not match blob fields.
            MRMSetHeaderError: If failed to set header.
            MRMWriteDatasetError: If failed to write dataset.
        """
        if not self._writer.is_open:
            self._writer.open(self._paths)
        if not self._writer.get_shard_header():
            self._writer.set_shard_header(self._header)
        if not isinstance(raw_data, list):
            raise ParamTypeError('raw_data', 'list')
        for each_raw in raw_data:
            if not isinstance(each_raw, dict):
                raise ParamTypeError('raw_data item', 'dict')
        self._verify_based_on_schema(raw_data)
        return self._writer.write_raw_data(raw_data, True, parallel_writer)

    def set_header_size(self, header_size):
        """
        Set the size of header which contains shard information, schema information, \
        page meta information, etc. The larger the header, the more training data \
        a single Mindrecord file can store.

        Args:
            header_size (int): Size of header, between 16KB and 128MB.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidHeaderSizeError: If failed to set header size.

        """
        return self._writer.set_header_size(header_size)

    def set_page_size(self, page_size):
        """
        Set the size of page which mainly refers to the block to store training data, \
        and the training data will be split into raw page and blob page in mindrecord. \
        The larger the page, the more training data a single page can store.

        Args:
           page_size (int): Size of page, between 32KB and 256MB.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidPageSizeError: If failed to set page size.
        """
        return self._writer.set_page_size(page_size)

    def commit(self):
        """
        Flush data to disk and generate the corresponding database files.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open MindRecord File.
            MRMSetHeaderError: If failed to set header.
            MRMIndexGeneratorError: If failed to create index generator.
            MRMGenerateIndexError: If failed to write to database.
            MRMCommitError: If failed to flush data to disk.
        """
        if not self._writer.is_open:
            self._writer.open(self._paths)
        # permit commit without data
        if not self._writer.get_shard_header():
            self._writer.set_shard_header(self._header)
        ret = self._writer.commit()
        if self._index_generator is True:
            if self._append:
                self._generator = ShardIndexGenerator(self._file_name, self._append)
            elif len(self._paths) >= 1:
                self._generator = ShardIndexGenerator(os.path.realpath(self._paths[0]), self._append)
            self._generator.build()
            self._generator.write_to_db()

        mindrecord_files = []
        index_files = []
        # change the file mode to 600
        for item in self._paths:
            if os.path.exists(item):
                os.chmod(item, stat.S_IRUSR | stat.S_IWUSR)
                mindrecord_files.append(item)
            index_file = item + ".db"
            if os.path.exists(index_file):
                os.chmod(index_file, stat.S_IRUSR | stat.S_IWUSR)
                index_files.append(index_file)

        logger.info("The list of mindrecord files created are: {}, and the list of index files are: {}".format(
            mindrecord_files, index_files))

        return ret

    def _validate_array(self, k, v):
        """
        Validate array item in schema

        Args:
           k (str): Key in dict.
           v (dict): Sub dict in schema

        Returns:
            bool, whether the array item is valid.
            str, error message.
        """
        if v['type'] not in VALID_ARRAY_ATTRIBUTES:
            error = "Field '{}' contain illegal " \
                    "attribute '{}'.".format(k, v['type'])
            return False, error
        if 'shape' in v:
            if isinstance(v['shape'], list) is False:
                error = "Field '{}' contain illegal " \
                        "attribute '{}'.".format(k, v['shape'])
                return False, error
        else:
            error = "Field '{}' contains illegal attributes.".format(v)
            return False, error
        return True, ''

    def _validate_schema(self, content):
        """
        Validate schema and return validation result and error message.

        Args:
           content (dict): Dict of raw schema.

        Returns:
            bool, whether the schema is valid.
            str, error message.
        """
        error = ''
        if not content:
            error = 'Schema content is empty.'
            return False, error
        if isinstance(content, dict) is False:
            error = 'Schema content should be dict.'
            return False, error
        for k, v in content.items():
            if not re.match(r'^[0-9a-zA-Z\_]+$', k):
                error = "Field '{}' should be composed of " \
                        "'0-9' or 'a-z' or 'A-Z' or '_'.".format(k)
                return False, error
            if v and isinstance(v, dict):
                if len(v) == 1 and 'type' in v:
                    if v['type'] not in VALID_ATTRIBUTES:
                        error = "Field '{}' contain illegal " \
                                "attribute '{}'.".format(k, v['type'])
                        return False, error
                elif len(v) == 2 and 'type' in v:
                    res_1, res_2 = self._validate_array(k, v)
                    if res_1 is not True:
                        return res_1, res_2
                else:
                    error = "Field '{}' contains illegal attributes.".format(v)
                    return False, error
            else:
                error = "Field '{}' should be dict.".format(k)
                return False, error
        return True, error
