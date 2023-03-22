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
This module is to write data into mindrecord.
"""
import os
import platform
import queue
import re
import stat
import time
import multiprocessing as mp
import numpy as np
from mindspore import log as logger
from .shardwriter import ShardWriter
from .shardreader import ShardReader
from .shardheader import ShardHeader
from .shardindexgenerator import ShardIndexGenerator
from .shardutils import MIN_SHARD_COUNT, MAX_SHARD_COUNT, VALID_ATTRIBUTES, VALID_ARRAY_ATTRIBUTES, \
    check_filename, VALUE_TYPE_MAP, SUCCESS
from .common.exceptions import ParamValueError, ParamTypeError, MRMInvalidSchemaError, MRMDefineIndexError

__all__ = ['FileWriter']


class FileWriter:
    r"""
    Class to write user defined raw data into MindRecord files.

    Note:
        After the MindRecord file is generated, if the file name is changed,
        the file may fail to be read.

    Args:
        file_name (str): File name of MindRecord file.
        shard_num (int, optional): The Number of MindRecord files.
            It should be between [1, 1000]. Default: 1.
        overwrite (bool, optional): Whether to overwrite if the file already exists. Default: False.

    Raises:
        ParamValueError: If `file_name` or `shard_num` or `overwrite` is invalid.

    Examples:
        >>> from mindspore.mindrecord import FileWriter
        >>> schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
        >>> indexes = ["file_name", "label"]
        >>> data = [{"file_name": "1.jpg", "label": 0,
        ...          "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"},
        ...         {"file_name": "2.jpg", "label": 56,
        ...          "data": b"\xe6\xda\xd1\xae\x07\xb8>\xd4\x00\xf8\x129\x15\xd9\xf2q\xc0\xa2\x91YFUO\x1dsE1"},
        ...         {"file_name": "3.jpg", "label": 99,
        ...          "data": b"\xaf\xafU<\xb8|6\xbd}\xc1\x99[\xeaj+\x8f\x84\xd3\xcc\xa0,i\xbb\xb9-\xcdz\xecp{T\xb1"}]
        >>> writer = FileWriter(file_name="test.mindrecord", shard_num=1, overwrite=True)
        >>> schema_id = writer.add_schema(schema_json, "test_schema")
        >>> status = writer.add_index(indexes)
        >>> status = writer.write_raw_data(data)
        >>> status = writer.commit()
    """

    def __init__(self, file_name, shard_num=1, overwrite=False):
        if platform.system().lower() == "windows":
            file_name = file_name.replace("\\", "/")
        check_filename(file_name)
        self._file_name = file_name

        if shard_num is not None:
            if isinstance(shard_num, int):
                if shard_num < MIN_SHARD_COUNT or shard_num > MAX_SHARD_COUNT:
                    raise ParamValueError("Parameter shard_num's value: {} should between {} and {}."
                                          .format(shard_num, MIN_SHARD_COUNT, MAX_SHARD_COUNT))
            else:
                raise ParamValueError("Parameter shard_num's type is not int.")
        else:
            raise ParamValueError("Parameter shard_num is None.")

        if not isinstance(overwrite, bool):
            raise ParamValueError("Parameter overwrite's type is not bool.")

        self._shard_num = shard_num
        self._index_generator = True
        suffix_shard_size = len(str(self._shard_num - 1))

        if self._shard_num == 1:
            self._paths = [self._file_name]
        else:
            self._paths = ["{}{}".format(self._file_name,
                                         str(x).rjust(suffix_shard_size, '0'))
                           for x in range(self._shard_num)]

        self._overwrite = overwrite
        self._append = False
        self._flush = False
        self._header = ShardHeader()
        self._writer = ShardWriter()
        self._generator = None

        # parallel write mode
        self._parallel_writer = None
        self._writers = None
        self._queue = None
        self._workers = None
        self._index_workers = None

    @classmethod
    def open_for_append(cls, file_name):
        r"""
        Open MindRecord file and get ready to append data.

        Args:
            file_name (str): String of MindRecord file name.

        Returns:
            FileWriter, file writer object for the opened MindRecord file.

        Raises:
            ParamValueError: If file_name is invalid.
            FileNameError: If path contains invalid characters.
            MRMOpenError: If failed to open MindRecord file.
            MRMOpenForAppendError: If failed to open file for appending data.

        Examples:
            >>> from mindspore.mindrecord import FileWriter
            >>> schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
            >>> data = [{"file_name": "1.jpg", "label": 0,
            ...          "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"}]
            >>> writer = FileWriter(file_name="test.mindrecord", shard_num=1, overwrite=True)
            >>> schema_id = writer.add_schema(schema_json, "test_schema")
            >>> status = writer.write_raw_data(data)
            >>> status = writer.commit()
            >>> write_append = FileWriter.open_for_append("test.mindrecord")
            >>> status = write_append.write_raw_data(data)
            >>> status = write_append.commit()
        """
        if platform.system().lower() == "windows":
            file_name = file_name.replace("\\", "/")
        check_filename(file_name)

        # construct ShardHeader
        reader = ShardReader()
        reader.open(file_name, False)
        header = ShardHeader(reader.get_header())
        reader.close()

        instance = cls("append")
        instance.init_append(file_name, header)
        return instance

    def init_append(self, file_name, header):
        self._append = True

        if platform.system().lower() == "windows":
            self._file_name = file_name.replace("\\", "/")
        else:
            self._file_name = file_name

        self._header = header
        self._writer.open_for_append(self._file_name)

    def add_schema(self, content, desc=None):
        """
        The schema is added to describe the raw data to be written.

        Note:
            Please refer to the Examples of class: `mindspore.mindrecord.FileWriter` .

        Args:
            content (dict): Dictionary of schema content.
            desc (str, optional): String of schema description, Default: None.

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
        schema is added through `add_schema` .

        Note:
            The index fields should be primitive type. e.g. int/float/str.
            If the function is not called, the fields of the primitive type
            in schema are set as indexes by default.

            Please refer to the Examples of class: `mindspore.mindrecord.FileWriter` .

        Args:
            index_fields (list[str]): fields from schema.

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

    def open_and_set_header(self):
        """
        Open writer and set header which stores meta information. The function is only used for parallel \
        writing and is called before the `write_raw_data` .

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open MindRecord file.
            MRMSetHeaderError: If failed to set header.
        """
        logger.warning("This interface will be deleted or invisible in the future.")

        if not self._writer.is_open:
            ret = self._writer.open(self._paths, self._overwrite)
        if not self._writer.get_shard_header():
            return self._writer.set_shard_header(self._header)
        return ret

    def write_raw_data(self, raw_data, parallel_writer=False):
        """
        Convert raw data into a series of consecutive MindRecord \
        files after the raw data is verified against the schema.

        Note:
            Please refer to the Examples of class: `mindspore.mindrecord.FileWriter` .

        Args:
           raw_data (list[dict]): List of raw data.
           parallel_writer (bool, optional): Write raw data in parallel if it equals to True. Default: False.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            ParamTypeError: If index field is invalid.
            MRMOpenError: If failed to open MindRecord file.
            MRMValidateDataError: If data does not match blob fields.
            MRMSetHeaderError: If failed to set header.
            MRMWriteDatasetError: If failed to write dataset.
            TypeError: If parallel_writer is not bool.
        """
        if not isinstance(parallel_writer, bool):
            raise TypeError("The parameter `parallel_writer` must be bool.")

        if self._parallel_writer is None:
            self._parallel_writer = parallel_writer
        if self._parallel_writer != parallel_writer:
            raise RuntimeError("The parameter `parallel_writer` must be consistent during use.")
        if not self._parallel_writer:
            if not self._writer.is_open:
                self._writer.open(self._paths, self._overwrite)
            if not self._writer.get_shard_header():
                self._writer.set_shard_header(self._header)
            if not isinstance(raw_data, list):
                raise ParamTypeError('raw_data', 'list')
            if self._flush and not self._append:
                raise RuntimeError("Not allowed to call `write_raw_data` on flushed MindRecord files." \
                                   "When creating new Mindrecord files, please remove `commit` before " \
                                   "`write_raw_data`. In other cases, when appending to existing MindRecord files, " \
                                   "please call `open_for_append` first and then `write_raw_data`.")
            for each_raw in raw_data:
                if not isinstance(each_raw, dict):
                    raise ParamTypeError('raw_data item', 'dict')
            self._verify_based_on_schema(raw_data)
            return self._writer.write_raw_data(raw_data, True, parallel_writer)

        ## parallel write mode
        # init the _writers and launch the workers
        if self._writers is None:
            self._writers = [None] * len(self._paths)  # writers used by worker
            self._queue = mp.Queue(len(self._paths) * 2)  # queue for worker
            self._workers = [None] * len(self._paths)  # worker process
            for i, path in enumerate(self._paths):
                self._writers[i] = ShardWriter()
                self._writers[i].open([path], self._overwrite)
                self._writers[i].set_shard_header(self._header)

                # launch the workers for parallel write
                self._queue._joincancelled = True  # pylint: disable=W0212
                p = mp.Process(target=self._write_worker, args=(i, self._queue))
                p.daemon = True
                p.start()
                logger.info("Start worker process(pid:{}) to parallel write.".format(p.pid))
                self._workers[i] = p

        # fill the self._queue
        check_interval = 0.5  # 0.5s
        start_time = time.time()
        while True:
            try:
                self._queue.put(raw_data, block=False)
            except queue.Full:
                if time.time() - start_time > check_interval:
                    start_time = time.time()
                    logger.warning("Because there are too few MindRecord file shards, the efficiency of parallel " \
                                   "writing is too low. You can stop the current task and add the parameter " \
                                   "`shard_num` of `FileWriter` to upgrade the task.")

                # check the status of worker process
                for i in range(len(self._paths)):
                    if not self._workers[i].is_alive():
                        raise RuntimeError("Worker process(pid:{}) has stopped abnormally. Please check " \
                                           "the above log".format(self._workers[i].pid))
                continue
            return SUCCESS

    def set_header_size(self, header_size):
        """
        Set the size of header which contains shard information, schema information, \
        page meta information, etc. The larger a header, the more data \
        the MindRecord file can store. If the size of header is larger than \
        the default size (16MB), users need to call the API to set a proper size.

        Args:
            header_size (int): Size of header, between 16*1024(16KB) and
                128*1024*1024(128MB).


        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidHeaderSizeError: If failed to set header size.

        Examples:
            >>> from mindspore.mindrecord import FileWriter
            >>> writer = FileWriter(file_name="test.mindrecord", shard_num=1)
            >>> status = writer.set_header_size(1 << 25) # 32MB
        """
        return self._writer.set_header_size(header_size)

    def set_page_size(self, page_size):
        """
        Set the size of page that represents the area where data is stored, \
        and the areas are divided into two types: raw page and blob page. \
        The larger a page, the more data the page can store. If the size of \
        a sample is larger than the default size (32MB), users need to call the API \
        to set a proper size.

        Args:
           page_size (int): Size of page, between 32*1024(32KB) and
               256*1024*1024(256MB).

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMInvalidPageSizeError: If failed to set page size.

        Examples:
            >>> from mindspore.mindrecord import FileWriter
            >>> writer = FileWriter(file_name="test.mindrecord", shard_num=1)
            >>> status = writer.set_page_size(1 << 26)  # 64MB
        """
        return self._writer.set_page_size(page_size)

    def commit(self):  # pylint: disable=W0212
        """
        Flush data in memory to disk and generate the corresponding database files.

        Note:
            Please refer to the Examples of class: `mindspore.mindrecord.FileWriter` .

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open MindRecord file.
            MRMSetHeaderError: If failed to set header.
            MRMIndexGeneratorError: If failed to create index generator.
            MRMGenerateIndexError: If failed to write to database.
            MRMCommitError: If failed to flush data to disk.
            RuntimeError: Parallel write failed.
        """
        if not self._parallel_writer:
            self._flush = True
            if not self._writer.is_open:
                self._writer.open(self._paths, self._overwrite)
            # permit commit without data
            if not self._writer.get_shard_header():
                self._writer.set_shard_header(self._header)
            self._writer.commit()
            if self._index_generator:
                if self._append:
                    self._generator = ShardIndexGenerator(self._file_name, self._append)
                elif len(self._paths) >= 1:
                    self._generator = ShardIndexGenerator(os.path.realpath(self._paths[0]), self._append)
                self._generator.build()
                self._generator.write_to_db()
        else:
            # maybe a empty mindrecord, so need check _writers
            if self._writers is None:
                self._writers = [None] * len(self._paths)
                for i, path in enumerate(self._paths):
                    self._writers[i] = ShardWriter()
                    self._writers[i].open(path, self._overwrite)
                    self._writers[i].set_shard_header(self._header)

            self._parallel_commit()

        # change the file mode to 600
        mindrecord_files = []
        index_files = []
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

        return SUCCESS

    def _index_worker(self, i):
        """The worker do the index generator"""
        generator = ShardIndexGenerator(os.path.realpath(self._paths[i]), False)
        generator.build()
        generator.write_to_db()

    def _parallel_commit(self):
        """Parallel commit"""
        # if some workers stopped, error may occur
        alive_count = 0
        for i in range(len(self._paths)):
            if self._workers[i].is_alive():
                alive_count += 1
        if alive_count != len(self._paths):
            raise RuntimeError("Parallel write worker error, please check the log file.")

        # send EOF to worker process
        for _ in range(len(self._paths)):
            while True:
                try:
                    self._queue.put("EOF", block=False)
                except queue.Full:
                    time.sleep(1)
                    continue
                break

        # wait the worker processing
        while True:
            alive_count = 0
            for i in range(len(self._paths)):
                if self._workers[i].is_alive():
                    alive_count += 1
            if alive_count == 0:
                break
            time.sleep(1)
            logger.info("Waiting for all the parallel workers to finish.")

        del self._queue

        # wait for worker process stop
        for index in range(len(self._paths)):
            while True:
                logger.info("Waiting for the worker process(pid:{}) to process all the data.".format(
                    self._workers[index].pid))
                if self._workers[index].is_alive():
                    time.sleep(1)
                    continue
                elif self._workers[index].exitcode != 0:
                    raise RuntimeError("Worker process(pid:{}) has stopped abnormally. Please check " \
                                       "the above log".format(self._workers[index].pid))
                break

        if self._index_generator:
            # use parallel index workers to generator index
            self._index_workers = [None] * len(self._paths)
            for index in range(len(self._paths)):
                p = mp.Process(target=self._index_worker, args=(index,))
                p.daemon = True
                p.start()
                logger.info("Start worker process(pid:{}) to generate index.".format(p.pid))
                self._index_workers[index] = p

            # wait the index workers stop
            for index in range(len(self._paths)):
                self._index_workers[index].join()

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
                    if not res_1:
                        return res_1, res_2
                else:
                    error = "Field '{}' contains illegal attributes.".format(v)
                    return False, error
            else:
                error = "Field '{}' should be dict.".format(k)
                return False, error
        return True, error

    def _write_worker(self, i, in_queue):
        """The worker do the data check and write to disk for parallel mode"""
        while True:
            # try to get new raw_data from master
            try:
                raw_data = in_queue.get(block=False)
            except queue.Empty:
                continue

            # get EOF from master, worker should commit and stop
            if raw_data == "EOF":
                ret = self._writers[i].commit()
                if ret != SUCCESS:
                    raise RuntimeError("Commit the {}th shard of MindRecord file failed.".format(i))
                break

            # check the raw_data
            if not isinstance(raw_data, list):
                raise ParamTypeError('raw_data', 'list')
            for each_raw in raw_data:
                if not isinstance(each_raw, dict):
                    raise ParamTypeError('raw_data item', 'dict')

            self._verify_based_on_schema(raw_data)
            self._writers[i].write_raw_data(raw_data, True, False)
