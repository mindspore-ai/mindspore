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
# ============================================================================
"""test mindrecord exception"""
import os
import pytest

import numpy as np
from utils import get_data

from mindspore import log as logger
from mindspore.mindrecord import FileWriter, FileReader, MindPage
from mindspore.mindrecord import ParamValueError, MRMGetMetaError

FILES_NUM = 4

def remove_one_file(x):
    if os.path.exists(x):
        os.remove(x)

def remove_file(file_name):
    x = file_name
    remove_one_file(x)
    x = file_name + ".db"
    remove_one_file(x)
    for i in range(FILES_NUM):
        x = file_name + str(i)
        remove_one_file(x)
        x = file_name + str(i) + ".db"
        remove_one_file(x)

def test_cv_file_writer_shard_num_none():
    """test cv file writer when shard num is None."""
    with pytest.raises(Exception, match="Parameter shard_num is None."):
        FileWriter("/tmp/123454321", None)

def test_cv_file_writer_overwrite_int():
    """
    Feature: Overwriting in FileWriter
    Description: invalid parameter
    Expectation: exception occur
    """
    with pytest.raises(Exception, match="Parameter overwrite's type is not bool."):
        FileWriter("/tmp/123454321", 4, 1)

def test_cv_file_writer_shard_num_str():
    """test cv file writer when shard num is string."""
    with pytest.raises(Exception, match="Parameter shard_num's type is not int."):
        FileWriter("/tmp/123454321", "20")

def test_cv_page_reader_consumer_num_none():
    """test cv page reader when consumer number is None."""
    with pytest.raises(Exception, match="Parameter num_consumer is None."):
        MindPage("dummy.mindrecord", None)


def test_cv_page_reader_consumer_num_str():
    """test cv page reader when consumer number is string."""
    with pytest.raises(Exception, match="Parameter num_consumer is not int."):
        MindPage("dummy.mindrecord", "2")


def test_nlp_file_reader_consumer_num_none():
    """test nlp file reader when consumer number is None."""
    with pytest.raises(Exception, match="Parameter num_consumer is None."):
        FileReader("dummy.mindrecord", None)


def test_nlp_file_reader_consumer_num_str():
    """test nlp file reader when consumer number is string."""
    with pytest.raises(Exception, match="Parameter num_consumer is not int."):
        FileReader("dummy.mindrecord", "4")


def create_cv_mindrecord(files_num, file_name):
    writer = FileWriter(file_name, files_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()


def test_lack_partition_and_db():
    """
    Feature: FileReader
    Description: test file reader when mindrecord file does not exist
    Expectation: exception occur
    """
    with pytest.raises(RuntimeError) as err:
        reader = FileReader('dummy.mindrecord')
        reader.close()
    assert "is not exists" in str(err.value)

def test_lack_db():
    """
    Feature: FileReader
    Description: test file reader when db file does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(1, file_name)
    os.remove("{}.db".format(file_name))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name)
        reader.close()
    assert ".db is not exists" in str(err.value)
    remove_file(file_name)

def test_lack_some_partition_and_db():
    """
    Feature: FileReader
    Description: test file reader when some partition and db do not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name + "0")
        reader.close()
    assert "can not be found. Please check whether the mindrecord file exists" \
           " and do not rename the mindrecord file." in str(err.value)
    remove_file(file_name)

def test_lack_some_partition_first():
    """
    Feature: FileReader
    Description: test file reader when first partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name + "0")
        reader.close()
    assert "is not exists" in str(err.value)
    remove_file(file_name)

def test_lack_some_partition_middle():
    """
    Feature: FileReader
    Description: test file reader when some partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[1]))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name + "0")
        reader.close()
    assert "can not be found. Please check whether the mindrecord file exists" \
           " and do not rename the mindrecord file." in str(err.value)
    remove_file(file_name)

def test_lack_some_partition_last():
    """
    Feature: FileReader
    Description: test file reader when last partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name + "0")
        reader.close()
    assert "can not be found. Please check whether the mindrecord file exists" \
           " and do not rename the mindrecord file." in str(err.value)
    remove_file(file_name)

def test_mindpage_lack_some_partition():
    """
    Feature: MindPage
    Description: test page reader when some partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(RuntimeError) as err:
        MindPage(file_name + "0")
    assert "is not exists" in str(err.value)
    remove_file(file_name)

def test_lack_some_db():
    """
    Feature: FileReader
    Description: test file reader when some db does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(RuntimeError) as err:
        reader = FileReader(file_name + "0")
        reader.close()
    assert ".db exists and do not rename the mindrecord file and meta file." in str(err.value)
    remove_file(file_name)

def test_invalid_mindrecord():
    """
    Feature: FileReader
    Description: test file reader when the content of mindrecord is illegal
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(file_name, 'w') as f:
        dummy = 's' * 100
        f.write(dummy)
    with open(file_name + '.db', 'w') as f:
        dummy = 's' * 100
        f.write(dummy)
    with pytest.raises(RuntimeError) as err:
        FileReader(file_name)
    assert "Invalid file, the size of mindrecord file header is larger than the upper limit." in str(err.value)
    remove_file(file_name)
    remove_file(file_name + '.db')

def test_invalid_db():
    """
    Feature: FileReader
    Description: test file reader when the content of db is illegal
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    create_cv_mindrecord(1, file_name)
    os.remove(file_name + ".db")
    with open(file_name + ".db", 'w') as f:
        f.write('just for test')
    with pytest.raises(RuntimeError) as err:
        FileReader(file_name)
    assert "Failed to execute the sql [ SELECT NAME from SHARD_NAME; ] " \
           "while verifying meta file" in str(err.value)
    remove_file(file_name)

def test_overwrite_invalid_mindrecord():
    """
    Feature: FileWriter
    Description: test file writer when overwrite invalid mindreocrd file
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(file_name, 'w') as f:
        f.write('just for test')
    with pytest.raises(RuntimeError) as err:
        create_cv_mindrecord(1, file_name)
    assert 'Invalid file, mindrecord files already exist. Please check file path:' in str(err.value)
    remove_file(file_name)

def test_overwrite_invalid_db():
    """
    Feature: FileWriter
    Description: test file writer when overwrite invalid db file
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(file_name + '.db', 'w') as f:
        f.write('just for test')
    with pytest.raises(RuntimeError) as err:
        create_cv_mindrecord(1, file_name)
    assert 'Invalid file, mindrecord files already exist. Please check file path:' in str(err.value)
    remove_file(file_name)

def test_read_after_close():
    """
    Feature: FileReader
    Description: test file reader when close read
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    create_cv_mindrecord(1, file_name)
    reader = FileReader(file_name)
    reader.close()
    count = 0
    for index, x in enumerate(reader.get_next()):
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 0
    remove_file(file_name)

def test_file_read_after_read():
    """
    Feature: FileReader
    Description: test file reader when finish read
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    create_cv_mindrecord(1, file_name)
    reader = FileReader(file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()
    cnt = 0
    for index, x in enumerate(reader.get_next()):
        cnt = cnt + 1
        logger.info("#item{}: {}".format(index, x))
    assert cnt == 0
    remove_file(file_name)


def test_cv_file_writer_shard_num_greater_than_1000():
    """
    Feature: FileWriter
    Description: test cv file writer shard number greater than 1000
    Expectation: exception occur
    """
    with pytest.raises(ParamValueError) as err:
        FileWriter('dummy.mindrecord', 1001)
    assert "Parameter shard_num's value: 1001 should between 1 and 1000." in str(err.value)


def test_add_index_without_add_schema():
    """
    Feature: FileWriter
    Description: test add index without adding schema
    Expectation: exception occur
    """
    with pytest.raises(MRMGetMetaError) as err:
        fw = FileWriter('dummy.mindrecord')
        fw.add_index(["label"])
    assert 'Failed to get meta info' in str(err.value)

def test_mindpage_pageno_pagesize_not_int():
    """
    Feature: MindPage
    Description: test page reader when some partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    reader = MindPage(file_name + "0")
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "label"

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_id(0, "0", 1)

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_id(0, 0, "b")

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_name("822", "e", 1)

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_name("822", 0, "qwer")

    with pytest.raises(RuntimeError, match=r"Invalid data, "
                       r"category_id: 99999 must be in the range \[0, 10\]."):
        reader.read_at_page_by_id(99999, 0, 1)
    remove_file(file_name)


def test_mindpage_filename_not_exist():
    """
    Feature: FileWrite
    Description: test page reader when some partition does not exist
    Expectation: exception occur
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(4, file_name)
    reader = MindPage(file_name + "0")
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "file_name"

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    with pytest.raises(RuntimeError, match=r"Invalid data, "
                       r"category_id: 9999 must be in the range \[0, 10\]."):
        reader.read_at_page_by_id(9999, 0, 1)

    with pytest.raises(RuntimeError, match="category_name: abc.jpg could not found."):
        reader.read_at_page_by_name("abc.jpg", 0, 1)

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_name(1, 0, 1)

    remove_file(file_name)

def test_invalid_schema():
    """
    Feature: FileWrite
    Description: test invalid schema
    Expectation: exception occur
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    writer = FileWriter(mindrecord_file_name)

    # string  =>  str
    schema = {"file_name": {"type": "str"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # int32  =>  np.int32
    schema = {"file_name": {"type": "string"},
              "label": {"type": "np.int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # float64  =>  np.float64
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "np.float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # int64  =>  int8
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int8", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # int64  =>  uint64
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "uint64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # bytes  =>  byte
    schema = {"file_name": {"type": "strint"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "byte"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # float32  => float3
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float3", "shape": [2, 88]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # string with shape
    schema = {"file_name": {"type": "string", "shape": [-1]},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

    # bytes with shape
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes", "shape": [100]}}
    with pytest.raises(Exception, match="Schema format is error"):
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

def test_write_with_invalid_data():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # field: file_name  =>  filename
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"filename": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"filename": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"filename": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"filename": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"filename": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"filename": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # field: data  =>  image
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "image": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "image": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "image": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "image": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "image": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "image": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # string type with int value
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": 1, "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": 2, "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": 3, "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": 4, "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": 5, "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": 6, "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # field with int64 type, but the real data is string
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": "cat", "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": "dog", "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.6], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": "bird", "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": "mouse", "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": "tiger", "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": "lion", "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # bytes field is string
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": "image bytes abc"},
                {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": "image bytes def"},
                {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": "image bytes ghi"},
                {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": "image bytes jkl"},
                {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": "image bytes mno"},
                {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": "image bytes pqr"}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # field is not numpy type
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": [3, 6, 9],
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": [1, 4, 7],
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": [7, 6, 3],
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": [2, 8, 0],
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": [3, 1, 2],
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": [7, 6, 7],
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # not enough field
    with pytest.raises(RuntimeError, match="There is no valid data which can be written"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
                 "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
                 "data": bytes("image bytes pqr", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "label": {"type": "int32"},
                  "score": {"type": "float64"},
                  "mask": {"type": "int64", "shape": [-1]},
                  "segments": {"type": "float32", "shape": [2, 2]},
                  "data": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

    # more field is ok
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
             "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
             "data": bytes("image bytes abc", encoding='UTF-8'), "test": 0},
            {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
             "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
             "data": bytes("image bytes def", encoding='UTF-8'), "test": 1},
            {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
             "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
             "data": bytes("image bytes ghi", encoding='UTF-8'), "test": 2},
            {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
             "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
             "data": bytes("image bytes jkl", encoding='UTF-8'), "test": 3},
            {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
             "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
             "data": bytes("image bytes mno", encoding='UTF-8'), "test": 4},
            {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
             "segments": np.array([[4.2, 6.3], [8.9, 81.8]], dtype=np.float32),
             "data": bytes("image bytes pqr", encoding='UTF-8'), "test": 5}
            ]
    writer = FileWriter(mindrecord_file_name)
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int64", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    writer.add_schema(schema, "data is so cool")
    writer.write_raw_data(data)
    writer.commit()

    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")
