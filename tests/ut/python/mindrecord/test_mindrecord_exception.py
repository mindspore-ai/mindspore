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
from mindspore.mindrecord import FileWriter, FileReader, MindPage, SUCCESS
from mindspore.mindrecord import MRMOpenError, MRMGenerateIndexError, ParamValueError, MRMGetMetaError, \
    MRMFetchDataError

CV_FILE_NAME = "./imagenet.mindrecord"
NLP_FILE_NAME = "./aclImdb.mindrecord"
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

@pytest.fixture
def fixture_cv_file():
    """add/remove file"""
    remove_file(CV_FILE_NAME)
    yield "yield_fixture_data"
    remove_file(CV_FILE_NAME)

@pytest.fixture
def fixture_nlp_file():
    """add/remove file"""
    remove_file(NLP_FILE_NAME)
    yield "yield_fixture_data"
    remove_file(NLP_FILE_NAME)

def test_cv_file_writer_shard_num_none():
    """test cv file writer when shard num is None."""
    with pytest.raises(Exception, match="Shard num is illegal."):
        FileWriter("/tmp/123454321", None)


def test_cv_file_writer_shard_num_str():
    """test cv file writer when shard num is string."""
    with pytest.raises(Exception, match="Shard num is illegal."):
        FileWriter("/tmp/123454321", "20")


def test_cv_page_reader_consumer_num_none():
    """test cv page reader when consumer number is None."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        MindPage(CV_FILE_NAME + "0", None)


def test_cv_page_reader_consumer_num_str():
    """test cv page reader when consumer number is string."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        MindPage(CV_FILE_NAME + "0", "2")


def test_nlp_file_reader_consumer_num_none():
    """test nlp file reader when consumer number is None."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        FileReader(NLP_FILE_NAME + "0", None)


def test_nlp_file_reader_consumer_num_str():
    """test nlp file reader when consumer number is string."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        FileReader(NLP_FILE_NAME + "0", "4")


def create_cv_mindrecord(files_num):
    writer = FileWriter(CV_FILE_NAME, files_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()


def test_lack_partition_and_db():
    """test file reader when mindrecord file does not exist."""
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader('dummy.mindrecord')
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_db(fixture_cv_file):
    """test file reader when db file does not exist."""
    create_cv_mindrecord(1)
    os.remove("{}.db".format(CV_FILE_NAME))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME)
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_some_partition_and_db(fixture_cv_file):
    """test file reader when some partition and db do not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_some_partition_first(fixture_cv_file):
    """test file reader when first partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_some_partition_middle(fixture_cv_file):
    """test file reader when some partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[1]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_some_partition_last(fixture_cv_file):
    """test file reader when last partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_mindpage_lack_some_partition(fixture_cv_file):
    """test page reader when some partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(MRMOpenError) as err:
        MindPage(CV_FILE_NAME + "0")
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_lack_some_db(fixture_cv_file):
    """test file reader when some db does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)


def test_invalid_mindrecord():
    """test file reader when the content of mindrecord is illegal."""
    with open(CV_FILE_NAME, 'w') as f:
        dummy = 's' * 100
        f.write(dummy)
    with pytest.raises(MRMOpenError) as err:
        FileReader(CV_FILE_NAME)
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)
    os.remove(CV_FILE_NAME)

def test_invalid_db(fixture_cv_file):
    """test file reader when the content of db is illegal."""
    create_cv_mindrecord(1)
    os.remove("imagenet.mindrecord.db")
    with open('imagenet.mindrecord.db', 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMOpenError) as err:
        FileReader('imagenet.mindrecord')
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_overwrite_invalid_mindrecord(fixture_cv_file):
    """test file writer when overwrite invalid mindreocrd file."""
    with open(CV_FILE_NAME, 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMOpenError) as err:
        create_cv_mindrecord(1)
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.' \
           in str(err.value)

def test_overwrite_invalid_db(fixture_cv_file):
    """test file writer when overwrite invalid db file."""
    with open('imagenet.mindrecord.db', 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMGenerateIndexError) as err:
        create_cv_mindrecord(1)
    assert '[MRMGenerateIndexError]: error_code: 1347690612, ' \
           'error_msg: Failed to generate index.' in str(err.value)

def test_read_after_close(fixture_cv_file):
    """test file reader when close read."""
    create_cv_mindrecord(1)
    reader = FileReader(CV_FILE_NAME)
    reader.close()
    count = 0
    for index, x in enumerate(reader.get_next()):
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 0

def test_file_read_after_read(fixture_cv_file):
    """test file reader when finish read."""
    create_cv_mindrecord(1)
    reader = FileReader(CV_FILE_NAME)
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


def test_cv_file_writer_shard_num_greater_than_1000():
    """test cv file writer shard number greater than 1000."""
    with pytest.raises(ParamValueError) as err:
        FileWriter(CV_FILE_NAME, 1001)
    assert 'Shard number should between' in str(err.value)


def test_add_index_without_add_schema():
    with pytest.raises(MRMGetMetaError) as err:
        fw = FileWriter(CV_FILE_NAME)
        fw.add_index(["label"])
    assert 'Failed to get meta info' in str(err.value)

def test_mindpage_pageno_pagesize_not_int(fixture_cv_file):
    """test page reader when some partition does not exist."""
    create_cv_mindrecord(4)
    reader = MindPage(CV_FILE_NAME + "0")
    fields = reader.get_category_fields()
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("label")
    assert ret == SUCCESS, 'failed on setting category field.'

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

    with pytest.raises(MRMFetchDataError, match="Failed to fetch data by category."):
        reader.read_at_page_by_id(99999, 0, 1)


def test_mindpage_filename_not_exist(fixture_cv_file):
    """test page reader when some partition does not exist."""
    create_cv_mindrecord(4)
    reader = MindPage(CV_FILE_NAME + "0")
    fields = reader.get_category_fields()
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("file_name")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    with pytest.raises(MRMFetchDataError):
        reader.read_at_page_by_id(9999, 0, 1)

    with pytest.raises(MRMFetchDataError):
        reader.read_at_page_by_name("abc.jpg", 0, 1)

    with pytest.raises(ParamValueError):
        reader.read_at_page_by_name(1, 0, 1)

    _ = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
         for x in range(FILES_NUM)]

def test_invalid_schema():
    mindrecord_file_name = "test.mindrecord"
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
    mindrecord_file_name = "test.mindrecord"

    # field: file_name  =>  filename
    with pytest.raises(Exception, match="Failed to write dataset"):
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

    # field: mask  =>  masks
    with pytest.raises(Exception, match="Failed to write dataset"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "masks": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": 91, "score": 5.4, "masks": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": 61, "score": 6.4, "masks": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": 29, "score": 8.1, "masks": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": 78, "score": 7.7, "masks": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": 37, "score": 9.4, "masks": np.array([7, 6, 7], dtype=np.int64),
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
    with pytest.raises(Exception, match="Failed to write dataset"):
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

    # field: label  =>  labels
    with pytest.raises(Exception, match="Failed to write dataset"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "labels": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "labels": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "labels": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "labels": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "labels": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "labels": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
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

    # field: score  =>  scores
    with pytest.raises(Exception, match="Failed to write dataset"):
        remove_one_file(mindrecord_file_name)
        remove_one_file(mindrecord_file_name + ".db")

        data = [{"file_name": "001.jpg", "label": 43, "scores": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
                 "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
                 "data": bytes("image bytes abc", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": 91, "scores": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
                 "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
                 "data": bytes("image bytes def", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": 61, "scores": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
                 "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
                 "data": bytes("image bytes ghi", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": 29, "scores": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
                 "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
                 "data": bytes("image bytes jkl", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": 78, "scores": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
                 "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
                 "data": bytes("image bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": 37, "scores": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
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

    # string type with int value
    with pytest.raises(Exception, match="Failed to write dataset"):
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
    with pytest.raises(Exception, match="Failed to write dataset"):
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
    with pytest.raises(Exception, match="Failed to write dataset"):
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
    with pytest.raises(Exception, match="Failed to write dataset"):
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
    with pytest.raises(Exception, match="Failed to write dataset"):
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
