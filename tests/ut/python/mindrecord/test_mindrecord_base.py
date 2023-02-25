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
# ============================================================================
"""test mindrecord base"""
import os
import uuid
import pytest
import numpy as np
from utils import get_data, get_nlp_data

from mindspore import log as logger
from mindspore.mindrecord import FileWriter, FileReader, MindPage, SUCCESS

FILES_NUM = 4

def remove_one_file(file):
    if os.path.exists(file):
        os.remove(file)


def remove_multi_files(file_name, file_num):
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(file_num)]
    for x in paths:
        remove_one_file("{}".format(x))
        remove_one_file("{}.db".format(x))


def test_write_read_process():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
             "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
             "data": bytes("image bytes abc", encoding='UTF-8')},
            {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
             "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
             "data": bytes("image bytes def", encoding='UTF-8')},
            {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
             "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
             "data": bytes("image bytes ghi", encoding='UTF-8')},
            {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
             "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
             "data": bytes("image bytes jkl", encoding='UTF-8')},
            {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
             "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
             "data": bytes("image bytes mno", encoding='UTF-8')},
            {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
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

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 6
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    remove_one_file("{}".format(mindrecord_file_name))
    remove_one_file("{}.db".format(mindrecord_file_name))


def test_write_read_process_with_define_index_field():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9], dtype=np.int64),
             "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
             "data": bytes("image bytes abc", encoding='UTF-8')},
            {"file_name": "002.jpg", "label": 91, "score": 5.4, "mask": np.array([1, 4, 7], dtype=np.int64),
             "segments": np.array([[5.1, 9.1], [2.0, 65.4]], dtype=np.float32),
             "data": bytes("image bytes def", encoding='UTF-8')},
            {"file_name": "003.jpg", "label": 61, "score": 6.4, "mask": np.array([7, 6, 3], dtype=np.int64),
             "segments": np.array([[0.0, 5.6], [3.0, 16.3]], dtype=np.float32),
             "data": bytes("image bytes ghi", encoding='UTF-8')},
            {"file_name": "004.jpg", "label": 29, "score": 8.1, "mask": np.array([2, 8, 0], dtype=np.int64),
             "segments": np.array([[5.9, 7.2], [4.0, 89.0]], dtype=np.float32),
             "data": bytes("image bytes jkl", encoding='UTF-8')},
            {"file_name": "005.jpg", "label": 78, "score": 7.7, "mask": np.array([3, 1, 2], dtype=np.int64),
             "segments": np.array([[0.6, 8.1], [5.3, 49.3]], dtype=np.float32),
             "data": bytes("image bytes mno", encoding='UTF-8')},
            {"file_name": "006.jpg", "label": 37, "score": 9.4, "mask": np.array([7, 6, 7], dtype=np.int64),
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
    writer.add_index(["label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 6
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    remove_one_file("{}".format(mindrecord_file_name))
    remove_one_file("{}.db".format(mindrecord_file_name))


def test_cv_file_writer_tutorial(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: tutorial for cv dataset writer
    Expectation: generated mindrecord file
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)


def test_cv_file_append_writer():
    """tutorial for cv dataset append writer."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, 4)
    writer = FileWriter(mindrecord_file_name, 4)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data[0:5])
    writer.commit()
    write_append = FileWriter.open_for_append(mindrecord_file_name + "0")
    write_append.write_raw_data(data[5:10])
    write_append.commit()
    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, 4)


def test_cv_file_append_writer_absolute_path():
    """tutorial for cv dataset append writer."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, 4)
    writer = FileWriter(mindrecord_file_name, 4)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data[0:5])
    writer.commit()
    write_append = FileWriter.open_for_append(mindrecord_file_name + "0")
    write_append.write_raw_data(data[5:10])
    write_append.commit()
    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, 4)


def test_cv_file_writer_loop_and_read():
    """tutorial for cv dataset loop writer."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    writer = FileWriter(mindrecord_file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for row in data:
        writer.write_raw_data([row])
    writer.commit()

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_file_reader_tutorial():
    """tutorial for cv file reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_file_reader_file_list():
    """tutorial for cv file partial reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = FileReader([mindrecord_file_name + str(x) for x in range(FILES_NUM)])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_file_reader_partial_tutorial():
    """tutorial for cv file partial reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
        if count == 5:
            reader.close()
    assert count == 5

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_page_reader_tutorial():
    """tutorial for cv page reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = MindPage(mindrecord_file_name + "0")
    fields = reader.get_category_fields()
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("label")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 3
    assert row[0]['label'] == 13

    row1 = reader.read_at_page_by_name("822", 0, 1)
    assert len(row1) == 1
    assert len(row1[0]) == 3
    assert row1[0]['label'] == 822

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_page_reader_tutorial_by_file_name():
    """tutorial for cv page reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = MindPage(mindrecord_file_name + "0")
    fields = reader.get_category_fields()
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("file_name")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 3
    assert row[0]['label'] == 490

    row1 = reader.read_at_page_by_name("image_00007.jpg", 0, 1)
    assert len(row1) == 1
    assert len(row1[0]) == 3
    assert row1[0]['label'] == 13

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_page_reader_tutorial_new_api():
    """tutorial for cv page reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = MindPage(mindrecord_file_name + "0")
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "file_name"

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 3
    assert row[0]['label'] == 490

    row1 = reader.read_at_page_by_name("image_00007.jpg", 0, 1)
    assert len(row1) == 1
    assert len(row1[0]) == 3
    assert row1[0]['label'] == 13

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_nlp_file_writer_tutorial(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: tutorial for nlp file writer
    Expectation: generated mindrecord file
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = list(get_nlp_data("../data/mindrecord/testAclImdbData/pos",
                             "../data/mindrecord/testAclImdbData/vocab.txt",
                             10))
    nlp_schema_json = {"id": {"type": "string"}, "label": {"type": "int32"},
                       "rating": {"type": "float32"},
                       "input_ids": {"type": "int64",
                                     "shape": [1, -1]},
                       "input_mask": {"type": "int64",
                                      "shape": [1, -1]},
                       "segment_ids": {"type": "int64",
                                       "shape": [1, -1]}
                       }
    writer.add_schema(nlp_schema_json, "nlp_schema")
    writer.add_index(["id", "rating"])
    writer.write_raw_data(data)
    writer.commit()
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)


def test_nlp_file_reader_tutorial():
    """tutorial for nlp file reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_nlp_file_writer_tutorial(mindrecord_file_name, remove_file=False)
    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 6
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()
    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_nlp_page_reader_tutorial():
    """tutorial for nlp page reader."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_nlp_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    reader = MindPage(mindrecord_file_name + "0")
    fields = reader.get_category_fields()
    assert fields == ['id', 'rating'], \
        'failed on getting candidate category fields.'

    ret = reader.set_category_field("rating")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 6
    logger.info("row[0]: {}".format(row[0]))

    row1 = reader.read_at_page_by_name("7", 0, 1)
    assert len(row1) == 1
    assert len(row1[0]) == 6
    logger.info("row1[0]: {}".format(row1[0]))

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_cv_file_writer_shard_num_10():
    """test file writer when shard num equals 10."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, 10)
    writer = FileWriter(mindrecord_file_name, 10)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    remove_multi_files(mindrecord_file_name, 10)


def test_cv_file_writer_absolute_path():
    """test cv file writer when file name is absolute path."""
    file_name = "/tmp/" + str(uuid.uuid4())
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    remove_multi_files(file_name, FILES_NUM)


def test_cv_file_writer_without_data():
    """test cv file writer without data."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    writer = FileWriter(mindrecord_file_name, 1)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.commit()
    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 0
    reader.close()
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_cv_file_writer_no_blob():
    """test cv file writer without blob data."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    writer = FileWriter(mindrecord_file_name, 1)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}}
    writer.add_schema(cv_schema_json, "no_blob_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()
    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        count += 1
        assert len(x) == 2
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_cv_file_writer_no_raw():
    """test cv file writer without raw data."""
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    writer = FileWriter(mindrecord_file_name)
    data = list(get_nlp_data("../data/mindrecord/testAclImdbData/pos",
                             "../data/mindrecord/testAclImdbData/vocab.txt",
                             10))
    nlp_schema_json = {"input_ids": {"type": "int64",
                                     "shape": [1, -1]},
                       "input_mask": {"type": "int64",
                                      "shape": [1, -1]},
                       "segment_ids": {"type": "int64",
                                       "shape": [1, -1]}
                       }
    writer.add_schema(nlp_schema_json, "no_raw_schema")
    writer.write_raw_data(data)
    writer.commit()
    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        count += 1
        assert len(x) == 3
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_write_read_process_with_multi_bytes():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"file_name": "001.jpg", "label": 43,
             "image1": bytes("image1 bytes abc", encoding='UTF-8'),
             "image2": bytes("image1 bytes def", encoding='UTF-8'),
             "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image1 bytes mno", encoding='UTF-8')},
            {"file_name": "002.jpg", "label": 91,
             "image1": bytes("image2 bytes abc", encoding='UTF-8'),
             "image2": bytes("image2 bytes def", encoding='UTF-8'),
             "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image2 bytes mno", encoding='UTF-8')},
            {"file_name": "003.jpg", "label": 61,
             "image1": bytes("image3 bytes abc", encoding='UTF-8'),
             "image2": bytes("image3 bytes def", encoding='UTF-8'),
             "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image3 bytes mno", encoding='UTF-8')},
            {"file_name": "004.jpg", "label": 29,
             "image1": bytes("image4 bytes abc", encoding='UTF-8'),
             "image2": bytes("image4 bytes def", encoding='UTF-8'),
             "image3": bytes("image4 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image4 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image4 bytes mno", encoding='UTF-8')},
            {"file_name": "005.jpg", "label": 78,
             "image1": bytes("image5 bytes abc", encoding='UTF-8'),
             "image2": bytes("image5 bytes def", encoding='UTF-8'),
             "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image5 bytes mno", encoding='UTF-8')},
            {"file_name": "006.jpg", "label": 37,
             "image1": bytes("image6 bytes abc", encoding='UTF-8'),
             "image2": bytes("image6 bytes def", encoding='UTF-8'),
             "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image6 bytes mno", encoding='UTF-8')}
            ]
    writer = FileWriter(mindrecord_file_name)
    schema = {"file_name": {"type": "string"},
              "image1": {"type": "bytes"},
              "image2": {"type": "bytes"},
              "image3": {"type": "bytes"},
              "label": {"type": "int32"},
              "image4": {"type": "bytes"},
              "image5": {"type": "bytes"}}
    writer.add_schema(schema, "data is so cool")
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 7
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader2 = FileReader(file_name=mindrecord_file_name, columns=["image1", "image2", "image5"])
    count = 0
    for index, x in enumerate(reader2.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader2.close()

    reader3 = FileReader(file_name=mindrecord_file_name, columns=["image2", "image4"])
    count = 0
    for index, x in enumerate(reader3.get_next()):
        assert len(x) == 2
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader3.close()

    reader4 = FileReader(file_name=mindrecord_file_name, columns=["image5", "image2"])
    count = 0
    for index, x in enumerate(reader4.get_next()):
        assert len(x) == 2
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader4.close()

    reader5 = FileReader(file_name=mindrecord_file_name, columns=["image5", "image2", "label"])
    count = 0
    for index, x in enumerate(reader5.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader5.close()

    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_write_read_process_with_multi_array():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"source_sos_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([13, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)},
            {"source_sos_ids": np.array([11, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([16, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([113, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([119, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([128, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([133, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([139, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([148, 49, 50, 51], dtype=np.int64)},
            {"source_sos_ids": np.array([21, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([26, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([213, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([219, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([228, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([233, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([239, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([248, 49, 50, 51], dtype=np.int64)},
            {"source_sos_ids": np.array([31, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([36, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([313, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([319, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([328, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([333, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([339, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([348, 49, 50, 51], dtype=np.int64)},
            {"source_sos_ids": np.array([41, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([46, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([413, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([419, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([428, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([433, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([439, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([448, 49, 50, 51], dtype=np.int64)},
            {"source_sos_ids": np.array([51, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([56, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "source_eos_ids": np.array([513, 14, 15, 16, 17, 18], dtype=np.int64),
             "source_eos_mask": np.array([519, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
             "target_sos_ids": np.array([528, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([533, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([539, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([548, 49, 50, 51], dtype=np.int64)}
            ]
    writer = FileWriter(mindrecord_file_name)
    schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
              "source_sos_mask": {"type": "int64", "shape": [-1]},
              "source_eos_ids": {"type": "int64", "shape": [-1]},
              "source_eos_mask": {"type": "int64", "shape": [-1]},
              "target_sos_ids": {"type": "int64", "shape": [-1]},
              "target_sos_mask": {"type": "int64", "shape": [-1]},
              "target_eos_ids": {"type": "int64", "shape": [-1]},
              "target_eos_mask": {"type": "int64", "shape": [-1]}}
    writer.add_schema(schema, "data is so cool")
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 8
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["source_eos_ids", "source_eos_mask",
                                                                 "target_sos_ids", "target_sos_mask",
                                                                 "target_eos_ids", "target_eos_mask"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 6
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["source_sos_ids",
                                                                 "target_sos_ids",
                                                                 "target_eos_mask"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["target_eos_mask",
                                                                 "source_eos_mask",
                                                                 "source_sos_mask"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["target_eos_ids"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 1
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_write_read_process_with_multi_bytes_and_array():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    data = [{"file_name": "001.jpg", "label": 4,
             "image1": bytes("image1 bytes abc", encoding='UTF-8'),
             "image2": bytes("image1 bytes def", encoding='UTF-8'),
             "source_sos_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image1 bytes mno", encoding='UTF-8'),
             "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)},
            {"file_name": "002.jpg", "label": 5,
             "image1": bytes("image2 bytes abc", encoding='UTF-8'),
             "image2": bytes("image2 bytes def", encoding='UTF-8'),
             "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image2 bytes mno", encoding='UTF-8'),
             "source_sos_ids": np.array([11, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([16, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "target_sos_ids": np.array([128, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([133, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([139, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([148, 49, 50, 51], dtype=np.int64)},
            {"file_name": "003.jpg", "label": 6,
             "source_sos_ids": np.array([21, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([26, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "target_sos_ids": np.array([228, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([233, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([239, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "image1": bytes("image3 bytes abc", encoding='UTF-8'),
             "image2": bytes("image3 bytes def", encoding='UTF-8'),
             "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image3 bytes mno", encoding='UTF-8'),
             "target_eos_mask": np.array([248, 49, 50, 51], dtype=np.int64)},
            {"file_name": "004.jpg", "label": 7,
             "source_sos_ids": np.array([31, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([36, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image4 bytes abc", encoding='UTF-8'),
             "image2": bytes("image4 bytes def", encoding='UTF-8'),
             "image3": bytes("image4 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image4 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image4 bytes mno", encoding='UTF-8'),
             "target_sos_ids": np.array([328, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([333, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([339, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([348, 49, 50, 51], dtype=np.int64)},
            {"file_name": "005.jpg", "label": 8,
             "source_sos_ids": np.array([41, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([46, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "target_sos_ids": np.array([428, 29, 30, 31, 32], dtype=np.int64),
             "target_sos_mask": np.array([433, 34, 35, 36, 37, 38], dtype=np.int64),
             "image1": bytes("image5 bytes abc", encoding='UTF-8'),
             "image2": bytes("image5 bytes def", encoding='UTF-8'),
             "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image5 bytes mno", encoding='UTF-8'),
             "target_eos_ids": np.array([439, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([448, 49, 50, 51], dtype=np.int64)},
            {"file_name": "006.jpg", "label": 9,
             "source_sos_ids": np.array([51, 2, 3, 4, 5], dtype=np.int64),
             "source_sos_mask": np.array([56, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "target_sos_ids": np.array([528, 29, 30, 31, 32], dtype=np.int64),
             "image1": bytes("image6 bytes abc", encoding='UTF-8'),
             "image2": bytes("image6 bytes def", encoding='UTF-8'),
             "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image6 bytes mno", encoding='UTF-8'),
             "target_sos_mask": np.array([533, 34, 35, 36, 37, 38], dtype=np.int64),
             "target_eos_ids": np.array([539, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
             "target_eos_mask": np.array([548, 49, 50, 51], dtype=np.int64)}
            ]

    writer = FileWriter(mindrecord_file_name)
    schema = {"file_name": {"type": "string"},
              "image1": {"type": "bytes"},
              "image2": {"type": "bytes"},
              "source_sos_ids": {"type": "int64", "shape": [-1]},
              "source_sos_mask": {"type": "int64", "shape": [-1]},
              "image3": {"type": "bytes"},
              "image4": {"type": "bytes"},
              "image5": {"type": "bytes"},
              "target_sos_ids": {"type": "int64", "shape": [-1]},
              "target_sos_mask": {"type": "int64", "shape": [-1]},
              "target_eos_ids": {"type": "int64", "shape": [-1]},
              "target_eos_mask": {"type": "int64", "shape": [-1]},
              "label": {"type": "int32"}}
    writer.add_schema(schema, "data is so cool")
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 13
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["source_sos_ids", "source_sos_mask",
                                                                 "target_sos_ids"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["image2", "source_sos_mask",
                                                                 "image3", "target_sos_ids"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 4
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["target_sos_ids", "image4",
                                                                 "source_sos_ids"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["target_sos_ids", "image5",
                                                                 "image4", "image3", "source_sos_ids"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 5
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    reader = FileReader(file_name=mindrecord_file_name, columns=["target_eos_mask", "image5", "image2",
                                                                 "source_sos_mask", "label"])
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 5
        for field in x:
            if isinstance(x[field], np.ndarray):
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 6
    reader.close()

    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")


def test_write_read_process_without_ndarray_type():
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

    # field: mask derivation type is int64, but schema type is int32
    data = [{"file_name": "001.jpg", "label": 43, "score": 0.8, "mask": np.array([3, 6, 9]),
             "segments": np.array([[5.0, 1.6], [65.2, 8.3]], dtype=np.float32),
             "data": bytes("image bytes abc", encoding='UTF-8')}
            ]
    writer = FileWriter(mindrecord_file_name)
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"},
              "score": {"type": "float64"},
              "mask": {"type": "int32", "shape": [-1]},
              "segments": {"type": "float32", "shape": [2, 2]},
              "data": {"type": "bytes"}}
    writer.add_schema(schema, "data is so cool")
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 6
        for field in x:
            if isinstance(x[field], np.ndarray):
                print("output: {}, input: {}".format(x[field], data[count][field]))
                assert (x[field] == data[count][field]).all()
            else:
                assert x[field] == data[count][field]
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 1
    reader.close()

    remove_one_file(mindrecord_file_name)
    remove_one_file(mindrecord_file_name + ".db")

def test_cv_file_overwrite_01():
    """
    Feature: Overwriting in FileWriter
    Description: full mindrecord files exist
    Expectation: generated new mindrecord files
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    writer = FileWriter(mindrecord_file_name, FILES_NUM, True)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)

def test_cv_file_overwrite_02():
    """
    Feature: Overwriting in FileWriter
    Description: lack 1 mindrecord file
    Expectation: generated new mindrecord files
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)
    # remove 1 mindrecord file
    os.remove(mindrecord_file_name + "0")

    writer = FileWriter(mindrecord_file_name, FILES_NUM, True)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)

def test_cv_file_overwrite_03():
    """
    Feature: Overwriting in FileWriter
    Description: lack 1 db file
    Expectation: generated new mindrecord files
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)
    # remove 1 db file
    os.remove(mindrecord_file_name + "0" + ".db")

    writer = FileWriter(mindrecord_file_name, FILES_NUM, True)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)

def test_cv_file_overwrite_04():
    """
    Feature: Overwriting in FileWriter
    Description: lack 1 db file and mindrecord file
    Expectation: generated new mindrecord files
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, FILES_NUM)
    test_cv_file_writer_tutorial(mindrecord_file_name, remove_file=False)

    os.remove(mindrecord_file_name + "0")
    os.remove(mindrecord_file_name + "0" + ".db")

    writer = FileWriter(mindrecord_file_name, FILES_NUM, True)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(mindrecord_file_name + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()

    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_mindrecord_commit_exception_01():
    """
    Feature: commit excepiton
    Description: write_raw_data after commit
    Expectation: exception occur
    """

    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_multi_files(mindrecord_file_name, 4)

    with pytest.raises(RuntimeError) as err:
        writer = FileWriter(mindrecord_file_name, 4)
        data = get_data("../data/mindrecord/testImageNetData/")
        cv_schema_json = {"file_name": {"type": "string"},
                          "label": {"type": "int64"}, "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.write_raw_data(data[0:5])
        writer.commit()
        writer.write_raw_data(data[5:10])

    assert 'Not allowed to call `write_raw_data` on flushed MindRecord files.' in str(err.value)
    remove_multi_files(mindrecord_file_name, 4)


def test_cv_file_overwrite_exception_01():
    """
    Feature: Overwriting in FileWriter
    Description: default write mode, detect mindrecord file
    Expectation: exception occur
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(mindrecord_file_name + "0", 'w'):
        pass
    with pytest.raises(RuntimeError) as err:
        writer = FileWriter(mindrecord_file_name, FILES_NUM)
        data = get_data("../data/mindrecord/testImageNetData/")
        cv_schema_json = {"file_name": {"type": "string"},
                          "label": {"type": "int64"}, "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.write_raw_data(data)
    assert 'Invalid file, mindrecord files already exist. Please check file path:' in str(err.value)
    remove_multi_files(mindrecord_file_name, FILES_NUM)

def test_cv_file_overwrite_exception_02():
    """
    Feature: Overwriting in FileWriter
    Description: default write mode, detect db file
    Expectation: exception occur
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(mindrecord_file_name + "0" + ".db", 'w'):
        pass
    with pytest.raises(RuntimeError) as err:
        writer = FileWriter(mindrecord_file_name, FILES_NUM)
        data = get_data("../data/mindrecord/testImageNetData/")
        cv_schema_json = {"file_name": {"type": "string"},
                          "label": {"type": "int64"}, "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.write_raw_data(data)
    assert 'Invalid file, mindrecord files already exist. Please check file path:' in str(err.value)
    remove_multi_files(mindrecord_file_name, FILES_NUM)


def test_file_writer_parallel(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: parallel for writer
    Expectation: generated mindrecord file
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # single file
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")
    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()
    if remove_file:
        remove_one_file(file_name)
        remove_one_file(file_name + ".db")

    # write_raw_data with empty
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")
    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    with pytest.raises(RuntimeError):
        writer.write_raw_data([])

    # multi files
    # len(data) > FILES_NUM which is parallel size
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(10):
        writer.write_raw_data(data, True)
    writer.commit()
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)

    # len(data) < FILES_NUM which is parallel size
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(2):
        writer.write_raw_data(data[0:2], True)
    writer.commit()
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)

    # write_raw_data(.., True) and write_raw_data(.., False)
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    with pytest.raises(RuntimeError):
        writer.write_raw_data(data[0:2], True)
        writer.write_raw_data(data[0:2])

    # without write_raw_data
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.commit()
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)

    # write_raw_data with empty
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    with pytest.raises(RuntimeError):
        writer.write_raw_data([], True)
        writer.commit()
