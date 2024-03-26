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
from mindspore.mindrecord import FileWriter, FileReader, MindPage
from mindspore.mindrecord import set_enc_key, set_enc_mode, set_dec_mode, set_hash_mode

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
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "label"

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
    fields = reader.candidate_fields
    assert fields == ['id', 'rating'], \
        'failed on getting candidate category fields.'

    reader.category_field = "rating"

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    row = reader.read_at_page_by_id(0, 0, 1)
    assert len(row) == 1
    assert len(row[0]) == 6
    logger.info("row[0]: {}".format(row[0]))

    row1 = reader.read_at_page_by_name("7.0", 0, 1)
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
    with pytest.raises(RuntimeError) as err:
        writer.write_raw_data(data)
    assert "There is no valid data which can be written by 'write_raw_data' to mindrecord file." in str(err.value)

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


def test_file_writer_schema_len(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: writer for schema and len
    Expectation: SUCCESS
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    ## single mindrecord file
    # 1. empty file
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")
    writer = FileWriter(file_name, 1)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.commit()

    # get the schema & len
    reader = FileReader(file_name)
    assert cv_schema_json == reader.schema()
    assert reader.len() == 0
    if remove_file:
        remove_one_file(file_name)
        remove_one_file(file_name + ".db")

    # 2. with 10 samples
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")
    writer = FileWriter(file_name, 1)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    data = get_data("../data/mindrecord/testImageNetData/")
    writer.add_schema(cv_schema_json, "img_schema")
    writer.write_raw_data(data)
    writer.commit()

    # get the schema & len
    reader = FileReader(file_name)
    assert cv_schema_json == reader.schema()
    assert reader.len() == 10
    if remove_file:
        remove_one_file(file_name)
        remove_one_file(file_name + ".db")

    ## multi mindrecord file
    # 1. empty file
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.commit()

    # get the schema & len
    reader = FileReader(file_name + "0")
    assert cv_schema_json == reader.schema()
    assert reader.len() == 0
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)

    # 2. with samples
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    data = get_data("../data/mindrecord/testImageNetData/")
    writer.add_schema(cv_schema_json, "img_schema")
    writer.write_raw_data(data)
    writer.commit()

    # get the schema & len
    reader = FileReader(file_name + "0")
    assert cv_schema_json == reader.schema()
    assert reader.len() == 10
    if remove_file:
        remove_multi_files(file_name, FILES_NUM)


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
    reader = FileReader(file_name)
    assert reader.len() == 50
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
    # len(data) > 2 which is parallel size
    remove_multi_files(file_name, 2)
    writer = FileWriter(file_name, 2)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(10):
        writer.write_raw_data(data, True)
    writer.commit()
    reader = FileReader([file_name + '0',
                         file_name + '1'])
    assert reader.len() == 100
    if remove_file:
        remove_multi_files(file_name, 2)

    # len(data) < 2 which is parallel size
    remove_multi_files(file_name, 2)
    writer = FileWriter(file_name, 2)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(2):
        writer.write_raw_data(data[0:2], True)
    writer.commit()
    reader = FileReader([file_name + '0',
                         file_name + '1'])
    assert reader.len() == 4
    if remove_file:
        remove_multi_files(file_name, 2)

    # write_raw_data(.., True) and write_raw_data(.., False)
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    with pytest.raises(RuntimeError) as e:
        writer.write_raw_data(data[0:2], True)
        writer.write_raw_data(data[0:2])
    assert "The parameter `parallel_writer` must be consistent during use" in str(e)

    # without write_raw_data
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.commit()
    reader = FileReader([file_name + '0',
                         file_name + '1',
                         file_name + '2',
                         file_name + '3'])
    assert reader.len() == 0
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

    # write_raw_data parameter parallel_writers is not bool
    remove_multi_files(file_name, FILES_NUM)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    with pytest.raises(TypeError) as e:
        writer.write_raw_data([], 18)
    assert "The parameter `parallel_writer` must be bool." in str(e)


def test_mindpage_with_single_multi_bytes_fields(file_name=None, remove_file=True):
    """
    Feature: MindPage
    Description: search and check value
    Expectation: SUCCESS
    """

    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # single bytes field
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        data = [{"file_name": str(i) + ".jpg", "label": i,
                 "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" +
                         int(i).to_bytes(length=4, byteorder='big', signed=True)}]
        writer.write_raw_data(data)
    writer.commit()

    mind_page = MindPage(file_name)

    # get all the index fields
    fields = mind_page.candidate_fields
    print(fields)

    # set the field to be retrieved
    mind_page.category_field = "file_name"

    # get all the group info
    info = mind_page.read_category_info()
    print(info)

    # get the row by id which is from category info
    row_by_id = mind_page.read_at_page_by_id(0, 0, 1)
    assert len(row_by_id) == 1
    assert row_by_id[0]['file_name'] == '0.jpg'
    assert row_by_id[0]['label'] == 0
    assert row_by_id[0]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                int(0).to_bytes(length=4, byteorder='big', signed=True)

    # get the row by name which is from category info
    row_by_name = mind_page.read_at_page_by_name("8.jpg", 0, 1)
    assert len(row_by_name) == 1
    assert row_by_name[0]['file_name'] == '8.jpg'
    assert row_by_name[0]['label'] == 8
    assert row_by_name[0]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(8).to_bytes(length=4, byteorder='big', signed=True)

    # single bytes field and got >1 result
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        data = [{"file_name": str(i % 4) + ".jpg", "label": i,
                 "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" +
                         int(i).to_bytes(length=4, byteorder='big', signed=True)}]
        writer.write_raw_data(data)
    writer.commit()

    mind_page = MindPage(file_name)

    # get all the index fields
    fields = mind_page.candidate_fields

    # set the field to be retrieved
    mind_page.category_field = "file_name"

    # get all the group info
    info = mind_page.read_category_info()
    print(info)

    # get the row by name which is from category info
    row_by_name = mind_page.read_at_page_by_name("1.jpg", 0, 5)
    assert len(row_by_name) == 3
    assert row_by_name[0]['file_name'] == '1.jpg'
    assert row_by_name[0]['label'] == 1
    assert row_by_name[0]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(1).to_bytes(length=4, byteorder='big', signed=True)
    assert row_by_name[1]['file_name'] == '1.jpg'
    assert row_by_name[1]['label'] == 5
    assert row_by_name[1]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(5).to_bytes(length=4, byteorder='big', signed=True)
    assert row_by_name[2]['file_name'] == '1.jpg'
    assert row_by_name[2]['label'] == 9
    assert row_by_name[2]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(9).to_bytes(length=4, byteorder='big', signed=True)

    # multi ndarray and got >1 result
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"},
                   "amask": {"type": "int32", "shape": [10, 15]}, "mask": {"type": "int32", "shape": [15, 15]}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        mask = np.ones([10, 15], dtype=np.int32)
        mask[i][i] = i
        mask2 = np.ones([15, 15], dtype=np.int32)
        mask2[i][i] = i * 10000000
        data = [{"file_name": str(i % 4) + ".jpg", "label": i,
                 "amask": mask, "mask": mask2}]
        writer.write_raw_data(data)
    writer.commit()

    mind_page = MindPage(file_name)

    # get all the index fields
    fields = mind_page.candidate_fields

    # set the field to be retrieved
    mind_page.category_field = "file_name"

    # get all the group info
    info = mind_page.read_category_info()
    print(info)

    # get the row by name which is from category info
    row_by_name = mind_page.read_at_page_by_name("2.jpg", 0, 5)
    assert len(row_by_name) == 2
    assert row_by_name[0]['file_name'] == '2.jpg'
    assert row_by_name[0]['label'] == 2
    mask = np.ones([10, 15], dtype=np.int32)
    mask[2][2] = 2
    assert (row_by_name[0]['amask'] == mask).all()
    mask2 = np.ones([15, 15], dtype=np.int32)
    mask2[2][2] = 2 * 10000000
    assert (row_by_name[0]['mask'] == mask2).all()

    assert row_by_name[1]['file_name'] == '2.jpg'
    assert row_by_name[1]['label'] == 6
    mask = np.ones([10, 15], dtype=np.int32)
    mask[6][6] = 6
    assert (row_by_name[1]['amask'] == mask).all()
    mask2 = np.ones([15, 15], dtype=np.int32)
    mask2[6][6] = 6 * 10000000
    assert (row_by_name[1]['mask'] == mask2).all()

    # multi bytes field and ndarray and got >1 result
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"},
                   "mask": {"type": "int32", "shape": [10, 15]}, "a_data": {"type": "bytes"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        mask = np.ones([10, 15], dtype=np.int32)
        mask[i][i] = i * 999999
        data = [{"file_name": str(i % 4) + ".jpg", "label": i,
                 "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" +
                         int(i).to_bytes(length=4, byteorder='big', signed=True),
                 "mask": mask, "a_data": b"\x10c\xb3w\xa8" + int(i).to_bytes(length=4, byteorder='big', signed=True)}]
        writer.write_raw_data(data)
    writer.commit()

    mind_page = MindPage(file_name)

    # get all the index fields
    fields = mind_page.candidate_fields

    # set the field to be retrieved
    mind_page.category_field = "file_name"

    # get all the group info
    info = mind_page.read_category_info()
    print(info)

    # get the row by name which is from category info
    row_by_name = mind_page.read_at_page_by_name("2.jpg", 0, 5)
    assert len(row_by_name) == 2
    assert row_by_name[0]['file_name'] == '2.jpg'
    assert row_by_name[0]['label'] == 2
    assert row_by_name[0]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(2).to_bytes(length=4, byteorder='big', signed=True)
    mask = np.ones([10, 15], dtype=np.int32)
    mask[2][2] = 2 * 999999
    assert (row_by_name[0]['mask'] == mask).all()
    assert row_by_name[0]['a_data'] == b"\x10c\xb3w\xa8" + int(2).to_bytes(length=4, byteorder='big', signed=True)

    assert row_by_name[1]['file_name'] == '2.jpg'
    assert row_by_name[1]['label'] == 6
    assert row_by_name[1]['data'] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                                  int(6).to_bytes(length=4, byteorder='big', signed=True)
    mask = np.ones([10, 15], dtype=np.int32)
    mask[6][6] = 6 * 999999
    assert (row_by_name[1]['mask'] == mask).all()
    assert row_by_name[1]['a_data'] == b"\x10c\xb3w\xa8" + int(6).to_bytes(length=4, byteorder='big', signed=True)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")


def test_filereader_and_check_result(file_name=None, remove_file=True):
    """
    Feature: FileReader
    Description: read and check value
    Expectation: SUCCESS
    """

    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"},
                   "a_mask": {"type": "int32", "shape": [10, 15]}, "a_data": {"type": "bytes"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    writer.add_index(indexes)
    for i in range(10):
        mask = np.ones([10, 15], dtype=np.int32)
        mask[i][i] = i * 999999
        data = [{"file_name": str(i) + ".jpg", "label": i,
                 "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" +
                         int(i).to_bytes(length=4, byteorder='big', signed=True),
                 "a_mask": mask, "a_data": b"\x10c\xb3w\xa8" + int(i).to_bytes(length=4, byteorder='big', signed=True)}]
        writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(file_name)
    count = 0
    for i, x in enumerate(reader.get_next()):
        assert len(x) == 5
        assert x['file_name'] == str(i) + ".jpg"
        assert x["label"] == i
        assert x["data"] == b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff" + \
                            int(i).to_bytes(length=4, byteorder='big', signed=True)
        mask = np.ones([10, 15], dtype=np.int32)
        mask[i][i] = i * 999999
        assert (x["a_mask"] == mask).all()
        assert x["a_data"] == b"\x10c\xb3w\xa8" + int(i).to_bytes(length=4, byteorder='big', signed=True)
        count = count + 1
    assert count == 10
    reader.close()

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")


def file_writer_encode_and_integrity_check(file_name=None, remove_file=True, encode=None, enc_mode=None,
                                           hash_mode=None, dec_mode=None):
    """
    Feature: FileWriter
    Description: writer for encode or integrity check
    Expectation: SUCCESS
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # mindrecord file without encode and hash to be compared with
    file_name_no_encode_no_hash = file_name + "_no_encode_no_hash"
    remove_one_file(file_name_no_encode_no_hash)
    remove_one_file(file_name_no_encode_no_hash + ".db")
    writer = FileWriter(file_name_no_encode_no_hash)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()
    reader = FileReader(file_name_no_encode_no_hash)
    assert reader.len() == 50

    set_enc_key(encode)
    set_enc_mode(enc_mode)
    set_hash_mode(hash_mode)

    # single file with encode and hash
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

    if encode is not None or hash_mode is not None:
        origin_size = os.path.getsize(file_name)
        new_size = os.path.getsize(file_name_no_encode_no_hash)
        if origin_size == new_size:
            raise RuntimeError("Encode and hash file is same as origin file.")

    if dec_mode is not None:
        set_enc_key(encode)
        set_dec_mode(dec_mode)

    # FileReader open the file
    reader = FileReader(file_name)
    assert reader.len() == 50

    # check the .decrypt_mindrecord dir permission
    if encode is not None:
        real_path_filename = os.path.realpath(file_name)
        parent_dir = os.path.dirname(real_path_filename)
        permission = os.popen("ls -a -l " + parent_dir +
                              " | grep \".decrypt_mindrecord\" | awk -F \" \" '{print $1;}'").readline()
        assert permission.startswith("drwx------")

    # MindPage open the file
    reader = MindPage(file_name)
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "label"

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

    # open for append
    write_append = FileWriter.open_for_append(file_name)
    write_append.write_raw_data(data[5:10])
    write_append.commit()

    # test read
    reader = FileReader(file_name)
    assert reader.len() == 55

    # test mindpage
    reader = MindPage(file_name)
    fields = reader.candidate_fields
    assert fields == ['file_name', 'label'], \
        'failed on getting candidate category fields.'

    reader.category_field = "label"

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

    set_enc_key(None)
    set_enc_mode()
    set_dec_mode(None)
    set_hash_mode(None)

    # remove the enc & hashed file
    if remove_file:
        remove_one_file(file_name)
        remove_one_file(file_name + ".db")

    # remove the origin file
    if remove_file:
        remove_one_file(file_name_no_encode_no_hash)
        remove_one_file(file_name_no_encode_no_hash + ".db")


def test_file_writer_encode_integrity_check(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: writer for encode and integrity check
    Expectation: SUCCESS
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    file_writer_encode_and_integrity_check(file_name, True, "0123456789012345", "AES-GCM", "sha3_512")
    file_writer_encode_and_integrity_check(file_name, True, "0123456789012345", "AES-CBC", None)
    file_writer_encode_and_integrity_check(file_name, True, None, "SM4-CBC", "sha3_384")
    file_writer_encode_and_integrity_check(file_name, True, None, "SM4-CBC", None)

    def encrypt(f_in, file_size, f_out, key):
        offset = 64 * 1024 * 1024    ## read the offset 64M
        current_offset = 0           ## use this to seek file

        while True:
            if file_size - current_offset >= offset:
                read_size = offset
            elif file_size - current_offset > 0:
                read_size = file_size - current_offset
            else:
                # have read the entire file
                break

            if f_in.seek(current_offset) == -1:
                raise RuntimeError("Seek the file failed.")

            data = f_in.read(read_size)

            # encrypt the data
            data_int = int.from_bytes(data, 'big')
            key_int = int.from_bytes(key.encode('utf-8'), 'big')

            data_out = (data_int ^ key_int).to_bytes(len(data), 'big')

            # write to the file
            f_out.write(int(len(data_out)).to_bytes(length=4, byteorder='big', signed=True))
            f_out.write(data_out)

            current_offset += read_size

    def decrypt(f_in, file_size, f_out, key):
        current_offset = 0           ## use this to seek file
        length = int().from_bytes(f_in.read(4), byteorder='big', signed=True)
        while length != 0:
            # current_offset is the encrypted data
            current_offset += 4
            if f_in.seek(current_offset) == -1:
                f_in.close()
                raise RuntimeError("Seek the file: {} to position: {} failed.".format(filename, current_offset))

            data = f_in.read(length)
            data_int = int.from_bytes(data, 'big')
            key_int = int.from_bytes(key.encode('utf-8'), 'big')
            data_out = (data_int ^ key_int).to_bytes(len(data), 'big')

            # write to decrypt file
            f_out.write(data_out)

            # current_offset is the length of next encrypted data block
            current_offset += length
            if f_in.seek(current_offset) == -1:
                f_in.close()
                raise RuntimeError("Seek the file: {} to position: {} failed.".format(filename, current_offset))

            length = int().from_bytes(f_in.read(4), byteorder='big', signed=True)

    def udf_hash(data, pre_hash):
        cur_hash = hash(data + pre_hash)
        return str(cur_hash).encode('utf-8')

    file_writer_encode_and_integrity_check(file_name, True, "0123456789012345", encrypt, None, decrypt)
    file_writer_encode_and_integrity_check(file_name, True, None, "AES-CBC", udf_hash, None)
    file_writer_encode_and_integrity_check(file_name, True, "0123456780abcdef", encrypt, udf_hash, decrypt)


def test_file_writer_encode_integrity_check_with_exception(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: writer for encode and integrity check with exception
    Expectation: SUCCESS
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    ## 1. create with encode and hash check
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_384")

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()

    ## 1.1 read with only encode
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    ## 1.2 read with only hash
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_384")

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    ## 1.3 read without encode and hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    ## 2. create with encode, without hash check
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()

    ## 2.1 read with encode and hash check
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_256")

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not hashed." in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not hashed." in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not hashed." in str(err.value)

    ## 2.2 read without encode, with hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_384")

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    ## 2.3 read without encode, without hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is encrypted. You need to configure" in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    ## 3. create without encode, with hash check
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_512")

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()

    ## 3.1 read with encode, with hash check
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not encrypted." in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not encrypted." in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not encrypted." in str(err.value)

    ## 3.2 read with encode, without hash check
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    ## 3.3 read without encode and hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is hashed. You need to configure" in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    ## 4. create without encode, without hash check
    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data, True)
    writer.commit()

    ## 4.1 read with encode, with hash check
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode("sha512")

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    ## 4.2 read with encode, without hash check
    set_enc_key("abcdefghijklmnop")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not encrypted. You can set" in str(err.value)

    ## 4.3 read without encode, with hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode("sha384")

    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "The mindrecord file is not hashed. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # MindPage open the file
        reader = MindPage(file_name)
        fields = reader.candidate_fields
        assert fields == ['file_name', 'label'], \
            'failed on getting candidate category fields.'
    assert "The mindrecord file is not hashed. You can set" in str(err.value)

    with pytest.raises(RuntimeError) as err:
        # open for append
        write_append = FileWriter.open_for_append(file_name)
        write_append.write_raw_data(data[5:10])
        write_append.commit()
    assert "The mindrecord file is not hashed. You can set" in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key(None)
    set_enc_mode()
    set_dec_mode(None)
    set_hash_mode(None)


def test_file_writer_encode_integrity_check_with_exception_invalid_key(file_name=None, remove_file=True):
    """
    Feature: FileWriter
    Description: writer for encode and integrity check with exception invalid key
    Expectation: SUCCESS
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key("zxcvasdfqwerbnm,")
    set_enc_mode("AES-CBC")
    set_hash_mode("sha256")

    # writer with auto shard
    with pytest.raises(RuntimeError) as err:
        writer = FileWriter(file_name, 2)
    assert "When encode mode or hash check is enabled, the automatic sharding function is unavailable." \
        in str(err.value)

    set_enc_key("zxcvasdfqwerbnm,")
    set_enc_mode("AES-CBC")
    set_hash_mode(None)

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data)
    writer.commit()

    ## read with other encode key
    set_enc_key("zxcvabdfqwerbnm,")
    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "Failed to decrypt data, please check if enc_key and enc_mode / dec_mode is valid." in str(err.value)

    ## read with other encode mode
    set_enc_key("zxcvasdfqwerbnm,")
    set_enc_mode("AES-GCM")
    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert "Failed to decrypt data, please check if enc_key and enc_mode / dec_mode is valid." in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    ## enable hash check
    set_enc_key(None)
    set_enc_mode("AES-CBC")
    set_hash_mode("sha3_256")

    writer = FileWriter(file_name)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    for _ in range(5):
        writer.write_raw_data(data)
    writer.commit()

    ## read with other hash mode
    set_hash_mode("sha256")
    with pytest.raises(RuntimeError) as err:
        # FileReader open the file
        reader = FileReader(file_name)
        assert reader.len() == 50
    assert " hash check fail. " in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    set_enc_key(None)
    set_enc_mode()
    set_dec_mode(None)
    set_hash_mode(None)


def create_empty_file(file_name):
    """Create empty file"""
    remove_one_file(file_name)

    f = open(file_name, 'w')
    f.close()
    assert os.path.exists(file_name)


def test_read_empty_file(file_name=None, remove_file=True):
    """
    Feature: FileReader
    Description: Read empty file
    Expectation: With exception
    """
    if not file_name:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    # single file
    create_empty_file(file_name)
    create_empty_file(file_name + ".db")

    with pytest.raises(RuntimeError) as err:
        _ = FileReader(file_name)
    assert "Invalid file, the size of mindrecord file: " in str(err.value)

    remove_one_file(file_name)
    remove_one_file(file_name + ".db")

    # multi files
    file_name1 = file_name + "1"
    create_empty_file(file_name1)
    create_empty_file(file_name1 + ".db")
    file_name2 = file_name + "2"
    create_empty_file(file_name2)
    create_empty_file(file_name2 + ".db")

    with pytest.raises(RuntimeError) as err:
        _ = FileReader([file_name1, file_name2])
    assert "Invalid file, the size of mindrecord file: " in str(err.value)

    remove_one_file(file_name1)
    remove_one_file(file_name1 + ".db")
    remove_one_file(file_name2)
    remove_one_file(file_name2 + ".db")
