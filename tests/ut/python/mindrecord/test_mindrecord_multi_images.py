# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test write multiple images"""
import os
import numpy as np
from utils import get_two_bytes_data, get_multi_bytes_data

from mindspore import log as logger
from mindspore.mindrecord import FileReader, FileWriter

MAP_FILE_NAME = "../data/mindrecord/testTwoImageData/cityscapes_train_19.txt"
MAP_FILE_FAKE_NAME = "../data/mindrecord/testTwoImageData/cityscapes_train_19_fake.txt"
DIFF_SHAPE_FILE_NAME = "../data/mindrecord/testImageNetData/cityscapes_train_19_fake.txt"
FILES_NUM = 1


def read(filename, fields_num=5):
    count = 0
    reader = FileReader(filename)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == fields_num
        count = count + 1
        logger.info("data: {}".format(x))
    assert count == 5
    reader.close()


def test_write_two_images_mindrecord():
    """test two images to mindrecord"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_two_bytes_data(MAP_FILE_NAME)
    cv_schema_json = {"img_data": {"type": "bytes"}, "label_data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "two_images_schema")
    writer.write_raw_data(data)
    writer.commit()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, 2)

    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)


def test_write_two_images_mindrecord_whole_field():
    """test two images to mindrecord"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)
    writer = FileWriter(file_name, FILES_NUM)
    data = get_two_bytes_data(MAP_FILE_NAME)
    cv_schema_json = {"id": {"type": "int32"}, "file_name": {"type": "string"},
                      "label_name": {"type": "string"}, "img_data": {"type": "bytes"},
                      "label_data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "two_images_schema")
    writer.write_raw_data(data)
    writer.commit()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, 5)

    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)


def test_write_two_diff_shape_images_mindrecord():
    """test two different shape images to mindrecord"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)
    bytes_num = 2
    writer = FileWriter(file_name, FILES_NUM)
    data = get_multi_bytes_data(DIFF_SHAPE_FILE_NAME, bytes_num)
    cv_schema_json = {"image_{}".format(i): {"type": "bytes"}
                      for i in range(bytes_num)}
    writer.add_schema(cv_schema_json, "two_images_schema")
    writer.write_raw_data(data)
    writer.commit()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, bytes_num)
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)


def test_write_multi_images_mindrecord():
    """test multiple images to mindrecord"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)
    bytes_num = 10
    writer = FileWriter(file_name, FILES_NUM)
    data = get_multi_bytes_data(MAP_FILE_FAKE_NAME, bytes_num)
    cv_schema_json = {"image_{}".format(i): {"type": "bytes"}
                      for i in range(bytes_num)}
    writer.add_schema(cv_schema_json, "multi_images_schema")
    writer.write_raw_data(data)
    writer.commit()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, bytes_num)
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)


def test_write_two_images_and_array_mindrecord():
    """test two image images and array to mindrecord"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)

    bytes_num = 2
    writer = FileWriter(file_name, FILES_NUM)
    data = get_multi_bytes_data(DIFF_SHAPE_FILE_NAME, bytes_num)

    for index, _ in enumerate(data):
        data[index].update({"input_ids": np.array([12, 45, 95, 0, 5, 66])})

    cv_schema_json = {"image_{}".format(i): {"type": "bytes"}
                      for i in range(bytes_num)}
    cv_schema_json.update({"id": {"type": "int64"},
                           "input_ids": {"type": "int64",
                                         "shape": [-1]}})
    writer.add_schema(cv_schema_json, "two_images_schema")
    writer.write_raw_data(data)
    writer.commit()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, bytes_num + 2)

    if os.path.exists("{}".format(file_name + ".db")):
        os.remove(file_name + ".db")
    if os.path.exists("{}".format(file_name)):
        os.remove(file_name)
