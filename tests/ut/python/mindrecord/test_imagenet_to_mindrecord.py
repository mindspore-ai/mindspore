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
"""test imagenet to mindrecord tool"""
import os
import pytest

from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import ImageNetToMR

IMAGENET_MAP_FILE = "../data/mindrecord/testImageNetDataWhole/labels_map.txt"
IMAGENET_IMAGE_DIR = "../data/mindrecord/testImageNetDataWhole/images"
PARTITION_NUMBER = 4

@pytest.fixture
def fixture_file():
    """add/remove file"""
    def remove_one_file(x):
        if os.path.exists(x):
            os.remove(x)
    def remove_file(file_name):
        x = file_name
        remove_one_file(x)
        x = file_name + ".db"
        remove_one_file(x)
        for i in range(PARTITION_NUMBER):
            x = file_name + str(i)
            remove_one_file(x)
            x = file_name + str(i) + ".db"
            remove_one_file(x)

    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    yield "yield_fixture_data"
    remove_file(file_name)

def read(filename):
    """test file reade"""
    count = 0
    reader = FileReader(filename)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 20
    reader.close()

def test_imagenet_to_mindrecord(fixture_file):
    """test transform imagenet dataset to mindrecord."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR,
                                        file_name, PARTITION_NUMBER)
    imagenet_transformer.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(file_name + str(i))
        assert os.path.exists(file_name + str(i) + ".db")
    read([file_name + "0",
          file_name + "1",
          file_name + "2",
          file_name + "3"])

def test_imagenet_to_mindrecord_default_partition_number(fixture_file):
    """
    test transform imagenet dataset to mindrecord
    when partition number is default.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR,
                                        file_name, 1)
    imagenet_transformer.transform()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name)

def test_imagenet_to_mindrecord_partition_number_0(fixture_file):
    """
    test transform imagenet dataset to mindrecord
    when partition number is 0.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception, match="Invalid parameter value"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR,
                                            file_name, 0)
        imagenet_transformer.transform()

def test_imagenet_to_mindrecord_partition_number_none(fixture_file):
    """
    test transform imagenet dataset to mindrecord
    when partition number is none.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception,
                       match="The parameter partition_number must be int"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR,
                                            file_name, None)
        imagenet_transformer.transform()

def test_imagenet_to_mindrecord_illegal_filename(fixture_file):
    """
    test transform imagenet dataset to mindrecord
    when file name contains illegal character.
    """
    filename = "imagenet_not_*ok"
    with pytest.raises(Exception, match="File name should not contains"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR, filename,
                                            PARTITION_NUMBER)
        imagenet_transformer.transform()

def test_imagenet_to_mindrecord_illegal_1_filename(fixture_file):
    """
    test transform imagenet dataset to mindrecord
    when file name end with '/'.
    """
    filename = "imagenet/path/"
    with pytest.raises(Exception, match="File path can not end with '/'"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR, filename,
                                            PARTITION_NUMBER)
        imagenet_transformer.transform()
