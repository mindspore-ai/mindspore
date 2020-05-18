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
MINDRECORD_FILE = "../data/mindrecord/testImageNetDataWhole/imagenet.mindrecord"
PARTITION_NUMBER = 4


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


def test_imagenet_to_mindrecord():
    """test transform imagenet dataset to mindrecord."""
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR,
                                        MINDRECORD_FILE, PARTITION_NUMBER)
    imagenet_transformer.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(MINDRECORD_FILE + str(i))
        assert os.path.exists(MINDRECORD_FILE + str(i) + ".db")
    read(MINDRECORD_FILE + "0")
    for i in range(PARTITION_NUMBER):
        os.remove(MINDRECORD_FILE + str(i))
        os.remove(MINDRECORD_FILE + str(i) + ".db")


def test_imagenet_to_mindrecord_default_partition_number():
    """
    test transform imagenet dataset to mindrecord
    when partition number is default.
    """
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR,
                                        MINDRECORD_FILE)
    imagenet_transformer.transform()
    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + ".db")
    read(MINDRECORD_FILE)
    os.remove("{}".format(MINDRECORD_FILE))
    os.remove("{}.db".format(MINDRECORD_FILE))


def test_imagenet_to_mindrecord_partition_number_0():
    """
    test transform imagenet dataset to mindrecord
    when partition number is 0.
    """
    with pytest.raises(Exception, match="Invalid parameter value"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR,
                                            MINDRECORD_FILE, 0)
        imagenet_transformer.transform()


def test_imagenet_to_mindrecord_partition_number_none():
    """
    test transform imagenet dataset to mindrecord
    when partition number is none.
    """
    with pytest.raises(Exception,
                       match="The parameter partition_number must be int"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR,
                                            MINDRECORD_FILE, None)
        imagenet_transformer.transform()


def test_imagenet_to_mindrecord_illegal_filename():
    """
    test transform imagenet dataset to mindrecord
    when file name contains illegal character.
    """
    filename = "not_*ok"
    with pytest.raises(Exception, match="File name should not contains"):
        imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                            IMAGENET_IMAGE_DIR, filename,
                                            PARTITION_NUMBER)
        imagenet_transformer.transform()
