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
"""test cifar100 to mindrecord tool"""
import os
import pytest

from mindspore import log as logger
from mindspore.mindrecord import Cifar100ToMR
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import MRMOpenError
from mindspore.mindrecord import SUCCESS

CIFAR100_DIR = "../data/mindrecord/testCifar100Data"
MINDRECORD_FILE = "./cifar100.mindrecord"

@pytest.fixture
def fixture_file():
    """add/remove file"""
    def remove_file(x):
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))
        if os.path.exists("{}_test".format(x)):
            os.remove("{}_test".format(x))
        if os.path.exists("{}_test.db".format(x)):
            os.remove("{}_test.db".format(x))

    remove_file(MINDRECORD_FILE)
    yield "yield_fixture_data"
    remove_file(MINDRECORD_FILE)

def test_cifar100_to_mindrecord_without_index_fields(fixture_file):
    """test transform cifar100 dataset to mindrecord without index fields."""
    cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, MINDRECORD_FILE)
    ret = cifar100_transformer.transform()
    assert ret == SUCCESS, "Failed to transform from cifar100 to mindrecord"
    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + "_test")
    read()

def test_cifar100_to_mindrecord(fixture_file):
    """test transform cifar100 dataset to mindrecord."""
    cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, MINDRECORD_FILE)
    cifar100_transformer.transform(['fine_label', 'coarse_label'])
    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + "_test")
    read()


def read():
    """test file reader"""
    count = 0
    reader = FileReader(MINDRECORD_FILE)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 4
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 16
    reader.close()

    count = 0
    reader = FileReader(MINDRECORD_FILE + "_test")
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 4
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 4
    reader.close()

def test_cifar100_to_mindrecord_illegal_file_name(fixture_file):
    """
    test transform cifar100 dataset to mindrecord
    when file name contains illegal character.
    """
    filename = "./:no_ok"
    with pytest.raises(Exception, match="File name should not contains"):
        cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, filename)
        cifar100_transformer.transform()

def test_cifar100_to_mindrecord_filename_start_with_space(fixture_file):
    """
    test transform cifar10 dataset to mindrecord
    when file name starts with space.
    """
    filename = "./ no_ok"
    with pytest.raises(Exception,
                       match="File name should not start/end with space"):
        cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, filename)
        cifar100_transformer.transform()

def test_cifar100_to_mindrecord_filename_contain_space(fixture_file):
    """
    test transform cifar10 dataset to mindrecord
    when file name contains space.
    """
    filename = "./yes  ok"
    cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, filename)
    cifar100_transformer.transform()
    assert os.path.exists(filename)
    assert os.path.exists(filename + "_test")

def test_cifar100_to_mindrecord_directory(fixture_file):
    """
    test transform cifar10 dataset to mindrecord
    when destination path is directory.
    """
    with pytest.raises(MRMOpenError,
                       match="MindRecord File could not open successfully"):
        cifar100_transformer = Cifar100ToMR(CIFAR100_DIR,
                                            CIFAR100_DIR)
        cifar100_transformer.transform()

def test_cifar100_to_mindrecord_filename_equals_cifar100(fixture_file):
    """
    test transform cifar10 dataset to mindrecord
    when destination path equals source path.
    """
    with pytest.raises(MRMOpenError,
                       match="MindRecord File could not open successfully"):
        cifar100_transformer = Cifar100ToMR(CIFAR100_DIR,
                                            CIFAR100_DIR + "/train")
        cifar100_transformer.transform()
