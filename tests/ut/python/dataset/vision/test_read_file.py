# Copyright 2025 Huawei Technologies Co., Ltd
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
Testing read_file
"""
import numpy
import os
from pathlib import Path
import pytest

import mindspore
import mindspore.dataset.vision as visions

TEST_DATA_DATASET_FUNC ="../data/dataset/"


def invalid_param(filename_param, error, error_msg):
    """ checking correct error and message with invalid parameter """
    with pytest.raises(error) as error_info:
        mindspore.dataset.vision.read_file(filename_param)
    assert error_msg in str(error_info.value)


def test_read_file_normal():
    """
    Feature: read_file
    Description: Test read_file by reading the contents of a file in binary mode
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    output = visions.read_file(filename)
    assert output.shape == (159109,)
    assert output.dtype == numpy.uint8
    assert output[0] == 255
    assert output[1] == 216
    assert output[2] == 255
    assert output[50000] == 0
    assert output[100000] == 132
    assert output[150000] == 64
    assert output[159106] == 63
    assert output[159107] == 255
    assert output[159108] == 217


def test_read_file_exception():
    """
    Feature: read_file
    Description: Test read_file with invalid filename
    Expectation: Error is caught when the filename is invalid
    """

    def test_invalid_param(filename_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            visions.read_file(filename_param)
        assert error_msg in str(error_info.value)

    # Test with a not exist filename
    wrong_filename = "this_file_is_not_exist"
    error_message = "Invalid file path, " + wrong_filename + " does not exist."
    test_invalid_param(wrong_filename, RuntimeError, error_message)

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0.1, TypeError, error_message)


def test_readfile_operation_01():
    """
    Feature: read_file operation
    Description: Testing the normal functionality of the read_file operator
    Expectation: The Output is equal to the expected output
    """
    # test jpg
    filename = os.path.join(TEST_DATA_DATASET_FUNC, 'apple.jpg')
    output = mindspore.dataset.vision.read_file(filename)
    assert output.shape == (159109,)
    assert output.dtype == numpy.uint8
    assert output[0] == 255
    assert output[1] == 216
    assert output[2] == 255

    # test 0
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    filename = os.path.join(base_dir, 'test1.txt')
    Path(filename).touch()
    output = mindspore.dataset.vision.read_file(filename)
    assert output.shape == (0,)
    assert output.dtype == numpy.uint8


def test_readfile_exception_01():
    """
    Feature: read_file operation
    Description: Testing the read_file Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # no param
    error_message = "read_file() missing 1 required positional argument: 'filename'"
    with pytest.raises(TypeError) as error_info:
        mindspore.dataset.vision.read_file()
    assert error_message in str(error_info.value)

    # two param
    error_message = "read_file() takes 1 positional argument but 2 were given"
    with pytest.raises(TypeError) as error_info:
        mindspore.dataset.vision.read_file("1", 2)
    assert error_message in str(error_info.value)

    # invalid_type
    error_message = "Input filename is not of type"
    invalid_param(0.1, TypeError, error_message)

    # Test with a directory name
    wrong_filename = TEST_DATA_DATASET_FUNC
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    invalid_param(wrong_filename, RuntimeError, error_message)


if __name__ == "__main__":
    test_read_file_normal()
    test_read_file_exception()
    test_readfile_operation_01()
    test_readfile_exception_01()
