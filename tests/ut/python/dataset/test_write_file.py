# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing write_file
"""
import os
import numpy
import pytest

from mindspore import Tensor
from mindspore.dataset import vision


def test_write_file_normal():
    """
    Feature: write_file
    Description: Test the write_file by writing the data into a file using binary mode
    Expectation: The file should be writeen and removed
    """
    filename_1 = "../data/dataset/apple.jpg"
    data_1_numpy = numpy.fromfile(filename_1, dtype=numpy.uint8)
    data_1_tensor = Tensor.from_numpy(data_1_numpy)

    filename_2 = filename_1 + ".test_write_file"

    # Test writing numpy.ndarray
    vision.write_file(filename_2, data_1_numpy)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    assert data_2_numpy.shape == (159109,)

    # Test writing Tensor
    vision.write_file(filename_2, data_1_tensor)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    assert data_2_numpy.shape == (159109,)

    # Test writing empty numpy.ndarray
    empty_numpy = numpy.empty(0, dtype=numpy.uint8)
    vision.utils.write_file(filename_2, empty_numpy)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    assert data_2_numpy.shape == (0,)

    # Test writing empty Tensor
    empty_tensor = Tensor.from_numpy(empty_numpy)
    vision.utils.write_file(filename_2, empty_tensor)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    assert data_2_numpy.shape == (0,)


def test_write_file_exception():
    """
    Feature: write_file
    Description: Test the write_file with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(filename_param, data_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.write_file(filename_param, data_param)
        assert error_msg in str(error_info.value)

    filename_1 = "../data/dataset/apple.jpg"
    data_1_numpy = numpy.fromfile(filename_1, dtype=numpy.uint8)
    data_1_tensor = Tensor.from_numpy(data_1_numpy)

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, data_1_numpy, RuntimeError, error_message)

    # Test with an invalid filename
    wrong_filename = "/dev/cdrom/0"
    error_message = "No such file or directory"
    test_invalid_param(wrong_filename, data_1_tensor, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0, data_1_numpy, TypeError, error_message)

    # Test with an invalid type for the data
    filename_2 = filename_1 + ".test_write_file"
    error_message = "Input data is not of type"
    test_invalid_param(filename_2, 0, TypeError, error_message)

    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=float)
    error_message = "The type of the elements of data should be"
    test_invalid_param(filename_2, invalid_data, RuntimeError, error_message)

    # Test with invalid data
    error_message = "The data has invalid dimensions"
    invalid_data = numpy.ndarray(shape=(10, 10), dtype=numpy.uint8)
    test_invalid_param(filename_2, invalid_data, RuntimeError, error_message)


if __name__ == "__main__":
    test_write_file_normal()
    test_write_file_exception()
