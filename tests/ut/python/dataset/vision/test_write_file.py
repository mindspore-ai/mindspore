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
Testing write_file
"""
import numpy
import os
import pandas as pd
import platform
import pytest
import shutil

import mindspore
from mindspore import Tensor
from mindspore.dataset import vision


def invalid_param(filename_param, data_param, error, error_msg):
    """
    A function used for checking correct error and message with invalid parameter
    """
    with pytest.raises(error) as error_info:
        mindspore.dataset.vision.write_file(filename_param, data_param)
    assert error_msg in str(error_info.value)


def data_random(data_length, use_numpy=True):
    """function method"""
    filename_1 = os.path.join(os.getcwd(), "data_random")
    filename_2 = filename_1 + ".test_write_file"
    data_1_numpy = numpy.random.randint(256, size=data_length, dtype=numpy.uint8)
    if use_numpy:
        mindspore.dataset.vision.write_file(filename_2, data_1_numpy)
    else:
        data_1_tensor = mindspore.Tensor.from_numpy(data_1_numpy)
        mindspore.dataset.vision.write_file(filename_2, data_1_tensor)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    assert numpy.array_equal(data_1_numpy, data_2_numpy)


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


def test_write_file_operation_01():
    """
    Feature: write_file operation
    Description: Testing the normal functionality of the write_file operator
    Expectation: The Output is equal to the expected output
    """
    # Description: Test the write_file by writing the data into a file using binary mode
    create_file = r"./demo.csv"
    df1 = pd.DataFrame({"ID": ["a", "b", "c"], "name": ["Chris", "Lucy", "LIly"], "score": [70, 80, 90]})
    df1.to_csv(create_file)
    filename_1 = os.path.join(os.getcwd(), "demo.csv")
    data_1_numpy = numpy.fromfile(filename_1, dtype=numpy.uint8)
    filename_2 = filename_1 + "test_write_file"
    mindspore.dataset.vision.write_file(filename_2, data_1_numpy)
    data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
    os.remove(filename_2)
    os.remove(filename_1)
    assert numpy.array_equal(data_1_numpy, data_2_numpy)

    # Description: Test the write_file by writing the data into a file using binary mode
    if platform.system() == "Windows":
        with pytest.raises(OSError):
            if not os.path.exists(os.path.join(os.getcwd(), r"^^&&***")):
                os.makedirs("^^&&***")
    else:
        if not os.path.exists(os.path.join(os.getcwd(), r"^^&&***")):
            os.makedirs("^^&&***")
        create_file = r"./^^&&***/sdaa**&^%%.csv"
        df1 = pd.DataFrame({"ID": ["a", "b", "c"], "name": ["Chris", "Lucy", "LIly"], "score": [70, 80, 90]})
        df1.to_csv(create_file)
        filename_1 = os.path.join(os.getcwd(), "^^&&***", "sdaa**&^%%.csv")
        data_1_numpy = numpy.fromfile(filename_1, dtype=numpy.uint8)
        filename_2 = filename_1 + "test_write_file"
        mindspore.dataset.vision.write_file(filename_2, data_1_numpy)
        data_2_numpy = numpy.fromfile(filename_2, dtype=numpy.uint8)
        os.remove(filename_2)
        if os.path.exists("^^&&***") is True:
            shutil.rmtree("^^&&***")
        assert numpy.array_equal(data_1_numpy, data_2_numpy)

    # create random size 0 with Tensor or numpy
    data_random(0)
    data_random(0, use_numpy=False)

    # create random size 5 with Tensor or numpy
    data_random(5)
    data_random(5, use_numpy=False)

    # create random size 512 with Tensor or numpy
    data_random(512)
    data_random(512, use_numpy=False)

    # create random size 20mb with Tensor or numpy
    data_random(20 * 1024 * 1024)
    data_random(20 * 1024 * 1024, use_numpy=False)


def test_write_file_exception_01():
    """
    Feature: write_file operation
    Description: Testing the write_file Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test without a parameter
    error_message = "write_file() missing 2 required positional arguments: 'filename' and 'data'"
    with pytest.raises(TypeError) as error_info:
        mindspore.dataset.vision.write_file()
    assert error_message in str(error_info.value)

    # Test without a parameter
    error_message = "write_file() missing 1 required positional argument: 'data'"
    with pytest.raises(TypeError) as error_info:
        mindspore.dataset.vision.write_file("1")
    assert error_message in str(error_info.value)

    # Test without a parameter
    error_message = "write_file() takes 2 positional arguments but 3 were given"
    with pytest.raises(TypeError) as error_info:
        mindspore.dataset.vision.write_file("1", 2, 3)
    assert error_message in str(error_info.value)

    # Description: Test the write_file with invalid parameter
    # Test with an invalid type for the filename
    error_message = "Input filename is not of type <class 'str'>, but got: <class 'int'>."
    data_1_numpy = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    invalid_param(0, data_1_numpy, TypeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with an invalid filename
    wrong_filename = "./dev/cdrom/0"
    error_message = "No such file or directory"
    data_1_numpy = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    invalid_param(wrong_filename, data_1_numpy, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with an invalid type for the data
    error_message = "Input data is not of type <class 'numpy.ndarray'> " \
                    "or <class 'mindspore.common.tensor.Tensor'>, but got: <class 'int'>."
    invalid_param("../data/dataset/test_write_file", 0, TypeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=float)
    error_message = "The type of the elements of data should be UINT8, but got float64."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid unit elements
    error_message = "The data has invalid dimensions. It should have only one dimension, but got 2 dimensions."
    invalid_data = numpy.ndarray(shape=(10, 10), dtype=numpy.uint8)
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid string elements
    invalid_data = numpy.ndarray(shape=(10), dtype=str)
    error_message = "The type of the elements of data should be UINT8, but got string."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.uint16)
    error_message = "The type of the elements of data should be UINT8, but got uint16."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.uint32)
    error_message = "The type of the elements of data should be UINT8, but got uint32."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=bool)
    error_message = "The type of the elements of data should be UINT8, but got bool."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.int8)
    error_message = "The type of the elements of data should be UINT8, but got int8."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)

    # Description: Test the write_file with invalid parameter
    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10), dtype=bytes)
    error_message = "The type of the elements of data should be UINT8, but got bytes."
    invalid_param("../data/dataset/test_write_file", invalid_data, RuntimeError, error_message)


if __name__ == "__main__":
    test_write_file_normal()
    test_write_file_exception()
    test_write_file_operation_01()
    test_write_file_exception_01()
