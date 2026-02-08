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
Testing read_image
"""
import numpy
import os
import pytest

from mindspore.dataset import vision
from mindspore.dataset.vision import ImageReadMode

TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    data_dir7 = os.path.join(TEST_DATA_DATASET_FUNC, "testFormats", "apple.tiff")
    data_list.append(data_dir1)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    data_list.append(data_dir7)
    return data_list


def test_read_image_jpeg():
    """
    Feature: read_image
    Description: Read the contents of a JPEG image file
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/testFormats/apple.jpg"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 3)
    assert output.dtype == numpy.uint8
    assert output[0, 0, 0] == 221
    assert output[0, 0, 1] == 221
    assert output[0, 0, 2] == 221
    assert output[100, 200, 0] == 195
    assert output[100, 200, 1] == 60
    assert output[100, 200, 2] == 31
    assert output[225, 402, 0] == 181
    assert output[225, 402, 1] == 181
    assert output[225, 402, 2] == 173
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)

    filename = "../data/dataset/testFormats/apple_grayscale.jpg"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)


def test_read_image_png():
    """
    Feature: read_image
    Description: Read the contents of a PNG image file
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/testFormats/apple.png"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)

    filename = "../data/dataset/testFormats/apple_4_channels.png"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)


def test_read_image_bmp():
    """
    Feature: read_image
    Description: Read the contents of a BMP image file
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/testFormats/apple.bmp"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)


def test_read_image_tiff():
    """
    Feature: read_image
    Description: Read the contents of a TIFF image file
    Expectation: The Output is equal to the expected output
    """
    filename = "../data/dataset/testFormats/apple.tiff"
    output = vision.read_image(filename)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.UNCHANGED)
    assert output.shape == (226, 403, 3)
    output = vision.read_image(filename, ImageReadMode.GRAYSCALE)
    assert output.shape == (226, 403, 1)
    output = vision.read_image(filename, ImageReadMode.COLOR)
    assert output.shape == (226, 403, 3)


def test_read_image_exception():
    """
    Feature: read_image
    Description: Test read_image with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(filename_param, mode_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.read_image(filename_param, mode_param)
        assert error_msg in str(error_info.value)

    # Test with a not exist filename
    wrong_filename = "this_file_is_not_exist"
    error_message = "Invalid file path, " + wrong_filename + " does not exist."
    test_invalid_param(wrong_filename, ImageReadMode.COLOR, RuntimeError, error_message)

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, ImageReadMode.COLOR, RuntimeError, error_message)

    # Test with a not supported gif file
    wrong_filename = "../data/dataset/testFormats/apple.gif"
    error_message = "Failed to read file " + wrong_filename
    test_invalid_param(wrong_filename, ImageReadMode.COLOR, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0, ImageReadMode.UNCHANGED, TypeError, error_message)

    # Test with an invalid type for the mode
    filename = "../data/dataset/testFormats/apple.jpg"
    error_message = "Input mode is not of type"
    test_invalid_param(filename, "0", TypeError, error_message)


def test_read_image_operation_01():
    """
    Feature: read_image operation
    Description: Testing the normal functionality of the read_image operator
    Expectation: The Output is equal to the expected output
    """
    # use tiff file, set mode default
    read_image_file = vision.read_image(dir_data()[5])
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (226, 403, 3)

    # use png file, set mode default
    read_image_file = vision.read_image(dir_data()[3])
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (484, 508, 3)

    # use jpg file, set mode default
    read_image_file = vision.read_image(dir_data()[1])
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (432, 576, 3)

    # use bmp file, set mode default
    read_image_file = vision.read_image(dir_data()[2])
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (96, 120, 3)

    # use png file, set mode=ImageReadMode.GRAYSCALE
    read_image_file = vision.read_image(dir_data()[3], ImageReadMode.GRAYSCALE)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (484, 508, 1)

    # use jpg file, set mode=ImageReadMode.GRAYSCALE
    read_image_file = vision.read_image(dir_data()[1], ImageReadMode.GRAYSCALE)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (432, 576, 1)

    # use bmp file, set mode=ImageReadMode.GRAYSCALE
    read_image_file = vision.read_image(dir_data()[2], ImageReadMode.GRAYSCALE)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (96, 120, 1)

    # use png file, set mode=ImageReadMode.COLOR
    read_image_file = vision.read_image(dir_data()[3], ImageReadMode.COLOR)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (484, 508, 3)

    # use jpg file, set mode=ImageReadMode.COLOR
    read_image_file = vision.read_image(dir_data()[1], ImageReadMode.COLOR)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (432, 576, 3)

    # use bmp file, set mode=ImageReadMode.COLOR
    read_image_file = vision.read_image(dir_data()[2], ImageReadMode.COLOR)
    assert isinstance(read_image_file, numpy.ndarray)
    assert read_image_file.dtype == 'uint8'
    assert read_image_file.ndim == 3
    assert read_image_file.shape == (96, 120, 3)


def test_read_image_exception_01():
    """
    Feature: read_image operation
    Description: Testing the read_image Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # use gif file, set mode=ImageReadMode.COLOR
    gif_file = dir_data()[4]
    with pytest.raises(RuntimeError):
        vision.read_image(gif_file, ImageReadMode.COLOR)

    # set mode type error
    with pytest.raises(TypeError, match="Input mode is not of type <enum 'ImageReadMode'>, but got: <class 'str'>."):
        vision.read_image(dir_data()[1], '0')

    # use not exit file
    image = 12
    with pytest.raises(TypeError, match="Input filename is not of type <class 'str'>, but got: <class 'int'>."):
        vision.read_image(image, ImageReadMode.COLOR)

    # Incorrect path
    with pytest.raises(RuntimeError):
        vision.read_image(dir_data()[0], ImageReadMode.COLOR)

    # Missing parameters
    try:
        vision.read_image()
    except TypeError as e:
        assert "read_image() missing 1 required positional argument: 'filename'" in str(e)


if __name__ == "__main__":
    test_read_image_jpeg()
    test_read_image_png()
    test_read_image_bmp()
    test_read_image_tiff()
    test_read_image_exception()
    test_read_image_operation_01()
    test_read_image_exception_01()
