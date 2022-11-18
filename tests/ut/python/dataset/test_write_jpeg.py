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
Testing write_jpeg
"""
import os
import cv2
import numpy
import pytest

from mindspore import Tensor
from mindspore.dataset import vision


def test_write_jpeg_three_channels():
    """
    Feature: write_jpeg
    Description: Write the image containing three channels into a JPEG file
    Expectation: The file should be written and removed
    """

    def write_jpeg_three_channels(filename_param, image_param, quality_param=75):
        """
        a function used for writing with three channels image
        """
        vision.write_jpeg(filename_param, image_param, quality_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032, 3)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_bgr = cv2.imread(filename_1, mode)
    image_1_numpy = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)
    filename_2 = filename_1 + ".test_write_jpeg.jpg"

    # Test writing numpy.ndarray
    write_jpeg_three_channels(filename_2, image_1_numpy)

    # Test writing Tensor and quality 1, 75, 100
    for quality in (1, 75, 100):
        write_jpeg_three_channels(filename_2, image_1_tensor, quality)

    # Test with three channels 2268*4032*3 random uint8, the quality is 50
    image_random = numpy.ndarray(shape=(2268, 4032, 3), dtype=numpy.uint8)
    write_jpeg_three_channels(filename_2, image_random, 50)


def test_write_jpeg_one_channel():
    """
    Feature: write_jpeg
    Description: Write the grayscale image into a JPEG file
    Expectation: The file should be written and removed
    """

    def write_jpeg_one_channel(filename_param, image_param, quality_param=75):
        """
        a function used for writing with three channels image
        """
        vision.write_jpeg(filename_param, image_param, quality_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    filename_2 = filename_1 + ".test_write_jpeg.jpg"
    image_grayscale = cv2.cvtColor(image_1_numpy, cv2.COLOR_BGR2GRAY)
    image_grayscale_tensor = Tensor.from_numpy(image_grayscale)

    # Test writing numpy.ndarray
    write_jpeg_one_channel(filename_2, image_grayscale)

    # Test writing Tensor and quality 1, 75, 100
    for quality in (1, 75, 100):
        write_jpeg_one_channel(filename_2, image_grayscale_tensor, quality)

    # Test with three channels 2268*4032 random uint8
    image_random = numpy.ndarray(shape=(2268, 4032), dtype=numpy.uint8)
    write_jpeg_one_channel(filename_2, image_random)

    # Test with one channels 2268*4032*1 random uint8, the quality is 50
    image_random = numpy.ndarray(shape=(2268, 4032, 1), dtype=numpy.uint8)
    write_jpeg_one_channel(filename_2, image_random, 50)


def test_write_jpeg_exception():
    """
    Feature: write_jpeg
    Description: Test write_jpeg with an invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(filename_param, image_param, quality_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.write_jpeg(filename_param, image_param, quality_param)
        assert error_msg in str(error_info.value)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, image_1_numpy, 75, RuntimeError, error_message)

    # Test with an invalid filename
    wrong_filename = "/dev/cdrom/0"
    error_message = "No such file or directory"
    test_invalid_param(wrong_filename, image_1_tensor, 75, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0, image_1_numpy, 75, TypeError, error_message)

    # Test with an invalid type for the data
    filename_2 = filename_1 + ".test_write_jpeg.jpg"
    error_message = "Input image is not of type"
    test_invalid_param(filename_2, 0, 75, TypeError, error_message)

    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10, 10), dtype=float)
    error_message = "The type of the elements of image should be UINT8"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with only one dimension
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with four dimensions
    invalid_data = numpy.ndarray(shape=(1, 2, 3, 4), dtype=numpy.uint8)
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with two channels
    invalid_data = numpy.ndarray(shape=(2, 3, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid quality
    invalid_data = numpy.ndarray(shape=(2, 3, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with an invalid integer for the quality 0, 101
    error_message = "Invalid quality"
    test_invalid_param(filename_2, image_1_numpy, 0, RuntimeError, error_message)
    test_invalid_param(filename_2, image_1_numpy, 101, RuntimeError, error_message)

    # Test with an invalid type for the quality
    error_message = "Input quality is not of type"
    test_invalid_param(filename_2, image_1_numpy, 75.0, TypeError, error_message)


if __name__ == "__main__":
    test_write_jpeg_three_channels()
    test_write_jpeg_one_channel()
    test_write_jpeg_exception()
