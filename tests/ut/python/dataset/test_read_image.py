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
Testing read_image
"""
import numpy
import pytest

from mindspore.dataset import vision
from mindspore.dataset.vision import ImageReadMode


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


if __name__ == "__main__":
    test_read_image_jpeg()
    test_read_image_png()
    test_read_image_bmp()
    test_read_image_tiff()
    test_read_image_exception()
