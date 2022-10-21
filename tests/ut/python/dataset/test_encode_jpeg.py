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
Testing encode_jpeg
"""
import cv2
import numpy
import pytest

from mindspore import Tensor
from mindspore.dataset import vision


def test_encode_jpeg_three_channels():
    """
    Feature: encode_jpeg
    Description: Test encode_jpeg by encoding the three channels image as JPEG data according to the quality
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test with numpy:ndarray and default quality
    encoded_jpeg = vision.encode_jpeg(image_rgb)
    assert encoded_jpeg.dtype == numpy.uint8
    assert encoded_jpeg[0] == 255
    assert encoded_jpeg[1] == 216
    assert encoded_jpeg[2] == 255

    # Test with Tensor and quality
    input_tensor = Tensor.from_numpy(image_rgb)
    encoded_jpeg_75 = vision.encode_jpeg(input_tensor, 75)
    assert encoded_jpeg_75[1] == 216

    # Test with the minimum quality
    encoded_jpeg_0 = vision.encode_jpeg(input_tensor, 1)
    assert encoded_jpeg_0[1] == 216

    # Test with the maximum quality
    encoded_jpeg_100 = vision.encode_jpeg(input_tensor, 100)
    assert encoded_jpeg_100[1] == 216

    # Test with three channels 12*34*3 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 3), dtype=numpy.uint8)
    encoded_jpeg = vision.encode_jpeg(image_random)
    assert encoded_jpeg[1] == 216
    encoded_jpeg = vision.encode_jpeg(Tensor.from_numpy(image_random))
    assert encoded_jpeg[1] == 216


def test_encode_jpeg_one_channel():
    """
    Feature: encode_jpeg
    Description: Test encode_jpeg by encoding the one channel image as JPEG data
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with one channel image_grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    encoded_jpeg = vision.encode_jpeg(image_grayscale)
    assert encoded_jpeg[1] == 216
    encoded_jpeg = vision.encode_jpeg(Tensor.from_numpy(image_grayscale))
    assert encoded_jpeg[1] == 216

    # Test with one channel 12*34 random uint8
    image_random = numpy.ndarray(shape=(12, 34), dtype=numpy.uint8)
    encoded_jpeg = vision.encode_jpeg(image_random)
    assert encoded_jpeg[1] == 216
    encoded_jpeg = vision.encode_jpeg(Tensor.from_numpy(image_random))
    assert encoded_jpeg[1] == 216

    # Test with one channel 12*34*1 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 1), dtype=numpy.uint8)
    encoded_jpeg = vision.encode_jpeg(image_random)
    assert encoded_jpeg[1] == 216
    encoded_jpeg = vision.encode_jpeg(Tensor.from_numpy(image_random))
    assert encoded_jpeg[1] == 216


def test_encode_jpeg_exception():
    """
    Feature: encode_jpeg
    Description: Test encode_jpeg with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(image_param, quality_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.encode_jpeg(image_param, quality_param)
        assert error_msg in str(error_info.value)

    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with an invalid integer for the quality
    error_message = "Invalid quality"
    test_invalid_param(image, 0, RuntimeError, error_message)
    test_invalid_param(image, 101, RuntimeError, error_message)

    # Test with an invalid type for the quality
    error_message = "Input quality is not of type"
    test_invalid_param(image, 75.0, TypeError, error_message)

    # Test with an invalid image containing the float elements
    invalid_image = numpy.ndarray(shape=(10, 10, 3), dtype=float)
    error_message = "The type of the image data"
    test_invalid_param(invalid_image, 75, RuntimeError, error_message)

    # Test with an invalid type for the image
    error_message = "Input image is not of type"
    test_invalid_param("invalid_image", 75, TypeError, error_message)

    # Test with an invalid image with only one dimension
    invalid_image = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(invalid_image, 75, RuntimeError, error_message)

    # Test with an invalid image with four dimensions
    invalid_image = numpy.ndarray(shape=(10, 10, 10, 3), dtype=numpy.uint8)
    test_invalid_param(invalid_image, 75, RuntimeError, error_message)

    # Test with an invalid image with two channels
    invalid_image = numpy.ndarray(shape=(10, 10, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(invalid_image, 75, RuntimeError, error_message)


if __name__ == "__main__":
    test_encode_jpeg_three_channels()
    test_encode_jpeg_one_channel()
    test_encode_jpeg_exception()
