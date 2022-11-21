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
Testing encode_png
"""
import cv2
import numpy
import pytest

import mindspore


def test_encode_png_three_channels():
    """
    Feature: encode_png
    Description: Test encode_png by encoding the three channels image as PNG data according to the compression_level
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test with numpy:ndarray and default compression_level
    encoded_png = mindspore.dataset.vision.encode_png(image_rgb)
    assert encoded_png.dtype == numpy.uint8
    assert encoded_png[0] == 137
    assert encoded_png[1] == 80
    assert encoded_png[2] == 78
    assert encoded_png[3] == 71

    # Test with Tensor and compression_level
    input_tensor = mindspore.Tensor.from_numpy(image_rgb)
    encoded_png_6 = mindspore.dataset.vision.encode_png(input_tensor, 6)
    assert encoded_png_6[1] == 80

    # Test with the minimum compression_level
    encoded_png_0 = mindspore.dataset.vision.encode_png(input_tensor, 0)
    assert encoded_png_0[1] == 80

    # Test with the maximum compression_level
    encoded_png_9 = mindspore.dataset.vision.encode_png(input_tensor, 9)
    assert encoded_png_9[1] == 80

    # Test with three channels 12*34*3 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 3), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80


def test_encode_png_one_channel():
    """
    Feature: encode_png
    Description: Test encode_png by encoding the one channel image as PNG data
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with one channel image_grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    encoded_png = mindspore.dataset.vision.encode_png(image_grayscale)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_grayscale))
    assert encoded_png[1] == 80

    # Test with one channel 12*34 random uint8
    image_random = numpy.ndarray(shape=(12, 34), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80

    # Test with one channel 12*34*1 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 1), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80


def test_encode_png_exception():
    """
    Feature: encode_png
    Description: Test encode_png with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(image_param, compression_level_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            mindspore.dataset.vision.encode_png(image_param, compression_level_param)
        assert error_msg in str(error_info.value)

    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with an invalid integer for the compression_level
    error_message = "Invalid compression_level"
    test_invalid_param(image, -1, RuntimeError, error_message)
    test_invalid_param(image, 10, RuntimeError, error_message)

    # Test with an invalid type for the compression_level
    error_message = "Input compression_level is not of type"
    test_invalid_param(image, 6.0, TypeError, error_message)

    # Test with an invalid image containing the float elements
    invalid_image = numpy.ndarray(shape=(10, 10, 3), dtype=float)
    error_message = "The type of the image data"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid type for the image
    error_message = "Input image is not of type"
    test_invalid_param("invalid_image", 6, TypeError, error_message)

    # Test with an invalid image with only one dimension
    invalid_image = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid image with four dimensions
    invalid_image = numpy.ndarray(shape=(10, 10, 10, 3), dtype=numpy.uint8)
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid image with two channels
    invalid_image = numpy.ndarray(shape=(10, 10, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)


if __name__ == "__main__":
    test_encode_png_three_channels()
    test_encode_png_one_channel()
    test_encode_png_exception()
