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
import os
import pytest

import mindspore
import mindspore.dataset.vision.utils as v_trans
from mindspore import Tensor
from mindspore.dataset import vision

TEST_DATA_DATASET_FUNC ="../data/dataset/"


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


def test_encode_jpeg_operation_01():
    """
    Feature: encode_jpeg operation
    Description: Testing the normal functionality of the encode_jpeg operator
    Expectation: The Output is equal to the expected output
    """
    # encode_jpeg Normal Scenario: Verify Normal Functionality
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    encode_jpeg = v_trans.encode_jpeg(image_numpy)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert encode_jpeg.shape == (42977,)

    # encode_jpeg Normal scenario: quality parameter set to 75
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_jpeg = v_trans.encode_jpeg(image_tensor, 6)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert encode_jpeg.shape == (7468,)

    # encode_jpeg Normal scenario: quality parameter set to 1
    mode = cv2.IMREAD_UNCHANGED
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_numpy = cv2.imread(data_dir4, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_jpeg = v_trans.encode_jpeg(image_tensor, 1)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert encode_jpeg.shape == (1797,)

    # encode_jpeg Normal scenario: quality parameter set to 100
    mode = cv2.IMREAD_UNCHANGED
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image_numpy = cv2.imread(data_dir5, mode)
    assert image_numpy.shape[2] == 4
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGRA2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_jpeg = v_trans.encode_jpeg(image_tensor, 100)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert encode_jpeg.shape == (210343,)

    # encode_jpeg Normal scenario: quality parameter set to 50
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_jpeg = v_trans.encode_jpeg(image_tensor, 50)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert encode_jpeg.shape == (28684,)

    # encode_jpeg Normal scenario: The parameter `image` is a random number.
    image_random = numpy.random.randint(256, size=(1, 8000, 3), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_jpeg = mindspore.dataset.vision.encode_jpeg(image_tensor, 3)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert len(encode_jpeg.shape) == 1

    # encode_jpeg Normal scenario: The parameter `image` is a random number.
    image_random = numpy.random.randint(256, size=(600, 200, 3), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_jpeg = mindspore.dataset.vision.encode_jpeg(image_tensor)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert len(encode_jpeg.shape) == 1

    # encode_jpeg Normal scenario: The image parameter is 2D.
    image_random = numpy.random.randint(256, size=(876, 543), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_jpeg = mindspore.dataset.vision.encode_jpeg(image_tensor, 7)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert len(encode_jpeg.shape) == 1

    # encode_jpeg normal scenario: parameter image with channel set to 1
    image_random = numpy.random.randint(256, size=(224, 224, 1), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_jpeg = mindspore.dataset.vision.encode_jpeg(image_tensor)
    assert isinstance(encode_jpeg, numpy.ndarray)
    assert encode_jpeg.dtype == 'uint8'
    assert len(encode_jpeg.shape) == 1


def test_encode_jpeg_exception_01():
    """
    Feature: encode_jpeg operation
    Description: Testing the encode_jpeg Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # encode_jpeg exception scenario: image channel count is neither 1 nor 3
    with pytest.raises(RuntimeError, match="EncodeJpeg: The image has invalid "
                                           "channels. It should have 1 or 3 channels, but got 2 channels."):
        image_random = numpy.random.randint(256, size=(876, 543, 2), dtype=numpy.uint8)
        image_tensor = mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg(image_tensor)

    # encode_jpeg exception scenario: image is 1-dimensional data
    with pytest.raises(RuntimeError, match="EncodeJpeg: The image has invalid "
                                           "dimensions. It should have two or three dimensions, but got 1 dimensions."):
        image_random = numpy.random.randint(256, size=(224,), dtype=numpy.uint8)
        image_tensor = mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg(image_tensor)

    # encode_jpeg exception scenario: image is of type uint16
    try:
        image_random = numpy.random.randint(256, size=(224, 124, 1), dtype=numpy.uint16)
        v_trans.encode_jpeg(image_random)
    except RuntimeError as e:
        assert "The type of the image data should be UINT8, but got uint16" in str(e)

    # encode_jpeg exception scenario: no parameters passed
    try:
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg()
    except TypeError as e:
        assert "encode_jpeg() missing 1 required positional argument: 'image'" in str(e)

    # encode_jpeg exception scenario: quality parameter exceeds 100
    with pytest.raises(RuntimeError, match=r"EncodeJpeg: Invalid quality 101, should be in range of \[1, 100\]."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg(image_random, 101)

    # encode_jpeg exception scenario: quality parameter type is str
    with pytest.raises(TypeError, match="Input quality is not of type <class 'int'>, "
                                        "but got: <class 'str'>."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg(image_random, '9')

    # encode_jpeg exception scenario: quality parameter type is float
    with pytest.raises(TypeError, match="Input quality is not of type <class 'int'>, "
                                        "but got: <class 'float'>."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_jpeg(image_random, 9.0)


if __name__ == "__main__":
    test_encode_jpeg_three_channels()
    test_encode_jpeg_one_channel()
    test_encode_jpeg_exception()
    test_encode_jpeg_operation_01()
    test_encode_jpeg_exception_01()
