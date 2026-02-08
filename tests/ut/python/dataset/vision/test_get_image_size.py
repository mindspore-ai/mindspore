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
Test MindData vision utility get_image_size
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset.vision.utils as vision_utils
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_get_image_size_output_array():
    """
    Feature: get_image_size
    Description: Test get_image_size array
    Expectation: The returned result is as expected
    """
    expect = [2268, 4032]
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    input_array = vision.Decode()(img)
    output = vision_utils.get_image_size(input_array)
    assert expect == output


def test_get_image_size_output_img():
    """
    Feature: get_image_size
    Description: Test get_image_size image (Image.size is [H, W])
    Expectation: The returned result is as expected
    """
    expect = [2268, 4032]
    img = Image.open("../data/dataset/apple.jpg")
    output_size = vision_utils.get_image_size(img)
    assert expect == output_size


def test_get_image_size_invalid_input():
    """
    Feature: get_image_size
    Description: Test get_image_size invalid input
    Expectation: Correct error is raised as expected
    """

    def test_invalid_input(test_name, image, error, error_msg):
        logger.info("Test GetImageSize with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision_utils.get_image_size(image)
        assert error_msg in str(error_info.value)

    invalid_input = 1
    invalid_shape = np.array([1, 2, 3])
    test_invalid_input("invalid input", invalid_input, TypeError,
                       "Input image is not of type <class 'numpy.ndarray'> or <class 'PIL.Image.Image'>, "
                       "but got: <class 'int'>.")
    test_invalid_input("invalid input", invalid_shape, RuntimeError,
                       "GetImageSize: invalid parameter, image should have at least two dimensions, but got: 1")


def test_get_image_size_operation_01():
    """
    Feature: get_image_num_channels operation
    Description: Testing the normal functionality of the get_image_num_channels operator
    Expectation: The Output is equal to the expected output
    """
    # Test: get_image_num_channels array
    apple_image = os.path.join(TEST_DATA_DATASET_FUNC, "apple.jpg")
    expect = [2268, 4032]
    img = np.fromfile(apple_image, dtype=np.uint8)
    input_array = vision.Decode()(img)
    output = vision_utils.get_image_size(input_array)
    assert expect == output

    # Test:get_image_num_channels img(Tensor shape is HWC)
    apple_image = os.path.join(TEST_DATA_DATASET_FUNC, "apple.jpg")
    expect = [2268, 4032]
    img = Image.open(apple_image)
    output_size = vision_utils.get_image_size(img)
    img.close()
    assert expect == output_size


def test_get_image_size_exception_01():
    """
    Feature: get_image_num_channels operation
    Description: Testing the get_image_num_channels Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test:test get_image_num_channels invalid input
    image_error = 1
    with pytest.raises(TypeError) as error_info:
        vision_utils.get_image_size(image_error)
    assert "Input image is not of type <class 'numpy.ndarray'> or <class 'PIL.Image.Image'>, " \
           "but got: <class 'int'>." in str(error_info.value)

    # Test:test get_image_num_channels invalid input float
    image_error = 1.0
    with pytest.raises(TypeError) as error_info:
        vision_utils.get_image_size(image_error)
    assert "Input image is not of type <class 'numpy.ndarray'> or <class 'PIL.Image.Image'>, " \
           "but got: <class 'float'>." in str(error_info.value)


if __name__ == "__main__":
    test_get_image_size_output_array()
    test_get_image_size_output_img()
    test_get_image_size_invalid_input()
    test_get_image_size_operation_01()
    test_get_image_size_exception_01()
