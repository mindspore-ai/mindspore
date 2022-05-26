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
Test MindData vision utility get_image_num_channels
"""
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset.vision.utils as vision_utils
import mindspore.dataset.vision as vision
from mindspore import log as logger


def test_get_image_num_channels_output_array():
    """
    Feature: get_image_num_channels array
    Description: Test get_image_num_channels
    Expectation: The returned result is as expected
    """
    expect_output = 3
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    input_array = vision.Decode()(img)
    output = vision_utils.get_image_num_channels(input_array)
    assert expect_output == output


def test_get_image_num_channels_output_img():
    """
    Feature: get_image_num_channels img
    Description: Test get_image_num_channels
    Expectation: The returned result is as expected
    """
    testdata = "../data/dataset/apple.jpg"
    img = Image.open(testdata)
    expect_channel = 3
    output_channel = vision_utils.get_image_num_channels(img)
    assert expect_channel == output_channel


def test_get_image_num_channels_invalid_input():
    """
    Feature: get_image_num_channels
    Description: Test get_image_num_channels invalid input
    Expectation: Correct error is raised as expected
    """

    def test_invalid_input(test_name, image, error, error_msg):
        logger.info("Test get_image_num_channels with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision_utils.get_image_num_channels(image)
        assert error_msg in str(error_info.value)

    invalid_input = 1
    invalid_shape = np.array([1, 2, 3])
    test_invalid_input("invalid input", invalid_input, TypeError,
                       "Input image is not of type <class 'numpy.ndarray'> or <class 'PIL.Image.Image'>, "
                       "but got: <class 'int'>.")
    test_invalid_input("invalid input", invalid_shape, RuntimeError,
                       "GetImageNumChannels: invalid parameter, image should have at least two dimensions, but got: 1")


if __name__ == "__main__":
    test_get_image_num_channels_output_array()
    test_get_image_num_channels_output_img()
    test_get_image_num_channels_invalid_input()
