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
Test MindData vision utility get_image_num_channels
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.utils as vision_utils
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger

TEST_DATA_DATASET_FUNC ="../data/dataset/"


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


def test_get_image_num_channels_operation_01():
    """
    Feature: get_image_num_channels operation
    Description: Testing the normal functionality of the get_image_num_channels operator
    Expectation: The Output is equal to the expected output
    """
    # test func get_img_channel,eager model,input 3 channel Numpy data
    ex_output = 3
    img_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    img = np.fromfile(img_path, dtype=np.uint8)
    input_array = vision.Decode()(img)
    output = vision_utils.get_image_num_channels(input_array)
    assert output == ex_output

    # test func get_img_channel,pipeline model,input 3 channel PIL data
    img_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(img_path, shuffle=False, decode=True)
    op = [vision.ToPIL(), vision_utils.get_image_num_channels]
    data1 = dataset.map(operations=op)
    for i in data1.create_dict_iterator():
        out = i["image"]
        assert out == 3

    # test func get_img_channel,pipeline model,input 1 channel PIL data
    img_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(img_path, shuffle=False, decode=True)
    op = [vision.ToPIL(), vision.Grayscale(), vision_utils.get_image_num_channels]
    data1 = dataset.map(operations=op)
    for i in data1.create_dict_iterator():
        out = i["image"]
        assert out == 1

    # test func get_img_channel,eager model,input 1 channel PIL data
    ex_output = 1
    img = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(img) as image:
        # Convert the input PIL Image to grayscale.
        out_img = vision.Grayscale()(image)
        output = vision_utils.get_image_num_channels(out_img)
        assert output == ex_output

    # test func get_img_channel,eager model,input two-dimensional 1 channel Numpy data
    ex_output = 1
    # Input a two-dimensional array,1 channel
    out_img = np.array([[[1],
                         [1]],
                        [[1],
                         [1]]])
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output

    # test func get_img_channel,eager model,input three-dimensional 1 channel Numpy data
    ex_output = 1
    # Input a three-dimensional array.Single channel
    out_img = np.array([[1, 2, 3],
                        [1, 3, 2]])
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output

    # test func get_img_channel,eager model,input two-dimensional 2 channel Numpy data
    ex_output = 2
    # Input a two-dimensional array.
    out_img = np.array([[[1, 2],
                         [1, 3]],
                        [[1, 4],
                         [1, 5]]])
    output_ms = vision_utils.get_image_num_channels(out_img)

    assert output_ms == ex_output

    # test func get_img_channel,eager model,input three-dimensional 3 channel Numpy data
    ex_output = 3
    out_img = np.random.randn(20, 30, 3)
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output

    # test func get_img_channel,eager model,input 3 channel PIL data
    img_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(img_path) as img:
        expect_channel = 3
        output_channel = vision_utils.get_image_num_channels(img)
        assert expect_channel == output_channel

    # test func get_img_channel,eager model,input three-dimensional 4 channel Numpy data
    ex_output = 4
    # Input a three-dimensional array.
    out_img = np.array([[[1, 2, 3, 4],
                         [1, 2, 3, 4]],
                        [[1, 2, 3, 4],
                         [1, 2, 3, 4]]])
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output

    # test func get_img_channel,eager model,input four-dimensional 3 channel Numpy data
    ex_output = 3
    out_img = np.random.randn(20, 30, 3, 3)
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output


def test_get_image_num_channels_operation_02():
    """
    Feature: get_image_num_channels operation
    Description: Testing the normal functionality of the get_image_num_channels operator
    Expectation: The Output is equal to the expected output
    """
    # test func get_img_channel,eager model,input four-dimensional 4 channel Numpy data
    ex_output = 4
    out_img = np.random.randn(20, 30, 3, 4)
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output

    # test func get_img_channel,eager model,input seven-dimensional 7 channel Numpy data
    ex_output = 7
    out_img = np.random.randn(20, 30, 3, 4, 5, 6, 7)
    output_ms = vision_utils.get_image_num_channels(out_img)
    assert output_ms == ex_output


def test_get_image_num_channels_exception_01():
    """
    Feature: get_image_num_channels operation
    Description: Testing the get_image_num_channels Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test func get_img_channel,eager model,input one-dimensional Numpy data
    out_img = np.array([1, 2, 3])
    # One-dimensional array input
    with pytest.raises(RuntimeError, match='GetImageNumChannels: invalid parameter, '
                                           'image should have at least two dimensions, but got: 1'):
        vision_utils.get_image_num_channels(out_img)

    # test func get_img_channel,eager model,input int data
    with pytest.raises(TypeError, match="Input image is not of type <class 'numpy.ndarray'> "
                                        "or <class 'PIL.Image.Image'>, but got: <class 'int'>."):
        vision_utils.get_image_num_channels(1)

    # test func get_img_channel,eager model,input str data
    with pytest.raises(TypeError, match="Input image is not of type <class 'numpy.ndarray'> "
                                        "or <class 'PIL.Image.Image'>, but got: <class 'str'>."):
        vision_utils.get_image_num_channels("1")

    # test func get_img_channel,eager model,no input data
    with pytest.raises(TypeError, match=r"get_image_num_channels\(\) missing 1 required positional argument: 'image'"):
        vision_utils.get_image_num_channels()

    # test func get_img_channel,eager model,input NULL data
    with pytest.raises(TypeError, match="Input image is not of type <class 'numpy.ndarray'> "
                                        "or <class 'PIL.Image.Image'>, but got: <class 'str'>."):
        vision_utils.get_image_num_channels("")

    # test func get_img_channel,eager model,input tensor data
    out_img = [1, 2, 3]
    with pytest.raises(TypeError, match="Input image is not of type <class 'numpy.ndarray'> "
                                        "or <class 'PIL.Image.Image'>, but got: <class 'list'>."):
        vision_utils.get_image_num_channels(out_img)


if __name__ == "__main__":
    test_get_image_num_channels_output_array()
    test_get_image_num_channels_output_img()
    test_get_image_num_channels_invalid_input()
    test_get_image_num_channels_operation_01()
    test_get_image_num_channels_operation_02()
    test_get_image_num_channels_exception_01()
