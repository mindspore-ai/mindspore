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
Testing RandomColorAdjust in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False
























def test_random_color_adjust_eager():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with eager mode
    Expectation: Test runs successfully
    """
    image = np.random.random((28, 28, 3)).astype(np.float32)
    random_color_adjust = vision.RandomColorAdjust(contrast=0.5)
    out = random_color_adjust(image)
    assert out.shape == (28, 28, 3)


def test_random_color_adjust_invalid_dtype():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with invalid image dtype
    Expectation: RuntimeError raised
    """
    image = np.random.random((28, 28, 3)).astype(np.float64)

    # test AdjustContrast
    with pytest.raises(RuntimeError) as error_info:
        adjust_contrast = vision.RandomColorAdjust(contrast=0.5)
        _ = adjust_contrast(image)
    assert "Expecting tensor in type of (uint8, uint16, float32)" in str(error_info.value)

    # test AdjustSaturation
    with pytest.raises(RuntimeError) as error_info:
        image = np.random.random((28, 28, 3)).astype(np.float64)
        adjust_saturation = vision.RandomColorAdjust(saturation=2.0)
        _ = adjust_saturation(image)
    assert "Expecting tensor in type of (uint8, uint16, float32)" in str(error_info.value)


def test_random_color_adjust_operation_01():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the normal functionality of the RandomColorAdjust operator
    Expectation: The Output is equal to the expected output
    """
    # When the brightness parameter is set to 100.0, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 100.0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter contrast is 0.0, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 0.0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter contrast is set to 100.0, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 100.0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the saturation parameter is 0.0, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 1.0
    saturation = 0.0
    hue = 0.0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                    hue=hue)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the saturation parameter is set to 100.0, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 1.0
    saturation = 100.0
    hue = 0.5
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                    hue=hue)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass




def test_random_color_adjust_operation_03():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the normal functionality of the RandomColorAdjust operator
    Expectation: The Output is equal to the expected output
    """
    # When the input image format is BMP, the RandomColorAdjust interface call succeeds.
    with Image.open(image_bmp) as image:
        brightness = 1024
        contrast = (0.01, 0.02)
        saturation = 0.5
        hue = 0.5
        random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                                   saturation=saturation, hue=hue)
        _ = random_color_adjust_op(image)

    # When the input image format is PNG, the RandomColorAdjust interface call succeeds.
    image = cv2.imread(image_png)
    brightness = 0
    contrast = 0
    saturation = 0
    hue = 0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    _ = random_color_adjust_op(image)

    # When the brightness parameter is 100.4, the RandomColorAdjust interface call succeeds.
    image = np.random.randn(256, 256, 3).astype(np.uint8)
    brightness = 100.4
    contrast = 3.2
    saturation = 16.8
    hue = 0.3856
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    _ = random_color_adjust_op(image)

    # When the brightness parameter is (1, 10.4), the RandomColorAdjust interface call succeeds.
    image = cv2.imread(image_file)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    _ = random_color_adjust_op(image)

    # When the brightness parameter is 16777216, the RandomColorAdjust interface call succeeds.
    image = np.random.randint(0, 255, (464, 864, 3)).astype(np.uint8)
    brightness = (100.4, 16777216)
    contrast = (100.4, 16777216)
    saturation = (100.4, 16777216)
    hue = (0.4, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    _ = random_color_adjust_op(image)

    # When the input image format is JPG, the RandomColorAdjust interface call succeeds.
    image = cv2.imread(image_file)
    random_color_adjust_op = vision.RandomColorAdjust()
    _ = random_color_adjust_op(image)


def test_random_color_adjust_exception_01():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the RandomColorAdjust Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the brightness parameter is negative, the RandomColorAdjust interface call fails.
    brightness = -1.0
    with pytest.raises(ValueError, match="The input value of brightness cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is (0.5, 0.1), the RandomColorAdjust interface call fails.
    brightness = (0.5, 0.1)
    with pytest.raises(ValueError, match="brightness value should be in \\(min,max\\) format. Got \\(0.5, 0.1\\)."):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is a string, the RandomColorAdjust interface call fails.
    brightness = ("", "")
    with pytest.raises(TypeError, match="not supported"):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is empty, the RandomColorAdjust interface call fails.
    brightness = ""
    with pytest.raises(TypeError, match="Argument brightness"):
        vision.RandomColorAdjust(brightness=brightness)

    # When the contrast parameter is negative, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = -1.0
    with pytest.raises(ValueError, match="The input value of contrast cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # The first value of the parameter contrast is greater than the second value.
    brightness = 1.0
    contrast = (0.5, 0.1)
    with pytest.raises(ValueError, match="contrast value should be in \\(min,max\\) format. Got \\(0.5, 0.1\\)."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the parameter contrast is (“”, “”), the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = ("", "")
    with pytest.raises(TypeError, match="not supported"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the contrast parameter is empty, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = ""
    with pytest.raises(TypeError, match="Argument contrast"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the saturation parameter is negative, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = -1.0
    with pytest.raises(ValueError, match="The input value of saturation cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # The first value of the saturation parameter is greater than the second value.
    brightness = 1.0
    contrast = 1.0
    saturation = (0.5, 0.1)
    with pytest.raises(ValueError, match="saturation value should be in \\(min,max\\) format. Got \\(0.5, 0.1\\)."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the parameter saturation is (“”, “”), the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = ("", "")
    with pytest.raises(TypeError, match="not supported"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the saturation parameter is empty, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = ""
    with pytest.raises(TypeError, match="Argument saturation"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the hue parameter exceeds 0.5, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = 100.0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # When the hue parameter is negative, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = -1.0
    with pytest.raises(ValueError, match="The input value of hue cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # The first value of the hue parameter is greater than the second value.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = (0.5, 0.1)
    with pytest.raises(ValueError, match="hue value should be in \\(min,max\\) format. Got \\(0.5, 0.1\\)."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)


def test_random_color_adjust_exception_02():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the RandomColorAdjust Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the hue parameter is an empty tuple, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = ("", "")
    with pytest.raises(TypeError, match="not supported"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # When the hue parameter is empty, the RandomColorAdjust interface call fails.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = ""
    with pytest.raises(TypeError, match="Argument hue"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # When the input image format is GIF, the RandomColorAdjust interface call fails.
    with Image.open(image_gif) as image:
        random_color_adjust_op = vision.RandomColorAdjust()
        with pytest.raises(ValueError):
            random_color_adjust_op(image)

    # When the input channel is 1, the RandomColorAdjust interface call fails.
    image = np.random.randint(0, 255, (464, 864, 1)).astype(np.uint8)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)

    with pytest.raises(RuntimeError, match=r"RandomColorAdjust: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in channel "
                                           r"of \(3\). But got channel 1."):
        random_color_adjust_op(image)

    # When the input channel is 4, the RandomColorAdjust interface call fails.
    image = np.random.randint(0, 255, (464, 864, 4)).astype(np.uint8)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    with pytest.raises(RuntimeError, match=r"RandomColorAdjust: the channel of image tensor does "
                                           r"not match the requirement of operator. Expecting tensor in channel "
                                           r"of \(3\). But got channel 4."):
        random_color_adjust_op(image)

    # When inputting 4D data, the RandomColorAdjust interface call failed.
    image = np.random.randint(0, 255, (464, 864, 3, 3)).astype(np.uint8)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    with pytest.raises(RuntimeError, match=r"RandomColorAdjust: the dimension of image tensor does"
                                           r" not match the requirement of operator. Expecting tensor in dimension "
                                           r"of \(3\), in shape of <H, W, C>. But got dimension 4."):
        random_color_adjust_op(image)

    # When inputting 2D data, the RandomColorAdjust interface call fails.
    image = np.random.randint(0, 255, (464, 464)).astype(np.uint8)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    with pytest.raises(RuntimeError, match=r"RandomColorAdjust: the dimension of image tensor does"
                                           r" not match the requirement of operator. Expecting tensor in dimension "
                                           r"of \(3\), in shape of <H, W, C>. But got dimension 2."):
        random_color_adjust_op(image)

    # When input data is float16, the RandomColorAdjust interface call fails.
    image = np.random.randint(0, 255, (464, 464, 3)).astype(np.float16)
    brightness = (1, 10.4)
    contrast = 0.2
    saturation = (0, 10.8)
    hue = (-0.5, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast,
                                               saturation=saturation, hue=hue)
    with pytest.raises(RuntimeError):
        random_color_adjust_op(image)

    # When the brightness parameter is a NumPy array, the RandomColorAdjust interface call fails.
    brightness = np.array([1, 10.4])
    with pytest.raises(TypeError, match="Argument brightness with value \\[ 1.  10.4\\] is not of type"):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter exceeds 16777216, the RandomColorAdjust interface call fails.
    brightness = (1.05, 16777216.001)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColorAdjust(brightness=brightness)


def test_random_color_adjust_exception_03():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the RandomColorAdjust Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Setting unsupported parameters causes the RandomColorAdjust interface call to fail.
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    hue = (0.1, 0.5)
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomColorAdjust(brightness, contrast, saturation, hue, more_para)

    # When the brightness parameter is a 3-tuple, the RandomColorAdjust interface call fails.
    brightness = (1, 10.4, 15.0)
    with pytest.raises(TypeError, match="If brightness is a sequence, the length must be 2."):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is set, the RandomColorAdjust interface call fails.
    brightness = {1, 10.4}
    with pytest.raises(TypeError, match="Argument brightness with value {1, 10.4} is not of type"):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is negative, the RandomColorAdjust interface call fails.
    brightness = (-0.01, 10.4)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColorAdjust(brightness=brightness)

    # When the brightness parameter is negative, the RandomColorAdjust interface call fails.
    brightness = (0.01, 10.4)
    contrast = -0.01
    with pytest.raises(ValueError, match="The input value of contrast cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the parameter "contrast" is a 1-tuple, the RandomColorAdjust interface call fails.
    brightness = (0.01, 10.4)
    contrast = (0.01,)
    with pytest.raises(TypeError, match="If contrast is a sequence, the length must be 2."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the parameter `contrast` is a tensor, the RandomColorAdjust interface call fails.
    brightness = (0.01, 10.4)
    contrast = ms.Tensor([0, 0.5])
    with pytest.raises(TypeError) as e:
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)
    assert "Argument contrast with value {}".format(contrast) in str(e)
    assert "[<class 'numbers.Number'>, <class 'list'>, <class 'tuple'>], but got" in str(e)

    # When the parameter contrast exceeds 16777216, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 16777216.1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast)

    # When the saturation parameter is negative, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = -0.1
    with pytest.raises(ValueError, match="The input value of saturation cannot be negative."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the saturation parameter is negative, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = (-1, 2.5)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the parameter saturation is set to set, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = {1, 2.5}
    with pytest.raises(TypeError, match="Argument saturation with value {1, 2.5} is not of type "):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the parameter saturation is an np array, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = np.array([1, 2.5])
    with pytest.raises(TypeError, match="Argument saturation with value \\[1.  2.5\\] is not of type"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the saturation parameter is a 3-tuple, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = (1, 2.5, 5.0)
    with pytest.raises(TypeError, match="If saturation is a sequence, the length must be 2."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the parameter saturation is a string, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = "0.5"
    with pytest.raises(TypeError, match="Argument saturation with value 0.5 is not of type"):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)

    # When the saturation parameter exceeds 16777216, the RandomColorAdjust interface call fails.
    brightness = 2.4
    contrast = 3.2
    saturation = (0.2, 16777217)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)


def test_random_color_adjust_exception_04():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the RandomColorAdjust Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the hue parameter is negative, the RandomColorAdjust interface call fails.
    hue = -0.1
    with pytest.raises(ValueError, match="The input value of hue cannot be negative."):
        vision.RandomColorAdjust(hue=hue)

    # When the hue parameter is an np array, the RandomColorAdjust interface call fails.
    hue = np.array([-0.1, 0.1])
    with pytest.raises(TypeError, match="Argument hue with value \\[-0.1  0.1\\] is not of type"):
        vision.RandomColorAdjust(hue=hue)

    # When the hue parameter is less than -0.5, the RandomColorAdjust interface call fails.
    hue = [-0.51, 0.3]
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[-0.5, 0.5\\]."):
        vision.RandomColorAdjust(hue=hue)

    # When the hue parameter exceeds 0.5, the RandomColorAdjust interface call fails.
    hue = (-0.1, 0.6)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[-0.5, 0.5\\]."):
        vision.RandomColorAdjust(hue=hue)

    # When the hue parameter is a 3-tuple, the RandomColorAdjust interface call fails.
    hue = (-0.1, 0.2, 0.3)
    with pytest.raises(TypeError, match="If hue is a sequence, the length must be 2."):
        vision.RandomColorAdjust(hue=hue)
