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
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False


def util_test_random_color_adjust_error(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
    """
    Util function that tests the error message in case of grayscale images
    """

    transforms = [
        vision.Decode(True),
        vision.Grayscale(1),
        vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = trans.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # if input is grayscale, the output dimensions should be single channel, the following should fail
    random_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)
    with pytest.raises(RuntimeError) as info:
        data1 = data1.map(operations=random_adjust_op, input_columns=["image"])
        dataset_shape_1 = []
        for item1 in data1.create_dict_iterator(num_epochs=1):
            c_image = item1["image"]
            dataset_shape_1.append(c_image.shape)

    error_msg = "Expecting tensor in channel of (3)"

    assert error_msg in str(info.value)


def util_test_random_color_adjust_op(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0), plot=False):
    """
    Util function that tests RandomColorAdjust for a specific argument
    """

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    random_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)

    ctrans = [decode_op,
              random_adjust_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                 hue=hue),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))

        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01

        if plot:
            visualize_image(c_image, py_image, mse)


def test_random_color_adjust_op_brightness(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for brightness
    Expectation: The dataset is processed as expected
    """

    logger.info("test_random_color_adjust_op_brightness")

    util_test_random_color_adjust_op(brightness=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_brightness_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for brightness input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_brightness_error")

    util_test_random_color_adjust_error(brightness=(0.5, 0.5))


def test_random_color_adjust_op_contrast(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for contrast
    Expectation: The dataset is processed as expected
    """

    logger.info("test_random_color_adjust_op_contrast")

    util_test_random_color_adjust_op(contrast=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_contrast_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for contrast input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_contrast_error")

    util_test_random_color_adjust_error(contrast=(0.5, 0.5))


def test_random_color_adjust_op_saturation(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for saturation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_color_adjust_op_saturation")

    util_test_random_color_adjust_op(saturation=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_saturation_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for saturation input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_saturation_error")

    util_test_random_color_adjust_error(saturation=(0.5, 0.5))


def test_random_color_adjust_op_hue(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for hue
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_color_adjust_op_hue")

    util_test_random_color_adjust_op(hue=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_hue_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for hue input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_hue_error")

    util_test_random_color_adjust_error(hue=(0.5, 0.5))


def test_random_color_adjust_md5():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("Test RandomColorAdjust with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_adjust_op = vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_adjust_op, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_color_adjust_01_c_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)
    filename = "random_color_adjust_01_py_result.npz"
    save_and_check_md5_pil(data2, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


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


def test_random_color_adjust_operation_02():
    """
    Feature: RandomColorAdjust operation
    Description: Testing the normal functionality of the RandomColorAdjust operator
    Expectation: The Output is equal to the expected output
    """
    # When all parameters are lists, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.RandomColorAdjust([0.1, 0.5], [0.1, 0.5], [0.1, 0.5], [0.1, 0.3]),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When no parameters are set, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_color_adjust_op = vision.RandomColorAdjust()
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the brightness parameter is not set, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    contrast = 1.0
    saturation = 1.0
    hue = (0.1, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(contrast=contrast, saturation=saturation, hue=hue)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the contrast parameter is not set, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    saturation = 1.0
    hue = (0.1, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, saturation=saturation, hue=hue)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the saturation parameter is not set, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 1.0
    hue = (0.1, 0.5)
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, hue=hue)
    dataset = dataset.map(input_columns=["image"], operations=random_color_adjust_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the hue parameter is not set, the RandomColorAdjust interface call succeeds.
    dataset = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    random_color_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation)
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


if __name__ == "__main__":
    test_random_color_adjust_op_brightness(plot=True)
    test_random_color_adjust_op_brightness_error()
    test_random_color_adjust_op_contrast(plot=True)
    test_random_color_adjust_op_contrast_error()
    test_random_color_adjust_op_saturation(plot=True)
    test_random_color_adjust_op_saturation_error()
    test_random_color_adjust_op_hue(plot=True)
    test_random_color_adjust_op_hue_error()
    test_random_color_adjust_md5()
    test_random_color_adjust_eager()
    test_random_color_adjust_invalid_dtype()
    test_random_color_adjust_operation_01()
    test_random_color_adjust_operation_02()
    test_random_color_adjust_operation_03()
    test_random_color_adjust_exception_01()
    test_random_color_adjust_exception_02()
    test_random_color_adjust_exception_03()
    test_random_color_adjust_exception_04()
