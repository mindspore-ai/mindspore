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
Testing ResizedCrop op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore import log as logger
from util import diff_mse, visualize_list

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_resized_crop_op(plot=False):
    """
    Feature: ResizedCrop op
    Description: Test ResizedCrop op basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("test_resized_crop_op")

    decode_op = vision.Decode()
    standard_resize_op = vision.Resize((512, 1024), vision.Inter.LINEAR)
    resized_crop_op = vision.ResizedCrop(128, 256, 256, 512, (224, 224), vision.Inter.LINEAR)

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=standard_resize_op, input_columns=["image"])
    data1 = data1.map(operations=resized_crop_op, input_columns=["image"])

    crop_op = vision.Crop((128, 256), (256, 512))
    resize_op = vision.Resize((224, 224), vision.Inter.LINEAR)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)


    data2 = data2.map(operations=decode_op, input_columns=["image"])
    data2 = data2.map(operations=standard_resize_op, input_columns=["image"])
    data2 = data2.map(operations=crop_op, input_columns=["image"])
    data2 = data2.map(operations=resize_op, input_columns=["image"])

    num_iter = 0
    crop_and_resize_images = []
    original_images = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        crop_and_resize = item1["image"]
        original = item2["image"]
        # Note: crop and resize original image with the same result as the one applied ResizedCrop
        mse = diff_mse(crop_and_resize, original)

        assert mse == 0
        logger.info("resized_corp_op_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        crop_and_resize_images.append(crop_and_resize)
        original_images.append(original)
    if plot:
        visualize_list(original_images, crop_and_resize_images)


def test_crop_and_resize_callable():
    """
    Feature: ResizedCrop op
    Description: Test op in eager mode
    Expectation: Output image shape from op is verified
    """
    logger.info("test_crop_resize_callable")
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    decode_op = vision.Decode()
    img = decode_op(img)
    assert img.shape == (2268, 4032, 3)

    resized_crop_op = vision.ResizedCrop(283, 1008, 567, 2016, (256, 512), vision.Inter.LINEAR)
    img1 = resized_crop_op(img)
    assert img1.shape == (256, 512, 3)


def test_resized_crop_op_invalid_input():
    """
    Feature: ResizedCrop op
    Description: Test ResizedCrop op with invalid input
    Expectation: Correct error is raised as expected
    """
    def test_invalid_input(test_name, top, left, height, width, size, interpolation, error, error_msg):
        logger.info("Test Resize with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision.ResizedCrop(top, left, height, width, size, interpolation)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid top parameter type as a single int",
                       2.1, 3, 4, 5, (3, 3), vision.Inter.LINEAR, TypeError,
                       "Argument top with value 2.1 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid left parameter type as a single int",
                       2, 3.1, 4, 5, (3, 3), vision.Inter.LINEAR, TypeError,
                       "Argument left with value 3.1 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid height parameter type as a single int",
                       2, 3, 4.1, 5, (3, 3), vision.Inter.LINEAR, TypeError,
                       "Argument height with value 4.1 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid width parameter type as a single int",
                       2, 3, 4, 5.1, (3, 3), vision.Inter.LINEAR, TypeError,
                       "Argument width with value 5.1 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid size parameter type as a single number",
                       2, 3, 4, 5, 3.5, vision.Inter.LINEAR, TypeError,
                       "Argument size with value 3.5 is not of type [<class 'int'>,"
                       " <class 'list'>, <class 'tuple'>], but got <class 'float'>.")
    test_invalid_input("invalid size parameter shape",
                       2, 3, 4, 5, (2, 3, 4), vision.Inter.LINEAR, TypeError,
                       "Size should be a single integer or a list/tuple (h, w) of length 2.")
    test_invalid_input("invalid Interpolation value", 2, 3, 4, 5, (2, 3), None, TypeError,
                       "Argument interpolation with value None is not of type [<enum 'Inter'>],"
                       " but got <class 'NoneType'>.")


def test_resized_crop_op_invalid_range():
    """
    Feature: ResizedCrop op
    Description: Test ResizedCrop op with invalid input range of [0, 2147483647]
    Expectation: Correct error is raised as expected
    """
    def test_invalid_range(test_name, top, left, height, width, size, interpolation, error, error_msg):
        logger.info("Test Resize with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
            decode_op = vision.Decode()
            crop_and_resize_op = vision.ResizedCrop(top, left, height, width, size, interpolation)
            data = data.map(operations=decode_op, input_columns=["image"])
            data = data.map(operations=crop_and_resize_op, input_columns=["image"])
        assert error_msg in str(error_info.value)

    logger.info("test_resized_crop_op_invalid_range")
    test_invalid_range("invalid top value range", -1, 1, 256, 512, (224, 224), vision.Inter.LINEAR, ValueError,
                       "Input top is not within the required interval of [0, 2147483647].")
    test_invalid_range("invalid left value range", 1, -1, 256, 512, (224, 224), vision.Inter.LINEAR, ValueError,
                       "Input left is not within the required interval of [0, 2147483647].")
    test_invalid_range("invalid height value range", 1, 1, 0, 512, (224, 224), vision.Inter.LINEAR, ValueError,
                       "Input height is not within the required interval of [1, 2147483647].")
    test_invalid_range("invalid width value range", 1, 1, 256, 0, (224, 224), vision.Inter.LINEAR, ValueError,
                       "Input width is not within the required interval of [1, 2147483647].")
    test_invalid_range("invalid width value range", 1, 1, 256, 512, (0, 224), vision.Inter.LINEAR, ValueError,
                       "Input is not within the required interval of [1, 16777216].")


def test_resized_crop_nearest():
    """
    Feature: ResizedCrop op
    Description: Test RandomCropAndResize with Python transformations where image interpolation mode is Inter.NEAREST
    Expectation: The dataset is processed as expected
    """
    logger.info("test_resized_crop_nearest")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resized_crop_op = vision.ResizedCrop(128, 256, 256, 512, (224, 224), vision.Inter.NEAREST)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resized_crop_op, input_columns=["image"])
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use RandomResizedCrop by vision.Inter.NEAREST process {} images.".format(num_iter))


if __name__ == "__main__":
    test_resized_crop_op_invalid_input()
    test_crop_and_resize_callable()
    test_resized_crop_op(plot=True)
    test_resized_crop_op_invalid_range()
    test_resized_crop_nearest()
