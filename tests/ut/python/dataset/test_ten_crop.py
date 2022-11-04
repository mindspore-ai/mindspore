# Copyright 2020-2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing TenCrop in DE
"""
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def util_test_ten_crop(crop_size, vertical_flip=False, plot=False):
    """
    Utility function for testing TenCrop. Input arguments are given by other tests
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
    ]
    transform_1 = mindspore.dataset.transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(True),
        vision.TenCrop(crop_size, use_vertical_flip=vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 10 images
    ]
    transform_2 = mindspore.dataset.transforms.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        if plot:
            visualize_list(np.array([image_1] * 10), (image_2 * 255).astype(np.uint8).transpose(0, 2, 3, 1))

        # The output data should be of a 4D tensor shape, a stack of 10 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 10


def test_ten_crop_op_square(plot=False):
    """
    Feature: TenCrop op
    Description: Test TenCrop op for a square crop
    Expectation: Output's shape is equal to the expected output's shape
    """

    logger.info("test_ten_crop_op_square")
    util_test_ten_crop(200, plot=plot)


def test_ten_crop_op_rectangle(plot=False):
    """
    Feature: TenCrop op
    Description: Test TenCrop op for a rectangle crop
    Expectation: Output's shape is equal to the expected output's shape
    """

    logger.info("test_ten_crop_op_rectangle")
    util_test_ten_crop((200, 150), plot=plot)


def test_ten_crop_op_vertical_flip(plot=False):
    """
    Feature: TenCrop op
    Description: Test TenCrop op with vertical flip set to True
    Expectation: Output's shape is equal to the expected output's shape
    """

    logger.info("test_ten_crop_op_vertical_flip")
    util_test_ten_crop(200, vertical_flip=True, plot=plot)


def test_ten_crop_md5():
    """
    Feature: TenCrop op
    Description: Test TenCrop op for giving the same results in multiple run for a specific input (since deterministic)
    Expectation: Passes the md5 check test
    """
    logger.info("test_ten_crop_md5")

    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(True),
        vision.TenCrop((200, 100), use_vertical_flip=True),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 10 images
    ]
    transform_2 = mindspore.dataset.transforms.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "ten_crop_01_result.npz"
    save_and_check_md5_pil(data2, filename, generate_golden=GENERATE_GOLDEN)


def test_ten_crop_list_size_error_msg():
    """
    Feature: TenCrop op
    Description: Test TenCrop op when size arg has more than 2 elements
    Expectation: Error is raised as expected
    """
    logger.info("test_ten_crop_list_size_error_msg")

    with pytest.raises(TypeError) as info:
        _ = [
            vision.Decode(True),
            vision.TenCrop([200, 200, 200]),
            lambda images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 10 images
        ]
    error_msg = "Size should be a single integer or a list/tuple (h, w) of length 2."
    assert error_msg == str(info.value)


def test_ten_crop_invalid_size_error_msg():
    """
    Feature: TenCrop op
    Description: Test TenCrop op when size arg is not positive
    Expectation: Error is raised as expected
    """
    logger.info("test_ten_crop_invalid_size_error_msg")

    with pytest.raises(ValueError) as info:
        _ = [
            vision.Decode(True),
            vision.TenCrop(0),
            lambda images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 10 images
        ]
    error_msg = "Input is not within the required interval of [1, 16777216]."
    assert error_msg == str(info.value)

    with pytest.raises(ValueError) as info:
        _ = [
            vision.Decode(True),
            vision.TenCrop(-10),
            lambda images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 10 images
        ]

    assert error_msg == str(info.value)


def test_ten_crop_wrong_img_error_msg():
    """
    Feature: TenCrop op
    Description: Test TenCrop op when the input image is not in the correct format
    Expectation: Error is raised as expected
    """

    logger.info("test_ten_crop_wrong_img_error_msg")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.TenCrop(200),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    with pytest.raises(RuntimeError) as info:
        data.create_tuple_iterator(num_epochs=1).__next__()
    error_msg = \
        "map operation: [ToTensor] failed. The op is OneToOne, can only accept one tensor as input."
    assert error_msg in str(info.value)


if __name__ == "__main__":
    test_ten_crop_op_square(plot=True)
    test_ten_crop_op_rectangle(plot=True)
    test_ten_crop_op_vertical_flip(plot=True)
    test_ten_crop_md5()
    test_ten_crop_list_size_error_msg()
    test_ten_crop_invalid_size_error_msg()
    test_ten_crop_wrong_img_error_msg()
