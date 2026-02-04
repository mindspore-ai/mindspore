# Copyright 2025 Huawei Technologies Co., Ltd.
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
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
image_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")


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


def test_ten_crop_operation_01():
    """
    Feature: TenCrop operation
    Description: Testing the normal functionality of the TenCrop operator
    Expectation: The Output is equal to the expected output
    """
    # TenCrop operation: size=10, use_vertical_flip=False
    size = 10
    use_vertical_flip = False
    dataset = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    dataset = dataset.map(input_columns=["image"], operations=transforms)
    for data in dataset.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operation: size = (10, 30), use_vertical_flip = False
    size = (10, 30)
    use_vertical_flip = False
    dataset = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    dataset = dataset.map(input_columns=["image"], operations=transforms)
    for data in dataset.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operation: size = (10, 30), use_vertical_flip = True
    size = (10, 30)
    use_vertical_flip = True
    dataset = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    dataset = dataset.map(input_columns=["image"], operations=transforms)

    for data in dataset.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operation: size = [10, 20], use_vertical_flip = False
    size = [10, 20]
    use_vertical_flip = False
    ds2 = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    ds2 = ds2.map(input_columns=["image"], operations=transforms)

    for data in ds2.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operation: size = [150, 5], use_vertical_flip = False
    size = [150, 5]
    use_vertical_flip = False
    ds2 = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    ds2 = ds2.map(input_columns=["image"], operations=transforms)

    for data in ds2.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operation: size = [10, 20], use_vertical_flip = True
    size = [10, 20]
    use_vertical_flip = True
    ds2 = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    ds2 = ds2.map(input_columns=["image"], operations=transforms)

    for data in ds2.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10


def test_ten_crop_operation_02():
    """
    Feature: TenCrop operation
    Description: Testing the normal functionality of the TenCrop operator
    Expectation: The Output is equal to the expected output
    """
    # TenCrop operation: size = [150, 5], use_vertical_flip = True
    size = [150, 5]
    use_vertical_flip = True
    ds2 = ds.ImageFolderDataset(DATA_DIR_1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.TenCrop(size=size, use_vertical_flip=use_vertical_flip),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])
    ]
    ds2 = ds2.map(input_columns=["image"], operations=transforms)
    for data in ds2.create_dict_iterator(output_numpy=True):
        image = data["image"]
        assert len(image.shape) == 4
        assert image.shape[0] == 10

    # TenCrop operator in eager mode: size = [50, 60], use_vertical_flip = False
    image = Image.open(image_file)
    size = [50, 60]
    use_vertical_flip = False
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 60, 3)

    # TenCrop operator in eager mode: size = [50, 60], use_vertical_flip = False
    image = Image.open(image_file2)
    size = [50, 60]
    use_vertical_flip = False
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 60, 3)

    # TenCrop operator in eager mode: input is GIF
    image = Image.open(image_file1)
    size = (50, 60)
    use_vertical_flip = True
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 60)

    # TenCrop operator in eager mode: input is JPG, use_vertical_flip = True
    use_vertical_flip = True
    image = Image.open(image_file)
    size = (50, 60)
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 60, 3)

    # TenCrop operator in eager mode: input is JPG, use_vertical_flip = False
    use_vertical_flip = False
    image = Image.open(image_file)
    size = (50, 60)
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 60, 3)

    # TenCrop operator in eager mode: input is GIF
    image = Image.open(image_file1)
    size = 50
    ten_crop_op = vision.TenCrop(size)
    out = ten_crop_op(image)
    for i in out:
        assert np.array(i).shape == (50, 50)


def test_ten_crop_exception_01():
    """
    Feature: TenCrop operation
    Description: Testing the TenCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # TenCrop operator exception: size larger than the original image
    image = Image.open(image_file1)
    size = (1000, 1000)
    use_vertical_flip = True
    ten_crop_op = vision.TenCrop(size, use_vertical_flip)
    with pytest.raises(ValueError, match=r'Crop size \(1000, 1000\) is larger than input image size \(283, 212\)'):
        ten_crop_op(image)

    # TenCrop operator exception: size is float
    size = [0.5]
    with pytest.raises(TypeError, match=r'Size should be a single integer or a list\/tuple \(h, w\) of length 2'):
        vision.TenCrop(size)

    # TenCrop operator exception: size is 3-dimensional
    size = [50, 50, 50]
    with pytest.raises(TypeError, match=r'Size should be a single integer or a list\/tuple \(h, w\) of length 2'):
        vision.TenCrop(size)

    # TenCrop operator exception: input is list
    image = list(np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8))
    size = [50, 50]
    ten_crop_op = vision.TenCrop(size)
    with pytest.raises(TypeError, match=r"img should be PIL image. Got <class 'list'>. Use "
                                        r"Decode\(\) for encoded data or ToPIL\(\) for decoded data."):
        ten_crop_op(image)

    # TenCrop operator exception: size is 0
    size = [0, 0]
    with pytest.raises(ValueError, match=r'Input is not within the required interval of \[1, 16777216\].'):
        vision.TenCrop(size)

    # TenCrop operator exception: size equals a negative number
    size = [-2, -1]
    with pytest.raises(ValueError, match=r'Input is not within the required interval of \[1, 16777216\]'):
        vision.TenCrop(size)

    # TenCrop operator exception: size is 2147483648
    size = [0, 2147483648]
    with pytest.raises(ValueError, match=r'Input is not within the required interval of \[1, 16777216\]'):
        vision.TenCrop(size)

    # TenCrop operator exception: use_vertical_flip is int
    size = [50, 50]
    use_vertical_flip = 2
    with pytest.raises(TypeError,
                       match=r'Argument use_vertical_flip with value 2 is not of type \[\<class \'bool\'\>\]'):
        vision.TenCrop(size, use_vertical_flip)

    # TenCrop operator exception: input is numpy
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    size = [50, 50]
    ten_crop_op = vision.TenCrop(size)
    with pytest.raises(TypeError, match=r'img should be PIL image. Got \<class \'numpy.ndarray\'\>'):
        ten_crop_op(image)


if __name__ == "__main__":
    test_ten_crop_op_square(plot=True)
    test_ten_crop_op_rectangle(plot=True)
    test_ten_crop_op_vertical_flip(plot=True)
    test_ten_crop_md5()
    test_ten_crop_list_size_error_msg()
    test_ten_crop_invalid_size_error_msg()
    test_ten_crop_wrong_img_error_msg()
    test_ten_crop_operation_01()
    test_ten_crop_operation_02()
    test_ten_crop_exception_01()
