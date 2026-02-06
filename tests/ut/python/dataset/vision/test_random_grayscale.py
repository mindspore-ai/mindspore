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
Testing RandomGrayscale op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import save_and_check_md5_pil, visualize_list, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def test_random_grayscale_valid_prob(plot=False):
    """
    Feature: RandomGrayscale op
    Description: Test RandomGrayscale op with valid input
    Expectation: Passes the test
    """
    logger.info("test_random_grayscale_valid_prob")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        # Note: prob is 1 so the output should always be grayscale images
        vision.RandomGrayscale(1),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = trans.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_gray = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_gray.append(image1)
        image.append(image2)
    if plot:
        visualize_list(image, image_gray)


def test_random_grayscale_input_grayscale_images():
    """
    Feature: RandomGrayscale op
    Description: Test RandomGrayscale op with valid parameter with grayscale images as input
    Expectation: Passes the test
    """
    logger.info("test_random_grayscale_input_grayscale_images")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.Grayscale(1),
        # Note: If the input images is grayscale image with 1 channel.
        vision.RandomGrayscale(0.5),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = trans.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_gray = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_gray.append(image1)
        image.append(image2)

        assert len(image1.shape) == 3
        assert image1.shape[2] == 1
        assert len(image2.shape) == 3
        assert image2.shape[2] == 3

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_md5_valid_input():
    """
    Feature: RandomGrayscale op
    Description: Test RandomGrayscale with md5 comparison and valid parameter
    Expectation: Passes the md5 comparison test
    """
    logger.info("test_random_grayscale_md5_valid_input")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomGrayscale(0.8),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Check output images with md5 comparison
    filename = "random_grayscale_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_md5_no_param():
    """
    Feature: RandomGrayscale op
    Description: Test RandomGrayscale op with no parameter given
    Expectation: Passes the test
    """
    logger.info("test_random_grayscale_md5_no_param")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomGrayscale(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Check output images with md5 comparison
    filename = "random_grayscale_02_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_grayscale_invalid_param():
    """
    Feature: RandomGrayscale op
    Description: Test RandomGrayscale op with invalid parameter given
    Expectation: Error is raised as expected
    """
    logger.info("test_random_grayscale_invalid_param")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            vision.RandomGrayscale(1.5),
            vision.ToTensor()
        ]
        transform = trans.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_gray_scale_operation_01():
    """
    Feature: RandomGrayscale operation
    Description: Testing the normal functionality of the RandomGrayscale operator
    Expectation: The Output is equal to the expected output
    """
    # When using default parameter values, the RandomGrayscale interface is successfully called
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        vision.Decode(to_pil=True),
        vision.RandomGrayscale(),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When the input image format is jpg, the RandomGrayscale interface is successfully called
    with Image.open(image_jpg) as image:
        prob = 1.0
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        _ = random_erasing_op(image)

    # When the input image format is bmp, the RandomGrayscale interface is successfully called
    with Image.open(image_bmp) as image:
        prob = 0.8
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        _ = random_erasing_op(image)

    # When the input image format is png, the RandomGrayscale interface is successfully called
    with Image.open(image_png) as image:
        prob = 0.3
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        _ = random_erasing_op(image)

    # When the input image format is gif, the RandomGrayscale interface is successfully called
    with Image.open(image_gif) as image:
        prob = 0.9999999
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        _ = random_erasing_op(image)

    # When parameter prob is 0, the output grayscale image from the RandomGrayscale interface call is consistent with the PIL image
    with Image.open(image_jpg) as image:
        prob = 0
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        out = random_erasing_op(image)
        assert (np.array(image) == out).all


def test_random_gray_scale_exception_01():
    """
    Feature: RandomGrayscale operation
    Description: Testing the RandomGrayscale Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the input is numpy.ndarray, the RandomGrayscale interface call fails
    image = np.random.randint(0, 255, (300, 350, 3)).astype(np.uint8)
    prob = 1.0
    random_erasing_op = vision.RandomGrayscale(prob=prob)
    with pytest.raises(AttributeError):
        random_erasing_op(image)

    # When the input is a list, the RandomGrayscale interface call fails
    with Image.open(image_jpg) as image1:
        image = np.array(image1).tolist()
        prob = 1.0
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        with pytest.raises(AttributeError):
            random_erasing_op(image)

    # When parameter prob is greater than 1.0, the RandomGrayscale interface call fails
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is negative, the RandomGrayscale interface call fails
    prob = -0.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is a list, the RandomGrayscale interface call fails
    prob = [0.5]
    with pytest.raises(TypeError, match="Argument prob with value \\[0.5\\] is not of "
                                        "type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is a string, the RandomGrayscale interface call fails
    prob = "0.5"
    with pytest.raises(TypeError, match="Argument prob with value 0.5 is not of "
                                        "type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is of bool type, the RandomGrayscale interface call fails
    prob = True
    with pytest.raises(TypeError, match="Argument prob with value True is not of "
                                        "type \\(<class 'float'>, <class 'int'>\\)."):
        vision.RandomGrayscale(prob=prob)


if __name__ == "__main__":
    test_random_grayscale_valid_prob(True)
    test_random_grayscale_input_grayscale_images()
    test_random_grayscale_md5_valid_input()
    test_random_grayscale_md5_no_param()
    test_random_grayscale_invalid_param()
    test_random_gray_scale_operation_01()
    test_random_gray_scale_exception_01()
