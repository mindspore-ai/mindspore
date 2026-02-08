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
Testing Resize op in DE
"""
import cv2
import numpy as np
import os
import time
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.common.tensor import Tensor
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers, diff_mse

TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
DATA_HIGH = [[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]]
DATA_LOW = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
DATA_IMG = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
DATA_SECOND = [1, 2, 3, 4, 5, 6]
expect_output_one = [[[[1, 2, 3]], [[4, 5, 6]]], [[[7, 8, 9]], [[10, 11, 12]]]]
expect_output_two = [[[[1, 2, 3]]], [[[7, 8, 9]]]]
expect_output_three = [[1, 2], [3, 4], [5, 6]]
expect_output_four = [[[1], [2]], [[3], [4]], [[5], [6]]]

GENERATE_GOLDEN = False


def test_resize_op(plot=False):
    """
    Feature: Resize op
    Description: Test Resize op basic usage
    Expectation: The dataset is processed as expected
    """

    def test_resize_op_parameters(test_name, size, interpolation, plot):
        """
        Test resize_op
        """
        logger.info("Test resize: {0}".format(test_name))
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

        # define map operations
        decode_op = vision.Decode()
        resize_op = vision.Resize(size, interpolation)

        # apply map operations on images
        data1 = data1.map(operations=decode_op, input_columns=["image"])
        data2 = data1.map(operations=resize_op, input_columns=["image"])
        image_original = []
        image_resized = []
        for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_1 = item1["image"]
            image_2 = item2["image"]
            image_original.append(image_1)
            image_resized.append(image_2)
        if plot:
            visualize_list(image_original, image_resized)

    test_resize_op_parameters("Test single int for size", 100, Inter.LINEAR, plot=plot)
    test_resize_op_parameters("Test tuple for size", (100, 300), Inter.BILINEAR, plot=plot)
    test_resize_op_parameters("Test single int for size", 200, Inter.AREA, plot=plot)
    test_resize_op_parameters("Test single int for size", 400, Inter.PILCUBIC, plot=plot)


def test_resize_4d_input_1_size():
    """
    Feature: Resize
    Description: Test resize with 4 dimension input and one size parameter
    Expectation: resize successfully
    """
    logger.info("Test resize: Test single int for size with 4 dimension input")

    input_np_original = np.array(DATA_LOW, dtype=np.float32)
    expect_output = np.array(expect_output_one, dtype=np.float32)
    shape = (2, 2, 1, 3)
    input_np_original = input_np_original.reshape(shape)
    resize_op = vision.Resize(1)
    vidio_de_resized = resize_op(input_np_original)
    mse = diff_mse(vidio_de_resized, expect_output)
    assert mse < 0.01


def test_resize_4d_input_2_size():
    """
    Feature: Resize
    Description: Test resize with 4 dimension input and two size parameter
    Expectation: resize successfully
    """
    logger.info("Test resize: Test tuple for size with 4 dimension input")

    input_np_original = np.array(DATA_LOW, dtype=np.float32)
    expect_output = np.array(expect_output_two, dtype=np.float32)
    shape = (2, 2, 1, 3)
    input_np_original = input_np_original.reshape(shape)
    resize_op = vision.Resize((1, 1))
    vidio_de_resized = resize_op(input_np_original)
    mse = diff_mse(vidio_de_resized, expect_output)
    assert mse < 0.01


def test_resize_2d_input_2_size():
    """
    Feature: Resize
    Description: Test resize with 2 dimension input and two size parameter
    Expectation: resize successfully
    """
    logger.info("Test resize: Test single int for size with 2 dimension input")

    input_np_original = np.array(DATA_SECOND, dtype=np.float32)
    expect_output = np.array(expect_output_three, dtype=np.float32)
    shape = (2, 3)
    input_np_original = input_np_original.reshape(shape)
    resize_op = vision.Resize((3, 2))
    vidio_de_resized = resize_op(input_np_original)
    mse = diff_mse(vidio_de_resized, expect_output)
    assert mse < 0.01


def test_resize_3d_input_2_size():
    """
    Feature: Resize
    Description: Test resize with 3 dimension input and two size parameter
    Expectation: resize successfully
    """
    logger.info("Test resize: Test single int for size with 3 dimension input")

    input_np_original = np.array(DATA_SECOND, dtype=np.float32)
    expect_output = np.array(expect_output_four, dtype=np.float32)
    shape = (2, 3, 1)
    input_np_original = input_np_original.reshape(shape)
    resize_op = vision.Resize((3, 2))
    vidio_de_resized = resize_op(input_np_original)
    mse = diff_mse(vidio_de_resized, expect_output)
    assert mse < 0.01


def test_resize_op_antialias():
    """
    Feature: Resize op
    Description: Test Resize op basic usage where image interpolation mode is Inter.ANTIALIAS
    Expectation: The dataset is processed as expected
    """
    logger.info("Test resize for ANTIALIAS")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode(True)
    resize_op = vision.Resize(20, Inter.ANTIALIAS)

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, resize_op, vision.ToTensor()], input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use Resize by Inter.ANTIALIAS process {} images.".format(num_iter))
    assert num_iter == 3


def run_test_resize_md5(test_name, size, filename, seed, expected_size, to_pil=True, plot=False):
    """
    Run Resize with md5 check for python and C op versions
    """
    logger.info("Test Resize with md5 check: {0}".format(test_name))
    original_seed = config_get_set_seed(seed)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # Generate dataset
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    compose_ops = ds.transforms.Compose([vision.Decode(to_pil=to_pil), vision.Resize(size)])
    transformed_data = dataset.map(operations=compose_ops, input_columns=["image"])
    # Compare with expected md5 from images
    if to_pil:
        save_and_check_md5_pil(transformed_data, filename, generate_golden=GENERATE_GOLDEN)
    else:
        save_and_check_md5(transformed_data, filename, generate_golden=GENERATE_GOLDEN)
    for item in transformed_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        resized_image = item["image"]
        assert resized_image.shape == expected_size
    if plot:
        image_original = []
        image_resized = []
        original_data = dataset.map(operations=vision.Decode(), input_columns=["image"])
        for item1, item2 in zip(original_data.create_dict_iterator(num_epochs=1, output_numpy=True),
                                transformed_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_1 = item1["image"]
            image_2 = item2["image"]
            image_original.append(image_1)
            image_resized.append(image_2)
        visualize_list(image_original, image_resized)
    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_resize_md5_c(plot=False):
    """
    Feature: Resize op C version
    Description: Test C Resize op using md5 check
    Expectation: Passes the md5 check test
    """
    run_test_resize_md5("Test single int for size", 5, "resize_01_result_c.npz",
                        5, (5, 8, 3), to_pil=False, plot=plot)
    run_test_resize_md5("Test tuple for size", (5, 7), "resize_02_result_c.npz",
                        7, (5, 7, 3), to_pil=False, plot=plot)


def test_resize_md5_py(plot=False):
    """
    Feature: Resize op py version
    Description: Test python Resize op using md5 check
    Expectation: Passes the md5 check test
    """
    run_test_resize_md5("Test single int for size", 5, "resize_01_result_py.npz",
                        5, (5, 8, 3), to_pil=True, plot=plot)
    run_test_resize_md5("Test tuple for size", (5, 7), "resize_02_result_py.npz",
                        7, (5, 7, 3), to_pil=True, plot=plot)


def test_resize_op_invalid_input():
    """
    Feature: Resize op
    Description: Test Resize op with invalid input
    Expectation: Correct error is raised as expected
    """

    def test_invalid_input(test_name, size, interpolation, error, error_msg):
        logger.info("Test Resize with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision.Resize(size, interpolation)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid size parameter type as a single number", 4.5, Inter.LINEAR, TypeError,
                       "Size should be a single integer or a list/tuple (h, w) of length 2.")
    test_invalid_input("invalid size parameter shape", (2, 3, 4), Inter.LINEAR, TypeError,
                       "Size should be a single integer or a list/tuple (h, w) of length 2.")
    test_invalid_input("invalid size parameter type in a tuple", (2.3, 3), Inter.LINEAR, TypeError,
                       "Argument size at dim 0 with value 2.3 is not of type [<class 'int'>]")
    test_invalid_input("invalid interpolation value", (2.3, 3), None, KeyError, "None")


def test_resize_op_exception_c_interpolation():
    """
    Feature: Resize
    Description: Test Resize with unsupported interpolation values for NumPy input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_resize_op_exception_c_interpolation")

    image = cv2.imread("../data/dataset/apple.jpg")

    with pytest.raises(TypeError) as error_info:
        resize_op = vision.Resize(size=(100, 200), interpolation=Inter.ANTIALIAS)
        _ = resize_op(image)
    assert "img should be PIL image. Got <class 'numpy.ndarray'>." in str(error_info.value)


def test_resize_op_exception_py_interpolation():
    """
    Feature: Resize
    Description: Test Resize with unsupported interpolation values for PIL input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_resize_op_exception_py_interpolation")

    image = Image.open("../data/dataset/apple.jpg").convert("RGB")

    with pytest.raises(TypeError) as error_info:
        resize_op = vision.Resize(size=123, interpolation=Inter.PILCUBIC)
        _ = resize_op(image)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        resize_op = vision.Resize(size=456, interpolation=Inter.AREA)
        _ = resize_op(image)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)


def test_resize_performance():
    """
    Feature: Resize
    Description: Test Resize performance in eager mode after optimize ndarray to cde.Tensor without memcpy
    Expectation: SUCCESS
    """

    input_apple_jpg = "../data/dataset/apple.jpg"
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode()(img_bytes)
    _ = vision.Resize(224)(img_decode)

    s = time.time()
    for _ in range(1000):
        _ = vision.Resize(224)(img_decode)
    assert (time.time() - s) < 2.5  # Probably around 1.9 seconds


def test_resize_operation_01():
    """
    Feature: Resize operation
    Description: Testing the normal functionality of the Resize operator
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test size is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 1
    resize_op = vision.Resize(size=size)
    dataset = dataset.map(input_columns=["image"], operations=resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test size is a list sequence of length 2
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 520]
    resize_op = vision.Resize(size=size)
    dataset = dataset.map(input_columns=["image"], operations=resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test size is a tuple sequence of length 2
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    resize_op = vision.Resize(size=size)
    dataset = dataset.map(input_columns=["image"], operations=resize_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.LINEAR and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.LINEAR
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.LINEAR and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.LINEAR
    decode = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.NEAREST and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.NEAREST
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.NEAREST and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.NEAREST
    decode = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.BICUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.BICUBIC
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_resize_operation_02():
    """
    Feature: Resize operation
    Description: Testing the normal functionality of the Resize operator
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test interpolation is Inter.BICUBIC and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.BICUBIC
    decode = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.PILCUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.PILCUBIC
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.CUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.CUBIC
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.CUBIC and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.CUBIC
    decode = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.AREA and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = Inter.AREA
    decode = vision.Decode()
    resize_op = vision.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.ANTIALIAS and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size, interpolation=Inter.ANTIALIAS)
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.BILINEAR and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = vision.Decode()
    resize_op = vision.Resize(size, interpolation=Inter.BILINEAR)
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test interpolation is Inter.BILINEAR and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size, interpolation=Inter.BILINEAR)
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_resize_operation_03():
    """
    Feature: Resize operation
    Description: Testing the normal functionality of the Resize operator
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = vision.Decode(to_pil=True)
    resize_op = vision.Resize(size)
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test input is 3d numpy array
    image = np.random.randn(1024, 1024, 3)
    size = (500, 520)
    resize_op = vision.Resize(size, Inter.LINEAR)
    out = resize_op(image)
    assert out.shape == (500, 520, 3)

    # Resize operator:Test input is 2d numpy array
    image = np.random.randn(1024, 1024)
    size = (500, 520)
    resize_op = vision.Resize(size, Inter.LINEAR)
    out = resize_op(image)
    assert out.shape == (500, 520)

    # Resize operator:Test input is jpg image
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_file) as image:
        size = (50, 60)
        interpolation = Inter.BILINEAR
        resize_op = vision.Resize(size, interpolation)
        out = resize_op(image)
        assert np.array(out).shape == (50, 60, 3)

    # Resize operator:Test input is bmp image
    image_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_file3) as image:
        size = (50, 60)
        interpolation = Inter.BILINEAR
        resize_op = vision.Resize(size, interpolation)
        out = resize_op(image)
        assert np.array(out).shape == (50, 60, 3)

    # Resize operator:Test input is png image
    image_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_file2) as image:
        size = (50, 60)
        interpolation = Inter.BILINEAR
        resize_op = vision.Resize(size, interpolation)
        out = resize_op(image)
        assert np.array(out).shape == (50, 60, 4)

    # Resize operator:Test input is gif image
    image_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_file1) as image:
        size = (1000, 1000)
        interpolation = Inter.ANTIALIAS
        resize_op = vision.Resize(size, interpolation)
        out = resize_op(image)
        assert np.array(out).shape == (1000, 1000)

    # Resize operator:Test input is image opened using the cv2 method
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_2.jpg")
    image = cv2.imread(image_file)
    size = 128
    resize_op = vision.Resize(size, Inter.BICUBIC)
    out = resize_op(image)
    if np.array(image).shape[0] < np.array(image).shape[1]:
        assert out.shape[0] == 128
        assert out.shape[1] == (out.shape[1] / out.shape[0] * 128)
    else:
        assert out.shape[0] == (out.shape[0] / out.shape[1] * 128)
        assert out.shape[1] == 128

    # Resize operator:Test input is numpy list
    image = np.random.randn(256, 188, 1).tolist()
    size = (100, 100)
    resize_op = vision.Resize(size, Inter.BICUBIC)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        out = resize_op(image)
        assert np.array(out).shape == (100, 100, 1)

    # Resize operator:输入 4d numpy array，维度已扩展
    image = np.random.randn(56, 88, 3, 3)
    size = (100, 100)
    resize_op = vision.Resize(size, Inter.BICUBIC)
    output = resize_op(image)
    assert output.shape == (56, 100, 100, 3)


def test_resize_operation_04():
    """
    Feature: Resize operation
    Description: Testing the normal functionality of the Resize operator
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test eager interpolation_c is Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_2.jpg")
    with Image.open(image_file) as image:
        size = (250, 300)
        interpolation_c = Inter.PILCUBIC
        interpolation_py = Inter.BICUBIC
        resize_c_op = vision.Resize(size, interpolation=interpolation_c)
        resize_py_op = vision.Resize(size, interpolation=interpolation_py)
        with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
            out_c = resize_c_op(image)
            out_py = resize_py_op(image)
            assert (out_c == out_py).all()

    # Resize operator:Test eager interpolation_c is Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_2.jpg")
    with Image.open(image_file) as image:
        size = (np.array(image).shape[0], np.array(image).shape[1])
        interpolation_c = Inter.PILCUBIC
        interpolation_py = Inter.BICUBIC
        resize_c_op = vision.Resize(size, interpolation=interpolation_c)
        resize_py_op = vision.Resize(size, interpolation=interpolation_py)
        with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
            out_c = resize_c_op(image)
            out_py = resize_py_op(image)
            assert (out_c == np.array(out_py)).all()
            assert (np.array(image) == np.array(out_c)).all()

    # Resize operator:Test eager interpolation_c is Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_2.jpg")
    with Image.open(image_file) as image:
        size = [2500, 3000]
        interpolation_c = Inter.PILCUBIC
        interpolation_py = Inter.BICUBIC
        resize_c_op = vision.Resize(size, interpolation=interpolation_c)
        resize_py_op = vision.Resize(size, interpolation=interpolation_py)
        with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
            out_c = resize_c_op(image)
            out_py = resize_py_op(image)
            assert (out_c == out_py).all()
            assert np.array(out_c).shape == (2500, 3000, 3)


def test_resize_exception_01():
    """
    Feature: Resize operation
    Description: Testing the Resize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Resize operator:Test size is 0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is 16777216
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 16777216
    with pytest.raises(RuntimeError,
                       match="Resize: the resizing width or height is too big, it's 1000 times bigger than"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is 16777217
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is float
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 500.5
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is None
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = None
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is str
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 'test'
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is ""
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = ""
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_resize_exception_02():
    """
    Feature: Resize operation
    Description: Testing the Resize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Resize operator:Test size is a sequence of length 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence of length 3
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 500, 520]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing a float of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500.5, 500]
    with pytest.raises(TypeError, match="Argument size at dim 0 with value 500.5 is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'float'>"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing a str of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 'test']
    with pytest.raises(TypeError, match="Argument size at dim 1 with value test is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing bool of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, True]
    with pytest.raises(TypeError, match="Argument size at dim 1 with value True is not of type " + \
                                        "\\(<class 'int'>,\\), but got <class 'bool'>"):
        resize_op = vision.Resize(size=size)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is ""
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = ""
    with pytest.raises(TypeError, match="Argument interpolation with value \"\" is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        resize_op = vision.Resize(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_resize_exception_03():
    """
    Feature: Resize operation
    Description: Testing the Resize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Resize operator:Test interpolation is str
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = "test"
    with pytest.raises(TypeError, match="Argument interpolation with value test is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        resize_op = vision.Resize(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is None
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = None
    with pytest.raises(KeyError, match="Interpolation should not be None"):
        resize_op = vision.Resize(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is bool
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = True
    with pytest.raises(TypeError, match="Argument interpolation with value True is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'bool'>"):
        resize_op = vision.Resize(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test no para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        resize_op = vision.Resize()
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test more para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = Inter.LINEAR
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        resize_op = vision.Resize(size, interpolation, more_para)
        dataset = dataset.map(input_columns=["image"], operations=resize_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test input is a tensor
    image = Tensor(np.random.randn(10, 10, 3))
    size = (100, 100)
    resize_op = vision.Resize(size, Inter.BICUBIC)
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>"):
        resize_op(image)

    # Resize operator:Test size is tensor
    image = np.random.randn(256, 188, 1)
    size = Tensor([128, 128])
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = vision.Resize(size, Inter.BICUBIC)
        resize_op(image)

    # Resize operator:Test no image is transferred
    size = [128, 128]
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        resize_op = vision.Resize(size, Inter.BICUBIC)
        resize_op()

    # Resize operator:Test Interpolation mode PILCUBIC and 1 channel numpy data
    image = np.random.randn(1024, 1024, 1)
    size = (100, 100)
    resize_c_op = vision.Resize(size, interpolation=Inter.PILCUBIC)
    with pytest.raises(RuntimeError, match="Resize: Interpolation mode PILCUBIC " + \
                                           "only supports image with 3 channels, but got: <1024,1024,1>"):
        resize_c_op(image)

    # Resize operator:Test eager image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        size = [50, 100]
        interpolation_c = Inter.PILCUBIC
        resize_c_op = vision.Resize(size, interpolation=interpolation_c)
        with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
            resize_c_op(image)


if __name__ == "__main__":
    test_resize_op(plot=True)
    test_resize_4d_input_1_size()
    test_resize_4d_input_2_size()
    test_resize_2d_input_2_size()
    test_resize_3d_input_2_size()
    test_resize_op_antialias()
    test_resize_md5_c(plot=False)
    test_resize_md5_py(plot=False)
    test_resize_op_invalid_input()
    test_resize_op_exception_c_interpolation()
    test_resize_op_exception_py_interpolation()
    test_resize_performance()
    test_resize_operation_01()
    test_resize_operation_02()
    test_resize_operation_03()
    test_resize_operation_04()
    test_resize_exception_01()
    test_resize_exception_02()
    test_resize_exception_03()
