# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from PIL import Image
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers, diff_mse

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
