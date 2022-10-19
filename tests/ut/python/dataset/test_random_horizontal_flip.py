# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
Testing the random horizontal flip op in DE
"""
import numpy as np
import pytest

from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms as ops
import mindspore.dataset.vision as vision
from util import save_and_check_md5, save_and_check_md5_pil, visualize_list, visualize_image, diff_mse, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
FOUR_DIM_DATA = [[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                 [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]
FIVE_DIM_DATA = [[[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                  [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]]
FOUR_DIM_RES = [[[[3.0, 4.0, 3.0], [1.0, 2.0, 3.0]], [[7.0, 8.0, 3.0], [5.0, 6.0, 3.0]]],
                [[[11.0, 12.0, 3.0], [9.0, 10.0, 3.0]], [[15.0, 16.0, 3.0], [13.0, 14.0, 3.0]]]]
FIVE_DIM_RES = [[[[[3.0, 4.0, 3.0], [1.0, 2.0, 3.0]], [[7.0, 8.0, 3.0], [5.0, 6.0, 3.0]]],
                 [[[11.0, 12.0, 3.0], [9.0, 10.0, 3.0]], [[15.0, 16.0, 3.0], [13.0, 14.0, 3.0]]]]]


def h_flip(image):
    """
    Apply the random_horizontal
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[:, ::-1, :]
    return image


def test_random_horizontal_op(plot=False):
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with default probability
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):

        # with the seed value, we can only guarantee the first number generated
        if num_iter > 0:
            break

        image_h_flipped = item1["image"]
        image = item2["image"]
        image_h_flipped_2 = h_flip(image)

        mse = diff_mse(image_h_flipped, image_h_flipped_2)
        assert mse == 0
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image, image_h_flipped, mse, image_h_flipped_2)


def test_random_horizontal_valid_prob_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Cpp implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_valid_prob_c")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    random_horizontal_op = vision.RandomHorizontalFlip(0.8)
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=random_horizontal_op, input_columns=["image"])

    filename = "random_horizontal_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_valid_prob_py():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Python implementation using valid non-default input
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_horizontal_valid_prob_py")
    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomHorizontalFlip(0.8),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    filename = "random_horizontal_01_py_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_invalid_prob_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Cpp implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_horizontal_invalid_prob_c")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        random_horizontal_op = vision.RandomHorizontalFlip(1.5)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=random_horizontal_op,
                        input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_horizontal_invalid_prob_py():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with Python implementation using invalid input
    Expectation: Error is raised as expected
    """
    logger.info("test_random_horizontal_invalid_prob_py")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)

    try:
        transforms = [
            vision.Decode(True),
            # Note: Valid range of prob should be [0.0, 1.0]
            vision.RandomHorizontalFlip(1.5),
            vision.ToTensor()
        ]
        transform = ops.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_horizontal_comp(plot=False):
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op and compare between Python and Cpp image augmentation ops
    Expectation: Resulting datasets from the ops are expected to be the same
    """
    logger.info("test_random_horizontal_comp")
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    decode_op = vision.Decode()
    # Note: The image must be flipped if prob is set to be 1
    random_horizontal_op = vision.RandomHorizontalFlip(1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        # Note: The image must be flipped if prob is set to be 1
        vision.RandomHorizontalFlip(1),
        vision.ToTensor()
    ]
    transform = ops.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    images_list_c = []
    images_list_py = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_c = item1["image"]
        image_py = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        images_list_c.append(image_c)
        images_list_py.append(image_py)

        # Check if the output images are the same
        mse = diff_mse(image_c, image_py)
        assert mse < 0.001
    if plot:
        visualize_list(images_list_c, images_list_py, visualize_mode=2)


def test_random_horizontal_op_1():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with different fields
    Expectation: The dataset is processed as expected
    """
    logger.info("Test RandomHorizontalFlip with different fields.")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    data = data.map(operations=ops.Duplicate(), input_columns=["image"],
                    output_columns=["image", "image_copy"])
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    decode_op = vision.Decode()

    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=decode_op, input_columns=["image_copy"])
    data = data.map(operations=random_horizontal_op,
                    input_columns=["image", "image_copy"])

    num_iter = 0
    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = data1["image"]
        image_copy = data1["image_copy"]
        mse = diff_mse(image, image_copy)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1


def test_random_horizontal_flip_invalid_data():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip with invalid data
    Expectation: Error is raised as expected
    """

    invalid_type_img = np.random.random((32, 32, 3)).astype(np.str_)
    invalid_shape_img = np.random.random(32).astype(np.float32)
    random_horizontal_flip = vision.RandomHorizontalFlip(0.1)

    with pytest.raises(RuntimeError) as error_info:
        random_horizontal_flip(invalid_type_img)
    assert "RandomHorizontalFlip: the data type of image tensor does not match the requirement of operator." \
           in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        random_horizontal_flip(invalid_shape_img)
    assert "RandomHorizontalFlip: the image tensor should have at least two dimensions. You may need to perform " \
           "Decode first." in str(error_info.value)


def test_random_horizontal_flip_video_op_1d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim 1
    Expectation: Error is raised as expected
    """
    logger.info("Test RandomHorizontalFlip with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    try:
        random_horizontal_op(input_mindspore)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "RandomHorizontalFlip: the image tensor should have at least two dimensions. You may need to perform " \
               "Decode first." in str(e)


def test_random_horizontal_flip_video_op_4d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 4)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip with 4 dimension input")
    input_4_dim = np.array(FOUR_DIM_DATA).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    n_num = input_4_dim.size // (input_4_shape[-2] * input_4_shape[-1])
    input_3_dim = input_4_dim.reshape([n_num, input_4_shape[-2], input_4_shape[-1]])

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_4_dim = random_horizontal_op(input_4_dim)
    out_3_dim = random_horizontal_op(input_3_dim)
    out_3_dim = out_3_dim.reshape(input_4_shape)

    mse = diff_mse(out_4_dim, out_3_dim)
    assert mse < 0.001


def test_random_horizontal_flip_video_op_5d_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 5)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip with 5 dimension input")
    input_5_dim = np.array(FIVE_DIM_DATA).astype(np.uint8)
    input_5_shape = input_5_dim.shape
    n_num = input_5_dim.size // (input_5_shape[-2] * input_5_shape[-1])
    input_3_dim = input_5_dim.reshape([n_num, input_5_shape[-2], input_5_shape[-1]])

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_5_dim = random_horizontal_op(input_5_dim)
    out_3_dim = random_horizontal_op(input_3_dim)
    out_3_dim = out_3_dim.reshape(input_5_shape)

    mse = diff_mse(out_5_dim, out_3_dim)
    assert mse < 0.001


def test_random_horizontal_flip_video_op_precision_eager_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 4) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    out_mindspore = random_horizontal_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001


def test_random_horizontal_flip_video_op_precision_pipeline_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Test RandomHorizontalFlip op by processing tensor with dim more than 3 (dim 5) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test RandomHorizontalFlip pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    random_horizontal_op = vision.RandomHorizontalFlip(1.0)
    dataset = dataset.map(input_columns=["col1"], operations=random_horizontal_op)
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001


if __name__ == "__main__":
    test_random_horizontal_op(plot=True)
    test_random_horizontal_valid_prob_c()
    test_random_horizontal_valid_prob_py()
    test_random_horizontal_invalid_prob_c()
    test_random_horizontal_invalid_prob_py()
    test_random_horizontal_comp(plot=True)
    test_random_horizontal_op_1()
    test_random_horizontal_flip_invalid_data()
    test_random_horizontal_flip_video_op_1d_c()
    test_random_horizontal_flip_video_op_4d_c()
    test_random_horizontal_flip_video_op_5d_c()
    test_random_horizontal_flip_video_op_precision_eager_c()
    test_random_horizontal_flip_video_op_precision_pipeline_c()
