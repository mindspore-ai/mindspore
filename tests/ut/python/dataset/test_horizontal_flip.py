# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Testing HorizontalFlip Python API
"""
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"
FOUR_DIM_DATA = [[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                 [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]
FIVE_DIM_DATA = [[[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                  [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]]
FOUR_DIM_RES = [[[[3, 4, 3], [1, 2, 3]], [[7, 8, 3], [5, 6, 3]]],
                [[[11, 12, 3], [9, 10, 3]], [[15, 16, 3], [13, 14, 3]]]]
FIVE_DIM_RES = [[[[[3, 4, 3], [1, 2, 3]], [[7, 8, 3], [5, 6, 3]]],
                 [[[11, 12, 3], [9, 10, 3]], [[15, 16, 3], [13, 14, 3]]]]]


def test_horizontal_flip_pipeline(plot=False):
    """
    Feature: HorizontalFlip
    Description: Test HorizontalFlip in pipeline mode with Cpp implementation
    Expectation: Output is equal to the expected output
    """
    logger.info("test_horizontal_flip_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    horizontal_flip_op = vision.HorizontalFlip()
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=horizontal_flip_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        horizontal_flip_ms = data1["image"]
        original = data2["image"]
        horizontal_flip_cv = cv2.flip(original, 1)
        mse = diff_mse(horizontal_flip_ms, horizontal_flip_cv)
        logger.info("horizontal_flip_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, horizontal_flip_ms, mse, horizontal_flip_cv)


def test_horizontal_flip_eager():
    """
    Feature: HorizontalFlip
    Description: Test HorizontalFlip in eager mode
    Expectation: Output is equal to the expected output
    """
    logger.info("test_horizontal_flip_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = vision.HorizontalFlip()(img)
    img_cv = cv2.flip(img, 1)
    mse = diff_mse(img_ms, img_cv)
    assert mse == 0


def test_horizontal_flip_video_op_1d():
    """
    Feature: HorizontalFlip op
    Description: Test HorizontalFlip op by processing tensor with dim 1
    Expectation: Error is raised as expected
    """
    logger.info("Test HorizontalFlip with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    horizontal_flip_op = vision.HorizontalFlip()
    try:
        horizontal_flip_op(input_mindspore)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "HorizontalFlip: the image tensor should have at least two dimensions. You may need to perform " \
               "Decode first." in str(e)


def test_horizontal_flip_video_op_4d():
    """
    Feature: HorizontalFlip op
    Description: Test HorizontalFlip op by processing tensor with dim more than 3 (dim 4)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test HorizontalFlip with 4 dimension input")
    input_4_dim = np.array(FOUR_DIM_DATA).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    num_batch = input_4_shape[0]
    out_4_list = []
    batch_1d = 0
    while batch_1d < num_batch:
        out_4_list.append(cv2.flip(input_4_dim[batch_1d], 1))
        batch_1d += 1
    out_4_cv = np.array(out_4_list).astype(np.uint8)
    horizontal_flip_op = vision.HorizontalFlip()
    out_4_mindspore = horizontal_flip_op(input_4_dim)

    mse = diff_mse(out_4_mindspore, out_4_cv)
    assert mse < 0.001


def test_horizontal_flip_video_op_5d():
    """
    Feature: HorizontalFlip op
    Description: Test HorizontalFlip op by processing tensor with dim more than 3 (dim 5)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test HorizontalFlip with 5 dimension input")
    input_5_dim = np.array(FIVE_DIM_DATA).astype(np.uint8)
    input_5_shape = input_5_dim.shape
    num_batch_1d = input_5_shape[0]
    num_batch_2d = input_5_shape[1]
    out_5_list = []
    batch_1d = 0
    batch_2d = 0
    while batch_1d < num_batch_1d:
        while batch_2d < num_batch_2d:
            out_5_list.append(cv2.flip(input_5_dim[batch_1d][batch_2d], 1))
            batch_2d += 1
        batch_1d += 1
    out_5_cv = np.array(out_5_list).astype(np.uint8)
    horizontal_flip_op = vision.HorizontalFlip()
    out_5_mindspore = horizontal_flip_op(input_5_dim)

    mse = diff_mse(out_5_mindspore, out_5_cv)
    assert mse < 0.001


def test_horizontal_flip_video_op_precision_eager():
    """
    Feature: HorizontalFlip op
    Description: Test HorizontalFlip op by processing tensor with dim more than 3 (dim 4) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test HorizontalFlip eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    horizontal_flip_op = vision.HorizontalFlip()
    out_mindspore = horizontal_flip_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001


def test_horizontal_flip_video_op_precision_pipeline():
    """
    Feature: HorizontalFlip op
    Description: Test HorizontalFlip op by processing tensor with dim more than 3 (dim 5) in pipeline mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test HorizontalFlip pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    horizontal_flip_op = vision.HorizontalFlip()
    dataset = dataset.map(operations=horizontal_flip_op, input_columns=["col1"])
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001


if __name__ == "__main__":
    test_horizontal_flip_pipeline(plot=False)
    test_horizontal_flip_eager()
    test_horizontal_flip_video_op_1d()
    test_horizontal_flip_video_op_4d()
    test_horizontal_flip_video_op_5d()
    test_horizontal_flip_video_op_precision_eager()
    test_horizontal_flip_video_op_precision_pipeline()
