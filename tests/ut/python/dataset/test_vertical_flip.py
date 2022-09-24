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
Testing VerticalFlip Python API
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
FOUR_DIM_RES = [[[[5, 6, 3], [7, 8, 3]], [[1, 2, 3], [3, 4, 3]]],
                [[[13, 14, 3], [15, 16, 3]], [[9, 10, 3], [11, 12, 3]]]]
FIVE_DIM_RES = [[[[[5, 6, 3], [7, 8, 3]], [[1, 2, 3], [3, 4, 3]]],
                 [[[13, 14, 3], [15, 16, 3]], [[9, 10, 3], [11, 12, 3]]]]]


def test_vertical_flip_pipeline(plot=False):
    """
    Feature: VerticalFlip
    Description: Test VerticalFlip of Cpp implementation
    Expectation: Output is the same as expected output
    """
    logger.info("test_vertical_flip_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    vertical_flip_op = vision.VerticalFlip()
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=vertical_flip_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        vertical_flip_ms = data1["image"]
        original = data2["image"]
        vertical_flip_cv = cv2.flip(original, 0)
        mse = diff_mse(vertical_flip_ms, vertical_flip_cv)
        logger.info("vertical_flip_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, vertical_flip_ms, mse, vertical_flip_cv)


def test_vertical_flip_eager():
    """
    Feature: VerticalFlip
    Description: Test VerticalFlip in eager mode
    Expectation: Output is the same as expected output
    """
    logger.info("test_vertical_flip_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = vision.VerticalFlip()(img)
    img_cv = cv2.flip(img, 0)
    mse = diff_mse(img_ms, img_cv)
    assert mse == 0


def test_vertical_flip_video_op_1d():
    """
    Feature: VerticalFlip op
    Description: Test VerticalFlip op by processing tensor with dim 1
    Expectation:  Error is raised as expected
    """
    logger.info("Test VerticalFlip with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    vertical_flip_op = vision.VerticalFlip()
    try:
        vertical_flip_op(input_mindspore)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "VerticalFlip: the image tensor should have at least two dimensions. You may need to perform " \
               "Decode first." in str(e)


def test_vertical_flip_video_op_4d():
    """
    Feature: VerticalFlip op
    Description: Test VerticalFlip op by processing tensor with dim more than 3 (dim 4)
    Expectation: The dataset is processed successfully
    """
    logger.info("Test VerticalFlip with 4 dimension input")
    input_4_dim = np.array(FOUR_DIM_DATA).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    num_batch = input_4_shape[0]
    out_4_list = []
    batch_1d = 0
    while batch_1d < num_batch:
        out_4_list.append(cv2.flip(input_4_dim[batch_1d], 0))
        batch_1d += 1
    out_4_cv = np.array(out_4_list).astype(np.uint8)
    vertical_flip_op = vision.VerticalFlip()
    out_4_mindspore = vertical_flip_op(input_4_dim)

    mse = diff_mse(out_4_mindspore, out_4_cv)
    assert mse < 0.001


def test_vertical_flip_video_op_5d():
    """
    Feature: VerticalFlip op
    Description: process tensor with dim more than 3 (dim 5)
    Expectation: process successfully
    """
    input_5_dim = np.array(FIVE_DIM_DATA).astype(np.uint8)
    input_5_shape = input_5_dim.shape
    num_batch_1d = input_5_shape[0]
    num_batch_2d = input_5_shape[1]
    out_5_list = []
    batch_1d = 0
    batch_2d = 0
    while batch_1d < num_batch_1d:
        while batch_2d < num_batch_2d:
            out_5_list.append(cv2.flip(input_5_dim[batch_1d][batch_2d], 0))
            batch_2d += 1
        batch_1d += 1
    out_5_cv = np.array(out_5_list).astype(np.uint8)
    vertical_flip_op = vision.VerticalFlip()
    out_5_mindspore = vertical_flip_op(input_5_dim)

    mse = diff_mse(out_5_mindspore, out_5_cv)
    assert mse < 0.001


def test_vertical_flip_video_op_precision_eager():
    """
    Feature: VerticalFlip op
    Description: Test VerticalFlip op by processing tensor with dim more than 3 (dim 4) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test VerticalFlip eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    vertical_flip_op = vision.VerticalFlip()
    out_mindspore = vertical_flip_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001


def test_vertical_flip_video_op_precision_pipeline():
    """
    Feature: VerticalFlip op
    Description: Test VerticalFlip op by processing tensor with dim more than 3 (dim 5) in pipeline mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test VerticalFlip pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    vertical_flip_op = vision.VerticalFlip()
    dataset = dataset.map(operations=vertical_flip_op, input_columns=["col1"])
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001


if __name__ == "__main__":
    test_vertical_flip_pipeline(plot=False)
    test_vertical_flip_eager()
    test_vertical_flip_video_op_1d()
    test_vertical_flip_video_op_4d()
    test_vertical_flip_video_op_5d()
    test_vertical_flip_video_op_precision_eager()
    test_vertical_flip_video_op_precision_pipeline()
