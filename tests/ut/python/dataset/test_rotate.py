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
Testing Rotate Python API
"""
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from mindspore.dataset.vision.utils import Inter
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"
FOUR_DIM_DATA = [[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                 [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]
FIVE_DIM_DATA = [[[[[1, 2, 3], [3, 4, 3]], [[5, 6, 3], [7, 8, 3]]],
                  [[[9, 10, 3], [11, 12, 3]], [[13, 14, 3], [15, 16, 3]]]]]
FOUR_DIM_RES = [[[[3, 4, 3], [7, 8, 3]], [[1, 2, 3], [5, 6, 3]]],
                [[[11, 12, 3], [15, 16, 3]], [[9, 10, 3], [13, 14, 3]]]]
FIVE_DIM_RES = [[[[3, 4, 3], [7, 8, 3]], [[1, 2, 3], [5, 6, 3]]],
                [[[11, 12, 3], [15, 16, 3]], [[9, 10, 3], [13, 14, 3]]]]


def test_rotate_pipeline_with_expanding(plot=False):
    """
    Feature: Rotate
    Description: Test Rotate of Cpp implementation in pipeline mode with expanding
    Expectation: Output is the same as expected output
    """
    logger.info("test_rotate_pipeline_with_expanding")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    rotate_op = vision.Rotate(90, expand=True)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=rotate_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        rotate_ms = data1["image"]
        original = data2["image"]
        rotate_cv = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mse = diff_mse(rotate_ms, rotate_cv)
        logger.info("rotate_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, rotate_ms, mse, rotate_cv)


def test_rotate_video_op_1d():
    """
    Feature: Rotate
    Description: Test Rotate op by processing tensor with dim 1
    Expectation: Error is raised as expected
    """
    logger.info("Test Rotate with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    rotate_op = vision.Rotate(90, expand=False)
    try:
        rotate_op(input_mindspore)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Rotate: the image tensor should have at least two dimensions. You may need to perform " \
               "Decode first." in str(e)


def test_rotate_video_op_4d_without_expanding():
    """
    Feature: Rotate
    Description: Test Rotate op by processing tensor with dim more than 3 (dim 4) without expanding
    Expectation: Output is the same as expected output
    """
    logger.info("Test Rotate with 4 dimension input")
    input_4_dim = np.array(FOUR_DIM_DATA).astype(np.uint8)
    input_4_shape = input_4_dim.shape
    num_batch = input_4_shape[0]
    out_4_list = []
    batch_1d = 0
    while batch_1d < num_batch:
        out_4_list.append(cv2.rotate(input_4_dim[batch_1d], cv2.ROTATE_90_COUNTERCLOCKWISE))
        batch_1d += 1
    out_4_cv = np.array(out_4_list).astype(np.uint8)
    out_4_mindspore = vision.Rotate(90, expand=False)(input_4_dim)
    mse = diff_mse(out_4_mindspore, out_4_cv)
    assert mse < 0.001


def test_rotate_video_op_5d_without_expanding():
    """
    Feature: Rotate
    Description: Test Rotate op by processing tensor with dim more than 3 (dim 5) without expanding
    Expectation: Output is the same as expected output
    """
    logger.info("Test Rotate with 5 dimension input")
    input_5_dim = np.array(FIVE_DIM_DATA).astype(np.uint8)
    input_5_shape = input_5_dim.shape
    num_batch_1d = input_5_shape[0]
    num_batch_2d = input_5_shape[1]
    out_5_list = []
    batch_1d = 0
    batch_2d = 0
    while batch_1d < num_batch_1d:
        while batch_2d < num_batch_2d:
            out_5_list.append(cv2.rotate(input_5_dim[batch_1d][batch_2d], cv2.ROTATE_90_COUNTERCLOCKWISE))
            batch_2d += 1
        batch_1d += 1
    out_5_cv = np.array(out_5_list).astype(np.uint8)
    out_5_mindspore = vision.Rotate(90, expand=False)(input_5_dim)
    mse = diff_mse(out_5_mindspore, out_5_cv)
    assert mse < 0.001


def test_rotate_video_op_precision_eager():
    """
    Feature: Rotate op
    Description: Test Rotate op by processing tensor with dim more than 3 (dim 4) in eager mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test Rotate eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    rotate_op = vision.Rotate(90, expand=False)
    out_mindspore = rotate_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001


def test_rotate_video_op_precision_pipeline():
    """
    Feature: Rotate op
    Description: Test Rotate op by processing tensor with dim more than 3 (dim 5) in pipeline mode
    Expectation: The dataset is processed successfully
    """
    logger.info("Test Rotate pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    rotate_op = vision.Rotate(90, expand=False)
    dataset = dataset.map(operations=rotate_op, input_columns=["col1"])
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001


def test_rotate_pipeline_without_expanding():
    """
    Feature: Rotate
    Description: Test Rotate of Cpp implementation in pipeline mode without expanding
    Expectation: Output is the same as expected output
    """
    logger.info("test_rotate_pipeline_without_expanding")

    # Create a Dataset then decode and rotate the image
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize((64, 128))
    rotate_op = vision.Rotate(30)
    dataset = dataset.map(operations=decode_op, input_columns=["image"])
    dataset = dataset.map(operations=resize_op, input_columns=["image"])
    dataset = dataset.map(operations=rotate_op, input_columns=["image"])

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        rotate_img = data["image"]
        assert rotate_img.shape == (64, 128, 3)


def test_rotate_eager():
    """
    Feature: Rotate
    Description: Test Rotate in eager mode
    Expectation: Output is the same as expected output
    """
    logger.info("test_rotate_eager")
    img = cv2.imread(IMAGE_FILE)
    resize_img = vision.Resize((32, 64))(img)
    rotate_img = vision.Rotate(-90, expand=True)(resize_img)
    assert rotate_img.shape == (64, 32, 3)


def test_rotate_exception():
    """
    Feature: Rotate
    Description: Test Rotate with invalid parameters
    Expectation: Correct error is raised as expected
    """
    logger.info("test_rotate_exception")
    try:
        _ = vision.Rotate("60")
    except TypeError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "not of type [<class 'float'>, <class 'int'>]" in str(e)
    try:
        _ = vision.Rotate(30, Inter.BICUBIC, False, (0, 0, 0))
    except ValueError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "Value center needs to be a 2-tuple." in str(e)
    try:
        _ = vision.Rotate(-120, Inter.NEAREST, False, (-1, -1), (255, 255))
    except TypeError as e:
        logger.info("Got an exception in Rotate: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)


if __name__ == "__main__":
    test_rotate_pipeline_with_expanding(False)
    test_rotate_video_op_1d()
    test_rotate_video_op_4d_without_expanding()
    test_rotate_video_op_5d_without_expanding()
    test_rotate_video_op_precision_eager()
    test_rotate_video_op_precision_pipeline()
    test_rotate_pipeline_without_expanding()
    test_rotate_eager()
    test_rotate_exception()
