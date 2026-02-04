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
Testing Rotate Python API
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode
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

TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


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


def test_rotate_operation_01():
    """
    Feature: Rotate operation
    Description: Testing the normal functionality of the Rotate operator
    Expectation: The Output is equal to the expected output
    """
    # Rotate operator, eager mode, 1 channel
    degrees = 0
    resample = mode.Inter.BILINEAR
    expand = True
    center = (0, 0)
    fill_value = (0, 0, 0)
    image = np.random.randn(104, 560, 1)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    _ = rotate_op(image)

    # Rotate operator, eager mode, 2 channel
    degrees = 100
    resample = mode.Inter.BICUBIC
    expand = False
    center = (1, 1)
    fill_value = (255, 255, 255)
    image = np.random.randn(104, 560, 2)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    rotate_op(image)

    # Rotate operator, eager mode, 3 channel
    degrees = 361
    resample = mode.Inter.NEAREST
    center = (100, 200)
    image = np.random.randn(1124, 1560, 3)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, center=center)
    _ = rotate_op(image)

    # Rotate operator, eager mode, 20 channel
    degrees = 16777216
    resample = mode.Inter.NEAREST
    expand = True
    center = (-100, 200)
    fill_value = 0
    image = np.random.randn(1024, 50, 20)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center, fill_value=fill_value)
    rotate_op(image)

    # Rotate operator, eager mode, degrees equals -100
    degrees = -100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100.3, 200.8)
    fill_value = 255
    image = np.random.randn(24, 56, 2)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center, fill_value=fill_value)
    rotate_op(image)

    # Rotate operator, eager mode, degrees equals -16777216
    degrees = -16777216
    resample = mode.Inter.BICUBIC
    expand = True
    center = (-100.3, -200.9)
    fill_value = (0, 100, 255)
    image = np.random.randn(24, 56)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    _ = rotate_op(image)

    # Rotate operator, eager mode, input equals Pillow
    image = Image.open(image_jpg)
    degrees = 100.8
    expand = True
    fill_value = (0, 100, 100)
    rotate_op = vision.Rotate(degrees=degrees, expand=expand, fill_value=fill_value)
    _ = rotate_op(image)

    # Rotate operator, eager mode, input is a GIF image
    image = Image.open(image_gif)
    degrees = -100.8
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 0, 100)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    _ = rotate_op(image)


def test_rotate_operation_02():
    """
    Feature: Rotate operation
    Description: Testing the normal functionality of the Rotate operator
    Expectation: The Output is equal to the expected output
    """
    # Rotate operator, eager mode, input is a PNG image
    image = Image.open(image_png)
    degrees = 3600000
    resample = mode.Inter.BICUBIC
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 0)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    _ = rotate_op(image)

    # Rotate Operator, Pipeline Mode
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    degrees = 0
    resample = mode.Inter.BICUBIC
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 0)
    rotate_op = vision.Rotate(degrees=degrees, resample=resample, expand=expand, center=center,
                              fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=rotate_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Rotate Operator: Test Rotate op by processing tensor with dim more than 3 (dim 5) without expanding
    logger.info("Test Rotate with 5 dimension input")
    five_data_list = [FIVE_DIM_DATA, np.random.randint(255, size=(3, 3, 3, 3, 3))]
    for five_data in five_data_list:
        input_5_dim = np.array(five_data).astype(np.uint8)
        input_5_shape = input_5_dim.shape
        num_batch_1d = input_5_shape[0]
        num_batch_2d = input_5_shape[1]
        out_5_list = []
        batch_1d = 0

        while batch_1d < num_batch_1d:
            batch_2d = 0
            while batch_2d < num_batch_2d:
                out_5_list.append(cv2.rotate(input_5_dim[batch_1d][batch_2d], cv2.ROTATE_90_COUNTERCLOCKWISE))
                batch_2d += 1
            batch_1d += 1
        out_5_cv = np.array(out_5_list).astype(np.uint8)

        reshape_np = np.array(out_5_cv).reshape(np.array(five_data).shape)
        out_5_mindspore = vision.Rotate(90, expand=False)(input_5_dim)
        mse = diff_mse(out_5_mindspore, reshape_np)
        assert mse < 0.001

    # Rotate operator, input equals multidimensional
    image_list = [np.random.randn(200, 200, 3), np.random.randn(20, 20, 30, 20, 4),
                  np.random.randn(2, 2, 3, 2, 3, 2, 3, 2, 3, 3)]
    for image in image_list:
        rotate_op = vision.Rotate(100)
        rotate_op(image)


def test_rotate_operation_03():
    """
    Feature: Rotate operation
    Description: Testing the normal functionality of the Rotate operator
    Expectation: The Output is equal to the expected output
    """
    # Rotate Operator: Test Rotate op by processing tensor with dim more than 3 (dim 4) in eager mode
    logger.info("Test Rotate eager with 4 dimension input")
    input_mindspore = np.array(FOUR_DIM_DATA).astype(np.uint8)

    rotate_op = vision.Rotate(90, expand=False)
    out_mindspore = rotate_op(input_mindspore)
    mse = diff_mse(out_mindspore, np.array(FOUR_DIM_RES).astype(np.uint8))
    assert mse < 0.001

    # Rotate Operator: Test Rotate op by processing tensor with dim more than 3 (dim 5) in pipeline mode
    logger.info("Test Rotate pipeline with 5 dimension input")
    data = np.array(FIVE_DIM_DATA).astype(np.uint8)
    expand_data = np.expand_dims(data, axis=0)

    dataset = ds.NumpySlicesDataset(expand_data, column_names=["col1"], shuffle=False)
    rotate_op = vision.Rotate(90, expand=False)
    dataset = dataset.map(operations=rotate_op, input_columns=["col1"])
    for item in dataset.create_dict_iterator(output_numpy=True):
        mse = diff_mse(item["col1"], np.array(FIVE_DIM_RES).astype(np.uint8))
        assert mse < 0.001

    # Rotate Operator: Test Rotate of Cpp implementation in pipeline mode without expanding
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

    # Rotate Operator: Test Rotate in eager mode
    logger.info("test_rotate_eager")
    resample_list = [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC]
    degrees_list = [-361, -360, -1, 0, 1, 180, 360, 361, 10000]
    for resample in resample_list:
        for degrees in degrees_list:
            img = Image.open(IMAGE_FILE)
            _ = vision.Rotate(degrees, resample=resample, expand=True)(img)


def test_rotate_exception_01():
    """
    Feature: Rotate operation
    Description: Testing the Rotate Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rotate operator, exception testing, degrees too large
    degrees = 16777217
    with pytest.raises(ValueError, match="Input degrees is not within the required interval"):
        vision.Rotate(degrees=degrees)

    # Rotate operator, exception testing, degrees too small
    degrees = -16777217
    with pytest.raises(ValueError, match="Input degrees is not within the required interval"):
        vision.Rotate(degrees=degrees)

    # Rotate operator, exception testing, degrees is empty
    degrees = ""
    with pytest.raises(TypeError,
                       match=r'Argument degrees with value "" is not of type \[<class \'float\'>, '
                             r'<class \'int\'>\], but got <class \'str\'>.'):
        vision.Rotate(degrees=degrees)

    # Rotate operator, exception testing, degrees is None
    degrees = None
    with pytest.raises(TypeError,
                       match=r"Argument degrees with value None is not of type \[<class \'float\'>, "
                             r"<class \'int\'>\], but got <class \'NoneType\'>."):
        vision.Rotate(degrees=degrees)

    # Rotate operator, exception testing, degrees is list
    degrees = [100, 100]
    with pytest.raises(TypeError,
                       match=r"Argument degrees with value \[100, 100\] is not of type \[<class \'float\'>, "
                             r"<class \'int\'>\], but got <class \'list\'>."):
        vision.Rotate(degrees=degrees)

    # Rotate operator, exception testing, resample is None
    degrees = 100
    resample = None
    with pytest.raises(TypeError,
                       match=r"Argument resample with value None is not of type \[<enum 'Inter'>\], but got <class "
                             r"'NoneType'>."):
        vision.Rotate(degrees=degrees, resample=resample)

    # Rotate operator, exception testing, resample is empty
    resample = ""
    degrees = 7536
    with pytest.raises(TypeError, match=r"Argument resample with value \"\" is not of type \[<enum \'Inter\'>\], "
                                        r"but got <class \'str\'>."):
        vision.Rotate(degrees=degrees, resample=resample)

    # Rotate operator, exception testing, expand is None
    expand = None
    degrees = 100
    with pytest.raises(TypeError,
                       match=r"Argument expand with value None is not of type \[<class 'bool'>\], but got <class "
                             r"'NoneType'>."):
        vision.Rotate(degrees=degrees, expand=expand)

    # Rotate operator, exception testing, expand is empty
    expand = ""
    degrees = 100
    with pytest.raises(TypeError, match=r"Argument expand with value \"\" is not of type \[<class \'bool\'>\], " \
                                        r"but got <class \'str\'>."):
        vision.Rotate(degrees=degrees, expand=expand)

    # Rotate operator, exception testing, center is 3-tuple
    degrees = -100
    center = (100, 200, 300)
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center)

    # Rotate operator, exception testing, center is empty
    degrees = 200
    center = ""
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center)

    # Rotate operator, exception testing, center is list
    center = [-100, -200]
    degrees = 100
    with pytest.raises(ValueError, match="Value center needs to be a 2-tuple"):
        vision.Rotate(degrees=degrees, center=center)

    # Rotate operator, exception testing, fill_value is 256
    fill_value = 256
    degrees = 100
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]"):
        vision.Rotate(degrees=degrees, fill_value=fill_value)

    # Rotate operator, exception testing, fill_value is 2-tuple
    degrees = 100
    fill_value = (100, 100)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)

    # Rotate operator, exception testing, fill_value is empty
    degrees = 100
    fill_value = ""
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)


def test_rotate_exception_02():
    """
    Feature: Rotate operation
    Description: Testing the Rotate Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rotate operator, exception testing, fill_value is float
    degrees = 100
    fill_value = (10, 100, 25.5)
    with pytest.raises(TypeError, match="value 25.5 is not of type \\[<class 'int'>\\], but got <class 'float'>."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)

    # Rotate operator, exception testing, fill_value is 4-tuple
    degrees = 100
    fill_value = (10, 100, 255, 45)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)

    # Rotate operator, exception testing, fill_value is list
    degrees = 100
    fill_value = [10, 100, 255]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)

    # Rotate operator, input is 4-dimensional data
    degrees = 100
    image = np.random.randn(200, 200, 3, 3)
    rotate_op = vision.Rotate(degrees=degrees)
    rotate_op(image)

    # Rotate operator, exception testing, input is Zero-dimensional data
    degrees = 100
    image = 10
    rotate_op = vision.Rotate(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        rotate_op(image)

    # Rotate operator, exception testing, No arguments
    with pytest.raises(TypeError, match="missing a required argument"):
        vision.Rotate()

    # Rotate operator, exception testing, Multi-parameter
    degrees = 100
    resample = mode.Inter.BILINEAR
    expand = True
    center = (100, 200)
    fill_value = (100, 100, 100)
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.Rotate(degrees, resample, expand, center, fill_value, more_para)

    # Rotate operator, input is list
    degrees = 256
    image = np.random.randn(128, 32, 3)
    rotate_op = vision.Rotate(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        rotate_op(list(image))

    # Rotate operator, input is tensor
    degrees = 256
    image = np.random.randn(20, 30, 3)
    rotate_op = vision.Rotate(degrees=degrees)
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>."):
        rotate_op(ms.Tensor(image))

    # Rotate operator, input is tuple
    degrees = 100
    image = np.random.randn(20, 30, 3)
    rotate_op = vision.Rotate(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        rotate_op(tuple(image))

    # Rotate operator: Test Rotate op by processing tensor with dim 1
    logger.info("Test Rotate with 1 dimension input")
    data = [1]
    input_mindspore = np.array(data).astype(np.uint8)
    rotate_op = vision.Rotate(90, expand=False)
    with pytest.raises(RuntimeError,
                       match="Rotate: the image tensor should have at least two dimensions."):
        rotate_op(input_mindspore)

    # Rotate operator: Test Rotate op by processing tensor with dim more than 3 (dim 4) without expanding
    four_data_list = [FOUR_DIM_DATA, np.random.randint(255, size=(3, 3, 3, 3))]
    for four_data in four_data_list:
        logger.info("Test Rotate with 4 dimension input")
        input_4_dim = np.array(four_data).astype(np.uint8)
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

    # Rotate operator, Invalid parameter type for resample
    resample_list = [[], (1), 1, "1", ds]
    for resample in resample_list:
        img = Image.open(IMAGE_FILE)
        with pytest.raises(TypeError, match="Argument resample with value"):
            vision.Rotate(45, resample=resample, expand=False)(img)


def test_rotate_exception_03():
    """
    Feature: Rotate operation
    Description: Testing the Rotate Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rotate operator, Invalid parameter type for degrees
    degrees_list = [[], {}, (), "1", ds]
    for degrees in degrees_list:
        img = Image.open(IMAGE_FILE)
        with pytest.raises(TypeError, match="Argument degrees with value"):
            vision.Rotate(degrees)(img)

    # doct: Note that the expand flag assumes rotation around the center and no translation.
    center_list = [(10, 20), (100, 200), (1000, 2000), (-1, -2), (0, 0)]
    for center in center_list:
        img = Image.open(IMAGE_FILE)
        re_img = vision.Resize((200, 200))(img)
        _ = vision.Rotate(45, center=center, expand=True)(re_img)

    # Rotate operator: Test Rotate with invalid parameters
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

    # Rotate operator, exception testing, fill_value equals -1
    degrees = 100
    fill_value = -1
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        vision.Rotate(degrees=degrees, fill_value=fill_value)


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
    test_rotate_operation_01()
    test_rotate_operation_02()
    test_rotate_operation_03()
    test_rotate_exception_01()
    test_rotate_exception_02()
    test_rotate_exception_03()
