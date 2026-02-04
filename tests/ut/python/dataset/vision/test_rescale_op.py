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
Testing the rescale op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from util import visualize_image, diff_mse, save_and_check_md5

TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def rescale_np(image):
    """
    Apply the rescale
    """
    image = image / 255.0
    image = image - 1.0
    return image


def get_rescaled(image_id):
    """
    Reads the image using DE ops and then rescales using Numpy
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        image = item["image"].asnumpy()
        if num_iter == image_id:
            return rescale_np(image)
        num_iter += 1

    return None


def test_rescale_op(plot=False):
    """
    Feature: Rescale op
    Description: Test rescale op basic usage
    Expectation: Output is the same as expected output
    """
    logger.info("Test rescale")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(1.0 / 255.0, -1.0)

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    data2 = data1.map(operations=rescale_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_original = item1["image"]
        image_de_rescaled = item2["image"]
        image_np_rescaled = get_rescaled(num_iter)
        mse = diff_mse(image_de_rescaled, image_np_rescaled)
        assert mse < 0.001  # rounding error
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image_original, image_de_rescaled, mse, image_np_rescaled)


def test_rescale_md5():
    """
    Feature: Rescale op
    Description: Test rescale op with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("Test Rescale with md5 comparison")

    # generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(1.0 / 255.0, -1.0)

    # apply map operations on images
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=rescale_op, input_columns=["image"])

    # check results with md5 comparison
    filename = "rescale_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_rescale_operation_01():
    """
    Feature: Rescale operation
    Description: Testing the normal functionality of the Rescale operator
    Expectation: The Output is equal to the expected output
    """
    # Rescale operator:Test rescale is 100.0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    rescale = 100.0
    shift = -1.0
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns=["image"], operations=rescale_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Rescale operator:Test rescale is 0.0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    rescale = 0.0
    shift = -1.0
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns=["image"], operations=rescale_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Rescale operator:Test shift is 0.0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    rescale = 1.0
    shift = 0.0
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns=["image"], operations=rescale_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Rescale operator:Test shift is 1.0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    rescale = 1.0
    shift = 1.0
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns=["image"], operations=rescale_op)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Rescale operator:Test rescale is within the upper boundary value
    rescale = 16777216.0
    shift = -10
    image = np.random.randn(10, 10, 1)
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test shift is within the upper boundary value.
    rescale = 0.8985
    shift = 16777216.0
    image = np.random.randn(15, 15, 3)
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test shift is within the lower boundary value.
    rescale = 0.8985
    shift = -16777216.0
    image = np.random.randn(15, 15, 3)
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is jpg image
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_2.jpg")
    rescale = 0.86
    shift = -1.3
    with Image.open(image_file) as image:
        rescale_op = vision.Rescale(rescale, shift)
        _ = rescale_op(image)


def test_rescale_operation_02():
    """
    Feature: Rescale operation
    Description: Testing the normal functionality of the Rescale operator
    Expectation: The Output is equal to the expected output
    """
    # Rescale operator:Test input is gif image
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        rescale = 0.86
        shift = -1.3
        rescale_op = vision.Rescale(rescale, shift)
        _ = rescale_op(image)

    # Rescale operator:Test input is bmp image
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        rescale = 0.86
        shift = -1.3
        rescale_op = vision.Rescale(rescale, shift)
        _ = rescale_op(image)

    # Rescale operator:Test input is image opened using the cv2 method
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_1.jpg")
    image = cv2.imread(image_file)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is 3-d image 01
    image = np.random.randn(468, 368, 3).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is 3-d image 02
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is 2-d numpy array
    image = np.random.randint(-255, 255, (256, 128)).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is 4-d image
    image = np.random.randn(10, 468, 368, 3).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_png) as image:
        rescale = 0.86
        shift = -1.3
        rescale_op = vision.Rescale(rescale, shift)
        _ = rescale_op(image)

    # Rescale operator:Test input is 1-d numpy data
    image = np.random.randn(200,).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)

    # Rescale operator:Test input is 4 channel numpy array
    image = np.random.randn(128, 128, 4).astype(np.uint8)
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    _ = rescale_op(image)


def test_rescale_exception_01():
    """
    Feature: Rescale operation
    Description: Testing the Rescale Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rescale operator:Test rescale is not within the upper boundary value
    rescale = 16777217.0
    shift = -1.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(ValueError,
                       match="Input rescale is not within the required interval of \\[-16777216, 16777216\\]"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale<0
    rescale = -1.0
    shift = -1.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(RuntimeError, match="Rescale: rescale must be greater than or equal to 0, got:"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is not within the upper boundary value
    rescale = 1.6
    shift = 16777217.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(ValueError,
                       match="Input shift is not within the required interval of \\[-16777216, 16777216\\]"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is not within the lower boundary value
    rescale = 1.6
    shift = -16777217.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(ValueError,
                       match="Input shift is not within the required interval of \\[-16777216, 16777216\\]"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is str
    rescale = 'test'
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is bool
    rescale = True
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is None
    rescale = None
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is list
    rescale = [1.6]
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is tuple
    rescale = (0.1,)
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test rescale is ""
    rescale = ""
    shift = 17.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument rescale with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is str
    rescale = 1.6
    shift = 'test'
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is bool
    rescale = 1.6
    shift = True
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)


def test_rescale_exception_02():
    """
    Feature: Rescale operation
    Description: Testing the Rescale Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rescale operator:Test shift is None
    rescale = 1.6
    shift = None
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is list
    rescale = 1.6
    shift = [0.5]
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is tuple
    rescale = 1.6
    shift = (17.0,)
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test shift is ""
    rescale = 1.6
    shift = ""
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="Argument shift with value"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test input is 3d numpy list
    image = np.random.randn(128, 128, 3).astype(np.uint8).tolist()
    rescale = 0.86
    shift = -1.3
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test input is int
    image = 10
    rescale = 0.86
    shift = -1.3
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test input is tuple
    image = (10,)
    rescale = 0.86
    shift = -1.3
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op(image)

    # Rescale operator:Test no image is transferred
    rescale = 0.86
    shift = -1.3
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        rescale_op = vision.Rescale(rescale, shift)
        rescale_op()

    # Rescale operator:Test input is tensor
    image = Tensor(np.random.randn(10, 10, 3))
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>"):
        rescale_op = vision.Rescale(1.0 / 255.0, -1.0)
        rescale_op(image)

    # Rescale operator:Test no parameter is transferred.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    with pytest.raises(TypeError, match="missing a required argument"):
        rescale_op = vision.Rescale()
        dataset = dataset.map(input_columns=["image"], operations=rescale_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Rescale operator:Test the rescale parameter is not transferred.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    shift = 1.0
    with pytest.raises(TypeError, match="missing a required argument"):
        rescale_op = vision.Rescale(shift=shift)
        dataset = dataset.map(input_columns=["image"], operations=rescale_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_rescale_exception_03():
    """
    Feature: Rescale operation
    Description: Testing the Rescale Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Rescale operator:Test the shift parameter is not transferred.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    rescale = 1.0
    with pytest.raises(TypeError, match="missing a required argument"):
        rescale_op = vision.Rescale(rescale=rescale)
        dataset = dataset.map(input_columns=["image"], operations=rescale_op)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Rescale operator:Test one more parameter
    rescale = 0.8985
    shift = -16777216.0
    image = np.random.randn(15, 15, 3)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
        rescale_op = vision.Rescale(rescale, shift, test='test')
        _ = rescale_op(image)

    # Rescale operator:Test input is numpy list
    rescale = 1.6
    shift = 17.0
    image = np.random.randn(15, 15, 3).tolist()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        rescale_op = vision.Rescale(rescale, shift)
        _ = rescale_op(image)

    # Rescale operator:Test input is 2d list
    image = list(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        _ = rescale_op(image)

    # Rescale operator:Test input is 2d tuple
    image = tuple(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    rescale = 0.86
    shift = -1.3
    rescale_op = vision.Rescale(rescale, shift)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        _ = rescale_op(image)


if __name__ == "__main__":
    test_rescale_op(plot=True)
    test_rescale_md5()
    test_rescale_operation_01()
    test_rescale_operation_02()
    test_rescale_exception_01()
    test_rescale_exception_02()
    test_rescale_exception_03()
