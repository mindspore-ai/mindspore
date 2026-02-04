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
Testing Crop op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision as c_vision
import mindspore.dataset.vision.transforms as v_trans

from mindspore import log as logger
from util import visualize_image, diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train/")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    return data_list


def test_crop_pipeline(plot=False):
    """
    Feature: Crop op
    Description: Test Crop op in pipeline mode
    Expectation: Passes the equality test
    """
    logger.info("test_crop_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    crop_op = c_vision.Crop((0, 0), (20, 25))
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=crop_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        crop_ms = data1["image"]
        original = data2["image"]
        crop_expect = original[0:20, 0:25]
        mse = diff_mse(crop_ms, crop_expect)
        logger.info("crop_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, crop_ms, mse, crop_expect)


def test_crop_eager():
    """
    Feature: Crop op
    Description: Test Crop op in eager mode
    Expectation: Passes the equality test
    """
    logger.info("test_crop_eager")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.Crop((20, 50), (30, 50))(img)
    img_expect = img[20:50, 50:100]
    mse = diff_mse(img_ms, img_expect)
    assert mse == 0


def test_crop_exception():
    """
    Feature: Crop op
    Description: Test Crop op with invalid parameters
    Expectation: Correct error and message are thrown as expected
    """
    logger.info("test_crop_exception")
    try:
        _ = c_vision.Crop([-10, 0], [20])
    except ValueError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not within the required interval of [0, 2147483647]" in str(e)
    try:
        _ = c_vision.Crop([0, 5.2], [10, 10])
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not of type [<class 'int'>]" in str(e)
    try:
        _ = c_vision.Crop([0], [28])
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "Coordinates should be a list/tuple (y, x) of length 2." in str(
            e)
    try:
        _ = c_vision.Crop((0, 0), -1)
    except ValueError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not within the required interval of [1, 16777216]" in str(e)
    try:
        _ = c_vision.Crop((0, 0), (10.5, 15))
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "not of type [<class 'int'>]" in str(e)
    try:
        _ = c_vision.Crop((0, 0), (0, 10, 20))
    except TypeError as e:
        logger.info("Got an exception in Crop: {}".format(str(e)))
        assert "Size should be a single integer or a list/tuple (h, w) of length 2." in str(e)


def test_crop_operation_01():
    """
    Feature: Crop operation
    Description: Testing the normal functionality of the Crop operator
    Expectation: The Output is equal to the expected output
    """
    # Crop Normal Function: Test size is 10
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    coordinates = (0, 0)
    size = 10
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=crop_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Crop Normal Function: Test size is (100, 150)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    coordinates = (10, 20)
    size = (100, 150)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=crop_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Crop Normal Function: Test input is PIL
    with Image.open(dir_data()[1]) as image:
        coordinates = [300, 100]
        size = (584, 618)
        crop_op = v_trans.Crop(coordinates=coordinates, size=size)
        _ = crop_op(image)

    # Crop Normal Function: Test input is 2d
    image = np.random.randn(560, 560)
    coordinates = [0, 0]
    size = (560, 560)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    out = crop_op(image)
    assert (out == image).all()

    # Crop Normal Function: Test input.shape is (300, 300, 3)
    image = np.random.randint(0, 255, (300, 300, 3)).astype(np.uint8)
    coordinates = (200, 0)
    size = 1
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    _ = crop_op(image)

    # Crop Normal Function: Test input.shape is (658, 714, 4)
    image = np.random.randint(0, 255, (658, 714, 4)).astype(np.int32)
    coordinates = (100, 50)
    size = [200, 200]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    out = crop_op(image)
    assert (image[100:300, 50:250, 0:4] == out).all()

    # Crop Normal Function: Test input.shape is (800, 1024, 10)
    image = np.random.randint(0, 255, (800, 1024, 10)).astype(np.uint8)
    coordinates = (200, 0)
    size = 500
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    out = crop_op(image)
    assert (image[200:700, 0:500, 0:10] == out).all()

    # Crop Normal Function: Test input.shape is (1024, 2048, 1)
    image = np.random.randn(1024, 2048, 1)
    coordinates = (1, 1)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    out = crop_op(image)
    assert (image[1:381, 1:381, 0:1] == out).all()


def test_crop_exception_01():
    """
    Feature: Crop operation
    Description: Testing the Crop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Crop Exception Scenarios: Test input.shape is  (658, 714, 3, 3)
    image = np.random.randint(0, 255, (658, 714, 3, 3)).astype(np.uint8)
    coordinates = (1, 1)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    with pytest.raises(RuntimeError, match=r"input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 4"):
        crop_op(image)

    # Crop Exception Scenarios: Test input.shape is  (658,)
    image = np.random.randint(0, 255, (658,)).astype(np.uint8)
    coordinates = (10, 10)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    with pytest.raises(RuntimeError,
                       match="input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 1."
                             " You may need to perform Decode first."):
        crop_op(image)

    # Crop Exception Scenarios: Test coordinates exceed image.shape
    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    coordinates = (201, 1)
    size = 10
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    with pytest.raises(RuntimeError,
                       match=r"Crop: Crop height dimension: 211 exceeds image height: 200"):
        crop_op(image)

    # Crop Exception Scenarios: Test coordinates plus size exceed image.shape
    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    coordinates = (100, 100)
    size = [101, 50]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    with pytest.raises(RuntimeError,
                       match="Crop: Crop height dimension: 201 exceeds image height: 200"):
        crop_op(image)

    # Crop Exception Scenarios: Test coordinates less than 0
    with pytest.raises(ValueError, match="Input coordinates\\[0\\] is not within the "
                                          "required interval of \\[0, 2147483647\\]."):
        v_trans.Crop(coordinates=[-1, 100], size=100)

    # Crop Exception Scenarios: Test coordinates exceed 2147483647
    with pytest.raises(ValueError, match="Input coordinates\\[1\\] is not within the "
                                          "required interval of \\[0, 2147483647\\]."):
        v_trans.Crop(coordinates=[10, 2147483648], size=100)

    # Crop Exception Scenarios: Test coordinates is float
    with pytest.raises(TypeError, match="Argument coordinates\\[1\\] with value 10.0 is not of "
                                         "type \\[<class 'int'>\\], but got <class 'float'>."):
        v_trans.Crop(coordinates=[10, 10.0], size=100)

    # Crop Exception Scenarios: Test coordinates is True
    with pytest.raises(TypeError, match="Argument coordinates\\[1\\] with value True is not of "
                                         "type \\(<class 'int'>,\\), but got <class 'bool'>."):
        v_trans.Crop(coordinates=(10, True), size=100)

    # Crop Exception Scenarios: Test coordinates is np
    with pytest.raises(TypeError, match="Argument coordinates with value \\[10 10\\] is not of type \\[<class "
                                         "'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        v_trans.Crop(coordinates=np.array([10, 10]), size=100)

    # Crop Exception Scenarios: Test coordinates is int
    with pytest.raises(TypeError, match="Argument coordinates with value 20 is not of type \\[<class "
                                         "'list'>, <class 'tuple'>\\], but got <class 'int'>."):
        v_trans.Crop(coordinates=20, size=100)

    # Crop Exception Scenarios: Test coordinates is 3-tuple
    with pytest.raises(TypeError, match="Coordinates should be a list/tuple \\(y, x\\) of length 2."):
        v_trans.Crop(coordinates=(10, 10, 10), size=100)

    # Crop Exception Scenarios: Test coordinates is 1-list
    with pytest.raises(TypeError, match="Coordinates should be a list/tuple \\(y, x\\) of length 2."):
        v_trans.Crop(coordinates=[10], size=100)

    # Crop Exception Scenarios: Test no coordinates
    with pytest.raises(TypeError, match="missing a required argument: 'coordinates'"):
        v_trans.Crop(size=100)

    # Crop Exception Scenarios: Test size is float
    with pytest.raises(TypeError, match="Argument size with value 100.0 is not of type \\[<class "
                                         "'int'>, <class 'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        v_trans.Crop(coordinates=[10, 10], size=100.0)

    # Crop Exception Scenarios: Test size is 0
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        v_trans.Crop(coordinates=[10, 10], size=0)

    # Crop Exception Scenarios: Test size exceed 2147483647
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        v_trans.Crop(coordinates=[10, 10], size=[2147483648, 10])


def test_crop_exception_02():
    """
    Feature: Crop operation
    Description: Testing the Crop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Crop Exception Scenarios: Test size is True
    with pytest.raises(TypeError, match="Argument size\\[0\\] with value True is not of "
                                         "type \\(<class 'int'>,\\), but got <class 'bool'>."):
        v_trans.Crop(coordinates=[10, 10], size=[True, 10])

    # Crop Exception Scenarios: Test size is np
    with pytest.raises(TypeError, match="Argument size with value \\[20 20\\] is not of type \\[<class 'int'>, "
                                         "<class 'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        v_trans.Crop(coordinates=[10, 10], size=np.array([20, 20]))

    # Crop Exception Scenarios: Test size is 3-list
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        v_trans.Crop(coordinates=[10, 10], size=[10, 10, 10])

    # Crop Exception Scenarios: Test size is 1-tuple
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        v_trans.Crop(coordinates=[10, 10], size=(50,))

    # Crop Exception Scenarios: Test no size
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        v_trans.Crop(coordinates=[10, 10])

    # Crop Exception Scenarios: Test more Parameters
    with pytest.raises(TypeError, match="too many positional arguments"):
        v_trans.Crop([10, 10], (50, 50), 100)


if __name__ == "__main__":
    test_crop_pipeline(plot=False)
    test_crop_eager()
    test_crop_exception()
    test_crop_operation_01()
    test_crop_exception_01()
    test_crop_exception_02()
