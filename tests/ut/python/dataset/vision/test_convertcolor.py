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
Testing ConvertColor op in DE
"""
import cv2
import numpy as np
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as c_vision
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.vision.utils as mode
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
    data_list.append(data_dir1)
    return data_list


def convert_color(ms_convert, cv_convert, plot=False):
    """
    ConvertColor with different mode.
    """
    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    convertcolor_op = c_vision.ConvertColor(ms_convert)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=convertcolor_op,
                            input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=[
        "image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        convertcolor_ms = data1["image"]
        original = data2["image"]
        convertcolor_cv = cv2.cvtColor(original, cv_convert)
        mse = diff_mse(convertcolor_ms, convertcolor_cv)
        logger.info("convertcolor_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        num_iter += 1
        if plot:
            visualize_image(original, convertcolor_ms, mse, convertcolor_cv)


def test_convertcolor_pipeline(plot=False):
    """
    Feature: ConvertColor op
    Description: Test ConvertColor op in pipeline mode
    Expectation: Passes the equality test
    """
    logger.info("test_convertcolor_pipeline")
    convert_color(mode.ConvertMode.COLOR_BGR2GRAY, cv2.COLOR_BGR2GRAY, plot)
    convert_color(mode.ConvertMode.COLOR_BGR2RGB, cv2.COLOR_BGR2RGB, plot)
    convert_color(mode.ConvertMode.COLOR_BGR2BGRA, cv2.COLOR_BGR2BGRA, plot)


def test_convertcolor_eager():
    """
    Feature: ConvertColor op
    Description: Test ConvertColor op in eager mode
    Expectation: Passes the equality test
    """
    logger.info("test_convertcolor")
    img = cv2.imread(IMAGE_FILE)

    img_ms = c_vision.ConvertColor(mode.ConvertMode.COLOR_BGR2GRAY)(img)
    img_expect = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mse = diff_mse(img_ms, img_expect)
    assert mse == 0


def test_convert_color_operation_01():
    """
    Feature: ConvertColor operation
    Description: Testing the normal functionality of the ConvertColor operator
    Expectation: The Output is equal to the expected output
    """
    # Convert BGR images to BGRA images
    dataset = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA)
    image_folder_dataset = dataset.map(operations=convert_op, input_columns=["image"])
    for _ in image_folder_dataset.create_dict_iterator(output_numpy=True):
        pass

    # Convert BGR images to BGRA images
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2BGRA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGB images to RGBA images
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2RGBA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGRA images to BGR images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2BGR)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGBA images to RGB images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2RGB)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGR images to RGBA images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGBA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGB images to BGRA images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGRA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGBA images to BGR images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGR)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGRA images to RGB images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGB)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGR images to RGB images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2RGB)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGB images to BGR images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGRA images to RGBA images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2RGBA)
    out = convert_op(image)
    assert (gray_image == out).all()


def test_convert_color_operation_02():
    """
    Feature: ConvertColor operation
    Description: Testing the normal functionality of the ConvertColor operator
    Expectation: The Output is equal to the expected output
    """
    # Convert RGBA images to BGRA images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2BGRA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGR images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGR2GRAY)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGB images to GRAY images
    image = np.random.randint(0, 255, (3, 3, 3)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2GRAY)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert GRAY images to BGR images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGR)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert GRAY images to RGB images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGB)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert GRAY images to BGRA images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2BGRA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert GRAY images to RGBA images
    image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_GRAY2RGBA)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert BGRA images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_BGRA2GRAY)
    out = convert_op(image)
    assert (gray_image == out).all()

    # Convert RGBA images to GRAY images
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGBA2GRAY)
    out = convert_op(image)
    assert (gray_image == out).all()


def test_convert_color_exception_01():
    """
    Feature: ConvertColor operation
    Description: Testing the ConvertColor Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ConvertColor Exception Scenario: No convert_mode parameter value provided
    with pytest.raises(TypeError, match=r"missing a required argument: 'convert_mode'"):
        v_trans.ConvertColor()

    # ConvertColor Exception Scenario: convert_mode parameter passed as int
    with pytest.raises(TypeError, match=r"Argument convert_mode with value 1 is not of type \[<enum 'ConvertMode'>\],"
                                        r" but got <class 'int'>."):
        v_trans.ConvertColor(1)

    # ConvertColor Exception Scenario: convert_mode parameter passed as str
    with pytest.raises(TypeError, match=r"Argument convert_mode with value a is not of type \[<enum 'ConvertMode'>\],"
                                        r" but got <class 'str'>."):
        v_trans.ConvertColor('a')

    # ConvertColor Exception Scenario: The parameter convert_mode is passed as True
    with pytest.raises(TypeError,
                       match=r"Argument convert_mode with value True is not of type \[<enum 'ConvertMode'>\],"
                             r" but got <class 'bool'>."):
        v_trans.ConvertColor(True)

    # ConvertColor Exception Scenario: The parameter convert_mode is passed as None
    with pytest.raises(TypeError,
                       match=r"Argument convert_mode with value None is not of type \[<enum 'ConvertMode'>\], "
                             "but got <class 'NoneType'>."):
        v_trans.ConvertColor(None)

    # ConvertColor Exception Scenario: Input is one-dimensional data
    image = np.random.randint(0, 255, (3)).astype(np.uint8)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
    with pytest.raises(RuntimeError, match=r"input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 1."
                                           " You may need to perform Decode first."):
        convert_op(image)

    # ConvertColor Exception Scenario: Input is four-dimensional data
    image = np.random.randint(0, 255, (10, 10, 10, 3)).astype(np.uint8)
    convert_op = v_trans.ConvertColor(mode.ConvertMode.COLOR_RGB2BGR)
    with pytest.raises(RuntimeError, match=r"input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 4"):
        convert_op(image)


if __name__ == "__main__":
    test_convertcolor_pipeline(plot=False)
    test_convertcolor_eager()
    test_convert_color_operation_01()
    test_convert_color_operation_02()
    test_convert_color_exception_01()
