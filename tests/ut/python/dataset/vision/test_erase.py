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
Testing Erase op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.transforms.transforms as t_trans
from mindspore import log as logger
from util import visualize_image, diff_mse


DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_erase_op(plot=False):
    """
    Feature: Erase op
    Description: Test Erase pipeline
    Expectation: Pass without error
    """
    logger.info("test_erase_pipeline")

    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    erase_op = vision.Erase(1, 1, 2, 4)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=erase_op, input_columns=["image"])

    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        erase_ms = item1["image"]
        original = item2["image"]
        erase_cv = cv2.rectangle(original, (1, 1), (4, 2), 0, -1)

        mse = diff_mse(erase_ms, erase_cv)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01

        if plot:
            visualize_image(erase_ms, erase_cv, mse)


def test_func_erase_eager():
    """
    Feature: Erase op
    Description: Test Erase in eager mode
    Expectation: Output is the same as expected output
    """
    image1 = np.random.randint(0, 255, (30, 30, 3), dtype=np.int32)

    out1 = vision.Erase(1, 1, 2, 4, 30)(image1)
    out2 = cv2.rectangle(image1, (1, 1), (4, 2), 30, -1)

    mse = diff_mse(out1, out2)
    logger.info("mse is {}".format(mse))
    assert mse < 0.01


def test_erase_invalid_input():
    """
    Feature: Erase op
    Description: Test operation with invalid input
    Expectation: Throw exception as expected
    """

    def test_invalid_input(test_name, top, left, height, width, value, inplace, error, error_msg):
        logger.info("Test Erase with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision.Erase(top, left, height, width, value, inplace)
        print(error_info)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid top parameter Value", 999999999999, 10, 10, 10, 0,
                       False, ValueError, "Input top is not within the required interval of [0, 2147483647].")
    test_invalid_input("invalid top parameter type", 10.5, 10, 10, 10, 0, False, TypeError,
                       "Argument top with value 10.5 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid left parameter Value", 10, 999999999999, 10, 10, 0,
                       False, ValueError, "Input left is not within the required interval of [0, 2147483647].")
    test_invalid_input("invalid left parameter type", 10, 10.5, 10, 10, 0, False, TypeError,
                       "Argument left with value 10.5 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid height parameter Value", 10, 10, 999999999999, 10, 0,
                       False, ValueError, "Input height is not within the required interval of [1, 2147483647].")
    test_invalid_input("invalid height parameter type", 10, 10, 10.5, 10, 0, False, TypeError,
                       "Argument height with value 10.5 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid width parameter Value", 10, 10, 10, 999999999999, 0,
                       False, ValueError, "Input width is not within the required interval of [1, 2147483647].")
    test_invalid_input("invalid width parameter type", 10, 10, 10, 10.5, 0, False, TypeError,
                       "Argument width with value 10.5 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid value parameter Value", 10, 10, 10, 10, 999999999999,
                       False, ValueError, "Input value[0] is not within the required interval of [0, 255].")
    test_invalid_input("invalid value parameter shape", 10, 10, 10, 10, (2, 3), False, TypeError,
                       "value should be a single integer/float or a 3-tuple.")
    test_invalid_input("invalid inplace parameter type as a single number", 10, 10, 10, 10, 0, 0, TypeError,
                       "Argument inplace with value 0 is not of type [<class 'bool'>], but got <class 'int'>.")


def test_erase_operation_01():
    """
    Feature: Erase operation
    Description: Testing the normal functionality of the Erase operator
    Expectation: The Output is equal to the expected output
    """
    # Erase Normal Scenario: input HWC Numpy uint8 format
    image1 = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    v = (30, 5, 100)
    _ = vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Normal Scenario: input HWC Numpy int32 format
    image1 = np.random.randint(0, 255, (30, 40, 3), dtype=np.int32)
    v = (30, 5, 100)
    _ = vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Normal Scenario: input HWC Numpy float64 format
    image1 = np.random.randn(30, 60, 3).astype(np.float64)
    v = (30, 5, 100)
    _ = vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Normal Scenario: no argument value
    image = np.random.randint(0, 255, (3, 6, 3)).astype(np.uint8)
    _ = vision.Erase(top=1, left=1, height=2, width=2, inplace=True)(image)

    # Erase Normal Scenario: no argument inplace
    image = np.random.randn(3, 6, 3).astype(np.uint8)
    v = (30, 5, 100)
    _ = vision.Erase(top=1, left=2, height=1, width=3, value=v)(image)

    # Erase Normal Scenario: test erase func pipeline
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    data1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    v = (30, 5, 100)
    transforms_1 = [
        vision.Decode(False),
        vision.Resize(3),
        vision.Erase(1, 1, 2, 2, v)
    ]
    transform_1 = t_trans.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])
    expect_out = [[[176, 166, 157], [185, 175, 166], [137, 146, 164]],
                  [[176, 165, 159], [30, 5, 100], [30, 5, 100]], [[12, 55, 84], [30, 5, 100], [30, 5, 100]]]
    for i in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out = i["image"]
    assert (expect_out == out).all()

    # Erase Normal Scenario: input HWC Numpy float64 format
    image1 = np.random.randn(30, 60, 3).astype(np.float32)
    v = (30, 5, 100)
    _ = vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Normal Scenario: input PIL format
    image1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "class1", "1_1.jpg")
    image = Image.open(image1)
    v = (30, 50, 100)
    _ = vision.Erase(1, 4, 3, 10, v)(image)
    image.close()


def test_erase_exception_01():
    """
    Feature: Erase operation
    Description: Testing the Erase Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Erase Exception Scenario: input CHW format
    image1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train", "class1", "1_1.jpg")
    with Image.open(image1) as image:
        image = vision.ToNumpy()(image)
        image = vision.HWC2CHW()(image)
        v = (30, 50, 100)
        with pytest.raises(RuntimeError, match="Erase: channel of input image should be 3, but got: .*"):
            vision.Erase(1, 1, 2, 30, v)(image)

    # Erase Exception Scenario: no argument top
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'top'"):
        vision.Erase(left=3, height=20, width=20, value=v, inplace=False)(image)

    # Erase Exception Scenario: no argument left
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'left'"):
        vision.Erase(top=1, height=20, width=20, value=v, inplace=False)(image)

    # Erase Exception Scenario: no argument height
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'height'"):
        vision.Erase(top=1, left=20, width=20, value=v, inplace=False)(image)

    # Erase Exception Scenario: no argument width
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'width'"):
        vision.Erase(top=1, left=20, height=20, value=v, inplace=False)(image)

    # Erase Exception Scenario: invalid top parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input top is not within the required interval of \\[0, 2147483647\\]."):
        vision.Erase(2147483648, 10, 10, 10, 0, False)(image)

    # Erase Exception Scenario: invalid top parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument top with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10.5, 10, 10, 10, 0, False)(image)

    # Erase Exception Scenario: invalid left parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input left is not within the required interval of \\[0, 2147483647\\]."):
        vision.Erase(10, 2147483648, 10, 10, 0, False)(image)

    # Erase Exception Scenario: invalid left parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument left with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10.5, 10, 10, 0, False)(image)

    # Erase Exception Scenario: invalid height parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input height is not within the required interval of \\[1, 2147483647\\]."):
        vision.Erase(10, 10, 2147483648, 10, 0, False)(image)

    # Erase Exception Scenario: invalid height parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument height with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10, 10.5, 10, 0, False)(image)

    # Erase Exception Scenario: invalid width parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input width is not within the required interval of \\[1, 2147483647\\]."):
        vision.Erase(10, 10, 10, 2147483648, 0, False)(image)

    # Erase Exception Scenario: invalid width parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument width with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10, 10, 10.5, 0, False)(image)


def test_erase_exception_02():
    """
    Feature: Erase operation
    Description: Testing the Erase Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Erase Exception Scenario: invalid value over 255
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input value\\[0\\] is not within the required interval of \\[0, 255\\]."):
        vision.Erase(10, 10, 10, 10, 256, False)(image)

    # Erase Exception Scenario:  nvalid value is (2, 3)
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="value should be a single integer/float or a 3-tuple."):
        vision.Erase(10, 10, 10, 10, (2, 3), False)(image)

    # Erase Exception Scenario: invalid value is [2, 3]
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument value with value \\[2, 3\\] is not of type \\[<class 'float'>,"
                                        " <class 'int'>, <class 'tuple'>\\], but got <class 'list'>"):
        vision.Erase(10, 10, 10, 10, [2, 3], False)(image)

    # Erase Exception Scenario: invalid inplace is int
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument inplace with value 0 is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'int'>."):
        vision.Erase(10, 10, 10, 10, 0, 0)(image)

    # Erase Exception Scenario: input 2-D Numpy
    image1 = np.random.randn(30, 3).astype(np.float32)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="input tensor is not in shape of <H,W,C>, but got rank: 2"):
        vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Exception Scenario: input 1-D Numpy
    image1 = np.random.randn(30).astype(np.float32)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="input tensor is not in shape of <H,W,C>, but got rank: 1"):
        vision.Erase(1, 4, 20, 30, v)(image1)

    # Erase Exception Scenario: input 4 channel Numpy
    image1 = np.random.randn(30, 60, 4).astype(np.float32)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="channel of input image should be 3"):
        vision.Erase(1, 4, 20, 30, v)(image1)


if __name__ == "__main__":
    test_erase_op(plot=True)
    test_func_erase_eager()
    test_erase_invalid_input()
    test_erase_operation_01()
    test_erase_exception_01()
    test_erase_exception_02()
