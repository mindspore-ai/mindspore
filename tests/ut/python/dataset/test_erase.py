# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import visualize_image, diff_mse


DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


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


if __name__ == "__main__":
    test_erase_op(plot=True)
    test_func_erase_eager()
    test_erase_invalid_input()
