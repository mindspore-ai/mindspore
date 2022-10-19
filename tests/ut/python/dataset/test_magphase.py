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
Testing Magphase Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger


def test_magphase_pipeline():
    """
    Feature: Magphase
    Description: Test Magphase in pipeline mode
    Expectation: Output is equal to the expected output
    """
    logger.info("Test Magphase pipeline.")

    data1 = [[[3.0, -4.0], [-5.0, 12.0]]]
    expected = [5, 13, -0.927295, 1.965587]
    dataset = ds.NumpySlicesDataset(data1, column_names=["col1"], shuffle=False)
    magphase_window = audio.Magphase(power=1.0)
    dataset = dataset.map(operations=magphase_window, input_columns=["col1"],
                          output_columns=["mag", "phase"])
    for data1, data2 in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert abs(data1[0] - expected[0]) < 0.00001
        assert abs(data1[1] - expected[1]) < 0.00001
        assert abs(data2[0] - expected[2]) < 0.00001
        assert abs(data2[1] - expected[3]) < 0.00001

    logger.info("Finish testing Magphase.")


def test_magphase_eager():
    """
    Feature: Magphase
    Description: Test Magphase in eager mode
    Expectation: Output is equal to the expected output
    """
    logger.info("Test Magphase eager.")

    input_number = np.array([41, 67, 34, 0, 69, 24, 78, 58]).reshape((2, 2, 2)).astype("double")
    mag = np.array([78.54934755, 34., 73.05477397, 97.20082304]).reshape((2, 2)).astype("double")
    phase = np.array([1.02164342, 0, 0.33473684, 0.63938591]).reshape((2, 2)).astype("double")
    magphase_window = audio.Magphase()
    data1, data2 = magphase_window(input_number)
    assert (abs(data1 - mag) < 0.00001).all()
    assert (abs(data2 - phase) < 0.00001).all()

    logger.info("Finish testing Magphase.")


def test_magphase_exception():
    """
    Feature: Magphase
    Description: Test Magphase with invalid input
    Expectation: Correct error is raised as expected
    """
    logger.info("Test Magphase not callable.")

    try:
        input_number = np.array([1, 2, 3, 4]).reshape(4,).astype("double")
        magphase_window = audio.Magphase(power=2.0)
        _ = magphase_window(input_number)
    except RuntimeError as error:
        logger.info("Got an exception in Magphase: {}".format(str(error)))
        assert "the shape of input tensor does not match the requirement of operator" in str(error)
    try:
        input_number = np.array([1, 2, 3, 4]).reshape(1, 4).astype("double")
        magphase_window = audio.Magphase(power=2.0)
        _ = magphase_window(input_number)
    except RuntimeError as error:
        logger.info("Got an exception in Magphase: {}".format(str(error)))
        assert "the shape of input tensor does not match the requirement of operator" in str(error)
    try:
        input_number = np.array(['test', 'test']).reshape(1, 2)
        magphase_window = audio.Magphase(power=2.0)
        _ = magphase_window(input_number)
    except RuntimeError as error:
        logger.info("Got an exception in Magphase: {}".format(str(error)))
        assert "the data type of input tensor does not match the requirement of operator" in str(error)
    try:
        input_number = np.array([1, 2, 3, 4]).reshape(2, 2).astype("double")
        magphase_window = audio.Magphase(power=-1.0)
        _ = magphase_window(input_number)
    except ValueError as error:
        logger.info("Got an exception in Magphase: {}".format(str(error)))
        assert "Input power is not within the required interval of [0, 16777216]." in str(error)

    logger.info("Finish testing Magphase.")


if __name__ == "__main__":
    test_magphase_pipeline()
    test_magphase_eager()
    test_magphase_exception()
